"""Light per-datacube statistics via cumsum-based sliding windows.

This is the "pass 1" stats step in the 3-step sampling pipeline:
filter-nan → stats-light → stats-heavy → sample. It extends the cumsum
trick from `filter_nan` — whose original purpose was counting NaNs per
candidate window in O(1) per window amortized — to also produce `sum`,
`mean`, and `frac_wet` for the same candidate set, essentially for free
on top of the I/O. Heavy stats that cannot be computed with cumsum (max,
quantiles) are left to `stats-heavy`.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from multiprocessing import Pool
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
import zarr
from loguru import logger
from tqdm import tqdm

from ..metadata import encode_stats_metadata
from ..units import default_wet_threshold, detect_data_kind

if TYPE_CHECKING:
    from numpy.typing import NDArray


STAT_COLUMNS = ("t", "x", "y", "nan_count", "sum", "mean", "frac_wet")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add stats-light specific arguments to the parser."""
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr dataset.")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output Parquet file path. If not specified, auto-generated from parameters.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--time-depth", type=int, default=24, help="Time depth of datacubes.")
    parser.add_argument("--width", type=int, default=256, help="Spatial width of datacubes.")
    parser.add_argument("--height", type=int, default=256, help="Spatial height of datacubes.")
    parser.add_argument("--step-t", type=int, default=3, help="Time step between datacubes.")
    parser.add_argument("--step-x", type=int, default=16, help="X step between datacubes.")
    parser.add_argument("--step-y", type=int, default=16, help="Y step between datacubes.")
    parser.add_argument(
        "--max-nan", type=int, default=10000,
        help="Maximum NaN count per datacube (hard filter on output).",
    )
    parser.add_argument(
        "--wet-threshold", type=float, default=None,
        help="Wet-pixel threshold in the same units as the data var. "
             "If omitted, auto-detected: 0.1 mm/h for rain rate, 7 dBZ for reflectivity.",
    )
    parser.add_argument(
        "--data-kind", choices=["rainrate", "reflectivity"], default=None,
        help="Override the data-kind auto-detection from zarr attrs. "
             "Needed only if the variable has non-standard attributes.",
    )
    parser.add_argument(
        "--time-step-minutes", type=int, default=5,
        help="Expected time step between consecutive frames in minutes.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--data-var", type=str, default="RR", help="Name of the zarr data variable.")
    parser.add_argument("--time-var", type=str, default="time", help="Name of the zarr time variable.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")


def _dim_cumsum_window(
    arr: NDArray, dim: int, delta: int, dim_len: int,
) -> NDArray:
    """3D sliding-window sum along one axis via a prefix-sum difference.

    Generic version of `filter_nan._dim_nan_count`: works for any numeric
    dtype (int for counting, float for summing). For every window of size
    `delta` along `dim`, returns the sum of the elements inside that
    window. O(n) per axis regardless of `delta`.
    """
    # Use int32 (not the numpy default int64) for int inputs to halve memory.
    # A window count is bounded by Dt*w*h <= 2^31, safe in int32.
    cumsum = np.cumsum(arr, axis=dim, dtype=arr.dtype if arr.dtype.kind == "f" else np.int32)

    pad_width = [(1, 0) if i == dim else (0, 0) for i in range(arr.ndim)]
    padded = np.pad(cumsum, pad_width=pad_width, mode="constant", constant_values=0)

    slices_start = [slice(dim_len - delta) if i == dim else slice(None) for i in range(arr.ndim)]
    slices_end = [slice(delta, dim_len) if i == dim else slice(None) for i in range(arr.ndim)]

    return padded[tuple(slices_end)] - padded[tuple(slices_start)]


def _datacube_window_sum(
    arr: NDArray, deltas: tuple[int, int, int], dim_lengths: tuple[int, int, int],
) -> NDArray:
    """3-axis cumsum-window sum over a (T, X, Y) array."""
    s = _dim_cumsum_window(arr, dim=0, delta=deltas[0], dim_len=dim_lengths[0])
    s = _dim_cumsum_window(s,   dim=1, delta=deltas[1], dim_len=dim_lengths[1])
    s = _dim_cumsum_window(s,   dim=2, delta=deltas[2], dim_len=dim_lengths[2])
    return s


def _process_chunk(
    time_range: tuple[int, int],
    t_start_idx: int,
    data: zarr.Array,
    max_nan: int,
    wet_threshold: float,
    deltas: tuple[int, int, int],
    steps: tuple[int, int, int],
    valid_starts_gap: NDArray[np.int32],
) -> dict[str, NDArray]:
    """Compute cumsum-based stats for all candidate windows in a chunk.

    Memory-conscious: the three cumsum-window reductions (nan_count, sum,
    wet_count) are computed **sequentially**, not simultaneously — each
    one is produced, indexed at the survivor positions, then freed before
    the next one starts. This keeps peak resident memory per worker close
    to one windowed array instead of three.
    """
    start_t, end_t = time_range
    chunk = data[start_t + t_start_idx : end_t + t_start_idx, :, :].astype(np.float32, copy=False)
    dim_lengths = chunk.shape
    Dt, w, h = deltas
    total_px = Dt * w * h

    # Build the NaN mask once; we'll use it (a) to drive nan_count, and
    # (b) as the mask to zero-fill `chunk` in place for the sum pass.
    nan_mask = np.isnan(chunk)  # bool, 1 byte/element

    # --- Pass A: nan_count window ------------------------------------------------
    nan_count_win = _datacube_window_sum(
        nan_mask.astype(np.int16), deltas, dim_lengths,
    )  # int32 output from _dim_cumsum_window

    # Survivor set: fixed for the rest of this function.
    valid_mask = nan_count_win <= max_nan
    idx_t_rel, idx_x, idx_y = np.where(valid_mask)
    del valid_mask
    idx_t_rel = idx_t_rel.astype(np.int32)
    idx_x = idx_x.astype(np.int32)
    idx_y = idx_y.astype(np.int32)

    # Stride + time-continuity filter on absolute indices.
    idx_t_abs_rel = idx_t_rel + start_t
    time_mask = np.isin(idx_t_abs_rel, valid_starts_gap)
    idx_t_abs = idx_t_abs_rel + t_start_idx  # into the full-dataset axis
    stride_mask = (
        (idx_t_abs % steps[0] == 0)
        & (idx_x % steps[1] == 0)
        & (idx_y % steps[2] == 0)
    )
    keep = time_mask & stride_mask
    del time_mask, stride_mask

    idx_t_rel = idx_t_rel[keep]
    idx_x = idx_x[keep]
    idx_y = idx_y[keep]
    idx_t_abs = idx_t_abs[keep]
    del idx_t_abs_rel, keep

    nan_count = nan_count_win[idx_t_rel, idx_x, idx_y]
    del nan_count_win

    # --- Pass B: sum window -------------------------------------------------------
    # In-place zero-fill of the chunk before the cumsum pass. This avoids
    # allocating a separate `chunk_filled` array (would cost another ~640 MB
    # per worker at (95, 1400, 1200) float32).
    chunk[nan_mask] = 0.0
    sum_win = _datacube_window_sum(chunk, deltas, dim_lengths)
    sum_vals = sum_win[idx_t_rel, idx_x, idx_y]
    del sum_win

    # --- Pass C: wet_count window -------------------------------------------------
    # `chunk` is now zero where it was NaN, so a simple `> wet_threshold`
    # check is equivalent to (value > threshold AND not NaN).
    wet_mask_i = (chunk > wet_threshold).astype(np.int16)
    del chunk, nan_mask
    wet_count_win = _datacube_window_sum(wet_mask_i, deltas, dim_lengths)
    del wet_mask_i
    wet_count = wet_count_win[idx_t_rel, idx_x, idx_y]
    del wet_count_win

    # Derived stats.
    valid_count = total_px - nan_count
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_vals = np.where(valid_count > 0, sum_vals / valid_count, np.nan).astype(np.float32)
    frac_wet = wet_count.astype(np.float32) / total_px

    return {
        "t": idx_t_abs,
        "x": idx_x,
        "y": idx_y,
        "nan_count": nan_count,
        "sum": sum_vals.astype(np.float32),
        "mean": mean_vals,
        "frac_wet": frac_wet,
    }


def _parquet_writer(
    output_queue: Queue,
    filename: str,
    schema_metadata: dict[bytes, bytes],
) -> None:
    """Drain the queue and stream rows to a Parquet file.

    Each queue item is the dict returned by `_process_chunk`. We buffer
    into Arrow RecordBatches and append to a single ParquetWriter so the
    on-disk file stays a single self-contained parquet. `schema_metadata`
    carries the mlcast sampling parameters (Dt, w, h, stride, etc.) so
    downstream commands don't need to parse the filename.
    """
    schema = pa.schema(
        [
            ("t", pa.int32()),
            ("x", pa.int32()),
            ("y", pa.int32()),
            ("nan_count", pa.int32()),
            ("sum", pa.float32()),
            ("mean", pa.float32()),
            ("frac_wet", pa.float32()),
        ],
        metadata=schema_metadata,
    )
    writer = pq.ParquetWriter(filename, schema, compression="zstd")
    total_rows = 0
    try:
        while True:
            item = output_queue.get()
            if item is None:
                break
            if item["t"].size == 0:
                continue
            batch = pa.record_batch(
                [pa.array(item[c]) for c in STAT_COLUMNS],
                schema=schema,
            )
            writer.write_batch(batch)
            total_rows += batch.num_rows
    finally:
        writer.close()
    logger.info(f"Wrote {total_rows} rows to {filename}")


def run(args: argparse.Namespace) -> int:
    """Execute the stats-light command."""
    Dt = args.time_depth
    w = args.width
    h = args.height
    step_T = args.step_t
    step_X = args.step_x
    step_Y = args.step_y
    max_nan = args.max_nan
    n_workers = args.workers
    time_chunk_size = 3 * Dt

    logger.info(f"Opening Zarr dataset: {args.zarr_path}")
    try:
        zg = zarr.open(args.zarr_path, mode="r")
        data = zg[args.data_var]
        ds = xr.open_zarr(args.zarr_path)
        time_array_full = pd.DatetimeIndex(ds[args.time_var].values)
        logger.info(f"Full dataset shape: T={data.shape[0]}, X={data.shape[1]}, Y={data.shape[2]}")
        logger.info(f"Time range: {time_array_full[0]} to {time_array_full[-1]}")
        var_attrs = dict(ds[args.data_var].attrs)
    except Exception as e:
        logger.error(f"Error loading Zarr dataset: {e}")
        return 1

    # Detect or override the data kind, then resolve the wet-pixel threshold.
    if args.data_kind is not None:
        data_kind = args.data_kind
        logger.info(f"Data kind overridden via --data-kind: {data_kind}")
    else:
        try:
            data_kind = detect_data_kind(var_attrs)
        except ValueError as e:
            logger.error(str(e))
            return 1
        logger.info(
            f"Detected data kind: {data_kind} "
            f"(standard_name={var_attrs.get('standard_name')!r}, "
            f"units={var_attrs.get('units')!r})"
        )

    wet_threshold = (
        args.wet_threshold if args.wet_threshold is not None
        else default_wet_threshold(data_kind)
    )
    units_str = var_attrs.get("units", "?")
    logger.info(f"Wet-pixel threshold: {wet_threshold} {units_str}")

    start_date = pd.to_datetime(args.start_date) if args.start_date else time_array_full[0]
    end_date = pd.to_datetime(args.end_date) if args.end_date else time_array_full[-1]

    mask = (time_array_full >= start_date) & (time_array_full <= end_date)
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        logger.error(f"No data found between {start_date} and {end_date}")
        return 1

    t_start_idx = valid_indices[0]
    t_end_idx = valid_indices[-1] + 1
    size_T = t_end_idx - t_start_idx
    size_X = data.shape[1]
    size_Y = data.shape[2]
    time_array = time_array_full[t_start_idx:t_end_idx]

    logger.info(f"Filtered dataset shape: T={size_T}, X={size_X}, Y={size_Y}")
    logger.info(f"Filtered time range: {time_array[0]} to {time_array[-1]}")
    max_t = size_T - Dt + 1

    logger.info("Checking time continuity...")
    expected_step = pd.Timedelta(minutes=args.time_step_minutes)
    time_diffs = time_array[1:] - time_array[:-1]
    gaps = (time_diffs != expected_step).astype(int)
    window_sum = np.convolve(gaps, np.ones(Dt - 1, dtype=int), mode="valid")
    valid_starts_gap = np.where(window_sum == 0)[0]
    logger.info(f"Found {len(valid_starts_gap)} valid time starts without gaps")

    # Peak memory per worker ~= chunk (float32) + cumsum working set + nan_mask (bool).
    # The three cumsum reductions run sequentially, so only one window array is
    # alive at a time. ~2x filter_nan in practice.
    chunk_bytes = (time_chunk_size + Dt - 1) * size_X * size_Y * 4
    per_chunk_gb = 2 * chunk_bytes / (1024 ** 3)
    logger.info(f"Estimated memory per chunk: {per_chunk_gb:.2f} GB (pipelined cumsums)")
    logger.info(f"Estimated total memory ({n_workers} workers): {per_chunk_gb * n_workers:.2f} GB")

    t_starts = np.arange(0, max_t, time_chunk_size)
    t_ends = np.minimum(t_starts + time_chunk_size + Dt - 1, size_T)
    t_pairs = np.stack((t_starts, t_ends), axis=1)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    if args.output:
        output_file = args.output
    else:
        output_file = (
            f"stats_light_{start_str}-{end_str}_{Dt}x{w}x{h}"
            f"_{step_T}x{step_X}x{step_Y}_{max_nan}.parquet"
        )
    if os.path.exists(output_file) and not args.overwrite:
        logger.error(f"File {output_file} already exists. Use --overwrite to replace.")
        return 1
    logger.info(f"Output file: {output_file}")

    process_chunk_partial = partial(
        _process_chunk,
        t_start_idx=t_start_idx,
        data=data,
        max_nan=max_nan,
        wet_threshold=wet_threshold,
        deltas=(Dt, w, h),
        steps=(step_T, step_X, step_Y),
        valid_starts_gap=valid_starts_gap,
    )

    schema_metadata = encode_stats_metadata(
        {
            "zarr_path": args.zarr_path,
            "data_var": args.data_var,
            "time_var": args.time_var,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "time_step_minutes": args.time_step_minutes,
            "time_depth": Dt,
            "width": w,
            "height": h,
            "step_t": step_T,
            "step_x": step_X,
            "step_y": step_Y,
            "max_nan": max_nan,
            "wet_threshold": wet_threshold,
            "data_kind": data_kind,
            "units": var_attrs.get("units"),
        }
    )

    output_queue: Queue = Queue(maxsize=100)
    writer_thread = Thread(
        target=_parquet_writer, args=(output_queue, output_file, schema_metadata)
    )
    writer_thread.daemon = False
    writer_thread.start()

    with Pool(n_workers) as pool:
        for hits in tqdm(
            pool.imap(process_chunk_partial, t_pairs, chunksize=1),
            total=len(t_starts),
            desc="Processing time chunks",
        ):
            output_queue.put(hits)

    output_queue.put(None)
    writer_thread.join()

    logger.success("stats-light completed successfully")
    return 0
