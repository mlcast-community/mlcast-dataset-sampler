"""Filter datacubes based on NaN count.

This module processes a Zarr dataset and identifies valid datacube coordinates
where the number of NaN values is below a specified threshold.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import zarr
from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from numpy.typing import NDArray


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add filter-nan specific arguments to the parser."""
    parser.add_argument(
        "zarr_path",
        type=str,
        help="Path to the Zarr dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV file path. If not specified, auto-generated from parameters.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for filtering (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for filtering (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--time-depth",
        type=int,
        default=24,
        help="Time depth of datacubes (default: 24).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Spatial width of datacubes (default: 256).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Spatial height of datacubes (default: 256).",
    )
    parser.add_argument(
        "--step-t",
        type=int,
        default=3,
        help="Time step between datacubes (default: 3).",
    )
    parser.add_argument(
        "--step-x",
        type=int,
        default=16,
        help="X step between datacubes (default: 16).",
    )
    parser.add_argument(
        "--step-y",
        type=int,
        default=16,
        help="Y step between datacubes (default: 16).",
    )
    parser.add_argument(
        "--max-nan",
        type=int,
        default=10000,
        help="Maximum NaN count per datacube (default: 10000).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--data-var",
        type=str,
        default="RR",
        help="Name of the data variable in the Zarr dataset (default: RR).",
    )
    parser.add_argument(
        "--time-var",
        type=str,
        default="time",
        help="Name of the time variable in the Zarr dataset (default: time).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )


def _dim_nan_count(
    mask: NDArray[np.int16], dim: int, delta: int, dim_len: int
) -> NDArray[np.int32]:
    """Compute the number of NaN in each window along a dimension.

    Args:
        mask: Binary array indicating NaN positions.
        dim: Dimension along which to compute.
        delta: Window size along dimension.
        dim_len: Length of the dimension.

    Returns:
        Array of NaN counts for each window position.
    """
    cumsum = np.cumsum(mask, axis=dim, dtype=np.int32)

    # Pad with zeros at the start along 'dim'
    pad_width = [(1, 0) if i == dim else (0, 0) for i in range(3)]
    padded_cumsum = np.pad(cumsum, pad_width=pad_width, mode="constant", constant_values=0)

    # Rolling window difference
    slices_start = [slice(dim_len - delta) if i == dim else slice(None) for i in range(3)]
    slices_end = [slice(delta, dim_len) if i == dim else slice(None) for i in range(3)]

    return padded_cumsum[tuple(slices_end)] - padded_cumsum[tuple(slices_start)]


def _datacube_nan_count(
    chunk: NDArray, deltas: tuple[int, int, int], dim_lengths: tuple[int, int, int]
) -> NDArray[np.int32]:
    """Compute the number of NaN in each datacube within a chunk.

    Args:
        chunk: Data chunk of shape (T, X, Y).
        deltas: Window sizes (Dt, w, h).
        dim_lengths: Chunk dimensions (T, X, Y).

    Returns:
        Array of NaN counts for each possible datacube position.
    """
    nan_mask = np.isnan(chunk).astype(np.int16)

    # Number of NaN along time
    nans_t = _dim_nan_count(nan_mask, dim=0, delta=deltas[0], dim_len=dim_lengths[0])

    # Number of NaN along X x T
    nans_xt = _dim_nan_count(nans_t, dim=1, delta=deltas[1], dim_len=dim_lengths[1])

    # Number of NaN in the datacube (Y x X x T)
    return _dim_nan_count(nans_xt, dim=2, delta=deltas[2], dim_len=dim_lengths[2])


def _process_chunk(
    time_range: tuple[int, int],
    t_start_idx: int,
    data: zarr.Array,
    max_nan: int,
    deltas: tuple[int, int, int],
    steps: tuple[int, int, int],
    valid_starts_gap: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Process a single time chunk and return valid datacube indices.

    Args:
        time_range: Start and end indices of the chunk.
        t_start_idx: Index offset corresponding to start_date.
        data: Zarr array.
        max_nan: Maximum number of NaN in each datacube.
        deltas: Datacube dimensions (Dt, w, h).
        steps: Step sizes (step_T, step_X, step_Y).
        valid_starts_gap: Valid starting time indices without gaps.

    Returns:
        Tuple of (t_indices, x_indices, y_indices).
    """
    start_t, end_t = time_range

    # Load chunk from Zarr (T, X, Y)
    chunk = data[start_t + t_start_idx : end_t + t_start_idx, :, :]
    dim_lengths = chunk.shape

    # Compute NaN counts
    nans_cube_chunk = _datacube_nan_count(chunk, deltas, dim_lengths)
    del chunk

    # Apply threshold mask
    valid_mask = nans_cube_chunk <= max_nan
    del nans_cube_chunk

    # Get indices (relative to chunk)
    idx_t_rel, idx_x, idx_y = np.where(valid_mask)
    del valid_mask

    # Cast to int32
    idx_t_rel = idx_t_rel.astype(np.int32)
    idx_x = idx_x.astype(np.int32)
    idx_y = idx_y.astype(np.int32)

    # Convert relative time indices
    idx_t = idx_t_rel + start_t

    # Keep only time indices in valid_starts_gap
    time_mask = np.isin(idx_t, valid_starts_gap)
    idx_t = idx_t[time_mask] + t_start_idx  # convert to absolute index
    idx_x = idx_x[time_mask]
    idx_y = idx_y[time_mask]

    # Filter by step size
    stride_mask = (idx_t % steps[0] == 0) & (idx_x % steps[1] == 0) & (idx_y % steps[2] == 0)
    idx_t = idx_t[stride_mask]
    idx_x = idx_x[stride_mask]
    idx_y = idx_y[stride_mask]

    return idx_t, idx_x, idx_y


def _file_writer(output_queue: Queue, filename: str, batch_size: int = 1000) -> None:
    """Write results to file from queue in a dedicated thread."""
    with open(filename, "w") as f:
        f.write("t,x,y\n")
        batch = []

        while True:
            item = output_queue.get()

            if item is None:  # Sentinel to stop
                for t, x, y in batch:
                    f.write(f"{t},{x},{y}\n")
                break

            batch.extend(zip(*item))

            if len(batch) >= batch_size:
                for t, x, y in batch:
                    f.write(f"{t},{x},{y}\n")
                f.flush()
                batch = []

    logger.info(f"Results saved to {filename}")


def run(args: argparse.Namespace) -> int:
    """Execute the filter-nan command."""
    # Parameters
    Dt = args.time_depth
    w = args.width
    h = args.height
    step_T = args.step_t
    step_X = args.step_x
    step_Y = args.step_y
    max_nan = args.max_nan
    n_workers = args.workers
    time_chunk_size = 3 * Dt

    # Load dataset
    logger.info(f"Opening Zarr dataset: {args.zarr_path}")
    try:
        zg = zarr.open(args.zarr_path, mode="r")
        data = zg[args.data_var]
        time_array_full = pd.to_datetime(zg[args.time_var][:])

        logger.info(f"Full dataset shape: T={data.shape[0]}, X={data.shape[1]}, Y={data.shape[2]}")
        logger.info(f"Time range: {time_array_full[0]} to {time_array_full[-1]}")
    except Exception as e:
        logger.error(f"Error loading Zarr dataset: {e}")
        return 1

    # Filter dates
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

    # Check time continuity
    logger.info("Checking time continuity...")
    expected_step = pd.Timedelta("00:05:00")
    time_diffs = time_array[1:] - time_array[:-1]
    gaps = (time_diffs != expected_step).astype(int)

    window_sum = np.convolve(gaps, np.ones(Dt - 1, dtype=int), mode="valid")
    valid_starts_gap = np.where(window_sum == 0)[0]
    logger.info(f"Found {len(valid_starts_gap)} valid time starts without gaps")

    # Memory estimation
    estimated_chunk_memory_gb = (time_chunk_size * size_X * size_Y * 4) / (1024**3)
    logger.info(f"Estimated memory per chunk: {estimated_chunk_memory_gb:.2f} GB")
    logger.info(f"Estimated total memory: {(estimated_chunk_memory_gb * n_workers):.2f} GB")

    # Prepare time chunks
    t_starts = np.arange(0, max_t, time_chunk_size)
    t_ends = np.minimum(t_starts + time_chunk_size + Dt - 1, size_T)
    t_pairs = np.stack((t_starts, t_ends), axis=1)

    # Output file
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    if args.output:
        output_file = args.output
    else:
        output_file = f"valid_datacubes_{start_str}-{end_str}_{Dt}x{w}x{h}_{step_T}x{step_X}x{step_Y}_{max_nan}.csv"

    if os.path.exists(output_file) and not args.overwrite:
        logger.error(f"File {output_file} already exists. Use --overwrite to replace.")
        return 1

    logger.info(f"Output file: {output_file}")

    # Create partial function
    process_chunk_partial = partial(
        _process_chunk,
        t_start_idx=t_start_idx,
        data=data,
        max_nan=max_nan,
        deltas=(Dt, w, h),
        steps=(step_T, step_X, step_Y),
        valid_starts_gap=valid_starts_gap,
    )

    # Start writer thread
    output_queue: Queue = Queue(maxsize=100)
    writer_thread = Thread(target=_file_writer, args=(output_queue, output_file, 1000))
    writer_thread.daemon = False
    writer_thread.start()

    # Process chunks in parallel
    with Pool(n_workers) as pool:
        for hits in tqdm(
            pool.imap(process_chunk_partial, t_pairs, chunksize=1),
            total=len(t_starts),
            desc="Processing time chunks",
        ):
            output_queue.put(hits)

    # Signal writer thread to stop
    output_queue.put(None)
    writer_thread.join()

    logger.success("Filter-nan completed successfully")
    return 0
