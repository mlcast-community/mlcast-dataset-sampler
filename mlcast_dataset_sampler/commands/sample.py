"""Importance sampling of valid datacubes.

This module performs importance sampling on pre-filtered datacubes,
selecting samples based on rain rate intensity to balance the training dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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
    """Add sample-specific arguments to the parser."""
    parser.add_argument(
        "zarr_path",
        type=str,
        help="Path to the Zarr dataset.",
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV with valid datacube coordinates (from filter-nan).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV file path. If not specified, auto-generated from parameters.",
    )
    parser.add_argument(
        "--q-min",
        type=float,
        default=1e-4,
        help="Minimum selection probability (default: 1e-4).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Denominator in the exponential transform (default: 1.0).",
    )
    parser.add_argument(
        "--mean-weight",
        type=float,
        default=0.1,
        help="Factor weighting the mean rescaled rain rate (default: 0.1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of sampling trials per datacube (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    parser.add_argument(
        "--data-var",
        type=str,
        default="RR",
        help="Name of the data variable in the Zarr dataset (default: RR).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    parser.add_argument(
        "--time-chunk-size",
        type=int,
        default=None,
        help="Number of timesteps per processing chunk (default: 3 * time_depth).",
    )


def _parse_csv_filename(csv_path: str) -> dict:
    """Parse parameters from CSV filename.

    Expected format: valid_datacubes_YYYY-MM-DD-YYYY-MM-DD_DtxWxH_stepTxstepXxstepY_maxNan.csv
    """
    basename = os.path.basename(csv_path)
    name_arr = basename.split("_")

    dates = name_arr[2]
    date_parts = dates.split("-")
    start_date = "-".join(date_parts[0:3])
    end_date = "-".join(date_parts[3:6])

    Dt, w, h = name_arr[3].split("x")
    step_T, step_X, step_Y = name_arr[4].split("x")
    max_nan = name_arr[5].replace(".csv", "")

    return {
        "start_date": start_date,
        "end_date": end_date,
        "Dt": int(Dt),
        "w": int(w),
        "h": int(h),
        "step_T": int(step_T),
        "step_X": int(step_X),
        "step_Y": int(step_Y),
        "max_nan": int(max_nan),
    }


def _process_time_chunk(args: tuple) -> list[tuple[int, int, int]]:
    """Process a single time chunk: load zarr slice, evaluate all coords.

    Each worker opens its own zarr handle, reads one time chunk, and
    processes all coordinates that fall within that chunk.

    Args:
        args: Tuple of (zarr_path, data_var, chunk_t_start, data_t_end,
              coords, Dt, w, h, n_samples, seed, scale, q_min, mean_weight).

    Returns:
        List of accepted (t, x, y) tuples.
    """
    (
        zarr_path, data_var, chunk_t_start, data_t_end,
        coords, Dt, w, h, n_samples, seed, scale, q_min, mean_weight,
    ) = args

    # Each worker opens its own zarr handle (no pickle of large arrays)
    zg = zarr.open(zarr_path, mode="r")
    data = zg[data_var]

    # Single zarr read for the entire time chunk + overlap, done in-place
    # on float32 to avoid doubling peak memory when we apply the transform.
    chunk_data = np.asarray(data[chunk_t_start:data_t_end, :, :], dtype=np.float32)
    np.divide(chunk_data, -scale, out=chunk_data)
    np.exp(chunk_data, out=chunk_data)
    np.subtract(1.0, chunk_data, out=chunk_data)
    transformed_chunk = chunk_data  # alias for readability

    # One RNG per chunk (not per-coord). Offsetting by chunk_t_start keeps
    # runs reproducible while still producing independent streams per chunk,
    # so different coords actually get different draws.
    rng = np.random.default_rng(
        None if seed is None else seed + chunk_t_start
    )
    accepted = []

    for row in coords:
        it, ix, iy = int(row[0]), int(row[1]), int(row[2])
        t_local = it - chunk_t_start

        window = transformed_chunk[t_local : t_local + Dt, ix : ix + w, iy : iy + h]
        q = min(1.0, q_min + mean_weight * float(np.nanmean(window)))

        accepted_count = int(np.sum(rng.random(n_samples) <= q))
        if accepted_count:
            accepted.extend([(it, ix, iy)] * accepted_count)

    return accepted


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

            batch.extend(item)

            if len(batch) >= batch_size:
                for t, x, y in batch:
                    f.write(f"{t},{x},{y}\n")
                f.flush()
                batch = []

    logger.info(f"Results saved to {filename}")


def run(args: argparse.Namespace) -> int:
    """Execute the sample command."""
    start_time = time.time()

    # Parse parameters from CSV filename
    try:
        params = _parse_csv_filename(args.csv_path)
    except Exception as e:
        logger.error(f"Could not parse CSV filename: {e}")
        logger.error("Expected format: valid_datacubes_YYYY-MM-DD-YYYY-MM-DD_DtxWxH_stepTxstepXxstepY_maxNan.csv")
        return 1

    Dt = params["Dt"]
    w = params["w"]
    h = params["h"]

    logger.info(f"Parsed from CSV: Dt={Dt}, w={w}, h={h}")
    logger.info(f"Date range: {params['start_date']} to {params['end_date']}")

    # Load dataset (main process — just for shape info)
    logger.info(f"Opening Zarr dataset: {args.zarr_path}")
    try:
        zg = zarr.open(args.zarr_path, mode="r")
        data = zg[args.data_var]
        size_T, size_X, size_Y = data.shape
        logger.info(f"Dataset shape: T={size_T}, X={size_X}, Y={size_Y}")
    except Exception as e:
        logger.error(f"Error loading Zarr dataset: {e}")
        return 1

    # Output file
    if args.output:
        output_file = args.output
    else:
        output_file = (
            f"sampled_datacubes_{params['start_date']}-{params['end_date']}"
            f"_{Dt}x{w}x{h}"
            f"_{params['step_T']}x{params['step_X']}x{params['step_Y']}"
            f"_{params['max_nan']}.csv"
        )

    if os.path.exists(output_file) and not args.overwrite:
        logger.error(f"File {output_file} already exists. Use --overwrite to replace.")
        return 1

    logger.info(f"Output file: {output_file}")

    # Save metadata
    metadata = {
        "csv_input": args.csv_path,
        "zarr_path": args.zarr_path,
        "output_file": output_file,
        "start_date": params["start_date"],
        "end_date": params["end_date"],
        "Dt": Dt,
        "w": w,
        "h": h,
        "step_T": params["step_T"],
        "step_X": params["step_X"],
        "step_Y": params["step_Y"],
        "max_nan": params["max_nan"],
        "n_samples": args.n_samples,
        "workers": args.workers,
        "q_min": args.q_min,
        "mean_weight": args.mean_weight,
        "scale": args.scale,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_filename = output_file.replace(".csv", "_metadata.json")
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved run metadata to {metadata_filename}")

    # Load all coordinates and sort by time
    logger.info("Loading coordinates from CSV...")
    all_coords = pd.read_csv(
        args.csv_path,
        usecols=["t", "x", "y"],
        dtype={"t": "int32", "x": "int32", "y": "int32"},
        engine="c",
    )
    all_coords.sort_values("t", inplace=True)
    coord_values = all_coords.values  # (N, 3) array of [t, x, y]
    logger.info(f"Loaded {len(coord_values)} coordinates, t range: [{coord_values[0, 0]}, {coord_values[-1, 0]}]")

    # Define time chunks with overlap of Dt-1 (same strategy as filter_nan)
    time_chunk_size = args.time_chunk_size if args.time_chunk_size else 3 * Dt
    t_min = int(coord_values[0, 0])
    t_max = int(coord_values[-1, 0]) + 1  # exclusive upper bound for coord starts

    estimated_chunk_memory_gb = (time_chunk_size + Dt - 1) * size_X * size_Y * 4 / (1024**3)
    logger.info(f"Time chunk size: {time_chunk_size} (+ {Dt - 1} overlap = {time_chunk_size + Dt - 1} loaded)")
    logger.info(f"Estimated memory per chunk: {estimated_chunk_memory_gb:.2f} GB")
    logger.info(f"Estimated total memory ({args.workers} workers): {estimated_chunk_memory_gb * args.workers:.2f} GB")

    # Build work items: (chunk_t_start, data_t_end, coords_for_chunk)
    t_column = coord_values[:, 0]
    work_items = []

    for chunk_t_start in range(t_min, t_max, time_chunk_size):
        chunk_t_end = chunk_t_start + time_chunk_size
        data_t_end = min(chunk_t_start + time_chunk_size + Dt - 1, size_T)

        idx_lo = np.searchsorted(t_column, chunk_t_start, side="left")
        idx_hi = np.searchsorted(t_column, chunk_t_end, side="left")
        chunk_coords = coord_values[idx_lo:idx_hi]

        if len(chunk_coords) == 0:
            continue

        work_items.append((
            args.zarr_path, args.data_var, chunk_t_start, data_t_end,
            chunk_coords, Dt, w, h, args.n_samples, args.seed,
            args.scale, args.q_min, args.mean_weight,
        ))

    logger.info(f"Split into {len(work_items)} time chunks")

    # Start writer thread
    output_queue: Queue = Queue(maxsize=100)
    writer_thread = Thread(target=_file_writer, args=(output_queue, output_file, 1000))
    writer_thread.daemon = False
    writer_thread.start()

    # Process time chunks in parallel
    total_accepted = 0

    with Pool(args.workers) as pool:
        for accepted in tqdm(
            pool.imap(_process_time_chunk, work_items, chunksize=1),
            total=len(work_items),
            desc="Processing time chunks",
        ):
            if accepted:
                output_queue.put(accepted)
                total_accepted += len(accepted)

    logger.info(f"Accepted {total_accepted} / {len(coord_values)} datacubes")

    # Signal writer thread to stop
    output_queue.put(None)
    writer_thread.join()

    elapsed = time.time() - start_time
    logger.success(f"Sampling completed in {elapsed:.1f}s")
    return 0
