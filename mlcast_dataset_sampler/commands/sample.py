"""Importance sampling of valid datacubes.

This module performs importance sampling on pre-filtered datacubes,
selecting samples based on rain rate intensity to balance the training dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from multiprocessing import Pool
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Callable

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


def _acceptance_probability(data: NDArray, q_min: float, mean_weight: float) -> float:
    """Calculate acceptance probability based on data mean."""
    return min(1.0, q_min + mean_weight * np.nanmean(data))


def _process_datacube(
    coord: tuple[int, int, int],
    data: zarr.Array,
    n_samples: int,
    seed: int | None,
    scale: float,
    q_min: float,
    mean_weight: float,
    Dt: int,
    w: int,
    h: int,
) -> list[tuple[int, int, int]]:
    """Process a single datacube for importance sampling.

    Args:
        coord: Tuple of (it, ix, iy) coordinates.
        data: Zarr rain rate array.
        n_samples: Number of sampling trials.
        seed: Random seed (None for non-deterministic).
        scale: Scale factor for exponential transform.
        q_min: Minimum acceptance probability.
        mean_weight: Weight for mean contribution to probability.
        Dt: Time depth.
        w: Width.
        h: Height.

    Returns:
        List of accepted (it, ix, iy) tuples.
    """
    try:
        it, ix, iy = coord
        time_slice = slice(it, it + Dt)
        x_slice = slice(ix, ix + w)
        y_slice = slice(iy, iy + h)

        # Load data from Zarr
        chunk = data[time_slice, x_slice, y_slice]
        # Transform: rescale rain rate
        transformed = 1 - np.exp(-chunk / scale)

        # Calculate acceptance probability
        q = _acceptance_probability(transformed, q_min, mean_weight)

        # Generate random numbers
        rng = np.random.default_rng(seed)
        random_numbers = rng.random(n_samples)
        accepted_count = int(np.sum(random_numbers <= q))

        # Return accepted hits
        return [(it, ix, iy)] * accepted_count

    except Exception as e:
        logger.warning(f"Error processing datacube ({it}, {ix}, {iy}): {e}")
        return []


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

    # Load dataset
    logger.info(f"Opening Zarr dataset: {args.zarr_path}")
    try:
        zg = zarr.open(args.zarr_path, mode="r")
        data = zg[args.data_var]
        logger.info(f"Dataset shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading Zarr dataset: {e}")
        return 1

    # Output file
    if args.output:
        output_file = args.output
    else:
        output_file = f"sampled_datacubes_{params['start_date']}-{params['end_date']}_{Dt}x{w}x{h}_{params['step_T']}x{params['step_X']}x{params['step_Y']}_{params['max_nan']}.csv"

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

    # Start writer thread
    output_queue: Queue = Queue(maxsize=100)
    writer_thread = Thread(target=_file_writer, args=(output_queue, output_file, 1000))
    writer_thread.daemon = False
    writer_thread.start()

    # Create partial function
    process_partial = partial(
        _process_datacube,
        data=data,
        n_samples=args.n_samples,
        seed=args.seed,
        scale=args.scale,
        q_min=args.q_min,
        mean_weight=args.mean_weight,
        Dt=Dt,
        w=w,
        h=h,
    )

    # Process CSV in chunks
    chunksize = 16000
    pool_chunksize = max(1, chunksize // args.workers)

    with Pool(args.workers) as pool:
        pbar = tqdm(desc="Processing datacubes")

        for chunk in pd.read_csv(
            args.csv_path,
            usecols=["t", "x", "y"],
            dtype={"t": "int32", "x": "int32", "y": "int32"},
            engine="c",
            chunksize=chunksize,
        ):
            for hits in pool.imap(
                process_partial,
                chunk.values,
                chunksize=pool_chunksize,
            ):
                if hits:
                    output_queue.put(hits)
                pbar.update(1)

        pbar.close()

    # Signal writer thread to stop
    output_queue.put(None)
    writer_thread.join()

    elapsed = time.time() - start_time
    logger.success(f"Sampling completed in {elapsed:.1f}s")
    return 0
