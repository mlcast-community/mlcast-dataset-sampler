# mlcast-dataset-sampler

Utility to sample MLCast source datasets and generate training-ready data indices.

## Usage

Run directly with `uvx` (no installation needed):

```bash
uvx --from "git+https://github.com/mlcast-community/mlcast-dataset-sampler" mlcast.sample_dataset --help
```

Or clone the repo:

```bash
git clone https://github.com/mlcast-community/mlcast-dataset-sampler
cd mlcast-dataset-sampler
uv sync
uv run mlcast.sample_dataset --help
```

The sampler provides two commands that run in sequence:

### Step 1: Filter valid datacubes

Scan the dataset and identify valid datacube coordinates (handles time gaps and NaN regions):

```bash
uv run mlcast.sample_dataset filter-nan /path/to/radar.zarr \
    --start-date 2021-01-01 \
    --end-date 2024-12-31 \
    --time-depth 24 \
    --width 256 \
    --height 256 \
    --max-nan 10000
```

This outputs a CSV file with valid `(t, x, y)` coordinates.

### Step 2: Importance sampling

Perform importance sampling on the filtered coordinates, weighting by rain intensity:

```bash
uv run mlcast.sample_dataset sample /path/to/radar.zarr \
    valid_datacubes_2021-01-01-2024-12-31_24x256x256_3x16x16_10000.csv \
    --q-min 1e-4 \
    --mean-weight 0.1
```

This outputs a sampled CSV ready for training.

## Why importance sampling?

Equal-frequency sampling (used in some implementations) gives the same probability to all precipitation intensities. This causes models to hallucinate thunderstorms after ~30 minutes of lead time.

Importance sampling addresses this by:
- Setting a minimum selection probability (`--q-min`) for all samples
- Adding a weighted contribution based on mean rain rate (`--mean-weight`)

This keeps low-intensity samples in training while still oversampling interesting meteorological events.

## Using sampled data for training

The output CSV contains `(t, x, y)` indices that point directly into the source Zarr dataset. See the [example DataModule](https://github.com/DSIP-FBK/ConvGRU-Ensemble/blob/main/convgru-ens/datamodule.py) for a PyTorch Dataset implementation.

## CLI Reference

```bash
uv run mlcast.sample_dataset --help
uv run mlcast.sample_dataset filter-nan --help
uv run mlcast.sample_dataset sample --help
```

## License

Apache-2.0 OR BSD-3-Clause
