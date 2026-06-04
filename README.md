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

The workflow is: compute a **stats parquet** once (offline), then **importance-sample
from it on the fly** inside your training `Dataset`.

### Step 1: Compute per-datacube stats

Scan the dataset and write one row per valid datacube candidate (handles time gaps
and NaN regions), with cheap per-window statistics computed via cumsum windows:

```bash
uv run mlcast.sample_dataset stats /path/to/radar.zarr \
    --start-date 2021-01-01 \
    --end-date 2024-12-31 \
    --time-depth 24 \
    --width 256 \
    --height 256 \
    --max-nan 10000
```

This outputs a Parquet file with columns `t, x, y, nan_count, sum, mean, frac_wet`,
one row per surviving `(t, x, y)` candidate. The sampling parameters (datacube shape,
stride, date range, wet threshold, ...) are embedded in the parquet schema metadata,
so downstream code never has to parse the filename. Validate a file against the
contract with:

```bash
uv run mlcast.sample_dataset validate-stats stats_2021-01-01-2024-12-31_24x256x256_3x16x16_10000.parquet
```

### Step 2: Importance-sample in your Dataset

There is no separate sampling command. The stats parquet already carries a per-candidate
`mean`, so weighting is pure tabular arithmetic done at `DataLoader` construction —
no second pass, no re-read of the source Zarr:

```python
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import WeightedRandomSampler
from mlcast_dataset_sampler.sampling import importance_weights
from mlcast_dataset_sampler.stats_spec import read_metadata

mean = pq.read_table(stats_path, columns=["mean"]).column(0).to_numpy()
weights = importance_weights(mean, q_min=1e-4, scale=1.0, mean_weight=0.1)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

meta = read_metadata(stats_path)   # typed: meta.time_depth, meta.width, meta.height, ...
# Dataset.__getitem__(i) slices the Zarr at coords[i] using meta's datacube shape.
loader = DataLoader(dataset, sampler=sampler, batch_size=...)
```

`WeightedRandomSampler` draws with replacement proportional to the weights every epoch,
oversampling intense windows while `q_min` keeps low-intensity samples represented.

## Why importance sampling?

Equal-frequency sampling gives the same probability to all precipitation intensities.
This causes models to hallucinate thunderstorms after ~30 minutes of lead time.

Importance sampling addresses this by:
- Setting a minimum selection weight (`q_min`) for all samples
- Adding a contribution based on mean rain rate (`mean_weight`)

Because the weight is a function of the window `mean` (rather than an average of a
per-pixel transform), spiky windows with a few intense pixels are oversampled rather
than suppressed — desirable for training nowcasting models on extremes.

## CLI Reference

```bash
uv run mlcast.sample_dataset --help
uv run mlcast.sample_dataset stats --help
uv run mlcast.sample_dataset validate-stats --help
```

## License

Apache-2.0 OR BSD-3-Clause
