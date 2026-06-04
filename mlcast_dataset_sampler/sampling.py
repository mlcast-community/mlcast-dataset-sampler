"""Importance-sampling weights for datacube candidates.

The `stats` command precomputes a per-candidate `mean` column in the stats
parquet. Training code turns those means into sampling weights with
:func:`importance_weights` and feeds them to a
``torch.utils.data.WeightedRandomSampler`` — no separate sampling pass and
no second read of the source Zarr is needed::

    import pyarrow.parquet as pq
    from torch.utils.data import WeightedRandomSampler
    from mlcast_dataset_sampler.sampling import importance_weights

    mean = pq.read_table(stats_path, columns=["mean"]).column(0).to_numpy()
    w = importance_weights(mean)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

The weight is ``q_min + mean_weight * (1 - exp(-mean / scale))``. The
transform ``1 - exp(-mean / scale)`` is a smooth 0→1 saturating map of mean
intensity; ``q_min`` is a floor that keeps low-intensity windows in the mix
(so the model still sees light/no rain), and ``mean_weight`` scales how
hard wetter windows are oversampled.

Note this weighs on ``mean`` directly (``f(mean)``), which — unlike
averaging a per-pixel transform — does not penalise spiky windows, so
intense localised cells are oversampled rather than suppressed. That is
deliberate: nowcasting benefits from seeing extremes during training.
``WeightedRandomSampler`` normalises the weights, so only their *relative*
magnitudes matter (no need to clamp to 1).
"""

from __future__ import annotations

import numpy as np


def importance_weights(
    mean: np.ndarray,
    *,
    q_min: float = 1e-4,
    scale: float = 1.0,
    mean_weight: float = 0.1,
) -> np.ndarray:
    """Per-candidate sampling weights from the stats parquet `mean` column.

    Parameters
    ----------
    mean
        Per-candidate mean values (the ``mean`` column of a stats parquet).
        NaNs (all-NaN windows) propagate to NaN weights; filter them out
        before sampling if present.
    q_min
        Floor weight applied to every candidate, keeping low-intensity
        windows represented. Must be non-negative.
    scale
        Denominator in the saturating transform ``1 - exp(-mean / scale)``;
        larger values make the transform respond more slowly to intensity.
        Must be positive.
    mean_weight
        Multiplier on the transformed mean, scaling how strongly wetter
        windows are oversampled. Must be non-negative.

    Returns
    -------
    numpy.ndarray
        Float64 weights, same shape as ``mean``, suitable for
        ``torch.utils.data.WeightedRandomSampler``.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if q_min < 0:
        raise ValueError(f"q_min must be non-negative, got {q_min}")
    if mean_weight < 0:
        raise ValueError(f"mean_weight must be non-negative, got {mean_weight}")

    mean = np.asarray(mean, dtype=np.float64)
    return q_min + mean_weight * (1.0 - np.exp(-mean / scale))
