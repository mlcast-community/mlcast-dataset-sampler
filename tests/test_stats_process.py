"""Regression tests for `_process_chunk`'s stride-first refactor.

Two independent oracles guard the rewrite:

1. ``_process_chunk_old`` — a verbatim copy of the pre-refactor
   (all-positions ``np.where`` + ``np.isin``) implementation. The new code
   must be **byte-identical** to it on every column.
2. ``_brute_force`` — a naive per-window reference that sums each candidate
   window directly (no cumsum trick). The new code must match it in value
   (exact for integer columns, ``allclose`` for the float sum/mean, whose
   addition order legitimately differs).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlcast_dataset_sampler.commands.stats import (
    _datacube_window_sum,
    _dim_cumsum_window,
    _process_chunk,
)

STAT_KEYS = ("t", "x", "y", "nan_count", "sum", "mean", "frac_wet")


# --- Oracle 1: verbatim pre-refactor implementation --------------------------

def _process_chunk_old(
    time_range, t_start_idx, data, max_nan, wet_threshold, deltas, steps, valid_starts_gap
):
    start_t, end_t = time_range
    chunk = data[start_t + t_start_idx : end_t + t_start_idx, :, :].astype(np.float32, copy=False)
    dim_lengths = chunk.shape
    Dt, w, h = deltas
    total_px = Dt * w * h
    nan_mask = np.isnan(chunk)
    nan_count_win = _datacube_window_sum(nan_mask.astype(np.int16), deltas, dim_lengths)
    valid_mask = nan_count_win <= max_nan
    idx_t_rel, idx_x, idx_y = np.where(valid_mask)
    idx_t_rel = idx_t_rel.astype(np.int32)
    idx_x = idx_x.astype(np.int32)
    idx_y = idx_y.astype(np.int32)
    idx_t_abs_rel = idx_t_rel + start_t
    time_mask = np.isin(idx_t_abs_rel, valid_starts_gap)
    idx_t_abs = idx_t_abs_rel + t_start_idx
    stride_mask = (
        (idx_t_abs % steps[0] == 0) & (idx_x % steps[1] == 0) & (idx_y % steps[2] == 0)
    )
    keep = time_mask & stride_mask
    idx_t_rel = idx_t_rel[keep]
    idx_x = idx_x[keep]
    idx_y = idx_y[keep]
    idx_t_abs = idx_t_abs[keep]
    nan_count = nan_count_win[idx_t_rel, idx_x, idx_y]
    chunk[nan_mask] = 0.0
    sum_win = _datacube_window_sum(chunk, deltas, dim_lengths)
    sum_vals = sum_win[idx_t_rel, idx_x, idx_y]
    wet_mask_i = (chunk > wet_threshold).astype(np.int16)
    wet_count_win = _datacube_window_sum(wet_mask_i, deltas, dim_lengths)
    wet_count = wet_count_win[idx_t_rel, idx_x, idx_y]
    valid_count = total_px - nan_count
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_vals = np.where(valid_count > 0, sum_vals / valid_count, np.nan).astype(np.float32)
    frac_wet = wet_count.astype(np.float32) / total_px
    return {
        "t": idx_t_abs, "x": idx_x, "y": idx_y, "nan_count": nan_count,
        "sum": sum_vals.astype(np.float32), "mean": mean_vals, "frac_wet": frac_wet,
    }


# --- Oracle 2: naive per-window brute force ----------------------------------

def _brute_force(chunk, deltas, steps, max_nan, wet_threshold, start_t, t_start_idx, valid_start_mask):
    Dt, w, h = deltas
    step_t, step_x, step_y = steps
    T, X, Y = chunk.shape
    total_px = Dt * w * h
    # Every valid window start: it in [0, T-Dt], ix in [0, X-w], iy in [0, Y-h]
    # (inclusive upper bound — the final window on each axis is included).
    rows = []
    for it in range(T - Dt + 1):
        if not valid_start_mask[it + start_t]:
            continue
        if (it + start_t + t_start_idx) % step_t != 0:
            continue
        for ix in range(0, X - w + 1, step_x):
            for iy in range(0, Y - h + 1, step_y):
                win = chunk[it:it + Dt, ix:ix + w, iy:iy + h]
                filled = np.where(np.isnan(win), np.float32(0.0), win)
                nan_c = int(np.isnan(win).sum())
                if nan_c > max_nan:
                    continue
                valid = total_px - nan_c
                rows.append((
                    it + start_t + t_start_idx, ix, iy, nan_c,
                    float(filled.sum()),
                    float(filled.sum() / valid) if valid > 0 else np.nan,
                    int((filled > wet_threshold).sum()) / total_px,
                ))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    cols = list(zip(*rows)) if rows else [()] * 7
    return {k: np.array(c) for k, c in zip(STAT_KEYS, cols)}


def _lexsort(d):
    o = np.lexsort((d["y"], d["x"], d["t"]))
    return {k: d[k][o] for k in STAT_KEYS}


# --- Test data ---------------------------------------------------------------

def _make_data(seed, T_total, X, Y, nan_blocks=True):
    rng = np.random.default_rng(seed)
    data = (rng.random((T_total, X, Y), dtype=np.float32) ** 6) * 30.0
    # transient scattered NaNs
    data[rng.random((T_total, X, Y)) < 0.03] = np.nan
    if nan_blocks:
        # a static "out of domain" block (like the real radar mask)
        data[:, : X // 4, :] = np.nan
        # a fully-NaN frame region to exercise the mean==NaN path
        data[2:5, X // 2 : X // 2 + 14, Y // 2 : Y // 2 + 12] = np.nan
    return data


# step_t such that off_t != 0 is exercised via t_start_idx
CASES = [
    # (seed, deltas, steps, max_nan, wet_thr, start_t, t_start_idx)
    (0, (6, 12, 10), (2, 3, 3), 5, 0.5, 0, 0),
    (1, (6, 12, 10), (3, 4, 4), 50, 0.5, 0, 2),   # off_t = (-2)%3 = 1
    (2, (8, 10, 10), (2, 5, 5), 0, 1.0, 4, 1),    # off_t with start_t and t_start_idx
    (3, (6, 14, 12), (4, 3, 3), 6 * 14 * 12, 0.5, 0, 3),  # max_nan == total_px -> all-NaN survives
]


@pytest.mark.parametrize("seed,deltas,steps,max_nan,wet_thr,start_t,t_start_idx", CASES)
def test_byte_identical_to_old(seed, deltas, steps, max_nan, wet_thr, start_t, t_start_idx):
    T_total, X, Y = 60, 44, 40
    data = _make_data(seed, T_total, X, Y)
    size_T = T_total - t_start_idx
    Dt = deltas[0]
    # gap-free starts, with a couple of gaps punched in
    rng = np.random.default_rng(seed + 100)
    valid_starts_gap = np.arange(size_T - Dt + 1, dtype=np.int64)
    drop = rng.choice(valid_starts_gap, size=2, replace=False)
    valid_starts_gap = np.setdiff1d(valid_starts_gap, drop)
    valid_start_mask = np.zeros(size_T, dtype=bool)
    valid_start_mask[valid_starts_gap] = True

    end_t = size_T - start_t  # cover the rest of the filtered region from start_t
    time_range = (start_t, end_t)

    # _process_chunk zero-fills its chunk in place. For a zarr array `data[slice]`
    # returns a fresh array so that's harmless; for a plain numpy `data` the slice
    # is a *view*, so pass each impl its own copy to avoid cross-call mutation.
    new = _process_chunk(time_range, t_start_idx, data.copy(), max_nan, wet_thr, deltas, steps, valid_start_mask)
    old = _process_chunk_old(time_range, t_start_idx, data.copy(), max_nan, wet_thr, deltas, steps, valid_starts_gap)

    for k in ("t", "x", "y", "nan_count", "sum", "frac_wet"):
        assert np.array_equal(new[k], old[k]), f"column {k} differs from old impl"
    assert np.array_equal(new["mean"], old["mean"], equal_nan=True), "mean differs from old impl"
    # dtypes must match the parquet schema expectations
    for k in ("t", "x", "y", "nan_count"):
        assert new[k].dtype == np.int32, f"{k} dtype {new[k].dtype}"
    for k in ("sum", "mean", "frac_wet"):
        assert new[k].dtype == np.float32, f"{k} dtype {new[k].dtype}"


@pytest.mark.parametrize("seed,deltas,steps,max_nan,wet_thr,start_t,t_start_idx", CASES)
def test_matches_brute_force(seed, deltas, steps, max_nan, wet_thr, start_t, t_start_idx):
    T_total, X, Y = 60, 44, 40
    data = _make_data(seed, T_total, X, Y)
    size_T = T_total - t_start_idx
    Dt = deltas[0]
    valid_start_mask = np.ones(size_T, dtype=bool)  # all continuous for the oracle

    end_t = size_T - start_t
    chunk = data[start_t + t_start_idx : end_t + t_start_idx, :, :].astype(np.float32)  # snapshot copy

    new = _lexsort(_process_chunk(
        (start_t, end_t), t_start_idx, data.copy(), max_nan, wet_thr, deltas, steps, valid_start_mask
    ))
    ref = _brute_force(chunk, deltas, steps, max_nan, wet_thr, start_t, t_start_idx, valid_start_mask)

    assert new["t"].size == ref["t"].size, "different number of survivors"
    for k in ("t", "x", "y", "nan_count"):
        assert np.array_equal(new[k].astype(np.int64), ref[k].astype(np.int64)), f"{k} mismatch vs brute force"
    assert np.allclose(new["sum"], ref["sum"], rtol=1e-4, atol=1e-2), "sum mismatch vs brute force"
    assert np.allclose(new["mean"], ref["mean"], rtol=1e-4, atol=1e-3, equal_nan=True), "mean mismatch"
    assert np.allclose(new["frac_wet"], ref["frac_wet"], rtol=0, atol=1e-6), "frac_wet mismatch"


def test_window_sum_order_invariant_for_integers():
    rng = np.random.default_rng(0)
    mask = (rng.random((20, 30, 28)) < 0.3).astype(np.int16)
    deltas, dl = (5, 8, 7), mask.shape
    a = _datacube_window_sum(mask, deltas, dl, order=(0, 1, 2))
    b = _datacube_window_sum(mask, deltas, dl, order=(2, 1, 0))
    assert np.array_equal(a, b)


def test_window_sum_includes_last_window():
    # Output length must be dim_len - delta + 1, and the final window start
    # (the one the old off-by-one dropped) must hold the correct sum.
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 5, size=(11, 13, 9)).astype(np.int16)
    deltas, dl = (4, 5, 3), arr.shape
    out = _datacube_window_sum(arr, deltas, dl)
    assert out.shape == (dl[0] - deltas[0] + 1, dl[1] - deltas[1] + 1, dl[2] - deltas[2] + 1)
    # last window on every axis, compared to a direct sum
    lt, lx, ly = dl[0] - deltas[0], dl[1] - deltas[1], dl[2] - deltas[2]
    expected = int(arr[lt:lt + deltas[0], lx:lx + deltas[1], ly:ly + deltas[2]].sum())
    assert int(out[lt, lx, ly]) == expected


def test_empty_when_all_windows_fail():
    # max_nan = -1 -> nothing passes
    data = np.zeros((20, 30, 28), dtype=np.float32)
    mask = np.ones(20, dtype=bool)
    out = _process_chunk((0, 20), 0, data, -1, 0.5, (5, 8, 7), (2, 4, 4), mask)
    for k in STAT_KEYS:
        assert out[k].size == 0
