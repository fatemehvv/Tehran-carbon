"""Smoothed targets without leakage (fusion paper, stage 2: T-sigma; preregistration_fusion_v1.md).

The coarsened target is the monthly z smoothed with a Gaussian of sigma km by normalised convolution over the valid
cells, then standardised again over them:
  truth           smoothed over all valid domain cells (data.target.smooth_km; CellMonthDataset.z)
  training labels smoothed over the training cells of the fold only (split.fold_aware_target), so that no value of
                  the test block or its buffer enters a training label
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_field(z2: np.ndarray, mask: np.ndarray, sigma_cells: tuple[float, float]) -> np.ndarray:
    """(H, W) normalised Gaussian convolution of z2 over the finite cells inside mask, standardised again over
    them; NaN outside mask or where z2 is NaN."""
    ok = mask & np.isfinite(z2)
    out = np.full(z2.shape, np.nan)
    if ok.sum() < 2:
        return out
    num = gaussian_filter(np.where(ok, z2, 0.0), sigma_cells, mode="constant", cval=0.0)
    den = gaussian_filter(ok.astype(float), sigma_cells, mode="constant", cval=0.0)
    s = num[ok] / np.maximum(den[ok], 1e-12)
    out[ok] = (s - s.mean()) / s.std()
    return out


def sigma_cells(sigma_km: float, grid) -> tuple[float, float]:
    return float(sigma_km) / grid.km_y, float(sigma_km) / grid.km_x


def smooth_steps(z: np.ndarray, mask: np.ndarray, sigma_km: float, grid) -> np.ndarray:
    """(T, H, W) z smoothed step by step over the cells of mask."""
    sc = sigma_cells(sigma_km, grid)
    return np.stack([smooth_field(z[t], mask, sc) for t in range(z.shape[0])])
