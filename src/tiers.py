"""Tier A (defect-controlled repeat exposures / multi-exposure plates) and Tier B classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _angular_sep_deg(ra1: float, dec1: float, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    """Great-circle angular separation in degrees (vectorized)."""
    r1 = np.deg2rad(ra1)
    d1 = np.deg2rad(dec1)
    r2 = np.deg2rad(ra2)
    d2 = np.deg2rad(dec2)
    sin_d = np.sin(0.5 * (d2 - d1))
    sin_r = np.sin(0.5 * (r2 - r1))
    a = sin_d ** 2 + np.cos(d1) * np.cos(d2) * (sin_r ** 2)
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def assign_tiers(
    df: pd.DataFrame,
    max_sep_radius_frac: float = 0.5,
    min_dt_minutes: float = 5.0,
    max_dt_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Assign Tier A and Tier B flags.

    Tier A = exposures with an independent repeat of the same field
    (same source_archive & instrument, centers within max_sep_radius_frac * field_radius,
    5 min <= |dt| <= 24 h, and different exposure_id)
    OR multi-exposure plates (multi_exposure_plate == True).
    """
    df = df.reset_index(drop=True).copy()
    min_dt_days = min_dt_minutes / 1440.0
    max_dt_days = max_dt_hours / 24.0

    paired = np.zeros(len(df), dtype=bool)
    pair_count = np.zeros(len(df), dtype=np.int32)

    # Group by (source_archive, instrument) for fast sliding-window matching
    for (_, _), grp in df.groupby(["source_archive", "instrument"], sort=False):
        if len(grp) < 2:
            continue
        idx = grp.index.to_numpy()
        mjd = grp["mjd_mid"].to_numpy()
        order = np.argsort(mjd)
        idx_s = idx[order]
        mjd_s = mjd[order]
        ra_s = grp["ra_deg"].to_numpy()[order]
        dec_s = grp["dec_deg"].to_numpy()[order]
        fov_r_s = 0.25 * (grp["fov_ra_deg"].to_numpy()[order] + grp["fov_dec_deg"].to_numpy()[order])

        n = len(idx_s)
        for i in range(n):
            t_i = mjd_s[i]
            # Search window [t_i - max_dt, t_i + max_dt]
            lo = int(np.searchsorted(mjd_s, t_i - max_dt_days, side="left"))
            hi = int(np.searchsorted(mjd_s, t_i + max_dt_days, side="right"))
            if hi - lo <= 1:
                continue
            dt = np.abs(mjd_s[lo:hi] - t_i)
            cand_mask = (dt >= min_dt_days) & (dt <= max_dt_days)
            if not np.any(cand_mask):
                continue
            cand_indices = np.arange(lo, hi)[cand_mask]
            seps = _angular_sep_deg(ra_s[i], dec_s[i], ra_s[cand_indices], dec_s[cand_indices])
            max_sep = max_sep_radius_frac * np.minimum(fov_r_s[i], fov_r_s[cand_indices])
            hits = np.sum(seps <= max_sep)
            if hits > 0:
                orig_idx = idx_s[i]
                paired[orig_idx] = True
                pair_count[orig_idx] = int(hits)

    df["tier_a_paired"] = paired
    df["tier_a_pair_count"] = pair_count
    df["tier_a_multiexp"] = df["multi_exposure_plate"].fillna(False).astype(bool)
    df["is_tier_a"] = df["tier_a_paired"] | df["tier_a_multiexp"]
    df["is_tier_b"] = True
    return df
