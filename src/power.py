"""Sensitivity-weighted power analysis, stratified exposure metrics, and go/no-go verdict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_sensitivity_weights(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Compute sensitivity weight relative to POSS-I-E median 0.5s flash limit:
      w = min(1.0, 10^(0.4 * (m_flash_05s - m_flash_poss1e_med)))
    """
    df = df.copy()
    poss_mask = df["sub_archive"] == "POSS-I-E"
    if not np.any(poss_mask):
        raise ValueError("No POSS-I-E benchmark rows found.")
    m_poss_med = float(np.nanmedian(df.loc[poss_mask, "m_flash_05s"]))

    delta_m = df["m_flash_05s"].to_numpy() - m_poss_med
    w = np.minimum(1.0, np.power(10.0, 0.4 * delta_m))
    w = np.where(np.isfinite(w), w, 0.0)
    df["sens_weight"] = w

    area = df["fov_ra_deg"].to_numpy() * df["fov_dec_deg"].to_numpy()
    sf = df["shadow_fraction"].to_numpy()

    df["field_area_deg2"] = area
    df["shadow_area_deg2_eff"] = sf * area
    df["out_area_deg2_eff"] = (1.0 - sf) * area
    df["sens_shadow_area_deg2_eff"] = w * sf * area
    df["sens_out_area_deg2_eff"] = w * (1.0 - sf) * area
    return df, m_poss_med


def summarize_group(sub: pd.DataFrame, label: str) -> dict[str, Any]:
    raw_in = float(sub["shadow_area_deg2_eff"].sum())
    raw_out = float(sub["out_area_deg2_eff"].sum())
    sens_in = float(sub["sens_shadow_area_deg2_eff"].sum())
    sens_out = float(sub["sens_out_area_deg2_eff"].sum())
    return {
        "group": label,
        "n_exposures": int(len(sub)),
        "n_any_in_shadow": int(sub["any_in_shadow"].sum()),
        "n_fully_in_shadow": int(sub["fully_in_shadow"].sum()),
        "n_instruments_fully_in": int(sub.loc[sub["fully_in_shadow"], "instrument"].nunique()),
        "median_m_flash_05s": float(np.nanmedian(sub["m_flash_05s"])) if len(sub) else np.nan,
        "median_sens_weight": float(np.nanmedian(sub["sens_weight"])) if len(sub) else np.nan,
        "raw_shadow_eff_deg2": raw_in,
        "raw_out_eff_deg2": raw_out,
        "raw_shadow_frac": raw_in / (raw_in + raw_out) if (raw_in + raw_out) > 0 else 0.0,
        "sens_shadow_eff_deg2": sens_in,
        "sens_out_eff_deg2": sens_out,
        "sens_shadow_frac": sens_in / (sens_in + sens_out) if (sens_in + sens_out) > 0 else 0.0,
    }


def min_detectable_deficit(
    rho_grid: np.ndarray,
    e_in: float,
    e_out: float,
    sigma: float = 3.0,
    overdispersion: float = 1.0,
) -> np.ndarray:
    """Minimum fractional deficit delta detectable at Z=sigma."""
    if e_in <= 0 or e_out <= 0:
        return np.full_like(rho_grid, np.nan, dtype=float)
    inv_e = (1.0 / e_in) + (1.0 / e_out)
    return sigma * np.sqrt(overdispersion * inv_e / rho_grid)


def plot_power_curves(
    summary_map: dict[str, dict[str, Any]],
    rho_grid: np.ndarray,
    overdispersion: float,
    out_path: Path,
) -> None:
    """Plot 3-sigma and 5-sigma minimum detectable deficit curves (raw & sensitivity-weighted)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    colors = {
        "Tier A (all)": "#1f77b4",
        "Tier B (all)": "#ff7f0e",
        "Tier A + XY-scanned": "#2ca02c",
        "POSS-I-E": "#d62728",
    }

    for ax, mode, title in [
        (axes[0], "sens", "Sensitivity-Weighted (Relative to POSS-I-E)"),
        (axes[1], "raw", "Raw Unweighted (deg²·exposures)"),
    ]:
        for label, s in summary_map.items():
            e_in = s[f"{mode}_shadow_eff_deg2"]
            e_out = s[f"{mode}_out_eff_deg2"]
            c = colors.get(label, "#333333")
            d3 = min_detectable_deficit(rho_grid, e_in, e_out, sigma=3.0, overdispersion=1.0)
            d5_od = min_detectable_deficit(rho_grid, e_in, e_out, sigma=5.0, overdispersion=overdispersion)
            ax.loglog(rho_grid, d3, color=c, linestyle="-", linewidth=2, label=f"{label} (3σ, φ=1)")
            ax.loglog(rho_grid, d5_od, color=c, linestyle="--", linewidth=1.5, alpha=0.8, label=f"{label} (5σ, φ={overdispersion:g})")

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
        ax.axhline(0.5, color="gray", linestyle="-.", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Transient surface density ρ (deg⁻² plate⁻¹)")
        ax.set_title(title)
        ax.set_ylim(0.02, 5.0)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Minimum detectable fractional deficit Δ_min")
    axes[1].legend(fontsize=7.5, loc="upper right", ncol=1)
    fig.suptitle("Phase 1 Power Analysis: Minimum Detectable Earth-Shadow Deficit", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def evaluate_go_no_go(
    df: pd.DataFrame,
    min_ratio: float = 1.0,
    min_fully_in: int = 30,
    min_instruments: int = 3,
) -> dict[str, Any]:
    """Evaluate Go/No-Go on sensitivity-weighted Tier A vs POSS-I-E."""
    poss = df[df["sub_archive"] == "POSS-I-E"]
    non_poss = df[df["source_archive"] != "POSS-I"]

    tier_a = non_poss[non_poss["is_tier_a"]]
    tier_a_xy = non_poss[non_poss["is_tier_a"] | non_poss["has_xy_scans"]]
    tier_b = non_poss[non_poss["is_tier_b"]]

    s_poss = summarize_group(poss, "POSS-I-E")
    s_ta = summarize_group(tier_a, "Tier A (all)")
    s_taxy = summarize_group(tier_a_xy, "Tier A + XY-scanned")
    s_tb = summarize_group(tier_b, "Tier B (all)")

    poss_sens_in = s_poss["sens_shadow_eff_deg2"]
    ratio_ta = s_ta["sens_shadow_eff_deg2"] / poss_sens_in if poss_sens_in > 0 else 0.0
    ratio_taxy = s_taxy["sens_shadow_eff_deg2"] / poss_sens_in if poss_sens_in > 0 else 0.0
    ratio_tb = s_tb["sens_shadow_eff_deg2"] / poss_sens_in if poss_sens_in > 0 else 0.0

    go = (
        (ratio_ta >= min_ratio)
        and (s_ta["n_fully_in_shadow"] >= min_fully_in)
        and (s_ta["n_instruments_fully_in"] >= min_instruments)
    )

    return {
        "verdict": "GO" if go else "NO-GO",
        "tier_a_vs_poss1e_sens_ratio": ratio_ta,
        "tier_a_xy_vs_poss1e_sens_ratio": ratio_taxy,
        "tier_b_vs_poss1e_sens_ratio": ratio_tb,
        "summaries": {
            "POSS-I-E": s_poss,
            "Tier A (all)": s_ta,
            "Tier A + XY-scanned": s_taxy,
            "Tier B (all)": s_tb,
        },
    }
