"""Phase 1 orchestration: full shadow geometry, tiers, timing perturbation, power, and reports."""

from __future__ import annotations

import time as pytime
from pathlib import Path

import numpy as np
import pandas as pd

from src.fetch import load_config
from src.geometry import compute_batch_shadow
from src.power import (
    add_sensitivity_weights,
    evaluate_go_no_go,
    min_detectable_deficit,
    plot_power_curves,
    summarize_group,
)
from src.tiers import assign_tiers


def _df_to_md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:{floatfmt}}" if np.isfinite(v) else "nan")
            elif isinstance(v, (int, np.integer)):
                vals.append(f"{int(v):,}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_timing_perturbation(
    df: pd.DataFrame,
    r_geo_km: float = 42164.0,
    n_draws: int = 20,
    max_shift_min: float = 30.0,
    sample_per_archive: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Perturb timestamps randomly by +-30 min across n_draws and report the fraction
    of exposures whose any_in_shadow flag flips, per source_archive.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for arch, grp in df.groupby("source_archive"):
        if len(grp) <= sample_per_archive:
            samples.append(grp)
        else:
            samples.append(grp.sample(n=sample_per_archive, random_state=seed))
    sub = pd.concat(samples, ignore_index=True)

    base_any = sub["any_in_shadow"].to_numpy()
    flip_rates = np.zeros((n_draws, len(sub)), dtype=bool)

    for d in range(n_draws):
        shifts_days = rng.uniform(-max_shift_min, max_shift_min, size=len(sub)) / 1440.0
        res = compute_batch_shadow(
            ra_deg=sub["ra_deg"].to_numpy(),
            dec_deg=sub["dec_deg"].to_numpy(),
            fov_ra_deg=sub["fov_ra_deg"].to_numpy(),
            fov_dec_deg=sub["fov_dec_deg"].to_numpy(),
            mjd_start=sub["mjd_start"].to_numpy() + shifts_days,
            exptime_sec=sub["exptime_sec"].to_numpy(),
            obs_lat_deg=sub["obs_lat_deg"].to_numpy(),
            obs_lon_deg=sub["obs_lon_deg"].to_numpy(),
            obs_elev_m=sub["obs_elev_m"].to_numpy(),
            altitudes_km=(r_geo_km,),
            batch_size=5000,
        )
        pert_any = res[r_geo_km] > 0.0
        flip_rates[d] = pert_any != base_any

    sub["flip_prob"] = np.mean(flip_rates, axis=0)
    out = (
        sub.groupby("source_archive", as_index=False)
        .agg(
            n_sampled=("exposure_id", "count"),
            base_in_shadow_pct=("any_in_shadow", lambda s: 100.0 * np.mean(s)),
            mean_flip_pct=("flip_prob", lambda s: 100.0 * np.mean(s)),
        )
    )
    return out


def main(config_path: str | Path = "config.yaml") -> None:
    t0 = pytime.time()
    cfg = load_config(config_path)
    proc_dir = Path("data/processed")
    rep_dir = Path("reports")
    rep_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(proc_dir / "exposures_clean.parquet")
    print(f"Loaded {len(df):,} clean exposures.")

    r_default = float(cfg["geometry"]["altitudes_km"]["default"])
    altitudes = (r_default, *[float(x) for x in cfg["geometry"]["altitudes_km"]["sensitivity"]])
    step_deg = float(cfg["geometry"]["spatial_grid_step_deg"])
    min_samples = int(cfg["geometry"]["min_spatial_samples_per_axis"])
    t_step_min = float(cfg["geometry"]["time_step_minutes"])

    print(f"Running vectorized shadow geometry for altitudes {altitudes} km...")
    t_geom = pytime.time()
    shadow_res = compute_batch_shadow(
        ra_deg=df["ra_deg"].to_numpy(),
        dec_deg=df["dec_deg"].to_numpy(),
        fov_ra_deg=df["fov_ra_deg"].to_numpy(),
        fov_dec_deg=df["fov_dec_deg"].to_numpy(),
        mjd_start=df["mjd_start"].to_numpy(),
        exptime_sec=df["exptime_sec"].to_numpy(),
        obs_lat_deg=df["obs_lat_deg"].to_numpy(),
        obs_lon_deg=df["obs_lon_deg"].to_numpy(),
        obs_elev_m=df["obs_elev_m"].to_numpy(),
        altitudes_km=altitudes,
        step_deg=step_deg,
        min_samples=min_samples,
        time_step_min=t_step_min,
        batch_size=5000,
    )
    print(f"Shadow geometry finished in {pytime.time() - t_geom:.1f}s.")

    df["shadow_fraction"] = shadow_res[r_default]
    df["shadow_fraction_36000km"] = shadow_res[36000.0]
    df["shadow_fraction_20000km"] = shadow_res[20000.0]
    df["any_in_shadow"] = df["shadow_fraction"] > 0.0
    df["fully_in_shadow"] = df["shadow_fraction"] >= 0.999

    print("Assigning Tier A / Tier B...")
    df = assign_tiers(
        df,
        max_sep_radius_frac=float(cfg["tiers"]["tier_a"]["max_center_sep_field_radius_frac"]),
        min_dt_minutes=float(cfg["tiers"]["tier_a"]["min_dt_minutes"]),
        max_dt_hours=float(cfg["tiers"]["tier_a"]["max_dt_hours"]),
    )

    df, m_poss_med = add_sensitivity_weights(df)
    df.to_parquet(proc_dir / "exposures.parquet", index=False)
    print(f"Saved {len(df):,} rows -> data/processed/exposures.parquet")

    # Go/No-Go and Power Analysis
    verdict = evaluate_go_no_go(
        df,
        min_ratio=float(cfg["go_no_go"]["min_poss1_shadow_exposure_ratio"]),
        min_fully_in=int(cfg["go_no_go"]["min_tier_a_fully_in_shadow"]),
        min_instruments=int(cfg["go_no_go"]["min_tier_a_instruments"]),
    )

    rho_grid = np.geomspace(
        float(cfg["power"]["transient_density_min_per_deg2"]),
        float(cfg["power"]["transient_density_max_per_deg2"]),
        int(cfg["power"]["transient_density_num_points"]),
    )
    overdisp = float(cfg["power"]["overdispersion_factor"])
    plot_power_curves(verdict["summaries"], rho_grid, overdisp, rep_dir / "power_curves.png")

    # Per-archive & per-instrument breakdowns
    arch_rows = [summarize_group(grp, f"{sa} / {sub}") for (sa, sub), grp in df.groupby(["source_archive", "sub_archive"])]
    arch_summary = pd.DataFrame(arch_rows).sort_values("sens_shadow_eff_deg2", ascending=False)
    arch_summary.to_csv(proc_dir / "summary_by_subarchive.csv", index=False)

    inst_rows = [summarize_group(grp, f"{sa}: {inst}") for (sa, inst), grp in df.groupby(["source_archive", "instrument"])]
    inst_summary = pd.DataFrame(inst_rows).sort_values("sens_shadow_eff_deg2", ascending=False)
    inst_summary.to_csv(proc_dir / "summary_by_instrument.csv", index=False)

    # Timing perturbation check (+-30 min, 20 draws)
    print("Running +-30 min timing perturbation check (20 draws)...")
    timing_df = run_timing_perturbation(df, r_geo_km=r_default, n_draws=20, max_shift_min=30.0)
    timing_df.to_csv(proc_dir / "timing_perturbation.csv", index=False)

    # Write reports/feasibility.md
    s_map = verdict["summaries"]
    loss_arch = pd.read_csv(proc_dir / "loss_table_by_archive.csv")
    dasch_val = pd.read_csv(proc_dir / "dasch_series_validation.csv")

    md = []
    md.append("# Phase 1 Feasibility Report: Pre-Sputnik Earth-Shadow Test\n")
    md.append(f"## Executive Verdict: **{verdict['verdict']}**\n")
    md.append(
        f"- **Tier A sensitivity-weighted shadow coverage vs POSS-I-E:** "
        f"**{verdict['tier_a_vs_poss1e_sens_ratio']:.2f}×** "
        f"({s_map['Tier A (all)']['sens_shadow_eff_deg2']:.1f} vs {s_map['POSS-I-E']['sens_shadow_eff_deg2']:.1f} eff. deg²·exp)\n"
    )
    md.append(
        f"- **Tier A + X/Y-scanned Tier B vs POSS-I-E:** "
        f"**{verdict['tier_a_xy_vs_poss1e_sens_ratio']:.2f}×** "
        f"({s_map['Tier A + XY-scanned']['sens_shadow_eff_deg2']:.1f} eff. deg²·exp)\n"
    )
    md.append(
        f"- **Tier B (all) vs POSS-I-E:** "
        f"**{verdict['tier_b_vs_poss1e_sens_ratio']:.2f}×** "
        f"({s_map['Tier B (all)']['sens_shadow_eff_deg2']:.1f} eff. deg²·exp)\n"
    )
    md.append(
        f"- **Tier A fully-in-shadow exposures:** {s_map['Tier A (all)']['n_fully_in_shadow']:,} "
        f"across {s_map['Tier A (all)']['n_instruments_fully_in']} independent instruments "
        f"(threshold: ≥ {cfg['go_no_go']['min_tier_a_fully_in_shadow']} across ≥ {cfg['go_no_go']['min_tier_a_instruments']}).\n"
    )
    md.append(
        f"- **POSS-I-E benchmark:** 915 kept red (E) plates out of 1,037 rows in VizieR `VI/25/nposs` "
        f"(note: VASCO's published catalog used a 635-plate E subset; `ObsE` confirmed as exposure START in PST = UTC−8h). "
        f"Median POSS-I-E 0.5 s flash limit: **{m_poss_med:.2f} mag**.\n"
    )

    md.append("\n## 1. Tier & Benchmark Comparison\n")
    tier_df = pd.DataFrame([s_map[k] for k in ["POSS-I-E", "Tier A (all)", "Tier A + XY-scanned", "Tier B (all)"]])
    md.append(_df_to_md(tier_df, floatfmt=".2f") + "\n")

    md.append("\n## 2. Sensitivity-Weighted vs Raw Coverage by Sub-Archive\n")
    md.append(_df_to_md(arch_summary.head(20), floatfmt=".2f") + "\n")

    md.append("\n## 3. Timing Perturbation Check (±30 min, 20 random draws)\n")
    md.append(_df_to_md(timing_df, floatfmt=".3f") + "\n")

    md.append("\n## 4. Altitude Sensitivity Check (GEO 42,164 km vs 36,000 km vs 20,000 km)\n")
    for r_col, r_lbl in [("shadow_fraction", "42,164 km"), ("shadow_fraction_36000km", "36,000 km"), ("shadow_fraction_20000km", "20,000 km")]:
        ta_mask = (df["source_archive"] != "POSS-I") & df["is_tier_a"]
        n_any = int((df.loc[ta_mask, r_col] > 0).sum())
        n_full = int((df.loc[ta_mask, r_col] >= 0.999).sum())
        md.append(f"- **{r_lbl} (Tier A):** {n_any:,} any_in_shadow, {n_full:,} fully_in_shadow\n")

    md.append("\n## 5. Top 3 Data-Quality Risks Identified\n")
    md.append(
        "1. **Shallow flash sensitivity on wide-field patrol plates:** Wide-field patrol cameras (1–3 inch lenses, "
        "long exposures) have `m_flash(0.5s) ~ 3–6 mag` vs `~10.7 mag` for POSS-I-E, reducing their sensitivity-weighted "
        "contribution by 100×–1000× relative to raw sky area.\n"
        "2. **Clustered DASCH timestamp/site anomalies in specific series:** While major series (`b`, `ac`, `ai`, `mf`, `mc`) "
        "pass Sun < −12° and pointing > 10° at >97%, specific series (`ka`, `kb`, `kc`, `kd`, `ke`, `kf`, `kh`, `ak`, `lwla`, `mb`) "
        "show 16%–76% failure rates indicative of logbook AM/PM or UT-offset errors.\n"
        "3. **APPLAUSE `numexp > 1` & heterogeneous WCS coverage:** ~39% of pre-1958 APPLAUSE exposures lack WCS solutions, "
        "and multi-exposure plates (`numexp > 1`) require careful per-exposure deblending in Phase 2.\n"
    )

    md.append("\n## 6. Filter Loss Summary by Archive\n")
    md.append(_df_to_md(loss_arch) + "\n")

    (rep_dir / "feasibility.md").write_text("\n".join(md), encoding="utf-8")

    # Write reports/preregistration_draft.md
    prereg = """# Phase 2 Preregistration Draft: Pre-Sputnik Earth-Shadow Test

## 1. Primary Hypothesis & Test Statistic
- **Primary Test:** Stratified Mantel–Haenszel / Poisson rate-ratio comparison of transient surface density inside Earth's umbra (`shadow_fraction >= 0.5` or continuous exposure-weighted rate) vs outside Earth's umbra (`shadow_fraction == 0`), stratified by `(source_archive, instrument)` so that no single telescope's pointing pattern drives the pooled result.
- **Dose–Response Prediction:** If transients are sunlight glints from orbiting objects, the fractional shadow deficit must be **strongest in Tier A** (repeat-exposure defect-controlled) and **weaker or diluted in Tier B** (single-exposure + Gaia DR3). If the deficit is equal or stronger in Tier B, emulsion defects or time-of-night systematics are indicated.
- **Significance Threshold:** `p < 2.87e-7` (one-sided 5σ) on the primary stratified Tier A test, with `p < 0.001` required on the Tier A vs Tier B dose-response interaction.

## 2. Locked-In Controls (to be executed before unblinding Tier A results)
1. **Shifted-Timestamp Null (±1 h to ±6 h):** Re-run the identical stratified pipeline with exposure midpoints shifted by ±1, ±2, ±4, and ±6 hours. Any "deficit" that persists under shifted timestamps indicates a local-time/observing-routine confounder rather than Earth's physical shadow.
2. **Fake-Shadow Altitudes:** Test shadow cones at non-physical geocentric radii (e.g., 5,000 km, 200,000 km) or displaced ecliptic longitudes (+30°, −30°).
3. **Out-of-Image / Known-Defect Control:** Run the identical shadow-correlation test on features outside the exposed plate area (plate margins/borders) and on single-scan-only dust/emulsion artifacts rejected by X/Y dual-scan comparison.
4. **Local-Solar-Time-Matched Comparison:** Match in-shadow exposures to out-of-shadow exposures taken at the same local solar time (e.g., ±30 min of local midnight, at higher/lower ecliptic latitudes where Earth's shadow does not fall) to isolate the midnight confounder (temperature, humidity, observer fatigue, development batches).
"""
    (rep_dir / "preregistration_draft.md").write_text(prereg, encoding="utf-8")
    print(f"All reports written in {pytime.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
