"""Step 3 & 5: Primary Negative Binomial GLM & empirical placebo-shadow power/FPR calibration on sunlit plates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from astropy import units as u
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord, get_sun
from astropy.time import Time


def build_plate_analysis_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build plate-level clean candidate count table from strictly out-of-shadow QC-passed pilot pairs,
    plus expected QC-passing shadow plate counts by instrument from metadata only (blindness preserved).
    """
    res = pd.read_csv("data/processed/pilot_results.csv")
    qc_pairs = res[res["qc_pass_pair"]].copy()

    cands = pd.read_parquet("data/processed/pilot_candidates.parquet")
    clean = cands[
        cands["passes_partner_rule"]
        & cands["plate_a_qc_pass"]
        & cands["plate_b_qc_pass"]
        & (~cands["is_asteroid"].fillna(False))
    ]
    counts_by_exp = clean.groupby("exp_a").size().to_dict()

    exp = pd.read_parquet("data/processed/exposures.parquet").set_index("exposure_id")

    # Collect unique QC-passed plates (both exp1 and exp2 from QC-passing pairs)
    plate_rows = []
    seen = set()
    for _, r in qc_pairs.iterrows():
        for col in ("exp1", "exp2"):
            eid = r[col]
            if eid in seen or eid not in exp.index:
                continue
            seen.add(eid)
            er = exp.loc[eid]
            # Strict blindness guard: only shadow_fraction == 0.0
            assert float(er["shadow_fraction"]) == 0.0, f"Blindness violation: {eid}"
            assert float(er["shadow_fraction_36000km"]) == 0.0, f"Blindness violation: {eid}"
            assert float(er["shadow_fraction_20000km"]) == 0.0, f"Blindness violation: {eid}"
            plate_rows.append({
                "exposure_id": eid,
                "instrument": str(er["instrument"]),
                "ra_deg": float(er["ra_deg"]),
                "dec_deg": float(er["dec_deg"]),
                "mjd_mid": float(er["mjd_mid"]),
                "obs_lon_deg": float(er["obs_lon_deg"]),
                "pointing_alt_mid_deg": float(er["pointing_alt_mid_deg"]),
                "eff_area_deg2": 0.81 * float(er["fov_ra_deg"]) * float(er["fov_dec_deg"]),
                "cands_clean": int(counts_by_exp.get(eid, 0)),
            })

    df = pd.DataFrame(plate_rows)

    # Compute astronomical covariates
    sc = SkyCoord(df["ra_deg"].to_numpy() * u.deg, df["dec_deg"].to_numpy() * u.deg, frame="icrs")
    ecl = sc.transform_to(BarycentricTrueEcliptic())
    df["abs_b_ecl"] = np.abs(ecl.lat.deg)

    t_mid = Time(df["mjd_mid"].to_numpy(), format="mjd", scale="utc")
    sun_sc = get_sun(t_mid)
    df["solar_elong_deg"] = sc.separation(sun_sc).deg

    # Local solar time centered on midnight (-12 to +12 hr)
    ut_hr = (df["mjd_mid"].to_numpy() % 1.0) * 24.0
    lst_hr = np.mod(ut_hr + df["obs_lon_deg"].to_numpy() / 15.0, 24.0)
    df["lst_midnight_hr"] = np.mod(lst_hr + 12.0, 24.0) - 12.0

    alt = np.clip(df["pointing_alt_mid_deg"].to_numpy(), 10.0, 90.0)
    df["airmass"] = 1.0 / np.sin(np.deg2rad(alt))

    # Expected QC-passing shadow plates by instrument (metadata only, no shadow FITS opened)
    qc = pd.read_csv("data/processed/pilot_plate_qc.csv").drop_duplicates("plate_id")
    inst_qc = qc.groupby("instrument")["qc_pass"].mean().to_dict()
    exp_full = exp.reset_index()
    sh_a = exp_full[exp_full["is_tier_a"] & (exp_full["m_flash_05s"] >= 9.0) & (exp_full["shadow_fraction"] > 0)]
    sh_counts = sh_a.groupby("instrument")["shadow_fraction"].agg(["count", "mean"]).reset_index()
    sh_counts["qc_rate"] = sh_counts["instrument"].map(inst_qc).fillna(0.0)
    sh_counts["exp_qc_pass"] = sh_counts["count"] * sh_counts["qc_rate"]
    sh_counts["sf_list"] = [
        sh_a.loc[sh_a["instrument"] == inst, "shadow_fraction"].to_numpy(dtype=float)
        for inst in sh_counts["instrument"]
    ]
    return df, sh_counts


def _build_design_matrix(df: pd.DataFrame, shadow_col: str = "shadow_fraction") -> pd.DataFrame:
    # Standardize continuous covariates for numerical stability
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    X["shadow_fraction"] = df[shadow_col].to_numpy(dtype=float)
    for col in ("abs_b_ecl", "solar_elong_deg", "lst_midnight_hr", "airmass"):
        v = df[col].to_numpy(dtype=float)
        sd = float(np.std(v))
        X[col] = (v - float(np.mean(v))) / (sd if sd > 1e-9 else 1.0)
    inst_dummies = pd.get_dummies(df["instrument"], prefix="inst", drop_first=True, dtype=float)
    X = pd.concat([X, inst_dummies], axis=1)
    return X


def run_empirical_calibration(n_sim: int = 500, seed: int = 42) -> None:
    df, sh_counts = build_plate_analysis_table()
    # Keep instruments with >= 3 QC-passed sunlit plates so instrument fixed effects are estimable
    inst_n = df["instrument"].value_counts()
    keep_insts = set(inst_n[inst_n >= 3].index)
    df = df[df["instrument"].isin(keep_insts)].reset_index(drop=True)
    print(f"Sunlit QC-passed plates available for NB GLM: {len(df)} across {df['instrument'].nunique()} instruments")
    print(df.groupby("instrument")["cands_clean"].agg(["count", "mean", "std"]).round(2).to_string())

    # Estimate NB dispersion alpha via Poisson GLM + exact NB2 profile log-likelihood
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    df["shadow_fraction"] = 0.0
    X0 = _build_design_matrix(df).drop(columns=["shadow_fraction"])
    y0 = df["cands_clean"].to_numpy(dtype=float)
    off0 = np.log(df["eff_area_deg2"].to_numpy(dtype=float))
    pois_fit = sm.GLM(y0, X0, family=sm.families.Poisson(), offset=off0).fit()
    mu0 = np.maximum(pois_fit.fittedvalues, 1e-6)

    def _nb_nll(alpha: float) -> float:
        r = 1.0 / alpha
        ll = (
            gammaln(y0 + r) - gammaln(r) - gammaln(y0 + 1.0)
            + r * np.log(r / (r + mu0)) + y0 * np.log(mu0 / (r + mu0))
        )
        return -float(np.sum(ll))

    opt = minimize_scalar(_nb_nll, bounds=(0.01, 20.0), method="bounded")
    alpha_hat = float(opt.x)
    print(f"\nEstimated NB overdispersion alpha_hat = {alpha_hat:.4f} (Poisson Pearson chi2/df = {pois_fit.pearson_chi2 / pois_fit.df_resid:.2f})")

    # Build target placebo-shadow counts per instrument matching expected QC-passing shadow plates
    rng = np.random.default_rng(seed)
    inst_indices = {inst: df.index[df["instrument"] == inst].to_numpy() for inst in keep_insts}
    target_sh = []
    for _, r in sh_counts.iterrows():
        inst = r["instrument"]
        n_exp = int(round(float(r["exp_qc_pass"])))
        if n_exp > 0 and inst in inst_indices:
            avail = len(inst_indices[inst])
            target_sh.append((inst, min(n_exp, max(1, avail // 2)), n_exp, avail, r["sf_list"]))

    print("\nPlacebo shadow allocation per instrument (capped at 50% of available sunlit plates if sunlit < shadow):")
    for inst, n_take, n_exp, avail, _ in target_sh:
        print(f"  {inst[:45]:45s}: target={n_exp:2d}, using={n_take:2d} (sunlit avail={avail:2d})")

    deltas = [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = []
    for delta in deltas:
        rejects_1s = 0
        rejects_2s = 0
        valid_fits = 0
        rr_hats = []
        for _ in range(n_sim):
            sf_vec = np.zeros(len(df), dtype=float)
            for inst, n_take, _, _, sf_pool in target_sh:
                chosen = rng.choice(inst_indices[inst], size=n_take, replace=False)
                sf_vals = rng.choice(sf_pool, size=n_take, replace=True)
                sf_vec[chosen] = sf_vals

            y_sim = y0.copy()
            if delta > 0:
                mask_sh = sf_vec > 0
                p_keep = np.clip(1.0 - delta * sf_vec[mask_sh], 0.0, 1.0)
                y_sim[mask_sh] = rng.binomial(y_sim[mask_sh].astype(int), p_keep).astype(float)

            df["shadow_fraction"] = sf_vec
            X = _build_design_matrix(df)
            try:
                fit = sm.GLM(
                    y_sim,
                    X,
                    family=sm.families.NegativeBinomial(alpha=alpha_hat),
                    offset=off0,
                ).fit(maxiter=100)
                beta = float(fit.params["shadow_fraction"])
                se = float(fit.bse["shadow_fraction"])
                if not (np.isfinite(beta) and np.isfinite(se) and se > 0):
                    continue
                z = beta / se
                valid_fits += 1
                rr_hats.append(np.exp(beta))
                if z < -1.6448536:
                    rejects_1s += 1
                if abs(z) > 1.959964:
                    rejects_2s += 1
            except Exception:
                continue

        rate_1s = rejects_1s / max(1, valid_fits)
        rate_2s = rejects_2s / max(1, valid_fits)
        results.append({
            "injected_deficit": delta,
            "n_sim_valid": valid_fits,
            "power_or_fpr_1sided": round(rate_1s, 4),
            "power_or_fpr_2sided": round(rate_2s, 4),
            "median_rate_ratio": round(float(np.median(rr_hats)) if rr_hats else np.nan, 4),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv("data/processed/empirical_power_calibration.csv", index=False)
    print("\n--- EMPIRICAL PLACEBO POWER & FPR CALIBRATION (NB GLM, N=500) ---")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    run_empirical_calibration(n_sim=500, seed=42)
