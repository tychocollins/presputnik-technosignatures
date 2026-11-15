"""Phase 2a Pilot Analysis, MPC Asteroid Cross-match, Power Update, and Pre-Registration Freeze."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.asteroids import crossmatch_candidates_skybot
from src.power import min_detectable_deficit


def run_pilot_analysis(skip_mpc: bool = False) -> None:
    res_df = pd.read_csv("data/processed/pilot_results.csv")
    qc_df = pd.read_csv("data/processed/pilot_plate_qc.csv")
    cand_path = Path("data/processed/pilot_candidates.parquet")
    cand_ast_path = Path("data/processed/pilot_candidates_asteroids.parquet")

    ok_pairs = res_df[res_df["status"] == "ok"].copy()
    print(f"Pairs attempted: {len(res_df)}, succeeded: {len(ok_pairs)}, failed: {len(res_df) - len(ok_pairs)}")
    if len(res_df) > len(ok_pairs):
        print("Failure breakdown:")
        print(res_df[res_df["status"] != "ok"]["status"].value_counts().head(10))

    # 1. Bright-star Gaia match rate QC
    dropped = qc_df[~qc_df["qc_pass"]].copy()
    print(f"\nPlates QC'd: {len(qc_df)}, Passed (>=90% bright Gaia match): {int(qc_df['qc_pass'].sum())}, Dropped (<90%): {len(dropped)}")
    if len(dropped) > 0:
        print("Dropped plates:")
        print(dropped[["plate_id", "instrument", "n_bright", "n_bright_matched", "bright_match_rate"]].to_string(index=False))

    if cand_path.exists():
        cands = pd.read_parquet(cand_path)
    else:
        cands = pd.DataFrame()

    # 2. Asteroid cross-match via SkyBoT (one cone search per unique plate)
    exp_df = pd.read_parquet("data/processed/exposures.parquet")
    exp_idx = exp_df.set_index("exposure_id")
    if len(cands) > 0 and not skip_mpc:
        if cand_ast_path.exists():
            cands = pd.read_parquet(cand_ast_path)
        else:
            sub_mask = cands["passes_partner_rule"] & cands["plate_a_qc_pass"] & cands["plate_b_qc_pass"]
            unique_exps = cands.loc[sub_mask, "exp_a"].unique().tolist()
            plate_centers = {}
            for eid in unique_exps:
                r_e = exp_idx.loc[eid]
                rad = 0.6 * float(np.hypot(r_e["fov_ra_deg"], r_e["fov_dec_deg"])) + 0.1
                plate_centers[eid] = (float(r_e["ra_deg"]), float(r_e["dec_deg"]), max(rad, 0.5), float(r_e["mjd_mid"]))
            sub = crossmatch_candidates_skybot(cands[sub_mask].copy(), plate_centers, max_workers=4)
            cands["is_asteroid"] = False
            cands["ast_desig"] = None
            cands["ast_sep_arcsec"] = np.nan
            cands["ast_vmag"] = np.nan
            cands.loc[sub_mask, ["is_asteroid", "ast_desig", "ast_sep_arcsec", "ast_vmag"]] = sub[
                ["is_asteroid", "ast_desig", "ast_sep_arcsec", "ast_vmag"]
            ].values
            cands.to_parquet(cand_ast_path, index=False)

    # 3. Candidate rates by instrument
    ok_pairs["area_pair_deg2"] = 2.0 * ok_pairs["fov_deg2"]  # 2 plates per pair
    qc_pairs = ok_pairs[ok_pairs["qc_pass_pair"]].copy()

    def _agg_cands(df_c: pd.DataFrame, mask: pd.Series, col_name: str) -> pd.Series:
        if len(df_c) == 0:
            return pd.Series(dtype=int, name=col_name)
        return df_c[mask].groupby("instrument").size().rename(col_name)

    inst_tbl = ok_pairs.groupby("instrument").agg(
        pairs_ok=("exp1", "count"),
        pairs_qc=("qc_pass_pair", "sum"),
        area_ok_deg2=("area_pair_deg2", "sum"),
        cands_raw=("cands_raw_total", "sum"),
        cands_rule=("cands_rule_total", "sum"),
    )
    inst_qc_area = qc_pairs.groupby("instrument")["area_pair_deg2"].sum().rename("area_qc_deg2")
    inst_tbl = inst_tbl.join(inst_qc_area, how="left").fillna({"area_qc_deg2": 0.0})

    if len(cands) > 0:
        m_rule_qc = cands["passes_partner_rule"] & cands["plate_a_qc_pass"] & cands["plate_b_qc_pass"]
        inst_tbl = inst_tbl.join(_agg_cands(cands, m_rule_qc, "cands_rule_qc"), how="left").fillna({"cands_rule_qc": 0})
        if "is_asteroid" in cands.columns:
            m_clean = m_rule_qc & (~cands["is_asteroid"].astype(bool))
            inst_tbl = inst_tbl.join(_agg_cands(cands, m_clean, "cands_clean"), how="left").fillna({"cands_clean": 0})
        else:
            inst_tbl["cands_clean"] = inst_tbl["cands_rule_qc"]
    else:
        inst_tbl["cands_rule_qc"] = 0
        inst_tbl["cands_clean"] = 0

    inst_tbl["rate_raw_per_plate"] = inst_tbl["cands_raw"] / (2.0 * inst_tbl["pairs_ok"])
    inst_tbl["rate_rule_per_plate"] = inst_tbl["cands_rule"] / (2.0 * inst_tbl["pairs_ok"])
    inst_tbl["rate_clean_per_plate"] = np.where(
        inst_tbl["pairs_qc"] > 0, inst_tbl["cands_clean"] / (2.0 * inst_tbl["pairs_qc"]), np.nan
    )
    inst_tbl["rate_clean_per_deg2"] = np.where(
        inst_tbl["area_qc_deg2"] > 0, inst_tbl["cands_clean"] / inst_tbl["area_qc_deg2"], np.nan
    )

    print("\n--- CANDIDATE RATES BY INSTRUMENT ---")
    print(inst_tbl.round(4).to_string())

    # Totals
    tot_pairs_qc = int(inst_tbl["pairs_qc"].sum())
    tot_area_qc = float(inst_tbl["area_qc_deg2"].sum())
    tot_clean = int(inst_tbl["cands_clean"].sum())
    overall_per_plate = tot_clean / (2.0 * tot_pairs_qc) if tot_pairs_qc > 0 else 0.0
    overall_per_deg2 = tot_clean / tot_area_qc if tot_area_qc > 0 else 0.0
    print(f"\nOverall QC-passed clean candidates: {tot_clean} across {2*tot_pairs_qc} plates ({tot_area_qc:.1f} deg2)")
    print(f"Overall clean rate: {overall_per_plate:.4f} /plate, {overall_per_deg2:.5f} /deg2")

    # 4. Ecliptic latitude bins (10 deg bins of |b_ecl|)
    if len(cands) > 0:
        m_rule_qc = cands["passes_partner_rule"] & cands["plate_a_qc_pass"] & cands["plate_b_qc_pass"]
        sub_c = cands[m_rule_qc].copy()
        sub_c["abs_b_ecl_bin"] = pd.cut(sub_c["ecliptic_lat_deg"].abs(), bins=range(0, 91, 10), right=False)
        ecl_tbl = sub_c.groupby("abs_b_ecl_bin", observed=False).agg(
            cands_before_ast=("ra", "count"),
            asteroids=("is_asteroid", "sum") if "is_asteroid" in sub_c.columns else ("ra", lambda x: 0),
        )
        ecl_tbl["cands_after_ast"] = ecl_tbl["cands_before_ast"] - ecl_tbl["asteroids"]
        print("\n--- CANDIDATES BY |ECLIPTIC LATITUDE| (10 deg bins) ---")
        print(ecl_tbl.to_string())

        # 5. Time separation dt_hours bins
        dt_bins = [0, 0.5, 2.0, 6.0, 24.0, 1e6]
        dt_labels = ["<0.5h", "0.5-2h", "2-6h", "6-24h", ">24h"]
        qc_pairs["dt_bin"] = pd.cut(qc_pairs["dt_hours"], bins=dt_bins, labels=dt_labels, right=False)
        dt_pair_counts = qc_pairs.groupby("dt_bin", observed=False).size().rename("qc_pairs")
        sub_c["dt_bin"] = pd.cut(sub_c["dt_hours"], bins=dt_bins, labels=dt_labels, right=False)
        if "is_asteroid" in sub_c.columns:
            dt_cands = sub_c[~sub_c["is_asteroid"].astype(bool)].groupby("dt_bin", observed=False).size().rename("cands_clean")
        else:
            dt_cands = sub_c.groupby("dt_bin", observed=False).size().rename("cands_clean")
        dt_tbl = pd.concat([dt_pair_counts, dt_cands], axis=1).fillna(0)
        dt_tbl["rate_per_plate"] = np.where(dt_tbl["qc_pairs"] > 0, dt_tbl["cands_clean"] / (2.0 * dt_tbl["qc_pairs"]), np.nan)
        print("\n--- CANDIDATES BY PAIR TIME SEPARATION (dt_hours) ---")
        print(dt_tbl.round(4).to_string())

    # 6. Power analysis update using Phase 1 Tier A m_flash >= 9.0 areas
    exp = pd.read_parquet("data/processed/exposures.parquet")
    t_a9 = exp[exp["is_tier_a"] & (exp["m_flash_05s"] >= 9.0)]
    sh_area = float(t_a9["shadow_area_deg2_eff"].sum())
    out_area = float(t_a9["out_area_deg2_eff"].sum())
    mu_sh = overall_per_deg2 * sh_area
    # Scale out_area expectation proportionally
    print("\n--- UPDATED POWER ANALYSIS (Phase 1 Tier A m_flash >= 9.0) ---")
    print(f"Shadow area: {sh_area:.2f} deg2*exp, Out-of-shadow area: {out_area:.2f} deg2*exp")
    print(f"Measured clean candidate rate: {overall_per_deg2:.5f} /deg2 -> Expected shadow counts if null: {mu_sh:.1f}")
    if sh_area > 0 and out_area > 0 and overall_per_deg2 > 0:
        rho_arr = np.array([overall_per_deg2], dtype=float)
        mdd_80 = float(min_detectable_deficit(rho_arr, sh_area, out_area, sigma=2.487)[0])
        mdd_3s = float(min_detectable_deficit(rho_arr, sh_area, out_area, sigma=3.0)[0])
        mdd_5s = float(min_detectable_deficit(rho_arr, sh_area, out_area, sigma=5.0)[0])
        print(f"Min detectable deficit (alpha=0.05 one-sided, power=0.80, Z=2.487): {mdd_80:.3f}")
        print(f"Min detectable deficit (3-sigma): {mdd_3s:.3f}, (5-sigma): {mdd_5s:.3f}")
    else:
        mdd_80 = mdd_3s = mdd_5s = np.nan

    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"\nCommit hash: {commit_hash}")
    print(f"Summary: tot_clean={tot_clean}, tot_pairs_qc={tot_pairs_qc}, tot_area_qc={tot_area_qc:.2f} deg2, "
          f"rate_per_plate={overall_per_plate:.4f}, rate_per_deg2={overall_per_deg2:.5f}, "
          f"mu_sh={mu_sh:.1f}, mdd_80={mdd_80:.3f}, mdd_3s={mdd_3s:.3f}, mdd_5s={mdd_5s:.3f}")


if __name__ == "__main__":
    import sys
    run_pilot_analysis(skip_mpc="--skip-mpc" in sys.argv)
