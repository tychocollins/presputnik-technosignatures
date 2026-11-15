"""Amendment 1: Within-plate conditional test (Mantel-Haenszel + spatial defect profile) and >=1000 placebo power calibration."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


def load_sunlit_plates_and_candidates(
    extra_filter_fn=None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load QC-passed strictly sunlit plates and map their clean candidates to normalized (x, y) in [0, 1]^2.
    Returns (plates_df, cand_xy_by_exp, cell_weights_by_exp).
    """
    res = pd.read_csv("data/processed/pilot_results.csv")
    qc_pairs = res[res["qc_pass_pair"]]
    qc_exps = set(qc_pairs["exp1"]).union(set(qc_pairs["exp2"]))

    exp = pd.read_parquet("data/processed/exposures.parquet").set_index("exposure_id")
    fp = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    cands = pd.read_parquet("data/processed/pilot_candidates.parquet")

    clean = cands[
        cands["passes_partner_rule"]
        & cands["plate_a_qc_pass"]
        & cands["plate_b_qc_pass"]
        & (~cands["is_asteroid"].fillna(False))
    ].copy()
    if extra_filter_fn is not None:
        clean = extra_filter_fn(clean)

    plate_rows = []
    cand_xy: dict[str, np.ndarray] = {}
    local_grids: dict[str, np.ndarray] = {}

    for eid in sorted(qc_exps):
        if eid not in exp.index or eid not in fp.index:
            continue
        er = exp.loc[eid]
        # Strict blindness check
        assert float(er["shadow_fraction"]) == 0.0
        assert float(er["shadow_fraction_36000km"]) == 0.0
        assert float(er["shadow_fraction_20000km"]) == 0.0
        fr = fp.loc[eid]
        nx, ny = int(fr["nx"]), int(fr["ny"])
        wcs = WCS(fits.Header.fromstring(fr["wcs_header"]))
        grid = np.array(json.loads(fr["local_mlim_grid_json"]), dtype=float)
        if grid.shape != (8, 8):
            grid = np.full((8, 8), float(fr["m_lim_cal"]))
        local_grids[eid] = np.nan_to_num(grid, nan=float(fr["m_lim_cal"]))

        sub_c = clean[clean["exp_a"] == eid]
        if len(sub_c) > 0:
            sc = SkyCoord(sub_c["ra"].to_numpy() * u.deg, sub_c["dec"].to_numpy() * u.deg)
            x, y = wcs.world_to_pixel(sc)
            xn = np.clip(x / max(nx, 1), 0.0, 1.0)
            yn = np.clip(y / max(ny, 1), 0.0, 1.0)
            in_active = (xn >= 0.05) & (xn <= 0.95) & (yn >= 0.05) & (yn <= 0.95)
            cand_xy[eid] = np.column_stack([xn[in_active], yn[in_active]])
        else:
            cand_xy[eid] = np.zeros((0, 2), dtype=float)

        plate_rows.append({
            "exposure_id": eid,
            "instrument": str(er["instrument"]),
            "m_lim_cal": float(fr["m_lim_cal"]),
            "n_clean": len(cand_xy[eid]),
        })

    plates_df = pd.DataFrame(plate_rows)

    # Fit per-instrument 2D spatial defect profile on SUNLIT plates (8x8 normalized grid, smoothed)
    inst_profiles: dict[str, np.ndarray] = {}
    global_hist = np.ones((8, 8), dtype=float) * 0.5
    for _, r in plates_df.iterrows():
        xy = cand_xy[r["exposure_id"]]
        if len(xy) > 0:
            gx = np.clip((xy[:, 0] * 8).astype(int), 0, 7)
            gy = np.clip((xy[:, 1] * 8).astype(int), 0, 7)
            for ix, iy in zip(gx, gy):
                global_hist[iy, ix] += 1.0
    global_prof = global_hist / np.mean(global_hist)

    for inst, grp in plates_df.groupby("instrument"):
        h = np.zeros((8, 8), dtype=float)
        tot = 0
        for eid in grp["exposure_id"]:
            xy = cand_xy[eid]
            tot += len(xy)
            if len(xy) > 0:
                gx = np.clip((xy[:, 0] * 8).astype(int), 0, 7)
                gy = np.clip((xy[:, 1] * 8).astype(int), 0, 7)
                for ix, iy in zip(gx, gy):
                    h[iy, ix] += 1.0
        # Shrink toward global profile (empirical Bayes prior weight = 40 counts)
        prior = 40.0 * global_prof / 64.0
        prof = (h + prior) / np.mean(h + prior)
        inst_profiles[inst] = prof

    # Cell weight map per plate = spatial defect profile * local depth factor
    cell_weights: dict[str, np.ndarray] = {}
    for _, r in plates_df.iterrows():
        eid = r["exposure_id"]
        prof = inst_profiles[r["instrument"]]
        # Depth weighting: 10^(0.2 * (local_mlim - m_lim_cal)) clipped to [0.5, 2.0]
        d_m = np.clip(local_grids[eid] - r["m_lim_cal"], -1.5, 1.5)
        w_depth = np.power(10.0, 0.2 * d_m)
        w_tot = prof * w_depth
        cell_weights[eid] = w_tot / np.sum(w_tot)

    return plates_df, cand_xy, cell_weights


def _placebo_shadow_mask_and_share(
    xy: np.ndarray,
    w_grid: np.ndarray,
    target_sf: float,
    rng: np.random.Generator,
) -> tuple[int, int, float]:
    """
    Place a realistic linear/curved Earth-shadow boundary across [0.05, 0.95]^2 at a random angle
    with area fraction ~ target_sf. Returns (n_in, n_total, expected_share_p).
    """
    theta = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    # Center grid coordinates in [-0.45, 0.45]
    pts_g = np.linspace(0.05 + 0.90 / 16.0, 0.95 - 0.90 / 16.0, 8) - 0.5
    gx, gy = np.meshgrid(pts_g, pts_g)
    proj_g = c * gx + s * gy
    # Threshold corresponding to target_sf quantile of projected active plate
    thresh = float(np.quantile(proj_g, target_sf))
    in_cell = proj_g <= thresh
    p_share = float(np.sum(w_grid[in_cell]))
    p_share = float(np.clip(p_share, 0.02, 0.98))

    n_tot = len(xy)
    if n_tot == 0:
        return 0, 0, p_share
    proj_c = c * (xy[:, 0] - 0.5) + s * (xy[:, 1] - 0.5)
    n_in = int(np.sum(proj_c <= thresh))
    return n_in, n_tot, p_share


def _cmh_stat(n_in: np.ndarray, n_tot: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Return (Z_within, Mantel-Haenszel Rate Ratio)."""
    u_val = float(np.sum(n_in - n_tot * p))
    v_val = float(np.sum(n_tot * p * (1.0 - p)))
    z = u_val / np.sqrt(max(v_val, 1e-9))
    num = float(np.sum(n_in * (1.0 - p)))
    den = float(np.sum((n_tot - n_in) * p))
    rr = num / max(den, 1e-9)
    return z, rr


def run_within_plate_power(
    n_sim: int = 1000,
    seed: int = 42,
    scale_to_tier_ab: bool = False,
) -> pd.DataFrame:
    plates_df, cand_xy, cell_weights = load_sunlit_plates_and_candidates()
    exp = pd.read_parquet("data/processed/exposures.parquet")
    qc = pd.read_csv("data/processed/pilot_plate_qc.csv").drop_duplicates("plate_id")
    inst_qc = qc.groupby("instrument")["qc_pass"].mean().to_dict()

    # Partial-shadow plates (0.05 < shadow_fraction < 0.95) from metadata
    if scale_to_tier_ab:
        mask = (exp["is_tier_a"] | exp["is_tier_b"]) & (exp["m_flash_05s"] >= 9.0)
        label = "Tier A+B (m_flash >= 9.0)"
    else:
        mask = exp["is_tier_a"] & (exp["m_flash_05s"] >= 9.0)
        label = "Tier A (m_flash >= 9.0)"

    part = exp[mask & (exp["shadow_fraction"] > 0.05) & (exp["shadow_fraction"] < 0.95)]
    part_by_inst = part.groupby("instrument")["shadow_fraction"].agg(list).to_dict()

    avail_by_inst = {inst: grp["exposure_id"].tolist() for inst, grp in plates_df.groupby("instrument")}
    alloc = []
    tot_exp_qc = 0.0
    for inst, sf_list in part_by_inst.items():
        n_exp = len(sf_list) * inst_qc.get(inst, 0.0)
        tot_exp_qc += n_exp
        n_take = int(round(n_exp))
        if n_take > 0 and inst in avail_by_inst and len(avail_by_inst[inst]) > 0:
            alloc.append((inst, n_take, np.array(sf_list, dtype=float)))

    print(f"\n=== WITHIN-PLATE CONDITIONAL POWER ({label}) ===")
    print(f"Expected QC-passing partial-shadow plates: {tot_exp_qc:.1f} (using {sum(x[1] for x in alloc)} matched sunlit draws)")

    rng = np.random.default_rng(seed)

    def _draw_one_z(delta: float) -> tuple[float, float]:
        n_in_arr, n_tot_arr, p_arr = [], [], []
        for inst, n_take, sf_pool in alloc:
            # Sample plates with replacement if n_take > avail sunlit plates (simulates sampling from same instrument distribution)
            eids = rng.choice(avail_by_inst[inst], size=n_take, replace=(n_take > len(avail_by_inst[inst])))
            sfs = rng.choice(sf_pool, size=n_take, replace=True)
            for eid, sf in zip(eids, sfs):
                n_in, n_tot, p = _placebo_shadow_mask_and_share(cand_xy[eid], cell_weights[eid], float(sf), rng)
                if delta > 0 and n_in > 0:
                    n_kept = int(rng.binomial(n_in, 1.0 - delta))
                    n_tot = n_tot - (n_in - n_kept)
                    n_in = n_kept
                n_in_arr.append(n_in)
                n_tot_arr.append(n_tot)
                p_arr.append(p)
        return _cmh_stat(np.array(n_in_arr), np.array(n_tot_arr), np.array(p_arr))

    # Build >= 1,000 null placebo reference distribution
    null_zs = np.array([_draw_one_z(0.0)[0] for _ in range(n_sim)])
    crit_z_5pct = float(np.quantile(null_zs, 0.05))
    # Empirical FPR using 5% placebo rank threshold (on independent 1000 null draws)
    null_zs_eval = np.array([_draw_one_z(0.0)[0] for _ in range(n_sim)])
    emp_fpr = float(np.mean(null_zs_eval <= crit_z_5pct))
    print(f"Placebo 5th-percentile Z threshold = {crit_z_5pct:.3f} | Calibrated empirical FPR = {emp_fpr:.4f}")

    rows = []
    for delta in [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:
        zs, rrs = [], []
        for _ in range(n_sim):
            z, rr = _draw_one_z(delta)
            zs.append(z)
            rrs.append(rr)
        zs = np.array(zs)
        pwr = float(np.mean(zs <= crit_z_5pct))
        rows.append({
            "tier": label,
            "injected_deficit": delta,
            "placebo_calibrated_power": round(pwr, 4),
            "median_MH_rate_ratio": round(float(np.median(rrs)), 4),
        })
    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False))
    return res_df


if __name__ == "__main__":
    df_a = run_within_plate_power(n_sim=1000, seed=42, scale_to_tier_ab=False)
    df_ab = run_within_plate_power(n_sim=1000, seed=42, scale_to_tier_ab=True)
    pd.concat([df_a, df_ab], ignore_index=True).to_csv("data/processed/within_plate_power_calibration.csv", index=False)
