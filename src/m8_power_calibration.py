"""Step 3: 1,000-placebo power & FPR calibration on real m_flash >= 8.0 sunlit plates with depth-scaled deficit injection."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from src.declump_and_mask import build_masked_declumped_data
from src.within_plate_power import _cmh_stat


def depth_dilution_scale(m_flash: float, m_full: float = 10.67, m_floor: float = 7.0) -> float:
    """Linear depth-dilution scaling: 1.0 at m_flash >= 10.67 (POSS-I-E), scaled linearly below."""
    return float(np.clip((m_flash - m_floor) / (m_full - m_floor), 0.15, 1.0))


def run_m8_calibration(n_sim: int = 1000, seed: int = 42) -> None:
    # 1. Load m>=9 pilot plates (masked + 30" declumped)
    p9, ev9, _ = build_masked_declumped_data(cluster_arcsec=30.0)
    exp = pd.read_parquet("data/processed/exposures.parquet").set_index("exposure_id")
    p9["m_flash_05s"] = [float(exp.loc[e, "m_flash_05s"]) for e in p9["exposure_id"]]
    p9["is_tier_a"] = [bool(exp.loc[e, "is_tier_a"]) for e in p9["exposure_id"]]

    # 2. Load newly processed m>=8 sunlit plates
    p8 = pd.read_parquet("data/processed/sunlit_m8_plates.parquet")
    p8 = p8[p8["qc_pass"].fillna(False)].copy()
    ev8_df = pd.read_parquet("data/processed/sunlit_m8_events.parquet")

    # Merge into unified sunlit pool (deduplicating by exposure_id)
    event_xy: dict[str, np.ndarray] = dict(ev9)
    valid_masks: dict[str, np.ndarray] = {}
    local_grids: dict[str, np.ndarray] = {}

    # Re-extract valid_masks & local_grids for p9 from pilot_footprints
    fp9 = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    from src.declump_and_mask import build_masked_declumped_data as _bmd  # already called
    # For p8, grids are stored directly
    for _, r in p8.iterrows():
        eid = r["exposure_id"]
        if eid in event_xy:
            continue
        sub_ev = ev8_df[ev8_df["exposure_id"] == eid]
        event_xy[eid] = sub_ev[["x_norm", "y_norm"]].to_numpy(dtype=float) if len(sub_ev) else np.zeros((0, 2))
        valid_masks[eid] = np.array(json.loads(r["valid_grid_32_json"]), dtype=bool).reshape(32, 32)
        g = np.array(json.loads(r["local_mlim_grid_json"]), dtype=float)
        local_grids[eid] = np.nan_to_num(g, nan=float(r["m_lim_cal"]))

    # Combine plate metadata
    cols = ["exposure_id", "instrument", "is_tier_a", "m_flash_05s", "m_lim_cal", "n_events"]
    p8_sub = p8[cols].copy()
    p9_sub = p9[cols].copy()
    all_sun = pd.concat([p9_sub, p8_sub], ignore_index=True).drop_duplicates("exposure_id").reset_index(drop=True)
    print(f"Combined QC-passed sunlit pool: {len(all_sun)} plates ({len(p9_sub)} pilot + {len(p8_sub)} m>=8)")
    print(all_sun.groupby("instrument")["n_events"].agg(["count", "sum", "mean", "std"]).round(2).to_string())

    # Fit per-instrument 2D spatial defect profile on ALL sunlit plates
    global_hist = np.ones((8, 8), dtype=float) * 0.5
    for eid in all_sun["exposure_id"]:
        xy = event_xy[eid]
        if len(xy) > 0:
            gx = np.clip((xy[:, 0] * 8).astype(int), 0, 7)
            gy = np.clip((xy[:, 1] * 8).astype(int), 0, 7)
            for ix, iy in zip(gx, gy):
                global_hist[iy, ix] += 1.0
    global_prof = global_hist / np.mean(global_hist)

    inst_profs = {}
    for inst, grp in all_sun.groupby("instrument"):
        h = np.zeros((8, 8), dtype=float)
        for eid in grp["exposure_id"]:
            xy = event_xy[eid]
            if len(xy) > 0:
                gx = np.clip((xy[:, 0] * 8).astype(int), 0, 7)
                gy = np.clip((xy[:, 1] * 8).astype(int), 0, 7)
                for ix, iy in zip(gx, gy):
                    h[iy, ix] += 1.0
        prior = 40.0 * global_prof / 64.0
        inst_profs[inst] = (h + prior) / np.mean(h + prior)

    # For p9 plates, use uniform active valid_mask [0.05, 0.95]^2 if not in valid_masks, or cell_weights from build_masked_declumped_data
    _, _, cw9 = build_masked_declumped_data(cluster_arcsec=30.0)
    cell_weights_32: dict[str, np.ndarray] = dict(cw9)
    for _, r in all_sun.iterrows():
        eid = r["exposure_id"]
        if eid in cell_weights_32:
            continue
        prof8 = inst_profs[r["instrument"]]
        prof32 = np.repeat(np.repeat(prof8, 4, axis=0), 4, axis=1)
        d_m = np.clip(local_grids[eid] - r["m_lim_cal"], -1.5, 1.5)
        depth32 = np.repeat(np.repeat(np.power(10.0, 0.2 * d_m), 4, axis=0), 4, axis=1)
        w32 = prof32 * depth32 * valid_masks[eid].astype(float)
        s = float(np.sum(w32))
        cell_weights_32[eid] = w32 / (s if s > 0 else 1.0)

    # Expected QC-passing partial-shadow counts from metadata + observed QC rates
    exp_full = exp.reset_index()
    qc_obs = pd.read_csv("data/processed/pilot_plate_qc.csv").drop_duplicates("plate_id")
    inst_qc = qc_obs.groupby("instrument")["qc_pass"].mean().to_dict()

    avail_by_inst = {inst: grp["exposure_id"].tolist() for inst, grp in all_sun.groupby("instrument")}
    mflash_map = all_sun.set_index("exposure_id")["m_flash_05s"].to_dict()
    rng = np.random.default_rng(seed)
    g_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32) - 0.5
    gx, gy = np.meshgrid(g_pts, g_pts)

    results = []
    for label, mask in [
        ("PRIMARY: Tier A+B (m_flash >= 8.0)", (exp_full["is_tier_a"] | exp_full["is_tier_b"]) & (exp_full["m_flash_05s"] >= 8.0)),
        ("SECONDARY: Tier A (m_flash >= 8.0)", exp_full["is_tier_a"] & (exp_full["m_flash_05s"] >= 8.0)),
        ("SECONDARY: Tier A+B (m_flash >= 9.0)", (exp_full["is_tier_a"] | exp_full["is_tier_b"]) & (exp_full["m_flash_05s"] >= 9.0)),
    ]:
        part = exp_full[mask & (exp_full["shadow_fraction"] > 0.05) & (exp_full["shadow_fraction"] < 0.95)]
        part_by_inst = part.groupby("instrument")["shadow_fraction"].agg(list).to_dict()
        alloc = []
        for inst, sf_list in part_by_inst.items():
            n_take = int(round(len(sf_list) * inst_qc.get(inst, 0.0)))
            if n_take > 0 and inst in avail_by_inst and len(avail_by_inst[inst]) > 0:
                alloc.append((inst, n_take, np.array(sf_list, dtype=float)))

        def _draw_z(delta: float, scale_by_depth: bool = True) -> tuple[float, float]:
            ni, nt, pa = [], [], []
            for inst, n_take, sf_pool in alloc:
                eids = rng.choice(avail_by_inst[inst], size=n_take, replace=(n_take > len(avail_by_inst[inst])))
                sfs = rng.choice(sf_pool, size=n_take, replace=True)
                for eid, sf in zip(eids, sfs):
                    theta = rng.uniform(0.0, 2.0 * np.pi)
                    c, s = np.cos(theta), np.sin(theta)
                    proj_g = c * gx + s * gy
                    w32 = cell_weights_32[eid]
                    valid_proj = proj_g[w32 > 0]
                    thresh = float(np.quantile(valid_proj, float(sf))) if len(valid_proj) else 0.0
                    p = float(np.clip(np.sum(w32[proj_g <= thresh]), 0.02, 0.98))
                    xy = event_xy[eid]
                    n_tot = len(xy)
                    n_in = int(np.sum((c * (xy[:, 0] - 0.5) + s * (xy[:, 1] - 0.5)) <= thresh)) if n_tot > 0 else 0
                    if delta > 0 and n_in > 0:
                        d_eff = delta * (depth_dilution_scale(mflash_map[eid]) if scale_by_depth else 1.0)
                        kept = int(rng.binomial(n_in, max(0.0, 1.0 - d_eff)))
                        n_tot = n_tot - (n_in - kept)
                        n_in = kept
                    ni.append(n_in); nt.append(n_tot); pa.append(p)
            return _cmh_stat(np.array(ni), np.array(nt), np.array(pa))

        null_zs = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])
        crit_z = float(np.quantile(null_zs, 0.05))
        emp_fpr = float(np.mean(np.array([_draw_z(0.0)[0] for _ in range(n_sim)]) <= crit_z))
        print(f"\n=== {label} (N_plates={sum(x[1] for x in alloc)}) | crit_z={crit_z:.3f} | Placebo FPR={emp_fpr:.4f} ===")
        for delta in [0.20, 0.30, 0.40, 0.50]:
            zs_sc, rrs_sc = [], []
            for _ in range(n_sim):
                z, rr = _draw_z(delta, scale_by_depth=True)
                zs_sc.append(z); rrs_sc.append(rr)
            pwr_sc = float(np.mean(np.array(zs_sc) <= crit_z))
            results.append({
                "test": label, "delta_full_depth": delta, "crit_z": round(crit_z, 3),
                "placebo_fpr": round(emp_fpr, 4), "power_depth_scaled": round(pwr_sc, 4),
                "med_rr_depth_scaled": round(float(np.median(rrs_sc)), 4),
            })
            print(f"  delta={delta:.2f} (depth-scaled) -> power={pwr_sc:.4f}, med_RR={np.median(rrs_sc):.4f}")

    pd.DataFrame(results).to_csv("data/processed/m8_measured_power_calibration.csv", index=False)


if __name__ == "__main__":
    run_m8_calibration(n_sim=1000, seed=42)
