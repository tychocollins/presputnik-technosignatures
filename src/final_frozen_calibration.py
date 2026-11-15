"""Final Frozen Filter (v1.0) + Proximity-Veto Area Correction + Blend-Aware Gaia + Depth-Scaled 1000-Placebo Calibration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from src.declump_and_mask import _declump_events
from src.within_plate_power import _cmh_stat


def depth_dilution_scale(m_flash: float, m_full: float = 10.67, m_floor: float = 7.0) -> float:
    """Linear depth-dilution scaling: 1.0 at m_flash >= 10.67 (POSS-I-E), scaled linearly below."""
    return float(np.clip((m_flash - m_floor) / (m_full - m_floor), 0.15, 1.0))


def _apply_frozen_filter_to_plate(
    eid: str,
    df_all: pd.DataFrame,
    wcs: WCS,
    nx: int,
    ny: int,
    med_r: float,
    m_lim_cal: float,
    local_grid: np.ndarray,
    partner_df: pd.DataFrame | None = None,
    partner_wcs: WCS | None = None,
    partner_nx: int = 0,
    partner_ny: int = 0,
    partner_med_r: float = 3.0,
    partner_grid: np.ndarray | None = None,
    ast_coords: SkyCoord | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Apply Final Frozen Filter v1.0 to a single plate's saved catalog:
      - ellip <= 0.20, 0.7 <= r_psf/med_r <= 1.5
      - edge mask >= max(5%, 4 * med_r)
      - bright-star annuli (mag < m_lim_cal - 3.0, radius 20"-60")
      - 2"-15" proximity veto around ALL extracted sources
      - 32x32 valid_mask subtracting ALL three (edge + bright-star + 2"-15" proximity veto)
      - partner all-source > 15" & mag_margin >= 1.0 (if Tier A paired)
      - SkyBoT asteroid removal & 30" de-clustering
    Returns (events_xy_norm, valid_grid_32, unmasked_frac, n_raw_cands).
    """
    med_r_val = float(med_r) if np.isfinite(med_r) and med_r > 0 else 3.0
    g_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32)
    gx_n, gy_n = np.meshgrid(g_pts, g_pts)
    gx_flat, gy_flat = gx_n.ravel(), gy_n.ravel()

    border_x = max(0.05, 4.0 * med_r_val / max(nx, 1))
    border_y = max(0.05, 4.0 * med_r_val / max(ny, 1))
    grid_valid = (gx_flat >= border_x) & (gx_flat <= 1.0 - border_x) & (gy_flat >= border_y) & (gy_flat <= 1.0 - border_y)

    sc_grid = wcs.pixel_to_world(gx_flat * nx, gy_flat * ny)

    # 1. Bright-star annulus mask subtraction on 32x32 grid
    mag_all = df_all["mag_est"].to_numpy(dtype=float)
    bright_idx = np.where(np.isfinite(mag_all) & (mag_all < (m_lim_cal - 3.0)))[0]
    if len(bright_idx) > 0:
        sc_bright = SkyCoord(df_all["ra"].to_numpy()[bright_idx] * u.deg, df_all["dec"].to_numpy()[bright_idx] * u.deg)
        r_mask_as = np.clip(20.0 + 8.0 * np.maximum(0.0, (m_lim_cal - 3.0) - mag_all[bright_idx]), 20.0, 60.0)
        idx_g, idx_b, sep_gb, _ = sc_bright.search_around_sky(sc_grid, 60.0 * u.arcsec)
        for ib, ig, s in zip(idx_b, idx_g, sep_gb.arcsec):
            if s <= r_mask_as[ib]:
                grid_valid[ig] = False
    else:
        sc_bright = None
        r_mask_as = np.zeros(0)

    # 2. Proximity veto (2"-15" discs around all extracted sources) area subtraction on 32x32 grid
    sc_all = SkyCoord(df_all["ra"].to_numpy() * u.deg, df_all["dec"].to_numpy() * u.deg)
    if len(sc_all) > 0:
        idx_g, idx_s, sep_gs, _ = sc_all.search_around_sky(sc_grid, 15.0 * u.arcsec)
        # Subtract 15" disc area (minus tiny 2" core = 98.2% of 15" disc)
        veto_grid = np.zeros(len(gx_flat), dtype=bool)
        veto_grid[idx_g[sep_gs.arcsec >= 2.0]] = True
        grid_valid &= (~veto_grid)

    valid_grid_32 = grid_valid.reshape(32, 32)
    unmasked_frac = float(np.mean(valid_grid_32))

    # 3. Candidate morphology filter
    morph_ok = (
        df_all["psf_pass"].to_numpy()
        & (~df_all["gaia_hit"].to_numpy())
        & (df_all["ellip"].to_numpy() <= 0.20)
        & (df_all["r_psf"].to_numpy() >= 0.7 * med_r_val)
        & (df_all["r_psf"].to_numpy() <= 1.5 * med_r_val)
    )
    sub = df_all[morph_ok].copy().reset_index(drop=True)

    # 4. Partner plate check (Tier A)
    if len(sub) > 0 and partner_df is not None and partner_wcs is not None and partner_grid is not None:
        xb, yb = partner_wcs.world_to_pixel(SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg))
        in_b = (xb >= 0.0) & (xb <= float(partner_nx - 1)) & (yb >= 0.0) & (yb <= float(partner_ny - 1))
        sub = sub[in_b].copy().reset_index(drop=True)
        xb, yb = xb[in_b], yb[in_b]
        if len(sub) > 0:
            edge_px_b = np.minimum(np.minimum(xb, float(partner_nx - 1) - xb), np.minimum(yb, float(partner_ny - 1) - yb))
            edge_psf_b = edge_px_b / max(partner_med_r, 1.0)
            ng = partner_grid.shape[0]
            ix_b = np.clip((xb / max(partner_nx, 1) * ng).astype(int), 0, ng - 1)
            iy_b = np.clip((yb / max(partner_ny, 1) * ng).astype(int), 0, ng - 1)
            mag_margin = partner_grid[iy_b, ix_b] - sub["mag_est"].to_numpy(dtype=float)
            sc_sub = SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg)
            sc_b_all = SkyCoord(partner_df["ra"].to_numpy() * u.deg, partner_df["dec"].to_numpy() * u.deg)
            _, sep_b, _ = sc_sub.match_to_catalog_sky(sc_b_all)
            sub = sub[(sep_b.arcsec > 15.0) & (edge_psf_b >= 4.0) & (mag_margin >= 1.0)].copy().reset_index(drop=True)

    # 5. Edge + bright-star annulus + 2"-15" proximity veto on candidates
    if len(sub) > 0:
        xa, ya = wcs.world_to_pixel(SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg))
        xn, yn = xa / max(nx, 1), ya / max(ny, 1)
        keep = (xn >= border_x) & (xn <= 1.0 - border_x) & (yn >= border_y) & (yn <= 1.0 - border_y)
        sc_k = SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg)
        if sc_bright is not None and len(sc_bright) > 0:
            idx_k, idx_b, sep_kb, _ = sc_bright.search_around_sky(sc_k, 60.0 * u.arcsec)
            for ib, ik, s in zip(idx_b, idx_k, sep_kb.arcsec):
                if s <= r_mask_as[ib]:
                    keep[ik] = False
        # 2"-15" proximity veto against ALL extracted sources
        idx_k, idx_s, sep_ks, _ = sc_all.search_around_sky(sc_k, 15.0 * u.arcsec)
        for ik, s in zip(idx_k, sep_ks.arcsec):
            if 2.0 < s <= 15.0:
                keep[ik] = False
        sub = sub[keep].copy().reset_index(drop=True)
        xn, yn = xn[keep], yn[keep]
    else:
        xn, yn = np.zeros(0), np.zeros(0)

    # 6. SkyBoT asteroid removal
    if len(sub) > 0 and ast_coords is not None and len(ast_coords) > 0:
        sc_k = SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg)
        _, sep_ast, _ = sc_k.match_to_catalog_sky(ast_coords)
        ok_ast = sep_ast.arcsec > 20.0
        sub = sub[ok_ast].copy().reset_index(drop=True)
        xn, yn = xn[ok_ast], yn[ok_ast]

    # 7. 30" de-clustering
    ev_xy = _declump_events(sub["ra"].to_numpy(), sub["dec"].to_numpy(), xn, yn, 30.0) if len(sub) > 0 else np.zeros((0, 2))
    return ev_xy, valid_grid_32, unmasked_frac, len(sub)


def run_final_calibration(n_sim: int = 1000, seed: int = 42) -> None:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"Final Frozen Filter v1.0 | Base Commit: {commit_hash}")

    exp = pd.read_parquet("data/processed/exposures.parquet").set_index("exposure_id")
    pilot_res = pd.read_csv("data/processed/pilot_results.csv")
    qc_pairs = pilot_res[pilot_res["qc_pass_pair"]]
    fp9 = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    cands_old = pd.read_parquet("data/processed/pilot_candidates.parquet")
    ast_rows = cands_old[cands_old["is_asteroid"].fillna(False)]
    ast_by_exp = {eid: SkyCoord(grp["ra"].to_numpy() * u.deg, grp["dec"].to_numpy() * u.deg) for eid, grp in ast_rows.groupby("exp_a")}

    event_xy: dict[str, np.ndarray] = {}
    valid_masks: dict[str, np.ndarray] = {}
    local_grids: dict[str, np.ndarray] = {}
    plate_rows = []

    # 1. Reprocess saved pilot catalogs offline with Final Frozen Filter v1.0
    p9_cat_cache = {}
    for _, pr in qc_pairs.iterrows():
        e1, e2 = pr["exp1"], pr["exp2"]
        for ea, eb in [(e1, e2), (e2, e1)]:
            if ea in event_xy:
                continue
            for eid in (ea, eb):
                if eid not in p9_cat_cache:
                    p9_cat_cache[eid] = pd.read_parquet(f"data/processed/pilot_catalogs/{eid}.parquet")
            fra, frb = fp9.loc[ea], fp9.loc[eb]
            wcs_a = WCS(fits.Header.fromstring(fra["wcs_header"]))
            wcs_b = WCS(fits.Header.fromstring(frb["wcs_header"]))
            ga = np.nan_to_num(np.array(json.loads(fra["local_mlim_grid_json"]), dtype=float), nan=float(fra["m_lim_cal"]))
            gb = np.nan_to_num(np.array(json.loads(frb["local_mlim_grid_json"]), dtype=float), nan=float(frb["m_lim_cal"]))
            ev_xy, vg32, ufrac, n_raw = _apply_frozen_filter_to_plate(
                ea, p9_cat_cache[ea], wcs_a, int(fra["nx"]), int(fra["ny"]), float(fra["med_r_psf"]), float(fra["m_lim_cal"]), ga,
                partner_df=p9_cat_cache[eb], partner_wcs=wcs_b, partner_nx=int(frb["nx"]), partner_ny=int(frb["ny"]),
                partner_med_r=float(frb["med_r_psf"]), partner_grid=gb, ast_coords=ast_by_exp.get(ea),
            )
            event_xy[ea] = ev_xy
            valid_masks[ea] = vg32
            local_grids[ea] = ga
            er = exp.loc[ea]
            plate_rows.append({
                "exposure_id": ea, "instrument": str(er["instrument"]), "is_tier_a": bool(er["is_tier_a"]),
                "m_flash_05s": float(er["m_flash_05s"]), "m_lim_cal": float(fra["m_lim_cal"]),
                "n_src_all": len(p9_cat_cache[ea]), "unmasked_frac": ufrac, "n_events": len(ev_xy),
            })

    # 2. Reprocess saved sunlit_m8 catalogs offline with Final Frozen Filter v1.0
    m8_path = Path("data/processed/sunlit_m8_plates.parquet")
    if m8_path.exists():
        p8 = pd.read_parquet(m8_path)
        p8 = p8[p8["qc_pass"].fillna(False)].copy()
        for _, r in p8.iterrows():
            eid = r["exposure_id"]
            if eid in event_xy:
                continue
            cat_p = Path(f"data/processed/sunlit_m8_catalogs/{eid}.parquet")
            if not cat_p.exists():
                continue
            df_all = pd.read_parquet(cat_p)
            wcs = WCS(fits.Header.fromstring(r["wcs_header"]))
            ga = np.nan_to_num(np.array(json.loads(r["local_mlim_grid_json"]), dtype=float), nan=float(r["m_lim_cal"]))
            ev_xy, vg32, ufrac, _ = _apply_frozen_filter_to_plate(
                eid, df_all, wcs, int(r["nx"]), int(r["ny"]), float(r["med_r_psf"]), float(r["m_lim_cal"]), ga,
            )
            event_xy[eid] = ev_xy
            valid_masks[eid] = vg32
            local_grids[eid] = ga
            plate_rows.append({
                "exposure_id": eid, "instrument": str(r["instrument"]), "is_tier_a": bool(r["is_tier_a"]),
                "m_flash_05s": float(r["m_flash_05s"]), "m_lim_cal": float(r["m_lim_cal"]),
                "n_src_all": len(df_all), "unmasked_frac": ufrac, "n_events": len(ev_xy),
            })

    all_sun = pd.DataFrame(plate_rows)
    print(f"Unified QC-passed sunlit pool (Final Frozen Filter v1.0): {len(all_sun)} plates, {all_sun['n_events'].sum()} events")
    print(all_sun.groupby("instrument")["n_events"].agg(["count", "sum", "mean", "std"]).round(2).to_string())

    # Spatial defect profile fit on sunlit plates only
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

    cell_weights_32: dict[str, np.ndarray] = {}
    for _, r in all_sun.iterrows():
        eid = r["exposure_id"]
        prof32 = np.repeat(np.repeat(inst_profs[r["instrument"]], 4, axis=0), 4, axis=1)
        d_m = np.clip(local_grids[eid] - r["m_lim_cal"], -1.5, 1.5)
        depth32 = np.repeat(np.repeat(np.power(10.0, 0.2 * d_m), 4, axis=0), 4, axis=1)
        w32 = prof32 * depth32 * valid_masks[eid].astype(float)
        s = float(np.sum(w32))
        cell_weights_32[eid] = w32 / (s if s > 0 else 1.0)

    # Run 1,000-placebo depth-scaled calibration
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

        def _draw_z(delta: float) -> tuple[float, float]:
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
                        d_eff = delta * depth_dilution_scale(mflash_map[eid])
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
            zs, rrs = [], []
            for _ in range(n_sim):
                z, rr = _draw_z(delta)
                zs.append(z); rrs.append(rr)
            pwr = float(np.mean(np.array(zs) <= crit_z))
            results.append({
                "test": label, "delta_full_depth": delta, "crit_z": round(crit_z, 3),
                "placebo_fpr": round(emp_fpr, 4), "power_depth_scaled": round(pwr, 4),
                "med_rr_depth_scaled": round(float(np.median(rrs)), 4),
            })
            print(f"  delta={delta:.2f} (depth-scaled) -> power={pwr:.4f}, med_RR={np.median(rrs):.4f}")

    pd.DataFrame(results).to_csv("data/processed/final_frozen_power_calibration.csv", index=False)


if __name__ == "__main__":
    run_final_calibration(n_sim=1000, seed=42)
