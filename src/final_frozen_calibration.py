"""Final Frozen Filter (v1.2 — Amendment 1 Complete) + Ecliptic Time-Shifted Placebos + Gaia Star-Density Control.

Evaluates strictly on sunlit (shadow_fraction == 0 at 42,164 / 36,000 / 20,000 km) plates:
  1. Primary Test: Single-Plate Tier A+B (m_flash >= 9.0), within-plate conditional Mantel-Haenszel,
     Final Frozen Filter v1.2, 32x32 local Gaia star-density + detectability + sunlit defect-profile
     + mask expectation p_i, and >=1,000 time-shifted 3D ecliptic umbral placebos (comparing both
     WITH and WITHOUT the local Gaia star-density control, under 53.1% non-stellar dilution).
  2. Key Defect-Controlled Secondary: Strict Partner-Vetoed Paired Tier A (m_flash >= 8.0),
     measuring exact power across 20%--70% deficits and 80%-power MDD under 3D ecliptic umbral geometry.
  3. Secondary Single-Plate Tier A+B (m_flash >= 8.0) with and without Gaia star-density control.
"""

from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, BarycentricTrueEcliptic, EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from src.declump_and_mask import _declump_events
from src.geometry import points_in_umbra
from src.tiers import _angular_sep_deg
from src.within_plate_power import _cmh_stat


def depth_dilution_scale(m_flash: float, m_full: float = 10.67, m_floor: float = 7.0) -> float:
    """Linear depth-dilution scaling: 1.0 at m_flash >= 10.67 (POSS-I-E), scaled linearly below."""
    return float(np.clip((m_flash - m_floor) / (m_full - m_floor), 0.15, 1.0))


def _apply_conc_pk_veto_from_stamps(eid: str, df_all: pd.DataFrame, min_conc: float = 0.03) -> pd.DataFrame:
    """Apply conc_pk_9x9 >= 0.03 veto using saved sunlit_m8_stamps/<eid>.npz when present."""
    sp = Path(f"data/processed/sunlit_m8_stamps/{eid}.npz")
    if not sp.exists() or len(df_all) == 0:
        return df_all
    z = np.load(sp)
    arr = z["snr"]
    if arr.ndim != 3 or len(arr) == 0:
        return df_all
    cy, cx = arr.shape[1] // 2, arr.shape[2] // 2
    pks = np.max(arr[:, cy - 1 : cy + 2, cx - 1 : cx + 2], axis=(1, 2))
    b9 = np.sum(np.maximum(0.0, arr[:, cy - 4 : cy + 5, cx - 4 : cx + 5]), axis=(1, 2))
    bad = (pks / np.maximum(b9, 1e-6)) < min_conc
    if not np.any(bad):
        return df_all
    sc_bad = SkyCoord(z["ra"][bad] * u.deg, z["dec"][bad] * u.deg)
    sc_df = SkyCoord(df_all["ra"].to_numpy() * u.deg, df_all["dec"].to_numpy() * u.deg)
    _, d_bad, _ = sc_df.match_to_catalog_sky(sc_bad)
    out = df_all.copy()
    out.loc[d_bad.arcsec <= 1.0, "psf_pass"] = False
    return out


def compute_gaia_density_grid_32(
    df_all: pd.DataFrame,
    wcs: WCS,
    nx: int,
    ny: int,
    valid_grid_32: np.ndarray,
    sigma_cells: float = 1.5,
) -> np.ndarray:
    """Compute local 32x32 Gaia DR3 source density (psf_pass & gaia_hit to plate depth), normalized on valid cells."""
    g_mask = df_all["psf_pass"].to_numpy(dtype=bool) & df_all["gaia_hit"].to_numpy(dtype=bool)
    g_sub = df_all[g_mask]
    if len(g_sub) > 0:
        xg, yg = wcs.world_to_pixel(SkyCoord(g_sub["ra"].to_numpy() * u.deg, g_sub["dec"].to_numpy() * u.deg))
        h32, _, _ = np.histogram2d(
            np.clip(yg / max(ny, 1), 0.0, 0.99999),
            np.clip(xg / max(nx, 1), 0.0, 0.99999),
            bins=32,
            range=[[0.0, 1.0], [0.0, 1.0]],
        )
        sm32 = gaussian_filter(h32.astype(float), sigma=sigma_cells, mode="nearest") + 0.5 * max(float(np.mean(h32)), 0.1)
    else:
        sm32 = np.ones((32, 32), dtype=float)
    denom = float(np.mean(sm32[valid_grid_32])) if np.any(valid_grid_32) else float(np.mean(sm32))
    return sm32 / max(denom, 1e-9)


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
    partner_m_lim_cal: float = 0.0,
    partner_grid: np.ndarray | None = None,
    ast_coords: SkyCoord | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Apply Final Frozen Filter v1.2 to a single or partner-paired plate's saved catalog."""
    med_r_val = float(med_r) if np.isfinite(med_r) and med_r > 0 else 3.0
    g_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32)
    gx_n, gy_n = np.meshgrid(g_pts, g_pts)
    gx_flat, gy_flat = gx_n.ravel(), gy_n.ravel()

    border_x = max(0.05, 4.0 * med_r_val / max(nx, 1))
    border_y = max(0.05, 4.0 * med_r_val / max(ny, 1))
    grid_valid = (gx_flat >= border_x) & (gx_flat <= 1.0 - border_x) & (gy_flat >= border_y) & (gy_flat <= 1.0 - border_y)

    sc_grid = wcs.pixel_to_world(gx_flat * nx, gy_flat * ny)
    if partner_wcs is not None and partner_nx > 0 and partner_ny > 0:
        xb_g, yb_g = partner_wcs.world_to_pixel(sc_grid)
        b_pad = 4.0 * max(partner_med_r, 1.0)
        grid_valid &= (xb_g >= b_pad) & (xb_g <= partner_nx - 1 - b_pad) & (yb_g >= b_pad) & (yb_g <= partner_ny - 1 - b_pad)

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
        idx_g, _, sep_gs, _ = sc_all.search_around_sky(sc_grid, 15.0 * u.arcsec)
        veto_grid = np.zeros(len(gx_flat), dtype=bool)
        veto_grid[idx_g[sep_gs.arcsec >= 2.0]] = True
        grid_valid &= ~veto_grid

    valid_grid_32 = grid_valid.reshape(32, 32)
    unmasked_frac = float(np.mean(valid_grid_32))

    # 3. Candidate morphology + pixel polarity/saturation filter
    morph_ok = (
        df_all["psf_pass"].to_numpy()
        & (~df_all["gaia_hit"].to_numpy())
        & (df_all["ellip"].to_numpy() <= 0.20)
        & (df_all["r_psf"].to_numpy() >= 0.7 * med_r_val)
        & (df_all["r_psf"].to_numpy() <= 1.5 * med_r_val)
    )
    if "peak_snr" in df_all.columns:
        morph_ok &= (
            (df_all["peak_snr"].to_numpy() >= 4.0)
            & (df_all["min3_snr"].to_numpy() >= -3.0)
            & (df_all["n_sat_pix"].to_numpy() == 0)
        )
    sub = df_all[morph_ok].copy().reset_index(drop=True)

    # 4. Partner plate check + active region (<=120") + 60" neighbor registration (>=0.80)
    if len(sub) > 0 and partner_df is not None and partner_wcs is not None and partner_grid is not None and len(partner_df) > 0:
        sc_sub = SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg)
        xb, yb = partner_wcs.world_to_pixel(sc_sub)
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
            keep_b = (sep_b.arcsec > 15.0) & (sep_b.arcsec <= 120.0) & (edge_psf_b >= 4.0) & (mag_margin >= 1.0)

            bright_lim_ab = min(m_lim_cal, partner_m_lim_cal if partner_m_lim_cal > 0 else m_lim_cal) - 1.0
            ref_mask = df_all["psf_pass"].to_numpy() & np.isfinite(mag_all) & (mag_all < bright_lim_ab)
            if np.any(ref_mask) and np.any(keep_b):
                sc_ref = SkyCoord(df_all["ra"].to_numpy()[ref_mask] * u.deg, df_all["dec"].to_numpy()[ref_mask] * u.deg)
                _, sep_ref_ab, _ = sc_ref.match_to_catalog_sky(sc_b_all)
                for ic in np.where(keep_b)[0]:
                    in60 = sc_sub[ic].separation(sc_ref).arcsec <= 60.0
                    if np.sum(in60) >= 3 and float(np.mean(sep_ref_ab.arcsec[in60] <= 5.0)) < 0.80:
                        keep_b[ic] = False
            sub = sub[keep_b].copy().reset_index(drop=True)

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
        idx_k, _, sep_ks, _ = sc_all.search_around_sky(sc_k, 15.0 * u.arcsec)
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


def build_ecliptic_time_shifted_bank(
    pool_df: pd.DataFrame,
    exp_idx: pd.DataFrame,
    wcs_map: dict[str, tuple[WCS, int, int]],
    ev_map: dict[str, np.ndarray],
    vm_map: dict[str, np.ndarray],
    r_km: float = 42164.0,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build exact 3D nighttime time-shifted ecliptic umbral realizations (points_in_umbra) on near-ecliptic sunlit plates."""
    eids_all = pool_df["exposure_id"].tolist()
    ecl_c = SkyCoord(
        exp_idx.loc[eids_all, "ra_deg"].to_numpy() * u.deg,
        exp_idx.loc[eids_all, "dec_deg"].to_numpy() * u.deg,
    ).transform_to(BarycentricTrueEcliptic())
    near_eids = [eid for eid, b in zip(eids_all, np.abs(ecl_c.lat.deg)) if b <= 20.0]

    gp_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32)
    gxn, gyn = np.meshgrid(gp_pts, gp_pts)
    gx_f, gy_f = gxn.ravel(), gyn.ravel()

    bank: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for eid in near_eids:
            vg_f = vm_map[eid].ravel()
            if not np.any(vg_f):
                continue
            wcs, nx, ny = wcs_map[eid]
            sc_g = wcs.pixel_to_world(gx_f * nx, gy_f * ny)
            dhat_g = np.column_stack([
                np.cos(sc_g.dec.rad) * np.cos(sc_g.ra.rad),
                np.cos(sc_g.dec.rad) * np.sin(sc_g.ra.rad),
                np.sin(sc_g.dec.rad),
            ])
            ev = ev_map[eid]
            if len(ev) > 0:
                sc_e = wcs.pixel_to_world(ev[:, 0] * nx, ev[:, 1] * ny)
                dhat_e = np.column_stack([
                    np.cos(sc_e.dec.rad) * np.cos(sc_e.ra.rad),
                    np.cos(sc_e.dec.rad) * np.sin(sc_e.ra.rad),
                    np.sin(sc_e.dec.rad),
                ])
            else:
                dhat_e = np.zeros((0, 3), dtype=float)

            r = exp_idx.loc[eid]
            mjd0 = float(r["mjd_start"])
            dur_d = float(r["exptime_sec"]) / 86400.0
            ut_mid = ((-float(r["obs_lon_deg"]) / 15.0) % 24.0) / 24.0
            t_scan = Time(np.floor(mjd0) + ut_mid + np.arange(20, 386), format="mjd", scale="utc")
            rs_scan = get_sun(t_scan).cartesian.xyz.to_value(u.km).T
            usun_scan = rs_scan / np.linalg.norm(rs_scan, axis=1, keepdims=True)
            ra0_r, dec0_r = np.deg2rad(float(r["ra_deg"])), np.deg2rad(float(r["dec_deg"]))
            u0 = np.array([np.cos(dec0_r) * np.cos(ra0_r), np.cos(dec0_r) * np.sin(ra0_r), np.sin(dec0_r)])
            mjd_opp = float(t_scan.mjd[int(np.argmax(-usun_scan @ u0))])

            dd, hh = np.meshgrid(np.linspace(-14.0, 14.0, 57), np.linspace(-5.0, 5.0, 21))
            mc = mjd_opp + dd.ravel() + hh.ravel() / 24.0
            mc = mc[np.abs(mc - mjd0) >= 15.0]

            loc = EarthLocation(
                lat=float(r["obs_lat_deg"]) * u.deg,
                lon=float(r["obs_lon_deg"]) * u.deg,
                height=float(r["obs_elev_m"]) * u.m,
            )
            t_mid = Time(mc + 0.5 * dur_d, format="mjd", scale="utc")
            aa = AltAz(obstime=t_mid, location=loc)
            sun_alt = get_sun(t_mid).transform_to(aa).alt.deg
            pt_alt = SkyCoord(float(r["ra_deg"]) * u.deg, float(r["dec_deg"]) * u.deg).transform_to(aa).alt.deg
            mc = mc[(sun_alt < -12.0) & (pt_alt > 10.0)]
            if len(mc) == 0:
                continue

            t3 = Time(np.column_stack([mc, mc + 0.5 * dur_d, mc + dur_d]).ravel(), format="mjd", scale="utc")
            robs3 = loc.get_gcrs(t3).cartesian.xyz.to_value(u.km).T
            rsun3 = get_sun(t3).cartesian.xyz.to_value(u.km).T
            ug = points_in_umbra(robs3, dhat_g, rsun3, r_km).reshape(len(mc), 3, 1024).mean(axis=1) >= 0.5
            sfs = ug[:, vg_f].mean(axis=1)
            ok = np.where((sfs > 0.05) & (sfs < 0.95))[0]
            if len(ok) == 0:
                continue

            ug_ok = ug[ok]
            sfs_ok = sfs[ok]
            if len(dhat_e) > 0:
                ue_ok = points_in_umbra(robs3, dhat_e, rsun3, r_km).reshape(len(mc), 3, len(dhat_e)).mean(axis=1)[ok] >= 0.5
                nin_ok = np.sum(ue_ok, axis=1).astype(int)
            else:
                nin_ok = np.zeros(len(ok), dtype=int)
            bank[eid] = (sfs_ok, ug_ok, nin_ok)
    return bank


def _build_cell_weights(
    pool_df: pd.DataFrame,
    ev_map: dict[str, np.ndarray],
    vm_map: dict[str, np.ndarray],
    lg_map: dict[str, np.ndarray],
    gaia_map: dict[str, np.ndarray],
    use_gaia_density: bool,
    gaia_share: float = 0.469,
) -> dict[str, np.ndarray]:
    """Compute normalized 1024-cell expected weights p_i with or without local 32x32 Gaia star-density control."""
    gh = np.ones((8, 8), dtype=float) * 0.5
    for eid in pool_df["exposure_id"]:
        xy = ev_map[eid]
        if len(xy) > 0:
            for ix, iy in zip(np.clip((xy[:, 0] * 8).astype(int), 0, 7), np.clip((xy[:, 1] * 8).astype(int), 0, 7)):
                gh[iy, ix] += 1.0
    gp = gh / np.mean(gh)

    ip: dict[str, np.ndarray] = {}
    for inst, grp in pool_df.groupby("instrument"):
        h = np.zeros((8, 8), dtype=float)
        for eid in grp["exposure_id"]:
            xy = ev_map[eid]
            if len(xy) > 0:
                for ix, iy in zip(np.clip((xy[:, 0] * 8).astype(int), 0, 7), np.clip((xy[:, 1] * 8).astype(int), 0, 7)):
                    h[iy, ix] += 1.0
        pr = 40.0 * gp / 64.0
        ip[inst] = (h + pr) / np.mean(h + pr)

    cw: dict[str, np.ndarray] = {}
    for _, r in pool_df.iterrows():
        eid = r["exposure_id"]
        vg = vm_map[eid]
        p32 = np.repeat(np.repeat(ip[r["instrument"]], 4, axis=0), 4, axis=1)
        d32 = np.repeat(np.repeat(np.power(10.0, 0.2 * np.clip(lg_map[eid] - r["m_lim_cal"], -1.5, 1.5)), 4, axis=0), 4, axis=1)
        pd_term = p32 * d32
        pd_norm = pd_term / max(float(np.mean(pd_term[vg])) if np.any(vg) else float(np.mean(pd_term)), 1e-9)
        if use_gaia_density:
            raw_w = (gaia_share * gaia_map[eid] + (1.0 - gaia_share) * pd_norm) * vg.astype(float)
        else:
            raw_w = pd_norm * vg.astype(float)
        w_flat = raw_w.ravel()
        s = float(np.sum(w_flat))
        cw[eid] = w_flat / (s if s > 0 else 1.0)
    return cw


def _evaluate_ecliptic_single_pool(
    full_pool_df: pd.DataFrame,
    sub_eids: list[str],
    bank: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ev_map: dict[str, np.ndarray],
    vm_map: dict[str, np.ndarray],
    lg_map: dict[str, np.ndarray],
    gaia_map: dict[str, np.ndarray],
    target_n_plates: int,
    target_sf_pool: np.ndarray,
    use_gaia_density: bool,
    gaia_share: float,
    non_stellar_frac: float,
    deltas: list[float],
    label: str,
    n_sim: int,
    seed: int,
) -> list[dict]:
    """Run 1,000-placebo calibration on near-ecliptic sunlit plates using time-shifted 3D ecliptic umbral geometry."""
    cw = _build_cell_weights(full_pool_df, ev_map, vm_map, lg_map, gaia_map, use_gaia_density, gaia_share)
    mf_map = full_pool_df.set_index("exposure_id")["m_flash_05s"].to_dict()
    ecl_eids = [eid for eid in sub_eids if eid in bank]

    p_bank: dict[str, tuple[np.ndarray, int, np.ndarray, np.ndarray]] = {}
    for eid in ecl_eids:
        sfs_b, ug_b, nin_b = bank[eid]
        p_arr = np.clip(ug_b @ cw[eid], 0.02, 0.98)
        p_bank[eid] = (sfs_b, len(ev_map[eid]), nin_b, p_arr)

    # Pre-bin target shadow fractions across [0.05, 0.95] requiring genuine |eff_sfs - t_sf| <= 0.12 match
    sf_centers = np.linspace(0.075, 0.925, 18)
    bin_tables = []
    for c_sf in sf_centers:
        elig = []
        for eid in ecl_eids:
            sfs_b = p_bank[eid][0]
            ks0 = np.where(np.abs(sfs_b - c_sf) <= 0.12)[0]
            if len(ks0) > 0:
                elig.append((eid, False, ks0))
            ks1 = np.where(np.abs((1.0 - sfs_b) - c_sf) <= 0.12)[0]
            if len(ks1) > 0:
                elig.append((eid, True, ks1))
        bin_tables.append(elig)

    rng = np.random.default_rng(seed)

    # Diagnostic 1-pass unflipped Z across all distinct near-ecliptic sunlit plates
    z_1pass = []
    for _ in range(500):
        ni_1, nt_1, pa_1 = [], [], []
        for eid in ecl_eids:
            sfs_b, n_tot, nin_b, p_arr = p_bank[eid]
            k = int(rng.integers(0, len(sfs_b)))
            ni_1.append(int(nin_b[k]))
            nt_1.append(n_tot)
            pa_1.append(float(p_arr[k]))
        z_1pass.append(_cmh_stat(np.array(ni_1), np.array(nt_1), np.array(pa_1))[0])
    mean_1pass_z = float(np.mean(z_1pass))

    def _draw_z(delta: float, depth_sc: bool = True) -> tuple[float, float]:
        sfs_t = rng.choice(target_sf_pool, size=target_n_plates, replace=True)
        b_idx = np.clip(((sfs_t - 0.05) / 0.05).astype(int), 0, 17)
        ni = np.empty(target_n_plates, dtype=int)
        nt = np.empty(target_n_plates, dtype=int)
        pa = np.empty(target_n_plates, dtype=float)
        for i in range(target_n_plates):
            elig = bin_tables[b_idx[i]]
            eid, flip, ks = elig[int(rng.integers(0, len(elig)))]
            _, n_tot, nin_b, p_arr = p_bank[eid]
            k = int(ks[rng.integers(0, len(ks))])
            n_in = int(n_tot - nin_b[k]) if flip else int(nin_b[k])
            p = float(np.clip(1.0 - p_arr[k], 0.02, 0.98)) if flip else float(p_arr[k])
            if delta > 0 and n_in > 0:
                d_eff = delta * non_stellar_frac * (depth_dilution_scale(mf_map[eid]) if depth_sc else 1.0)
                kept = int(rng.binomial(n_in, max(0.0, 1.0 - d_eff)))
                n_tot -= n_in - kept
                n_in = kept
            ni[i], nt[i], pa[i] = n_in, n_tot, p
        return _cmh_stat(ni, nt, pa)

    null_zs = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])
    crit_z = float(np.quantile(null_zs, 0.05))
    indep_null = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])
    emp_fpr = float(np.mean(indep_null <= crit_z))
    analytic_fpr = float(np.mean(indep_null <= -1.6448536))
    print(
        f"\n=== {label} (N_plates={target_n_plates}, K_ecl_sunlit={len(ecl_eids)}, 1pass_Z={mean_1pass_z:+.2f}) "
        f"| crit_z={crit_z:.3f} | AnalyticFPR={analytic_fpr:.4f} | PlaceboRankFPR={emp_fpr:.4f} ==="
    )

    rows = []
    for delta in deltas:
        zs_d, rrs_d = zip(*[_draw_z(delta, True) for _ in range(n_sim)])
        zs_u, _ = zip(*[_draw_z(delta, False) for _ in range(n_sim)])
        pwr_d = float(np.mean(np.array(zs_d) <= crit_z))
        pwr_u = float(np.mean(np.array(zs_u) <= crit_z))
        med_rr = float(np.median(rrs_d))
        rows.append({
            "test": label,
            "use_gaia_density": use_gaia_density,
            "n_plates": target_n_plates,
            "k_sunlit_pool": len(ecl_eids),
            "one_pass_z": round(mean_1pass_z, 3),
            "delta_full_depth": delta,
            "non_stellar_frac": round(non_stellar_frac, 4),
            "crit_z": round(crit_z, 3),
            "analytic_fpr_neg1p645": round(analytic_fpr, 4),
            "placebo_fpr": round(emp_fpr, 4),
            "power_depth_scaled": round(pwr_d, 4),
            "power_unscaled": round(pwr_u, 4),
            "med_rr_depth_scaled": round(med_rr, 4),
        })
        print(f"  delta={delta:.2f} -> power(depth-scaled)={pwr_d:.4f} | power(unscaled)={pwr_u:.4f} | med_RR={med_rr:.4f}")
    return rows


def _evaluate_paired_tierA_3d_umbra(
    paired_df: pd.DataFrame,
    paired_ev: dict[str, np.ndarray],
    paired_vm: dict[str, np.ndarray],
    paired_lg: dict[str, np.ndarray],
    paired_gaia: dict[str, np.ndarray],
    part_tA: pd.DataFrame,
    inst_qc_pair: dict[str, float],
    deltas: list[float],
    label: str,
    n_sim: int,
    seed: int,
) -> list[dict]:
    """Evaluate Strict Partner-Vetoed Paired Tier A projecting 3D ecliptic umbral geometry onto instrument-matched pilot plates."""
    cw = _build_cell_weights(paired_df, paired_ev, paired_vm, paired_lg, paired_gaia, False, 0.0)
    avail = {inst: grp["exposure_id"].tolist() for inst, grp in paired_df.groupby("instrument")}
    mf_map = paired_df.set_index("exposure_id")["m_flash_05s"].to_dict()
    alloc_tA = [
        (inst, int(round(len(grp) * inst_qc_pair.get(inst, 0.0))), grp.index.to_numpy())
        for inst, grp in part_tA.groupby("instrument")
        if inst in avail and int(round(len(grp) * inst_qc_pair.get(inst, 0.0))) > 0
    ]
    n_pl = sum(x[1] for x in alloc_tA)

    gp_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32) - 0.5
    gxn, gyn = np.meshgrid(gp_pts, gp_pts)
    umb_masks: dict[int, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx_r, r in part_tA.iterrows():
            ra0, dec0 = float(r["ra_deg"]), float(r["dec_deg"])
            fx, fy = float(r["fov_ra_deg"]), float(r["fov_dec_deg"])
            dec_g = np.clip(dec0 + gyn.ravel() * fy, -89.9, 89.9)
            ra_g = ra0 + (gxn.ravel() * fx) / np.maximum(np.cos(np.deg2rad(dec_g)), 0.05)
            dhat = np.column_stack([
                np.cos(np.deg2rad(dec_g)) * np.cos(np.deg2rad(ra_g)),
                np.cos(np.deg2rad(dec_g)) * np.sin(np.deg2rad(ra_g)),
                np.sin(np.deg2rad(dec_g)),
            ])
            loc = EarthLocation(
                lat=float(r["obs_lat_deg"]) * u.deg,
                lon=float(r["obs_lon_deg"]) * u.deg,
                height=float(r["obs_elev_m"]) * u.m,
            )
            mjd0, dur_d = float(r["mjd_start"]), float(r["exptime_sec"]) / 86400.0
            t3 = Time([mjd0, mjd0 + 0.5 * dur_d, mjd0 + dur_d], format="mjd", scale="utc")
            ug = points_in_umbra(
                loc.get_gcrs(t3).cartesian.xyz.to_value(u.km).T,
                dhat,
                get_sun(t3).cartesian.xyz.to_value(u.km).T,
                42164.0,
            ).mean(axis=0) >= 0.5
            umb_masks[int(idx_r)] = ug.reshape(32, 32)

    rng = np.random.default_rng(seed)

    def _draw_z(delta: float, depth_sc: bool = True) -> tuple[float, float]:
        ni, nt, pa = [], [], []
        for inst, n_take, r_idxs in alloc_tA:
            eids = rng.choice(avail[inst], size=n_take, replace=(n_take > len(avail[inst])))
            m_idxs = rng.choice(r_idxs, size=n_take, replace=True)
            flips = rng.random(n_take) < 0.5
            for eid, midx, flip in zip(eids, m_idxs, flips):
                u32 = ~umb_masks[int(midx)] if flip else umb_masks[int(midx)]
                p_raw = float(np.sum(cw[eid][u32.ravel()]))
                if p_raw <= 0.05 or p_raw >= 0.95:
                    # Re-draw a shadow mask that genuinely straddles the valid partner-overlap footprint
                    for _ in range(10):
                        alt_m = int(rng.choice(list(umb_masks.keys())))
                        alt_f = bool(rng.random() < 0.5)
                        u_try = ~umb_masks[alt_m] if alt_f else umb_masks[alt_m]
                        pr_try = float(np.sum(cw[eid][u_try.ravel()]))
                        if 0.05 < pr_try < 0.95:
                            u32, p_raw = u_try, pr_try
                            break
                p = float(np.clip(p_raw, 0.02, 0.98))
                ev = paired_ev[eid]
                n_t = len(ev)
                if n_t > 0:
                    ix = np.clip((ev[:, 0] * 32).astype(int), 0, 31)
                    iy = np.clip((ev[:, 1] * 32).astype(int), 0, 31)
                    n_in = int(np.sum(u32[iy, ix]))
                else:
                    n_in = 0
                if delta > 0 and n_in > 0:
                    d_eff = delta * (depth_dilution_scale(mf_map[eid]) if depth_sc else 1.0)
                    kept = int(rng.binomial(n_in, max(0.0, 1.0 - d_eff)))
                    n_t -= n_in - kept
                    n_in = kept
                ni.append(n_in)
                nt.append(n_t)
                pa.append(p)
        return _cmh_stat(np.array(ni), np.array(nt), np.array(pa))

    null_zs = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])
    crit_z = float(np.quantile(null_zs, 0.05))
    indep_null = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])
    emp_fpr = float(np.mean(indep_null <= crit_z))
    analytic_fpr = float(np.mean(indep_null <= -1.6448536))
    print(
        f"\n=== {label} (N_plates={n_pl}, K_paired_sunlit={len(paired_df)}) "
        f"| crit_z={crit_z:.3f} | AnalyticFPR={analytic_fpr:.4f} | PlaceboRankFPR={emp_fpr:.4f} ==="
    )

    rows = []
    for delta in deltas:
        zs_d, rrs_d = zip(*[_draw_z(delta, True) for _ in range(n_sim)])
        zs_u, _ = zip(*[_draw_z(delta, False) for _ in range(n_sim)])
        pwr_d = float(np.mean(np.array(zs_d) <= crit_z))
        pwr_u = float(np.mean(np.array(zs_u) <= crit_z))
        med_rr = float(np.median(rrs_d))
        rows.append({
            "test": label,
            "use_gaia_density": False,
            "n_plates": n_pl,
            "k_sunlit_pool": len(paired_df),
            "one_pass_z": 0.0,
            "delta_full_depth": delta,
            "non_stellar_frac": 1.0,
            "crit_z": round(crit_z, 3),
            "analytic_fpr_neg1p645": round(analytic_fpr, 4),
            "placebo_fpr": round(emp_fpr, 4),
            "power_depth_scaled": round(pwr_d, 4),
            "power_unscaled": round(pwr_u, 4),
            "med_rr_depth_scaled": round(med_rr, 4),
        })
        print(f"  delta={delta:.2f} -> power(depth-scaled)={pwr_d:.4f} | power(unscaled)={pwr_u:.4f} | med_RR={med_rr:.4f}")
    return rows


def run_final_calibration(n_sim: int = 1000, seed: int = 42) -> None:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"Final Frozen Filter v1.2 (Amendment 1 Frozen) | Base Commit: {commit_hash}")

    exp = pd.read_parquet("data/processed/exposures.parquet")
    exp_idx = exp.set_index("exposure_id")
    pilot_res = pd.read_csv("data/processed/pilot_results.csv")
    qc9 = pd.read_csv("data/processed/pilot_plate_qc.csv")
    fp9 = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    p8 = pd.read_parquet("data/processed/sunlit_m8_plates.parquet")
    ap = pd.read_parquet("data/raw/applause_plate.parquet")
    dn = pd.read_parquet("data/raw/dasch_narrow_plates.parquet")
    dw = pd.read_parquet("data/raw/dasch_wide_plates.parquet")
    cands_old = pd.read_parquet("data/processed/pilot_candidates.parquet")
    ast_rows = cands_old[cands_old["is_asteroid"].fillna(False)]
    ast_by_exp = {eid: SkyCoord(grp["ra"].to_numpy() * u.deg, grp["dec"].to_numpy() * u.deg) for eid, grp in ast_rows.groupby("exp_a")}

    # Hard blindness guard across ALL pilot + sunlit_m8 exposures
    sun_ids = list(set(pilot_res["exp1"]).union(set(pilot_res["exp2"])).union(set(p8["exposure_id"])))
    for col in ["shadow_fraction", "shadow_fraction_36000km", "shadow_fraction_20000km"]:
        mx = float(exp_idx.loc[sun_ids, col].max())
        if mx > 0.0:
            raise RuntimeError(f"BLINDNESS GUARD ABORT: {col}={mx} > 0")
    print(f"Hard blindness guard passed on {len(sun_ids)} sunlit exposures (max shadow_fraction=0.0).")

    ap_map = dict(zip(ap["plate_id"].astype(str), ap["filter"].fillna("").astype(str).str.strip()))
    dasch = pd.concat([dn[["dasch_id", "plate_class"]], dw[["dasch_id", "plate_class"]]])
    dasch_map = dict(zip("DASCH_" + dasch["dasch_id"].astype(str), dasch["plate_class"].fillna("").astype(str).str.strip()))

    def get_emul(pid: str) -> str:
        s = str(pid)
        return dasch_map.get(s, "") if s.startswith("DASCH_") else ap_map.get(s.replace("APPLAUSE_", ""), ap_map.get(s, ""))

    exp["emul"] = exp["plate_id"].astype(str).map(get_emul)
    band_map = dict(zip(qc9["plate_id"].astype(str), qc9["phot_band"].astype(str)))
    pilot_res["emul_ok"] = pilot_res["plate1"].map(get_emul) == pilot_res["plate2"].map(get_emul)
    pilot_res["band_ok"] = pilot_res["plate1"].map(band_map) == pilot_res["plate2"].map(band_map)
    pilot_res["pair_ok"] = pilot_res["qc_pass_pair"] & pilot_res["emul_ok"] & pilot_res["band_ok"]

    # 1. Single-plate v1.2 sunlit pool (209 sunlit_m8 + 139 pilot QC plates, deduplicated)
    single_ev, single_vm, single_lg, single_gaia, single_wcs, single_rows = {}, {}, {}, {}, {}, []
    p8_qc = p8[p8["qc_pass"].fillna(False)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in p8_qc.iterrows():
            eid = r["exposure_id"]
            df_all = _apply_conc_pk_veto_from_stamps(eid, pd.read_parquet(f"data/processed/sunlit_m8_catalogs/{eid}.parquet"))
            wcs = WCS(fits.Header.fromstring(r["wcs_header"]))
            nx, ny = int(r["nx"]), int(r["ny"])
            ga = np.nan_to_num(np.array(json.loads(r["local_mlim_grid_json"]), dtype=float), nan=float(r["m_lim_cal"]))
            ev, vg, uf, _ = _apply_frozen_filter_to_plate(eid, df_all, wcs, nx, ny, float(r["med_r_psf"]), float(r["m_lim_cal"]), ga)
            single_ev[eid], single_vm[eid], single_lg[eid] = ev, vg, ga
            single_gaia[eid] = compute_gaia_density_grid_32(df_all, wcs, nx, ny, vg)
            single_wcs[eid] = (wcs, nx, ny)
            single_rows.append({
                "exposure_id": eid, "instrument": str(r["instrument"]),
                "m_flash_05s": float(r["m_flash_05s"]), "m_lim_cal": float(r["m_lim_cal"]),
                "unmasked_frac": uf, "n_events": len(ev),
            })

        for eid in fp9.index:
            if eid in single_ev or eid not in exp_idx.index:
                continue
            qm = qc9[qc9["exposure_id"] == eid] if "exposure_id" in qc9.columns else qc9[qc9["plate_id"] == fp9.loc[eid, "plate_id"]]
            if len(qm) == 0 or not bool(qm["qc_pass"].iloc[0]):
                continue
            fra = fp9.loc[eid]
            df_all = pd.read_parquet(f"data/processed/pilot_catalogs/{eid}.parquet")
            wcs = WCS(fits.Header.fromstring(fra["wcs_header"]))
            nx, ny = int(fra["nx"]), int(fra["ny"])
            ga = np.nan_to_num(np.array(json.loads(fra["local_mlim_grid_json"]), dtype=float), nan=float(fra["m_lim_cal"]))
            ev, vg, uf, _ = _apply_frozen_filter_to_plate(
                eid, df_all, wcs, nx, ny, float(fra["med_r_psf"]), float(fra["m_lim_cal"]), ga,
                ast_coords=ast_by_exp.get(eid),
            )
            single_ev[eid], single_vm[eid], single_lg[eid] = ev, vg, ga
            single_gaia[eid] = compute_gaia_density_grid_32(df_all, wcs, nx, ny, vg)
            single_wcs[eid] = (wcs, nx, ny)
            er = exp_idx.loc[eid]
            single_rows.append({
                "exposure_id": eid, "instrument": str(er["instrument"]),
                "m_flash_05s": float(er["m_flash_05s"]), "m_lim_cal": float(fra["m_lim_cal"]),
                "unmasked_frac": uf, "n_events": len(ev),
            })
        single_df = pd.DataFrame(single_rows)

        # 2. Strict Partner-Vetoed Paired Tier A v1.2 pool (102 emulsion+band-matched pilot plates)
        paired_ev, paired_vm, paired_lg, paired_gaia, paired_rows = {}, {}, {}, {}, []
        for _, pr in pilot_res[pilot_res["pair_ok"]].iterrows():
            for ea, eb in [(pr["exp1"], pr["exp2"]), (pr["exp2"], pr["exp1"])]:
                if ea in paired_ev:
                    continue
                dfa = pd.read_parquet(f"data/processed/pilot_catalogs/{ea}.parquet")
                dfb = pd.read_parquet(f"data/processed/pilot_catalogs/{eb}.parquet")
                fra, frb = fp9.loc[ea], fp9.loc[eb]
                wa, wb = WCS(fits.Header.fromstring(fra["wcs_header"])), WCS(fits.Header.fromstring(frb["wcs_header"]))
                nxa, nya = int(fra["nx"]), int(fra["ny"])
                ga = np.nan_to_num(np.array(json.loads(fra["local_mlim_grid_json"]), dtype=float), nan=float(fra["m_lim_cal"]))
                gb = np.nan_to_num(np.array(json.loads(frb["local_mlim_grid_json"]), dtype=float), nan=float(frb["m_lim_cal"]))
                ev, vg, uf, _ = _apply_frozen_filter_to_plate(
                    ea, dfa, wa, nxa, nya, float(fra["med_r_psf"]), float(fra["m_lim_cal"]), ga,
                    partner_df=dfb, partner_wcs=wb, partner_nx=int(frb["nx"]), partner_ny=int(frb["ny"]),
                    partner_med_r=float(frb["med_r_psf"]), partner_m_lim_cal=float(frb["m_lim_cal"]),
                    partner_grid=gb, ast_coords=ast_by_exp.get(ea),
                )
                paired_ev[ea], paired_vm[ea], paired_lg[ea] = ev, vg, ga
                paired_gaia[ea] = compute_gaia_density_grid_32(dfa, wa, nxa, nya, vg)
                er = exp_idx.loc[ea]
                paired_rows.append({
                    "exposure_id": ea, "instrument": str(er["instrument"]),
                    "m_flash_05s": float(er["m_flash_05s"]), "m_lim_cal": float(fra["m_lim_cal"]),
                    "unmasked_frac": uf, "n_events": len(ev),
                })
        paired_df = pd.DataFrame(paired_rows)

    print(f"Single-Plate v1.2 pool: {len(single_df)} plates, {single_df['n_events'].sum()} events (mean {single_df['n_events'].mean():.2f}/plate)")
    print(f"Paired Tier A v1.2 pool: {len(paired_df)} plates, {paired_df['n_events'].sum()} events (mean {paired_df['n_events'].mean():.2f}/plate)")

    single_bank = build_ecliptic_time_shifted_bank(single_df, exp_idx, single_wcs, single_ev, single_vm)
    print(
        f"Time-shifted 3D ecliptic umbral bank: {len(single_bank)} sunlit ecliptic plates "
        f"(m_flash >= 9.0: {sum(1 for e in single_bank if exp_idx.loc[e, 'm_flash_05s'] >= 9.0)})"
    )

    qc_comb = pd.concat([
        qc9[["plate_id", "instrument", "qc_pass"]],
        p8[p8["status"] == "ok"][["plate_id", "instrument", "qc_pass"]],
    ], ignore_index=True).drop_duplicates("plate_id")
    inst_qc_single = qc_comb.groupby("instrument")["qc_pass"].mean().to_dict()

    em_sub = pilot_res[pilot_res["emul_ok"]]
    emp_ps = em_sub.groupby("instrument")["pair_ok"].sum().to_dict()
    emp_pc = em_sub.groupby("instrument")["pair_ok"].count().to_dict()
    inst_qc_pair = {
        i: ((emp_ps.get(i, 0) + 5.0 * (p1 ** 1.5) * 0.982) / (emp_pc.get(i, 0) + 5.0) if p1 > 0 else 0.0)
        for i, p1 in inst_qc_single.items()
    }

    m8 = exp[exp["m_flash_05s"] >= 8.0].reset_index(drop=True)
    has_emul_p = np.zeros(len(m8), dtype=bool)
    for (_, _), grp in m8.groupby(["source_archive", "instrument"], sort=False):
        if len(grp) < 2:
            continue
        idx = grp.index.to_numpy()
        mjd = grp["mjd_mid"].to_numpy()
        order = np.argsort(mjd)
        idx_s, mjd_s = idx[order], mjd[order]
        ra_s, dec_s = grp["ra_deg"].to_numpy()[order], grp["dec_deg"].to_numpy()[order]
        fov_r = 0.25 * (grp["fov_ra_deg"].to_numpy()[order] + grp["fov_dec_deg"].to_numpy()[order])
        pid_s = grp["plate_id"].astype(str).to_numpy()[order]
        em_s = grp["emul"].to_numpy()[order]
        for i in range(len(idx_s)):
            lo = int(np.searchsorted(mjd_s, mjd_s[i] - 1.0, side="left"))
            hi = int(np.searchsorted(mjd_s, mjd_s[i] + 1.0, side="right"))
            if hi - lo <= 1:
                continue
            dt = np.abs(mjd_s[lo:hi] - mjd_s[i])
            cm = (dt >= 5.0 / 1440.0) & (dt <= 1.0) & (pid_s[lo:hi] != pid_s[i]) & (em_s[lo:hi] == em_s[i])
            if not np.any(cm):
                continue
            ci = np.arange(lo, hi)[cm]
            if np.any(_angular_sep_deg(ra_s[i], dec_s[i], ra_s[ci], dec_s[ci]) <= 0.5 * np.minimum(fov_r[i], fov_r[ci])):
                has_emul_p[idx_s[i]] = True
    m8["has_emul_p"] = has_emul_p

    part_tAB9 = exp[(exp["m_flash_05s"] >= 9.0) & (exp["shadow_fraction"] > 0.05) & (exp["shadow_fraction"] < 0.95)]
    part_tAB8 = m8[(m8["shadow_fraction"] > 0.05) & (m8["shadow_fraction"] < 0.95)]
    part_tA8 = m8[m8["has_emul_p"] & (m8["shadow_fraction"] > 0.05) & (m8["shadow_fraction"] < 0.95)].reset_index(drop=True)

    n_tAB9 = sum(int(round(len(sfs) * inst_qc_single.get(i, 0.0))) for i, sfs in part_tAB9.groupby("instrument")["shadow_fraction"])
    n_tAB8 = sum(int(round(len(sfs) * inst_qc_single.get(i, 0.0))) for i, sfs in part_tAB8.groupby("instrument")["shadow_fraction"])

    eids_m9 = single_df[single_df["m_flash_05s"] >= 9.0]["exposure_id"].tolist()
    eids_m8 = single_df["exposure_id"].tolist()

    all_rows = []
    # 1. Primary Test (m_flash >= 9.0): WITH vs WITHOUT local 32x32 Gaia star-density control
    all_rows.extend(_evaluate_ecliptic_single_pool(
        single_df, eids_m9, single_bank, single_ev, single_vm, single_lg, single_gaia,
        n_tAB9, part_tAB9["shadow_fraction"].to_numpy(), True, 0.469, 0.531, [0.20, 0.30, 0.40, 0.50],
        "PRIMARY: Single-Plate Tier A+B (m_flash >= 9.0, WITH Gaia density, 53.1% diluted)", n_sim, seed,
    ))
    all_rows.extend(_evaluate_ecliptic_single_pool(
        single_df, eids_m9, single_bank, single_ev, single_vm, single_lg, single_gaia,
        n_tAB9, part_tAB9["shadow_fraction"].to_numpy(), False, 0.0, 0.531, [0.20, 0.30, 0.40, 0.50],
        "PRIMARY ABLATION: Single-Plate Tier A+B (m_flash >= 9.0, NO Gaia density, 53.1% diluted)", n_sim, seed,
    ))
    all_rows.extend(_evaluate_ecliptic_single_pool(
        single_df, eids_m9, single_bank, single_ev, single_vm, single_lg, single_gaia,
        n_tAB9, part_tAB9["shadow_fraction"].to_numpy(), True, 0.469, 1.000, [0.20, 0.30, 0.40, 0.50],
        "PRIMARY UNDILUTED REF: Single-Plate Tier A+B (m_flash >= 9.0, WITH Gaia density, raw deficit)", n_sim, seed,
    ))
    # 2. Key Defect-Controlled Secondary: Strict Partner-Vetoed Paired Tier A (m_flash >= 8.0)
    all_rows.extend(_evaluate_paired_tierA_3d_umbra(
        paired_df, paired_ev, paired_vm, paired_lg, paired_gaia,
        part_tA8, inst_qc_pair, [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.75],
        "KEY SECONDARY: Strict Paired Tier A (m_flash >= 8.0, partner-vetoed)", n_sim, seed,
    ))
    # 3. Secondary Single-Plate Tier A+B (m_flash >= 8.0): WITH vs WITHOUT Gaia star-density control
    all_rows.extend(_evaluate_ecliptic_single_pool(
        single_df, eids_m8, single_bank, single_ev, single_vm, single_lg, single_gaia,
        n_tAB8, part_tAB8["shadow_fraction"].to_numpy(), True, 0.469, 0.531, [0.20, 0.30, 0.40, 0.50],
        "SECONDARY: Single-Plate Tier A+B (m_flash >= 8.0, WITH Gaia density, 53.1% diluted)", n_sim, seed,
    ))
    all_rows.extend(_evaluate_ecliptic_single_pool(
        single_df, eids_m8, single_bank, single_ev, single_vm, single_lg, single_gaia,
        n_tAB8, part_tAB8["shadow_fraction"].to_numpy(), False, 0.0, 0.531, [0.20, 0.30, 0.40, 0.50],
        "SECONDARY ABLATION: Single-Plate Tier A+B (m_flash >= 8.0, NO Gaia density, 53.1% diluted)", n_sim, seed,
    ))

    pd.DataFrame(all_rows).to_csv("data/processed/final_frozen_power_calibration.csv", index=False)


if __name__ == "__main__":
    run_final_calibration(n_sim=1000, seed=42)
