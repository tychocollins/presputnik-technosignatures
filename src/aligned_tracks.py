"""Pre-registered Secondary Test (Amendment 1): Aligned Multi-Flash Tracks & GEO Orientation Test.

Frozen specification:
1. Track definition: >= 3 clean, de-clustered candidates (all absent on partner plate for Tier A pairs)
   collinear within perpendicular scatter <= 1.0 PSF FWHM (2.3548 * med_r_psf pixels) and consecutive
   along-track spacing ratios in [0.75, 1.25] (+-25%), deduplicated to maximal chains.
2. Per-plate null: >= 1,000 Monte Carlo shuffles per plate drawing N positions from the plate's
   unmasked valid footprint weighted by depth-scaled local detectability and respecting the 30"
   de-clustering floor.
3. Orientation test: compare undirected track sky position angles theta in [0, 180) deg (E of N)
   against (a) the topocentric apparent drift PA of a GEO-altitude (r = 42,164 km) object during a
   sidereally tracked exposure (axial V-statistic cos(2*(theta - theta_GEO))), and (b) the plate/scanner
   pixel x/y axes (flagging tracks within <= 5 deg of either pixel axis).
4. Shadow combination: evaluate track counts inside vs. outside Earth's shadow only after unblinding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation, GCRS, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from src.declump_and_mask import _declump_events

FWHM_FACTOR = 2.354820045
SPACING_RATIO_MIN = 0.75
SPACING_RATIO_MAX = 1.25
MIN_AXIS_SEP_DEG = 20.0
R_GEO_KM = 42164.0
OMEGA_EARTH_RAD_S = 7.2921150e-5


def axial_diff_deg(pa1_deg: float | np.ndarray, pa2_deg: float | np.ndarray) -> float | np.ndarray:
    """Smallest undirected axial angle difference in [0, 90] degrees."""
    d = np.abs(np.asarray(pa1_deg) - np.asarray(pa2_deg)) % 180.0
    return np.minimum(d, 180.0 - d)


def predict_geo_drift_pa_deg(
    ra_deg: float,
    dec_deg: float,
    mjd_mid: float,
    obs_lat_deg: float,
    obs_lon_deg: float,
    obs_elev_m: float = 0.0,
) -> float:
    """Predict undirected sky PA [0, 180) deg (E of N) of apparent star-relative drift at r_GEO = 42,164 km.

    Uses a prograde circular orbit at geocentric radius r_GEO = 42,164 km at its declination apex
    (dot(dec_geo) = 0, dot(ra_geo) = +omega_Earth) observed topocentrically during a sidereally
    tracked exposure.
    """
    t0 = Time(mjd_mid, format="mjd")
    dt_s = 30.0
    t_minus = Time(mjd_mid - dt_s / 86400.0, format="mjd")
    t_plus = Time(mjd_mid + dt_s / 86400.0, format="mjd")
    loc = EarthLocation(lat=obs_lat_deg * u.deg, lon=obs_lon_deg * u.deg, height=obs_elev_m * u.m)

    obs_gcrs_0 = loc.get_gcrs(t0).cartesian.xyz.to_value(u.km)
    los = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").cartesian.xyz.value
    # Solve |obs_gcrs_0 + rho * los|^2 = R_GEO_KM^2 for topocentric range rho > 0
    b = 2.0 * float(np.dot(obs_gcrs_0, los))
    c = float(np.dot(obs_gcrs_0, obs_gcrs_0)) - R_GEO_KM**2
    disc = max(0.0, b * b - 4.0 * c)
    rho = 0.5 * (-b + np.sqrt(disc))
    pos_geo_0 = obs_gcrs_0 + rho * los

    # Prograde orbital angular velocity around Z_GCRS at GEO synchronous rate: v = omega_Earth x pos_geo_0
    vel_geo = np.array([-OMEGA_EARTH_RAD_S * pos_geo_0[1], OMEGA_EARTH_RAD_S * pos_geo_0[0], 0.0])
    pos_minus = pos_geo_0 - vel_geo * dt_s
    pos_plus = pos_geo_0 + vel_geo * dt_s

    obs_minus = loc.get_gcrs(t_minus).cartesian.xyz.to_value(u.km)
    obs_plus = loc.get_gcrs(t_plus).cartesian.xyz.to_value(u.km)
    topo_m = pos_minus - obs_minus
    topo_p = pos_plus - obs_plus

    sc_m = SkyCoord(x=topo_m[0], y=topo_m[1], z=topo_m[2], unit="km", representation_type="cartesian", frame="icrs")
    sc_p = SkyCoord(x=topo_p[0], y=topo_p[1], z=topo_p[2], unit="km", representation_type="cartesian", frame="icrs")
    pa = float(sc_m.position_angle(sc_p).to_value(u.deg)) % 180.0
    return pa


def plate_axes_pa_deg(wcs: WCS, nx: int, ny: int) -> tuple[float, float]:
    """Return undirected sky PAs [0, 180) deg of plate +x and +y pixel axes at plate center."""
    cx, cy = 0.5 * nx, 0.5 * ny
    step = max(10.0, 0.02 * min(nx, ny))
    sc0 = wcs.pixel_to_world(cx, cy)
    scx = wcs.pixel_to_world(cx + step, cy)
    scy = wcs.pixel_to_world(cx, cy + step)
    pa_x = float(sc0.position_angle(scx).to_value(u.deg)) % 180.0
    pa_y = float(sc0.position_angle(scy).to_value(u.deg)) % 180.0
    return pa_x, pa_y


def find_aligned_tracks(
    xy_px: np.ndarray,
    fwhm_px: float,
    ratio_min: float = SPACING_RATIO_MIN,
    ratio_max: float = SPACING_RATIO_MAX,
    max_points: int = 200,
) -> list[tuple[int, ...]]:
    """Find maximal collinear evenly-spaced tracks (>= 3 points) in pixel coordinates.

    Frozen tolerances:
    - Orthogonal distance of every track member to best-fit line through endpoints <= 1.0 * fwhm_px
    - Consecutive interval ratios d_{k+1} / d_k in [0.75, 1.25]
    """
    n = len(xy_px)
    if n < 3 or n > max_points:
        return []

    pts = np.asarray(xy_px, dtype=float)
    raw_chains: set[tuple[int, ...]] = set()

    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n):
            dvec = pts[j] - pi
            dist_ij = float(np.hypot(dvec[0], dvec[1]))
            if dist_ij <= 2.0 * fwhm_px:
                continue
            ux, uy = dvec[0] / dist_ij, dvec[1] / dist_ij
            nx_v, ny_v = -uy, ux

            rel = pts - pi
            perp = np.abs(rel[:, 0] * nx_v + rel[:, 1] * ny_v)
            col_idx = np.where(perp <= fwhm_px)[0]
            if len(col_idx) < 3:
                continue

            s_vals = rel[col_idx, 0] * ux + rel[col_idx, 1] * uy
            order = np.argsort(s_vals)
            c_idx = col_idx[order]
            c_s = s_vals[order]
            m = len(c_idx)

            # Find all valid evenly-spaced chains of length >= 3 along this sorted line
            # dp[(a, b)] = list of chains ending at indices (a, b) in 0..m-1
            chains_end: dict[tuple[int, int], list[list[int]]] = {}
            for a in range(m - 1):
                for b in range(a + 1, m):
                    dab = c_s[b] - c_s[a]
                    if dab <= 2.0 * fwhm_px:
                        continue
                    extended = False
                    for prev_a in range(a):
                        if (prev_a, a) not in chains_end:
                            continue
                        d_prev = c_s[a] - c_s[prev_a]
                        rat = dab / d_prev
                        if ratio_min <= rat <= ratio_max:
                            for ch in chains_end[(prev_a, a)]:
                                chains_end.setdefault((a, b), []).append(ch + [int(c_idx[b])])
                            extended = True
                    if not extended:
                        chains_end.setdefault((a, b), []).append([int(c_idx[a]), int(c_idx[b])])

            for ch_list in chains_end.values():
                for ch in ch_list:
                    if len(ch) >= 3:
                        # Refit orthogonal scatter against first-to-last endpoint of chain
                        p0, p1 = pts[ch[0]], pts[ch[-1]]
                        dv = p1 - p0
                        L = float(np.hypot(dv[0], dv[1]))
                        if L <= 0:
                            continue
                        nnx, nny = -dv[1] / L, dv[0] / L
                        sub_rel = pts[ch] - p0
                        max_perp = float(np.max(np.abs(sub_rel[:, 0] * nnx + sub_rel[:, 1] * nny)))
                        if max_perp <= fwhm_px:
                            canon = tuple(ch) if ch[0] < ch[-1] else tuple(reversed(ch))
                            raw_chains.add(canon)

    if not raw_chains:
        return []

    # Remove strict subsets of longer tracks
    sorted_chains = sorted(raw_chains, key=lambda t: (-len(t), t))
    maximal: list[tuple[int, ...]] = []
    maximal_sets: list[set[int]] = []
    for ch in sorted_chains:
        s_ch = set(ch)
        if not any(s_ch.issubset(ms) for ms in maximal_sets):
            maximal.append(ch)
            maximal_sets.append(s_ch)
    return maximal


def verify_applause_dual_scan_track(
    track_ra_deg: np.ndarray,
    track_dec_deg: np.ndarray,
    wcs_y_scan: WCS,
    df_y_scan: pd.DataFrame,
    med_r_y: float,
    match_tol_as: float = 5.0,
    pa_tol_deg: float = 3.0,
) -> bool:
    """APPLAUSE orthogonal-scan sub-test: verify a track on scan X is sky-fixed on rotated scan Y.

    Scanner artifacts rotate by 90 deg in sky PA between orthogonal X and Y scans, whereas a real
    plate emulsion/sky track remains at identical sky coordinates and identical sky PA.
    """
    if len(track_ra_deg) < 3 or len(df_y_scan) < 3:
        return False
    sc_tr = SkyCoord(track_ra_deg * u.deg, track_dec_deg * u.deg)
    sc_y = SkyCoord(df_y_scan["ra"].to_numpy(dtype=float) * u.deg, df_y_scan["dec"].to_numpy(dtype=float) * u.deg)
    idx_y, sep_y, _ = sc_tr.match_to_catalog_sky(sc_y)
    if np.any(sep_y.arcsec > match_tol_as):
        return False
    matched = df_y_scan.iloc[idx_y]
    xy_y = np.column_stack([matched["x"].to_numpy(dtype=float), matched["y"].to_numpy(dtype=float)])
    chains_y = find_aligned_tracks(xy_y, FWHM_FACTOR * med_r_y)
    if not any(len(ch) == len(track_ra_deg) for ch in chains_y):
        return False
    pa_x_tr = float(sc_tr[0].position_angle(sc_tr[-1]).to_value(u.deg)) % 180.0
    sc_y0 = wcs_y_scan.pixel_to_world(xy_y[0, 0], xy_y[0, 1])
    sc_y1 = wcs_y_scan.pixel_to_world(xy_y[-1, 0], xy_y[-1, 1])
    pa_y_tr = float(sc_y0.position_angle(sc_y1).to_value(u.deg)) % 180.0
    return float(axial_diff_deg(pa_x_tr, pa_y_tr)) <= pa_tol_deg


def sample_plate_shuffle_exact(
    n_raw_cands: int,
    nx_a: int,
    ny_a: int,
    bx: float,
    by: float,
    wcs_a: WCS,
    wcs_b: WCS,
    nx_b: int,
    ny_b: int,
    med_r_b: float,
    grid_a: np.ndarray,
    sc_br: SkyCoord | None,
    r_br: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_raw_cands positions respecting exact border, partner-overlap, and bright-star masks,
    then apply the exact frozen 30\" sky-arcsec _declump_events procedure used on the real data.
    """
    if n_raw_cands <= 0:
        return np.zeros((0, 2), dtype=float)

    ng = grid_a.shape[0]
    med_m = float(np.median(grid_a))
    # Frozen Amendment 1 detectability weight: 10^(0.2 * clip(local_mlim - med, -1.5, 1.5))
    cell_w = (10.0 ** (0.2 * np.clip(grid_a - med_m, -1.5, 1.5))).ravel()
    cell_p = cell_w / float(np.sum(cell_w))

    kept_x: list[float] = []
    kept_y: list[float] = []
    kept_ra: list[float] = []
    kept_dec: list[float] = []
    attempts = 0
    max_attempts = max(200, n_raw_cands * 25)
    while len(kept_x) < n_raw_cands and attempts < max_attempts:
        need = max(8, (n_raw_cands - len(kept_x)) * 2)
        attempts += need
        cells = rng.choice(len(cell_p), size=need, replace=True, p=cell_p)
        iy = cells // ng
        ix = cells % ng
        xs = (ix + rng.random(need)) / ng * nx_a
        ys = (iy + rng.random(need)) / ng * ny_a
        in_border = (xs / nx_a >= bx) & (xs / nx_a <= 1.0 - bx) & (ys / ny_a >= by) & (ys / ny_a <= 1.0 - by)
        xs, ys = xs[in_border], ys[in_border]
        if len(xs) == 0:
            continue
        sc_cand = wcs_a.pixel_to_world(xs, ys)
        xb, yb = wcs_b.world_to_pixel(sc_cand)
        ok = (xb >= 2.0 * med_r_b) & (xb <= nx_b - 1 - 2.0 * med_r_b) & (yb >= 2.0 * med_r_b) & (yb <= ny_b - 1 - 2.0 * med_r_b)
        if sc_br is not None and len(sc_br) > 0 and np.any(ok):
            ik, ib, s_kb, _ = sc_br.search_around_sky(sc_cand, 60.0 * u.arcsec)
            for j_b, j_k, s_v in zip(ib, ik, s_kb.arcsec):
                if s_v <= r_br[j_b]:
                    ok[j_k] = False
        for xv, yv, rav, decv in zip(xs[ok], ys[ok], sc_cand.ra.deg[ok], sc_cand.dec.deg[ok]):
            kept_x.append(float(xv))
            kept_y.append(float(yv))
            kept_ra.append(float(rav))
            kept_dec.append(float(decv))
            if len(kept_x) == n_raw_cands:
                break

    if len(kept_x) == 0:
        return np.zeros((0, 2), dtype=float)

    # Apply the exact same _declump_events(..., 30.0) sky-arcsec de-clustering as the real data
    xn = np.asarray(kept_x, dtype=float) / max(nx_a, 1)
    yn = np.asarray(kept_y, dtype=float) / max(ny_a, 1)
    ev_norm = _declump_events(np.asarray(kept_ra), np.asarray(kept_dec), xn, yn, 30.0)
    if len(ev_norm) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.column_stack([ev_norm[:, 0] * nx_a, ev_norm[:, 1] * ny_a])


def calibrate_pilot_sunlit_tracks(n_shuffles: int = 1000, seed: int = 20261114) -> dict[str, Any]:
    """Calibrate aligned-track rate and orientation statistics on strictly-sunlit emulsion-matched Tier A pilot plates."""
    res = pd.read_csv("data/processed/pilot_results.csv")
    qc = pd.read_csv("data/processed/pilot_plate_qc.csv")
    exp = pd.read_parquet("data/processed/exposures.parquet").set_index("exposure_id")
    fp = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    ap = pd.read_parquet("data/raw/applause_plate.parquet")
    dn = pd.read_parquet("data/raw/dasch_narrow_plates.parquet")
    dw = pd.read_parquet("data/raw/dasch_wide_plates.parquet")
    cands_old = pd.read_parquet("data/processed/pilot_candidates.parquet")

    # Hard blindness guard: strictly sunlit (shadow_fraction == 0)
    pilot_exps = list(set(res["exp1"]).union(set(res["exp2"])))
    max_shad = float(exp.loc[pilot_exps, "shadow_fraction"].max())
    if max_shad > 0.0:
        raise RuntimeError(f"BLINDNESS GUARD ABORT: pilot exposure has shadow_fraction={max_shad}")

    band_map = dict(zip(qc["plate_id"].astype(str), qc["phot_band"].astype(str)))
    ap_map = dict(zip(ap["plate_id"].astype(str), ap["filter"].fillna("").astype(str).str.strip()))
    dasch = pd.concat([dn[["dasch_id", "plate_class"]], dw[["dasch_id", "plate_class"]]])
    dasch_map = dict(zip("DASCH_" + dasch["dasch_id"].astype(str), dasch["plate_class"].fillna("").astype(str).str.strip()))

    def get_emul(pid: str) -> str:
        if pid.startswith("DASCH_"):
            return dasch_map.get(pid, "")
        return ap_map.get(pid.replace("APPLAUSE_", ""), ap_map.get(pid, ""))

    # Require QC-pass pair AND identical phot_band AND identical filter/plate_class
    qc_pairs = res[res["qc_pass_pair"]].copy()
    qc_pairs = qc_pairs[
        (qc_pairs["plate1"].map(band_map) == qc_pairs["plate2"].map(band_map))
        & (qc_pairs["plate1"].map(get_emul) == qc_pairs["plate2"].map(get_emul))
    ].reset_index(drop=True)

    ast_rows = cands_old[cands_old["is_asteroid"].fillna(False)]
    ast_by_exp = {
        eid: SkyCoord(grp["ra"].to_numpy() * u.deg, grp["dec"].to_numpy() * u.deg)
        for eid, grp in ast_rows.groupby("exp_a")
    }

    rng = np.random.default_rng(seed)
    null_totals = np.zeros(n_shuffles, dtype=int)
    null_unconfounded_totals = np.zeros(n_shuffles, dtype=int)
    obs_tracks: list[dict[str, Any]] = []
    null_v_geo_unconfounded: list[float] = []
    axis_sep_list: list[float] = []
    n_plates_eval = 0
    n_plates_unconfounded = 0
    n_plates_xy_scans = 0
    total_raw_cands_eval = 0
    total_cands_eval = 0

    seen_ea: set[str] = set()
    for _, pr in qc_pairs.iterrows():
        for ea, eb in [(pr["exp1"], pr["exp2"]), (pr["exp2"], pr["exp1"])]:
            if ea in seen_ea:
                continue
            seen_ea.add(ea)
            df_a = pd.read_parquet(f"data/processed/pilot_catalogs/{ea}.parquet")
            df_b = pd.read_parquet(f"data/processed/pilot_catalogs/{eb}.parquet")
            fr_a, fr_b = fp.loc[ea], fp.loc[eb]
            wcs_a = WCS(fits.Header.fromstring(fr_a["wcs_header"]))
            wcs_b = WCS(fits.Header.fromstring(fr_b["wcs_header"]))
            grid_b = np.nan_to_num(np.array(json.loads(fr_b["local_mlim_grid_json"]), dtype=float), nan=float(fr_b["m_lim_cal"]))
            grid_a = np.nan_to_num(np.array(json.loads(fr_a["local_mlim_grid_json"]), dtype=float), nan=float(fr_a["m_lim_cal"]))

            nx_a, ny_a = int(fr_a["nx"]), int(fr_a["ny"])
            nx_b, ny_b = int(fr_b["nx"]), int(fr_b["ny"])
            med_r_a = float(fr_a["med_r_psf"]) if np.isfinite(fr_a["med_r_psf"]) and fr_a["med_r_psf"] > 0 else 3.0
            med_r_b = float(fr_b["med_r_psf"]) if np.isfinite(fr_b["med_r_psf"]) and fr_b["med_r_psf"] > 0 else 3.0
            m_lim_a = float(fr_a["m_lim_cal"])

            # Bright-star mask & 32x32 valid grid on Plate A
            g_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32)
            gx_n, gy_n = np.meshgrid(g_pts, g_pts)
            gx_f, gy_f = gx_n.ravel(), gy_n.ravel()
            bx = max(0.05, 4.0 * med_r_a / max(nx_a, 1))
            by = max(0.05, 4.0 * med_r_a / max(ny_a, 1))
            valid_32 = (gx_f >= bx) & (gx_f <= 1.0 - bx) & (gy_f >= by) & (gy_f <= 1.0 - by)
            # Partner overlap check on 32x32 grid
            sc_g = wcs_a.pixel_to_world(gx_f * nx_a, gy_f * ny_a)
            xb_g, yb_g = wcs_b.world_to_pixel(sc_g)
            valid_32 &= (xb_g >= 2.0 * med_r_b) & (xb_g <= nx_b - 1 - 2.0 * med_r_b) & (yb_g >= 2.0 * med_r_b) & (yb_g <= ny_b - 1 - 2.0 * med_r_b)

            mag_all_a = df_a["mag_est"].to_numpy(dtype=float)
            br_idx = np.where(np.isfinite(mag_all_a) & (mag_all_a < m_lim_a - 3.0))[0]
            sc_br = SkyCoord(df_a["ra"].to_numpy()[br_idx] * u.deg, df_a["dec"].to_numpy()[br_idx] * u.deg) if len(br_idx) > 0 else None
            r_br = np.clip(20.0 + 8.0 * np.maximum(0.0, (m_lim_a - 3.0) - mag_all_a[br_idx]), 20.0, 60.0) if len(br_idx) > 0 else np.zeros(0)
            if sc_br is not None and len(sc_br) > 0:
                ig, ib, s_gb, _ = sc_br.search_around_sky(sc_g, 60.0 * u.arcsec)
                for j_b, j_g, s_v in zip(ib, ig, s_gb.arcsec):
                    if s_v <= r_br[j_b]:
                        valid_32[j_g] = False

            # Candidate filtering (frozen morphology + partner all-source 15" absence + partner m_lim margin + SkyBoT + bright mask)
            morph = (
                df_a["psf_pass"].to_numpy()
                & (~df_a["gaia_hit"].to_numpy())
                & (df_a["ellip"].to_numpy() <= 0.20)
                & (df_a["r_psf"].to_numpy() >= 0.7 * med_r_a)
                & (df_a["r_psf"].to_numpy() <= 1.5 * med_r_a)
            )
            sub = df_a[morph].copy().reset_index(drop=True)
            if len(sub) > 0:
                xa, ya = wcs_a.world_to_pixel(SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg))
                keep = (xa / nx_a >= bx) & (xa / nx_a <= 1.0 - bx) & (ya / ny_a >= by) & (ya / ny_a <= 1.0 - by)
                sc_s = SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg)
                xb, yb = wcs_b.world_to_pixel(sc_s)
                keep &= (xb >= 2.0 * med_r_b) & (xb <= nx_b - 1 - 2.0 * med_r_b) & (yb >= 2.0 * med_r_b) & (yb <= ny_b - 1 - 2.0 * med_r_b)
                ng = grid_b.shape[0]
                ix_b = np.clip((xb / max(nx_b, 1) * ng).astype(int), 0, ng - 1)
                iy_b = np.clip((yb / max(ny_b, 1) * ng).astype(int), 0, ng - 1)
                keep &= (grid_b[iy_b, ix_b] - sub["mag_est"].to_numpy(dtype=float)) >= 1.0
                if len(df_b) > 0:
                    sc_b_all = SkyCoord(df_b["ra"].to_numpy() * u.deg, df_b["dec"].to_numpy() * u.deg)
                    _, sep_b, _ = sc_s.match_to_catalog_sky(sc_b_all)
                    keep &= sep_b.arcsec > 15.0
                if ea in ast_by_exp:
                    _, sep_ast, _ = sc_s.match_to_catalog_sky(ast_by_exp[ea])
                    keep &= sep_ast.arcsec > 20.0
                if sc_br is not None and len(sc_br) > 0:
                    ik, ib, s_kb, _ = sc_br.search_around_sky(sc_s, 60.0 * u.arcsec)
                    for j_b, j_k, s_v in zip(ib, ik, s_kb.arcsec):
                        if s_v <= r_br[j_b]:
                            keep[j_k] = False
                sub = sub[keep].copy().reset_index(drop=True)
                xa, ya = xa[keep], ya[keep]
            else:
                xa, ya = np.zeros(0), np.zeros(0)

            # Exact 30" sky-arcsec de-clustering via _declump_events (identical to primary pipeline & null)
            n_raw_here = len(sub)
            if n_raw_here > 0:
                xn_a = xa / max(nx_a, 1)
                yn_a = ya / max(ny_a, 1)
                ev_norm = _declump_events(sub["ra"].to_numpy(dtype=float), sub["dec"].to_numpy(dtype=float), xn_a, yn_a, 30.0)
                xy_arr = np.column_stack([ev_norm[:, 0] * nx_a, ev_norm[:, 1] * ny_a]) if len(ev_norm) > 0 else np.zeros((0, 2), dtype=float)
            else:
                xy_arr = np.zeros((0, 2), dtype=float)

            erow = exp.loc[ea]
            theta_geo = predict_geo_drift_pa_deg(
                float(erow["ra_deg"]), float(erow["dec_deg"]), float(erow["mjd_mid"]),
                float(erow["obs_lat_deg"]), float(erow["obs_lon_deg"]), float(erow["obs_elev_m"]),
            )
            pa_x, pa_y = plate_axes_pa_deg(wcs_a, nx_a, ny_a)
            geo_axis_sep = float(min(axial_diff_deg(theta_geo, pa_x), axial_diff_deg(theta_geo, pa_y)))
            is_unconfounded = geo_axis_sep >= MIN_AXIS_SEP_DEG

            n_plates_eval += 1
            if is_unconfounded:
                n_plates_unconfounded += 1
            if bool(erow["has_xy_scans"]):
                n_plates_xy_scans += 1
            axis_sep_list.append(geo_axis_sep)
            total_raw_cands_eval += n_raw_here
            total_cands_eval += len(xy_arr)
            if len(xy_arr) < 3 and n_raw_here < 3:
                continue

            fwhm_px = FWHM_FACTOR * med_r_a
            tr_list = find_aligned_tracks(xy_arr, fwhm_px)
            for ch in tr_list:
                p0, p1 = xy_arr[ch[0]], xy_arr[ch[-1]]
                sc0 = wcs_a.pixel_to_world(p0[0], p0[1])
                sc1 = wcs_a.pixel_to_world(p1[0], p1[1])
                pa_tr = float(sc0.position_angle(sc1).to_value(u.deg)) % 180.0
                d_geo = float(axial_diff_deg(pa_tr, theta_geo))
                d_ax = float(min(axial_diff_deg(pa_tr, pa_x), axial_diff_deg(pa_tr, pa_y)))
                obs_tracks.append({
                    "exposure_id": ea, "n_points": len(ch), "pa_deg": round(pa_tr, 2),
                    "theta_geo_deg": round(theta_geo, 2), "geo_axis_sep_deg": round(geo_axis_sep, 2),
                    "unconfounded_ge20": is_unconfounded, "d_geo_deg": round(d_geo, 2),
                    "d_plate_axis_deg": round(d_ax, 2), "cos2_geo": float(np.cos(np.deg2rad(2.0 * d_geo))),
                })

            for b_i in range(n_shuffles):
                sh_xy = sample_plate_shuffle_exact(
                    n_raw_here, nx_a, ny_a, bx, by, wcs_a, wcs_b, nx_b, ny_b, med_r_b, grid_a, sc_br, r_br, rng
                )
                sh_tr = find_aligned_tracks(sh_xy, fwhm_px)
                if sh_tr:
                    null_totals[b_i] += len(sh_tr)
                    if is_unconfounded:
                        null_unconfounded_totals[b_i] += len(sh_tr)
                        for ch in sh_tr:
                            p0, p1 = sh_xy[ch[0]], sh_xy[ch[-1]]
                            sc0 = wcs_a.pixel_to_world(p0[0], p0[1])
                            sc1 = wcs_a.pixel_to_world(p1[0], p1[1])
                            pa_tr = float(sc0.position_angle(sc1).to_value(u.deg)) % 180.0
                            d_g = float(axial_diff_deg(pa_tr, theta_geo))
                            null_v_geo_unconfounded.append(float(np.cos(np.deg2rad(2.0 * d_g))))

    n_obs = len(obs_tracks)
    obs_unconf = [t for t in obs_tracks if t["unconfounded_ge20"]]
    null_mean = float(np.mean(null_totals))
    null_p95 = float(np.percentile(null_totals, 95.0))
    p_count = float((1 + np.sum(null_totals >= n_obs)) / (n_shuffles + 1))
    mean_v_obs_unconf = float(np.mean([t["cos2_geo"] for t in obs_unconf])) if obs_unconf else 0.0
    mean_v_null_unconf = float(np.mean(null_v_geo_unconfounded)) if null_v_geo_unconfounded else 0.0
    n_axis_aligned = sum(1 for t in obs_tracks if t["d_plate_axis_deg"] <= 5.0)
    n_geo_aligned_unconf = sum(1 for t in obs_unconf if t["d_geo_deg"] <= 10.0)

    # Power assessment at alpha = 0.05 one-sided:
    # 1) Track-count excess across ~108 expected QC-passing Tier A m>=8 partial-shadow plates
    from scipy.stats import poisson
    lam_shadow = null_mean * (107.6 / max(n_plates_eval, 1))
    k_crit = int(poisson.ppf(0.95, lam_shadow)) + 1
    # Find minimum added real orbital tracks Delta_T for >= 80% power
    delta_80 = next((d for d in range(1, 50) if 1.0 - poisson.cdf(k_crit - 1, lam_shadow + d) >= 0.80), -1)
    # 2) Unconfounded orientation V-test power (if M tracks found on |theta_GEO - scan_axis| >= 20 deg plates):
    # Under uniform PA, Var(cos 2d) = 0.5, SE(mean V) = sqrt(0.5 / M); for real GEO tracks (cos 2d ~ 0.95), 80% power at alpha=0.05 (z=1.645 + 0.842 = 2.487) needs M >= 0.5 * (2.487 / 0.95)^2 = 3.4 -> >= 4 unconfounded tracks.

    summary = {
        "n_emulsion_matched_qc_plates": n_plates_eval,
        "n_plates_geo_axis_sep_ge20": n_plates_unconfounded,
        "frac_plates_geo_axis_sep_ge20": round(n_plates_unconfounded / max(n_plates_eval, 1), 4),
        "geo_axis_sep_percentiles_deg": {
            "p25": round(float(np.percentile(axis_sep_list, 25)), 2),
            "p50": round(float(np.percentile(axis_sep_list, 50)), 2),
            "p75": round(float(np.percentile(axis_sep_list, 75)), 2),
            "p90": round(float(np.percentile(axis_sep_list, 90)), 2),
        },
        "n_pilot_plates_with_xy_scans": n_plates_xy_scans,
        "total_raw_clean_cands": total_raw_cands_eval,
        "total_clean_declustered_cands": total_cands_eval,
        "n_shuffles_per_plate": n_shuffles,
        "obs_tracks": n_obs,
        "null_tracks_mean": round(null_mean, 3),
        "null_tracks_95pct": round(null_p95, 1),
        "p_count_one_sided": round(p_count, 4),
        "obs_unconfounded_tracks": len(obs_unconf),
        "null_unconfounded_tracks_mean": round(float(np.mean(null_unconfounded_totals)), 3),
        "obs_unconfounded_mean_cos2_geo": round(mean_v_obs_unconf, 4),
        "null_unconfounded_mean_cos2_geo": round(mean_v_null_unconf, 4),
        "obs_unconfounded_within_10deg_geo": n_geo_aligned_unconf,
        "obs_within_5deg_plate_axis": n_axis_aligned,
        "expected_m8_tier_a_null_tracks": round(lam_shadow, 2),
        "m8_tier_a_crit_tracks_alpha05": k_crit,
        "m8_tier_a_extra_tracks_for_80pct_power": delta_80,
        "min_unconfounded_tracks_for_80pct_orient_power": 4,
        "tracks": obs_tracks,
    }
    out_p = Path("data/processed/aligned_tracks_sunlit_pilot_cal.json")
    out_p.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "tracks"}, indent=2))
    if obs_tracks:
        print("Observed sunlit tracks:", pd.DataFrame(obs_tracks).to_string(index=False))
    return summary


if __name__ == "__main__":
    calibrate_pilot_sunlit_tracks(n_shuffles=1000)
