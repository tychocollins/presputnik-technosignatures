"""Per-plate iterative TAN-SIP astrometric refit and monotonic-spline photometric calibration against Gaia DR3."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyvo
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from scipy.interpolate import PchipInterpolator


class _TimeoutSession(requests.Session):
    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", 60)
        return super().request(method, url, **kwargs)


def _tan_project(ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic (TAN) projection from (ra, dec) to (xi, eta) in degrees around (ra0, dec0)."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)
    dra = ra - ra0
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    sin_d0, cos_d0 = np.sin(dec0), np.cos(dec0)
    cos_c = sin_d0 * sin_d + cos_d0 * cos_d * np.cos(dra)
    cos_c = np.where(np.abs(cos_c) < 1e-9, 1e-9, cos_c)
    xi = (cos_d * np.sin(dra)) / cos_c
    eta = (cos_d0 * sin_d - sin_d0 * cos_d * np.cos(dra)) / cos_c
    return np.rad2deg(xi), np.rad2deg(eta)


def _tan_deproject(xi_deg: np.ndarray, eta_deg: np.ndarray, ra0_deg: float, dec0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse gnomonic (TAN) projection from (xi, eta) in degrees to (ra, dec) in degrees."""
    xi = np.deg2rad(xi_deg)
    eta = np.deg2rad(eta_deg)
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    sin_c, cos_c = np.sin(c), np.cos(c)
    sin_d0, cos_d0 = np.sin(dec0), np.cos(dec0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dec = np.where(rho == 0, dec0, np.arcsin(np.clip(cos_c * sin_d0 + (eta * sin_c * cos_d0) / rho, -1.0, 1.0)))
        ra = ra0 + np.where(rho == 0, 0.0, np.arctan2(xi * sin_c, rho * cos_d0 * cos_c - eta * sin_d0 * sin_c))
    return np.mod(np.rad2deg(ra), 360.0), np.rad2deg(dec)


def _poly_design(u_norm: np.ndarray, v_norm: np.ndarray, deg: int) -> np.ndarray:
    """2D polynomial design matrix up to total degree `deg`."""
    cols = []
    for total in range(deg + 1):
        for i in range(total + 1):
            j = total - i
            cols.append((u_norm ** i) * (v_norm ** j))
    return np.column_stack(cols)


def fetch_gaia_uniform(
    df_psf: pd.DataFrame,
    nx: int,
    ny: int,
    mjd_mid: float,
) -> pd.DataFrame:
    """Pick ~315 brightest PSF stars uniformly across a 3x3 grid and query 30\" TAP_UPLOAD."""
    if len(df_psf) == 0:
        return pd.DataFrame(columns=["source_id", "ra", "dec", "G", "BP", "RP"])

    ix = np.clip((df_psf["x"].to_numpy() / max(nx, 1) * 3).astype(int), 0, 2)
    iy = np.clip((df_psf["y"].to_numpy() / max(ny, 1) * 3).astype(int), 0, 2)
    sel_parts = []
    for gy in range(3):
        for gx in range(3):
            cell = df_psf[(ix == gx) & (iy == gy)].sort_values("flux", ascending=False).head(27)
            if len(cell) > 0:
                sel_parts.append(cell)
    sel = pd.concat(sel_parts, ignore_index=True) if sel_parts else df_psf.head(240)
    ra_q = sel["ra"].to_numpy(dtype=float)
    dec_q = sel["dec"].to_numpy(dtype=float)

    svc = pyvo.dal.TAPService("http://dc.g-vo.org/tap", session=_TimeoutSession())
    parts = []
    for start in range(0, len(ra_q), 60):
        end = min(len(ra_q), start + 60)
        tbl = Table({"idx": np.arange(start, end), "ra": ra_q[start:end], "dec": dec_q[start:end]})
        q = (
            "SELECT DISTINCT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, "
            "g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag "
            "FROM TAP_UPLOAD.u AS u JOIN gaia.dr3lite AS g "
            "ON 1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', u.ra, u.dec, 0.0083333))"
        )
        for attempt in range(2):
            try:
                res = svc.search(q, uploads={"u": tbl}).to_table().to_pandas()
                if len(res) > 0:
                    parts.append(res)
                break
            except Exception:
                if attempt == 1:
                    break
                import time
                time.sleep(1.5)

    if not parts:
        return pd.DataFrame(columns=["source_id", "ra", "dec", "G", "BP", "RP"])

    gdf = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["source_id"]).reset_index(drop=True)
    dt_yr = (float(mjd_mid) - 57388.0) / 365.25
    g_ra = gdf["ra"].to_numpy(dtype=float)
    g_dec = gdf["dec"].to_numpy(dtype=float)
    pmra = np.nan_to_num(gdf["pmra"].to_numpy(dtype=float), nan=0.0)
    pmdec = np.nan_to_num(gdf["pmdec"].to_numpy(dtype=float), nan=0.0)
    cos_dec = np.clip(np.cos(np.deg2rad(g_dec)), 1e-4, 1.0)

    gdf["ra"] = np.mod(g_ra + (pmra * dt_yr) / (3_600_000.0 * cos_dec), 360.0)
    gdf["dec"] = np.clip(g_dec + (pmdec * dt_yr) / 3_600_000.0, -90.0, 90.0)
    return gdf.rename(columns={
        "phot_g_mean_mag": "G",
        "phot_bp_mean_mag": "BP",
        "phot_rp_mean_mag": "RP",
    })[["source_id", "ra", "dec", "G", "BP", "RP"]]


def _fit_poly_cv(
    u_n: np.ndarray,
    v_n: np.ndarray,
    xi_t: np.ndarray,
    eta_t: np.ndarray,
    ra0: float,
    dec0: float,
    ra_g: np.ndarray,
    dec_g: np.ndarray,
    allow_orders: tuple[int, ...] = (3, 4, 5),
) -> tuple[int, np.ndarray, np.ndarray]:
    """Fit 2D TAN-SIP polynomial with 5-fold CV order selection and 3-sigma MAD clipping."""
    n = len(u_n)
    best_deg = 3
    best_cv = np.inf
    rng = np.random.default_rng(42)

    for deg in allow_orders:
        n_params = (deg + 1) * (deg + 2) // 2
        if n < max(3 * n_params, 30):
            continue
        if len(allow_orders) == 1:
            best_deg = deg
            break
        perm = rng.permutation(n)
        folds = np.array_split(perm, 5)
        cv_seps = []
        X_all = _poly_design(u_n, v_n, deg)
        for val_idx in folds:
            tr_mask = np.ones(n, dtype=bool)
            tr_mask[val_idx] = False
            c_xi, _, _, _ = np.linalg.lstsq(X_all[tr_mask], xi_t[tr_mask], rcond=None)
            c_eta, _, _, _ = np.linalg.lstsq(X_all[tr_mask], eta_t[tr_mask], rcond=None)
            ra_p, dec_p = _tan_deproject(X_all[val_idx] @ c_xi, X_all[val_idx] @ c_eta, ra0, dec0)
            sep = SkyCoord(ra_p * u.deg, dec_p * u.deg).separation(SkyCoord(ra_g[val_idx] * u.deg, dec_g[val_idx] * u.deg)).arcsec
            cv_seps.append(np.median(sep))
        med_cv = float(np.median(cv_seps))
        if med_cv < best_cv:
            best_cv = med_cv
            best_deg = deg

    X = _poly_design(u_n, v_n, best_deg)
    c_xi, _, _, _ = np.linalg.lstsq(X, xi_t, rcond=None)
    c_eta, _, _, _ = np.linalg.lstsq(X, eta_t, rcond=None)
    # 1 pass of 3-sigma MAD outlier rejection
    res_as = np.hypot((X @ c_xi - xi_t), (X @ c_eta - eta_t)) * 3600.0
    med_r = np.median(res_as)
    mad_r = max(1.4826 * np.median(np.abs(res_as - med_r)), 0.2)
    keep = res_as <= (med_r + 3.0 * mad_r)
    if np.sum(keep) >= (best_deg + 1) * (best_deg + 2) // 2 + 5:
        c_xi, _, _, _ = np.linalg.lstsq(X[keep], xi_t[keep], rcond=None)
        c_eta, _, _, _ = np.linalg.lstsq(X[keep], eta_t[keep], rcond=None)
    return best_deg, c_xi, c_eta


def _fit_monotonic_photometry(
    m_inst: np.ndarray,
    g_mag: np.ndarray,
    bp_mag: np.ndarray,
) -> tuple[PchipInterpolator | None, str, float]:
    """Fit a monotonic PCHIP spline from instrumental mag to Gaia G or BP on unsaturated stars."""
    # Choose BP vs G by smaller MAD scatter
    valid_g = np.isfinite(m_inst) & np.isfinite(g_mag)
    valid_bp = np.isfinite(m_inst) & np.isfinite(bp_mag)
    if np.sum(valid_g) < 15:
        return None, "none", np.nan

    def _mad(arr: np.ndarray) -> float:
        m = np.median(arr)
        return float(1.4826 * np.median(np.abs(arr - m)))

    mad_g = _mad(g_mag[valid_g] - m_inst[valid_g])
    mad_bp = _mad(bp_mag[valid_bp] - m_inst[valid_bp]) if np.sum(valid_bp) >= 15 else np.inf
    if mad_bp <= mad_g:
        band = "BP"
        y_raw, x_raw = bp_mag[valid_bp], m_inst[valid_bp]
    else:
        band = "G"
        y_raw, x_raw = g_mag[valid_g], m_inst[valid_g]

    # Exclude top 2% brightest (saturated) if > 200 matches
    order = np.argsort(x_raw)
    x_s, y_s = x_raw[order], y_raw[order]
    if len(x_s) > 200:
        cut = int(0.02 * len(x_s))
        x_s, y_s = x_s[cut:], y_s[cut:]

    n_bins = min(15, max(5, len(x_s) // 10))
    edges = np.quantile(x_s, np.linspace(0.0, 1.0, n_bins + 1))
    bx, by = [], []
    for k in range(n_bins):
        mask = (x_s >= edges[k]) & (x_s <= edges[k + 1] if k == n_bins - 1 else x_s < edges[k + 1])
        if np.sum(mask) >= 3:
            bx.append(float(np.median(x_s[mask])))
            by.append(float(np.median(y_s[mask])))

    if len(bx) < 4:
        return None, band, np.nan

    bx_arr = np.array(bx, dtype=float)
    by_arr = np.maximum.accumulate(np.array(by, dtype=float))
    # Ensure strictly increasing x for PchipInterpolator
    keep_x = np.concatenate([[True], np.diff(bx_arr) > 1e-4])
    bx_arr, by_arr = bx_arr[keep_x], by_arr[keep_x]
    if len(bx_arr) < 4:
        return None, band, np.nan

    interp = PchipInterpolator(bx_arr, by_arr, extrapolate=True)
    phot_mad = _mad(y_s - interp(x_s))
    return interp, band, phot_mad


def _estimate_turnover_mlim(mags: np.ndarray) -> float | None:
    """Estimate limiting magnitude from the source-count turnover (mode + 0.25 mag)."""
    valid = mags[np.isfinite(mags)]
    if len(valid) < 15:
        return None
    lo, hi = float(np.percentile(valid, 5)), float(np.percentile(valid, 99))
    if hi - lo < 0.5:
        return float(np.percentile(valid, 90))
    bins = np.arange(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2 + 0.5, 0.5)
    if len(bins) < 3:
        return float(np.percentile(valid, 90))
    counts, edges = np.histogram(valid, bins=bins)
    peak_idx = int(np.argmax(counts))
    return float(edges[peak_idx] + 0.25)


def calibrate_plate(
    df_cat: pd.DataFrame,
    wcs_init: WCS,
    nx: int,
    ny: int,
    mjd_mid: float,
    m_lim_archive: float,
    exptime_sec: float,
    rms_grid: np.ndarray | None = None,
    med_rms: float | None = None,
    gaia_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    """
    Run iterative TAN-SIP astrometric refit + monotonic photometric spline + local m_lim grid.
    Returns (updated_df_cat, qc_and_cal_summary, local_mlim_grid_8x8).
    """
    df = df_cat.copy()
    psf_mask = df["psf_pass"].to_numpy(dtype=bool)
    n_psf = int(np.sum(psf_mask))

    c_center = wcs_init.pixel_to_world(nx / 2.0, ny / 2.0)
    ra0, dec0 = float(c_center.ra.deg), float(c_center.dec.deg)

    if gaia_df is None:
        gaia_df = fetch_gaia_uniform(df.loc[psf_mask], nx, ny, mjd_mid)

    n_grid = 8
    fallback_grid = np.full((n_grid, n_grid), float(m_lim_archive), dtype=float)
    if rms_grid is not None and med_rms is not None and med_rms > 0:
        fallback_grid = float(m_lim_archive) - 2.5 * np.log10(np.clip(rms_grid / float(med_rms), 0.1, 10.0))

    if n_psf < 15 or len(gaia_df) < 15:
        df["gaia_hit"] = False
        df["gaia_sep_arcsec"] = np.nan
        df["mag_cal"] = np.nan
        df["mag_est"] = np.nan
        return df, {
            "sip_order": 0, "n_astrom_matches": 0, "med_resid_before_as": np.nan,
            "med_resid_after_as": np.nan, "phot_band": "none", "phot_mad": np.nan,
            "m_lim_cal": float(m_lim_archive), "m_flash_05s_cal": np.nan,
            "n_bright": 0, "n_bright_matched": 0, "bright_match_rate": np.nan, "qc_pass": False,
        }, fallback_grid

    sc_gaia = SkyCoord(gaia_df["ra"].to_numpy() * u.deg, gaia_df["dec"].to_numpy() * u.deg)
    c_center = wcs_init.pixel_to_world(nx / 2.0, ny / 2.0)
    ra0, dec0 = float(c_center.ra.deg), float(c_center.dec.deg)

    x_all = df["x"].to_numpy(dtype=float)
    y_all = df["y"].to_numpy(dtype=float)
    scale = max(nx, ny) / 2.0
    u_all = (x_all - nx / 2.0) / scale
    v_all = (y_all - ny / 2.0) / scale

    ra_cur = df["ra"].to_numpy(dtype=float).copy()
    dec_cur = df["dec"].to_numpy(dtype=float).copy()

    # Initial residuals before refit (on PSF sources within 30")
    sc_init_psf = SkyCoord(ra_cur[psf_mask] * u.deg, dec_cur[psf_mask] * u.deg)
    _, sep_init, _ = sc_init_psf.match_to_catalog_sky(sc_gaia)
    med_before = float(np.median(sep_init.arcsec[sep_init.arcsec <= 30.0])) if np.any(sep_init.arcsec <= 30.0) else np.nan

    # Iterative TAN-SIP refit: 30" -> 10" -> 5" -> 3"
    sip_order = 0
    best_deg = 3
    c_xi, c_eta = None, None
    flux_psf = df.loc[psf_mask, "flux"].to_numpy(dtype=float)
    bright_seed_idx = np.argsort(-flux_psf)[: min(1500, n_psf)]

    for step_idx, rad_as in enumerate([30.0, 10.0, 5.0, 3.0]):
        use_idx = np.where(psf_mask)[0]
        if step_idx == 0:
            use_idx = use_idx[bright_seed_idx]

        sc_u = SkyCoord(ra_cur[use_idx] * u.deg, dec_cur[use_idx] * u.deg)
        g_idx, sep2d, _ = sc_u.match_to_catalog_sky(sc_gaia)
        hit = sep2d.arcsec <= rad_as
        if np.sum(hit) < 20:
            break

        m_u = use_idx[hit]
        m_g = g_idx[hit]
        xi_t, eta_t = _tan_project(gaia_df["ra"].to_numpy()[m_g], gaia_df["dec"].to_numpy()[m_g], ra0, dec0)
        if step_idx == 0:
            # Robust median tangent-plane shift first to remove bulk ~15-20" offset without outlier pull
            xi_u, eta_u = _tan_project(ra_cur[m_u], dec_cur[m_u], ra0, dec0)
            d_xi = float(np.median(xi_t - xi_u))
            d_eta = float(np.median(eta_t - eta_u))
            xi_all, eta_all = _tan_project(ra_cur, dec_cur, ra0, dec0)
            ra_cur, dec_cur = _tan_deproject(xi_all + d_xi, eta_all + d_eta, ra0, dec0)
            continue
        orders = (3,) if rad_as >= 10.0 else (3, 4, 5)
        best_deg, c_xi, c_eta = _fit_poly_cv(
            u_all[m_u], v_all[m_u], xi_t, eta_t, ra0, dec0,
            gaia_df["ra"].to_numpy()[m_g], gaia_df["dec"].to_numpy()[m_g], allow_orders=orders,
        )
        sip_order = best_deg
        X_full = _poly_design(u_all, v_all, best_deg)
        ra_cur, dec_cur = _tan_deproject(X_full @ c_xi, X_full @ c_eta, ra0, dec0)

    # Safety guard: only keep refit if it improves median Gaia residual over archive WCS
    if sip_order > 0 and np.isfinite(med_before):
        sc_refit_psf = SkyCoord(ra_cur[psf_mask] * u.deg, dec_cur[psf_mask] * u.deg)
        _, sep_refit, _ = sc_refit_psf.match_to_catalog_sky(sc_gaia)
        med_refit = float(np.median(sep_refit.arcsec[sep_refit.arcsec <= 30.0])) if np.any(sep_refit.arcsec <= 30.0) else np.inf
        if med_refit > med_before:
            ra_cur = df["ra"].to_numpy(dtype=float).copy()
            dec_cur = df["dec"].to_numpy(dtype=float).copy()
            sip_order = 0

    df["ra"] = ra_cur
    df["dec"] = dec_cur

    # Final match on refit coordinates via fast 5.0" TAP_UPLOAD (PM-propagated)
    ra_psf = ra_cur[psf_mask]
    dec_psf = dec_cur[psf_mask]
    sep_psf = np.full(n_psf, np.inf)
    dt_yr = (float(mjd_mid) - 57388.0) / 365.25
    svc_m = pyvo.dal.TAPService("http://dc.g-vo.org/tap", session=_TimeoutSession())
    g_mag_psf = np.full(n_psf, np.nan)
    bp_mag_psf = np.full(n_psf, np.nan)

    for start in range(0, n_psf, 2000):
        end = min(n_psf, start + 2000)
        tbl = Table({"idx": np.arange(start, end), "ra": ra_psf[start:end], "dec": dec_psf[start:end]})
        q = (
            "SELECT u.idx, g.ra, g.dec, g.pmra, g.pmdec, g.phot_g_mean_mag, g.phot_bp_mean_mag "
            "FROM TAP_UPLOAD.u AS u "
            "JOIN gaia.dr3lite AS g "
            "ON 1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', u.ra, u.dec, 0.00138889))"
        )
        res = svc_m.search(q, uploads={"u": tbl}).to_table().to_pandas()
        if len(res) == 0:
            continue
        ia = res["idx"].to_numpy(dtype=int)
        gra = res["ra"].to_numpy(dtype=float)
        gdec = res["dec"].to_numpy(dtype=float)
        pmra = np.nan_to_num(res["pmra"].to_numpy(dtype=float), nan=0.0)
        pmdec = np.nan_to_num(res["pmdec"].to_numpy(dtype=float), nan=0.0)
        cdec = np.clip(np.cos(np.deg2rad(gdec)), 1e-4, 1.0)
        ra_p = np.mod(gra + (pmra * dt_yr) / (3_600_000.0 * cdec), 360.0)
        dec_p = np.clip(gdec + (pmdec * dt_yr) / 3_600_000.0, -90.0, 90.0)
        s_as = SkyCoord(ra_psf[ia] * u.deg, dec_psf[ia] * u.deg).separation(SkyCoord(ra_p * u.deg, dec_p * u.deg)).arcsec
        gm = res["phot_g_mean_mag"].to_numpy(dtype=float)
        bm = res["phot_bp_mean_mag"].to_numpy(dtype=float)
        for k in range(len(ia)):
            i = ia[k]
            if s_as[k] < sep_psf[i]:
                sep_psf[i] = float(s_as[k])
                g_mag_psf[i] = gm[k]
                bp_mag_psf[i] = bm[k]

    sep_as = np.full(len(df), np.inf)
    sep_as[psf_mask] = sep_psf
    matched_3as = sep_as <= 3.0
    matched_5as = sep_as <= 5.0
    df["gaia_hit"] = matched_5as
    df["gaia_sep_arcsec"] = np.where(np.isfinite(sep_as), sep_as, np.nan)

    # Photometric monotonic spline on matched PSF sources (<= 3.0")
    flux_all = df["flux"].to_numpy(dtype=float)
    m_inst = np.where(flux_all > 0, -2.5 * np.log10(np.maximum(flux_all, 1e-30)), np.nan)
    phot_mask_psf = (sep_psf <= 3.0) & np.isfinite(m_inst[psf_mask])
    interp, band, phot_mad = _fit_monotonic_photometry(
        m_inst[psf_mask][phot_mask_psf],
        g_mag_psf[phot_mask_psf],
        bp_mag_psf[phot_mask_psf],
    )
    if interp is not None:
        mag_cal = interp(m_inst)
        # Guard against collapsed/flat spline (e.g. heavy saturation flattening PCHIP)
        fin_cal = mag_cal[psf_mask & np.isfinite(mag_cal)]
        if len(fin_cal) >= 10 and (np.percentile(fin_cal, 75) - np.percentile(fin_cal, 25)) < 0.5:
            interp = None
        else:
            df["mag_cal"] = mag_cal
            df["mag_est"] = mag_cal
    if interp is None:
        g_mags = g_mag_psf[phot_mask_psf]
        if np.sum(np.isfinite(g_mags)) >= 5:
            zp = float(np.nanmedian(g_mags - m_inst[psf_mask][phot_mask_psf]))
            mag_cal = m_inst + zp
            df["mag_cal"] = mag_cal
            df["mag_est"] = mag_cal
            band = "G_zp"
        else:
            mag_cal = np.full(len(df), np.nan)
            df["mag_cal"] = mag_cal
            df["mag_est"] = mag_cal

    # Global & 8x8 local limiting magnitude from calibrated count turnover
    m_lim_cal = _estimate_turnover_mlim(mag_cal[psf_mask]) or float(m_lim_archive)
    local_grid = fallback_grid.copy()
    if np.any(np.isfinite(mag_cal[psf_mask])):
        ix = np.clip((x_all / max(nx, 1) * n_grid).astype(int), 0, n_grid - 1)
        iy = np.clip((y_all / max(ny, 1) * n_grid).astype(int), 0, n_grid - 1)
        for gy in range(n_grid):
            for gx in range(n_grid):
                cell_m = mag_cal[psf_mask & (ix == gx) & (iy == gy)]
                cell_lim = _estimate_turnover_mlim(cell_m)
                if cell_lim is not None:
                    local_grid[gy, gx] = cell_lim
                elif rms_grid is not None and med_rms is not None and med_rms > 0:
                    local_grid[gy, gx] = m_lim_cal - 2.5 * np.log10(np.clip(rms_grid[gy, gx] / float(med_rms), 0.1, 10.0))
                else:
                    local_grid[gy, gx] = m_lim_cal

    m_flash_05s_cal = m_lim_cal - 2.5 * np.log10(max(float(exptime_sec), 0.5) / 0.5)

    # New QC Gate: bright stars (mag_cal < m_lim_cal - 2.0) match >= 90% within 3" AND median residual <= 1.5"
    bright_mask = psf_mask & np.isfinite(mag_cal) & (mag_cal < (m_lim_cal - 2.0))
    n_bright = int(np.sum(bright_mask))
    if n_bright < 5 and n_psf >= 20:
        # Fallback to top 100 brightest PSF sources if turnover cut leaves < 5
        top_idx = np.where(psf_mask)[0][np.argsort(-flux_psf)[: min(100, n_psf)]]
        bright_mask = np.zeros(len(df), dtype=bool)
        bright_mask[top_idx] = True
        n_bright = int(np.sum(bright_mask))

    if n_bright >= 5:
        b_seps = sep_as[bright_mask]
        n_bright_matched = int(np.sum(b_seps <= 3.0))
        rate_3as = float(n_bright_matched / n_bright)
        med_after = float(np.median(b_seps))
        qc_pass = bool((rate_3as >= 0.90) and (med_after <= 1.5))
    else:
        n_bright_matched = 0
        rate_3as = np.nan
        med_after = np.nan
        qc_pass = False

    summary = {
        "sip_order": int(sip_order),
        "n_astrom_matches": int(np.sum(psf_mask & matched_3as)),
        "med_resid_before_as": med_before,
        "med_resid_after_as": med_after,
        "phot_band": band,
        "phot_mad": phot_mad,
        "m_lim_cal": float(m_lim_cal),
        "m_flash_05s_cal": float(m_flash_05s_cal),
        "n_bright": n_bright,
        "n_bright_matched": n_bright_matched,
        "bright_match_rate": rate_3as,
        "qc_pass": qc_pass,
    }
    return df, summary, local_grid
