"""Phase 2a Blind Pilot: out-of-shadow Tier A pair source extraction and Gaia DR3 matching."""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvo
import requests
import sep
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astroquery.xmatch import XMatch

from src.calibrate import calibrate_plate
from src.power import min_detectable_deficit


def build_stratified_pilot_sample(
    n_total: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Pick a stratified random sample of ~200 distinct-plate Tier A pairs
    with m_flash >= 9.0 that are strictly out of shadow at all 3 altitudes.
    """
    exp_df = pd.read_parquet("data/processed/exposures.parquet")
    zero_shadow_ids = set(
        exp_df.loc[
            (exp_df["shadow_fraction"] == 0.0)
            & (exp_df["shadow_fraction_36000km"] == 0.0)
            & (exp_df["shadow_fraction_20000km"] == 0.0),
            "exposure_id",
        ]
    )

    pool = pd.read_parquet("data/processed/pilot_pool_pairs.parquet")
    pool = pool[
        (pool["plate1"] != pool["plate2"])
        & pool["exp1"].isin(zero_shadow_ids)
        & pool["exp2"].isin(zero_shadow_ids)
    ].copy()

    # Stratified allocation across instruments (proportional with floor of min(avail, 10))
    rng = np.random.default_rng(seed)
    counts = pool.groupby(["source_archive", "instrument"]).size().reset_index(name="avail")
    # Allocate ~15 per instrument where available, remainder proportional
    allocs = {}
    for _, r in counts.iterrows():
        key = (r["source_archive"], r["instrument"])
        allocs[key] = min(int(r["avail"]), 12)
    remaining = n_total - sum(allocs.values())
    if remaining > 0:
        big_keys = [k for k, v in allocs.items() if counts.set_index(["source_archive", "instrument"]).loc[k, "avail"] > v]
        total_big = sum(counts.set_index(["source_archive", "instrument"]).loc[k, "avail"] - allocs[k] for k in big_keys)
        for k in big_keys:
            avail_extra = int(counts.set_index(["source_archive", "instrument"]).loc[k, "avail"]) - allocs[k]
            add = int(round(remaining * avail_extra / max(1, total_big)))
            allocs[k] += min(avail_extra, add)

    parts = []
    for key, n_take in allocs.items():
        if n_take <= 0:
            continue
        grp = pool[(pool["source_archive"] == key[0]) & (pool["instrument"] == key[1])]
        parts.append(grp.sample(n=min(n_take, len(grp)), random_state=seed))

    sample = pd.concat(parts, ignore_index=True).head(n_total)
    # Blindness assertion
    assert sample["exp1"].isin(zero_shadow_ids).all(), "BLINDNESS VIOLATION in exp1!"
    assert sample["exp2"].isin(zero_shadow_ids).all(), "BLINDNESS VIOLATION in exp2!"

    sample.to_csv("data/processed/pilot_sample_pairs.csv", index=False)
    return sample


def _get_applause_scan_and_wcs_map(plate_ids: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Map APPLAUSE_<plate_id> -> scan download URL and FITS WCS header string."""
    scn = pd.read_parquet("data/raw/applause_scan.parquet")
    scn["key"] = "APPLAUSE_" + scn["plate_id"].astype(str)
    sub = scn[scn["key"].isin(set(plate_ids))].drop_duplicates(subset=["key"], keep="first")
    urls = {
        row["key"]: (
            str(row["filename_scan"])
            if str(row["filename_scan"]).startswith("http")
            else f"https://www.plate-archive.org/files/{row['filename_scan']}"
        )
        for _, row in sub.iterrows()
        if pd.notna(row["filename_scan"])
    }

    # Fetch header_wcs from applause_dr4.solution for these plate_ids
    raw_pids = sorted({int(pid.replace("APPLAUSE_", "")) for pid in plate_ids})
    wcs_map: dict[str, str] = {}
    if raw_pids:
        svc = pyvo.dal.TAPService("https://www.plate-archive.org/tap")
        for start in range(0, len(raw_pids), 100):
            chunk = ",".join(str(x) for x in raw_pids[start : start + 100])
            q = (
                f"SELECT plate_id, header_wcs FROM applause_dr4.solution "
                f"WHERE plate_id IN ({chunk}) AND solution_num = 1"
            )
            res = svc.search(q).to_table().to_pandas()
            for _, r in res.iterrows():
                if pd.notna(r["header_wcs"]):
                    wcs_map[f"APPLAUSE_{int(r['plate_id'])}"] = str(r["header_wcs"])
    return urls, wcs_map


def _parse_wcs_from_string(wcs_str: str) -> WCS:
    """Parse WCS cards from APPLAUSE header_wcs string, skipping non-WCS/CONTINUE cards."""
    keep_prefixes = (
        "NAXIS", "CTYPE", "CRVAL", "CRPIX", "CD1_", "CD2_", "PC1_", "PC2_",
        "CDELT", "CROTA", "PV1_", "PV2_", "EQUINOX", "RADESYS", "LONPOLE",
        "LATPOLE", "WCSAXES", "A_", "B_", "AP_", "BP_",
    )
    lines = [l for l in wcs_str.splitlines() if l.startswith(keep_prefixes)]
    hdr = fits.Header.fromstring("\n".join(lines), sep="\n")
    return WCS(hdr, naxis=2)


def _compute_rms_grid(rms_map: np.ndarray, mask: np.ndarray, n_grid: int = 8) -> tuple[np.ndarray, float]:
    """Downsample 2D SEP background RMS map to an (n_grid x n_grid) array + global median RMS."""
    valid = np.where(~mask & np.isfinite(rms_map) & (rms_map > 0), rms_map, np.nan)
    med_rms = float(np.nanmedian(valid)) if np.any(np.isfinite(valid)) else float(np.nanmedian(rms_map))
    ny, nx = rms_map.shape
    grid = np.full((n_grid, n_grid), med_rms, dtype=float)
    ys = np.linspace(0, ny, n_grid + 1, dtype=int)
    xs = np.linspace(0, nx, n_grid + 1, dtype=int)
    for iy in range(n_grid):
        for ix in range(n_grid):
            patch = valid[ys[iy] : ys[iy + 1], xs[ix] : xs[ix + 1]]
            if np.any(np.isfinite(patch)):
                grid[iy, ix] = float(np.nanmedian(patch))
    return grid, med_rms


def extract_stamp_60as(
    data_sub: np.ndarray,
    rms_img: np.ndarray,
    sat_mask: np.ndarray,
    wcs: WCS,
    ra: float,
    dec: float,
    box_arcsec: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Extract a 60-arcsec SNR-normalized stamp and saturation mask centered on (ra, dec)."""
    ny, nx = data_sub.shape
    scales = wcs.proj_plane_pixel_scales()
    ps = float(np.mean([scales[0].to_value(u.arcsec), scales[1].to_value(u.arcsec)]))
    if not np.isfinite(ps) or ps <= 0:
        ps = 1.5
    half_px = max(12, min(120, int(round((box_arcsec / 2.0) / ps))))
    size = 2 * half_px + 1

    xc, yc = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    ix, iy = int(round(float(xc))), int(round(float(yc)))
    x0, x1 = ix - half_px, ix + half_px + 1
    y0, y1 = iy - half_px, iy + half_px + 1

    snr_stamp = np.full((size, size), np.nan, dtype=np.float32)
    sat_stamp = np.ones((size, size), dtype=bool)

    sx0, sx1 = max(0, x0), min(nx, x1)
    sy0, sy1 = max(0, y0), min(ny, y1)
    hits_border = (x0 < 0) or (y0 < 0) or (x1 > nx) or (y1 > ny)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x0, sy0 - y0
        dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
        snr_stamp[dy0:dy1, dx0:dx1] = (
            data_sub[sy0:sy1, sx0:sx1] / np.maximum(rms_img[sy0:sy1, sx0:sx1], 1e-6)
        ).astype(np.float32)
        sat_stamp[dy0:dy1, dx0:dx1] = sat_mask[sy0:sy1, sx0:sx1]
    return snr_stamp, sat_stamp, ps, hits_border


def _extract_plate_sources(
    fits_bytes: bytes | Path,
    override_wcs_str: str | None = None,
    return_image: bool = False,
) -> tuple[Any, ...] | None:
    """Run SEP source extraction + stellar PSF tagging + 8x8 local RMS map."""
    tmp_ctx = None
    if isinstance(fits_bytes, Path):
        fits_path = str(fits_bytes)
    else:
        tmp_ctx = tempfile.NamedTemporaryFile(suffix=".fits")
        tmp_ctx.write(fits_bytes)
        tmp_ctx.flush()
        fits_path = tmp_ctx.name
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            hdu = None
            for h in hdul:
                if h.data is not None and h.data.ndim == 2:
                    hdu = h
                    break
            if hdu is None:
                return None

            data = np.ascontiguousarray(hdu.data, dtype=np.float32)
            hdr = hdu.header
            ny, nx = data.shape

            # Handle TNX projection or missing celestial WCS (use APPLAUSE solution header_wcs)
            hdr_wcs = hdr.copy()
            if str(hdr_wcs.get("CTYPE1", "")).endswith("TNX"):
                hdr_wcs["CTYPE1"] = "RA---TAN"
                hdr_wcs["CTYPE2"] = "DEC--TAN"
            wcs = WCS(hdr_wcs, naxis=2)
            if (not wcs.has_celestial) and override_wcs_str:
                wcs = _parse_wcs_from_string(override_wcs_str)
            if not wcs.has_celestial:
                return None
    finally:
        if tmp_ctx is not None:
            tmp_ctx.close()

    # Mask zeros / borders (5% margin)
    mask = data == 0
    my, mx = int(0.05 * ny), int(0.05 * nx)
    mask[:my, :] = True
    mask[-my:, :] = True
    mask[:, :mx] = True
    mask[:, -mx:] = True

    # Auto-detect polarity (photographic negatives can be sky-bright or sky-dark)
    valid_vals = data[np.isfinite(data) & (~mask)]
    if len(valid_vals) < 1000:
        return None
    p05, p50, p995 = np.percentile(valid_vals, [0.5, 50.0, 99.5])
    if (p50 - p05) > 1.3 * (p995 - p50):
        data *= -1.0
        valid_vals *= -1.0

    # Flag clipped/saturated plateaus before background subtraction
    sat_mask = (~np.isfinite(data)) | (data == 0)
    v_max = float(np.max(valid_vals))
    v_min = float(np.min(valid_vals))
    if np.sum(valid_vals >= v_max - 1e-6) > 10:
        sat_mask |= data >= v_max - 1e-6
    if np.sum(valid_vals <= v_min + 1e-6) > 10:
        sat_mask |= data <= v_min + 1e-6
    del valid_vals

    bkg = sep.Background(data, mask=mask, bw=128, bh=128, fw=3, fh=3)
    data -= bkg.back()
    data_sub = data
    rms_img = np.maximum(bkg.rms(), 1e-6)
    rms = float(bkg.globalrms)
    if not np.isfinite(rms) or rms <= 0:
        return None

    # Flag negative wrap-around / emulsion holes in background-subtracted image
    sat_mask |= data_sub < -5.0 * rms_img

    rms_grid, med_rms = _compute_rms_grid(bkg.rms(), mask, n_grid=8)

    # Disable multi-threshold deblending (deblend_nthresh=1) to prevent overflow on plate artifacts
    sep.set_extract_pixstack(2_000_000)
    objs = sep.extract(data_sub, thresh=6.0 * rms, mask=mask, minarea=7, deblend_nthresh=1, deblend_cont=1.0)
    cols = ["ra", "dec", "x", "y", "flux", "a", "b", "ellip", "r_psf", "npix", "flag", "psf_pass", "peak_snr", "min3_snr", "n_sat_pix"]
    if len(objs) == 0:
        empty_df = pd.DataFrame(columns=cols)
        if return_image:
            return empty_df, wcs, (ny, nx), 3.0, rms_grid, med_rms, {"data_sub": data_sub, "rms_img": rms_img, "sat_mask": sat_mask}
        return empty_df, wcs, (ny, nx), 3.0, rms_grid, med_rms

    df = pd.DataFrame(
        {
            "x": objs["x"],
            "y": objs["y"],
            "flux": objs["flux"],
            "a": objs["a"],
            "b": objs["b"],
            "npix": objs["npix"],
            "flag": objs["flag"],
        }
    )
    df["ellip"] = np.where(df["a"] > 0, 1.0 - (df["b"] / df["a"]), np.nan)
    df["r_psf"] = 0.5 * (df["a"] + df["b"])

    # Pixel-level polarity & saturation checks for every source while image is in memory
    n_obj = len(df)
    peak_snr = np.full(n_obj, np.nan, dtype=np.float32)
    min3_snr = np.full(n_obj, np.nan, dtype=np.float32)
    n_sat_pix = np.zeros(n_obj, dtype=np.int32)
    xs_i = np.rint(df["x"].to_numpy()).astype(int)
    ys_i = np.rint(df["y"].to_numpy()).astype(int)
    for idx_o in range(n_obj):
        ix, iy = xs_i[idx_o], ys_i[idx_o]
        x0, x1 = max(0, ix - 1), min(nx, ix + 2)
        y0, y1 = max(0, iy - 1), min(ny, iy + 2)
        if x1 > x0 and y1 > y0:
            r_loc = float(np.median(rms_img[y0:y1, x0:x1]))
            p3 = data_sub[y0:y1, x0:x1]
            peak_snr[idx_o] = float(np.max(p3)) / r_loc
            min3_snr[idx_o] = float(np.min(p3)) / r_loc
        sx0, sx1 = max(0, ix - 2), min(nx, ix + 3)
        sy0, sy1 = max(0, iy - 2), min(ny, iy + 3)
        if sx1 > sx0 and sy1 > sy0:
            n_sat_pix[idx_o] = int(np.sum(sat_mask[sy0:sy1, sx0:sx1]))
    df["peak_snr"] = peak_snr
    df["min3_snr"] = min3_snr
    df["n_sat_pix"] = n_sat_pix

    # Tag stellar PSF filter rather than discarding non-PSF sources before saving catalog
    base_ok = (df["flag"] == 0) & (df["npix"] >= 5) & (df["b"] > 0) & (df["a"] > 0) & (df["flux"] > 0) & (df["ellip"] <= 0.4)
    med_r = float(df.loc[base_ok, "r_psf"].median()) if np.any(base_ok) else 3.0
    if np.sum(base_ok) > 10:
        df["psf_pass"] = base_ok & (df["r_psf"] >= 0.6 * med_r) & (df["r_psf"] <= 2.5 * med_r)
    else:
        df["psf_pass"] = base_ok

    sc = wcs.pixel_to_world(df["x"].to_numpy(), df["y"].to_numpy())
    df["ra"] = sc.ra.deg
    df["dec"] = sc.dec.deg
    df = df.reset_index(drop=True)
    if return_image:
        return df[cols], wcs, (ny, nx), med_r, rms_grid, med_rms, {"data_sub": data_sub, "rms_img": rms_img, "sat_mask": sat_mask}
    return df[cols], wcs, (ny, nx), med_r, rms_grid, med_rms


def _gaia_xmatch_with_pm(
    df_src: pd.DataFrame,
    mjd_mid: float,
    m_lim: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Propagate Gaia DR3 positions (J2016.0, MJD 57388.0) to plate epoch `mjd_mid`
    using pmra/pmdec, match within 5.0 arcsec, calibrate instrumental magnitudes,
    and compute bright-star (< m_lim - 2) Gaia match rate.
    """
    n = len(df_src)
    if n == 0:
        return np.zeros(0, dtype=bool), np.full(0, np.nan), {
            "zp": np.nan, "n_bright": 0, "n_bright_matched": 0, "bright_match_rate": np.nan, "qc_pass": True
        }

    class _TimeoutSession(requests.Session):
        def request(self, method, url, **kwargs):
            kwargs.setdefault("timeout", 60)
            return super().request(method, url, **kwargs)

    svc = pyvo.dal.TAPService("http://dc.g-vo.org/tap", session=_TimeoutSession())
    ra = df_src["ra"].to_numpy()
    dec = df_src["dec"].to_numpy()
    flux = df_src["flux"].to_numpy()

    dt_yr = (float(mjd_mid) - 57388.0) / 365.25
    matched_5as = np.zeros(n, dtype=bool)
    gaia_mag = np.full(n, np.nan)
    best_sep = np.full(n, np.inf)

    # Pre-match within 15 arcsec (0.00416667 deg) to capture high-PM stars, then apply exact PM propagation
    for start in range(0, n, 2000):
        end = min(n, start + 2000)
        tbl = Table({"idx": np.arange(start, end), "ra": ra[start:end], "dec": dec[start:end]})
        q = (
            "SELECT u.idx, g.ra, g.dec, g.pmra, g.pmdec, g.phot_g_mean_mag "
            "FROM TAP_UPLOAD.u AS u "
            "JOIN gaia.dr3lite AS g "
            "ON 1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', u.ra, u.dec, 0.00416667))"
        )
        res = svc.search(q, uploads={"u": tbl}).to_table().to_pandas()
        if len(res) == 0:
            continue

        idx_arr = res["idx"].to_numpy(dtype=int)
        g_ra = res["ra"].to_numpy(dtype=float)
        g_dec = res["dec"].to_numpy(dtype=float)
        pmra = np.nan_to_num(res["pmra"].to_numpy(dtype=float), nan=0.0)
        pmdec = np.nan_to_num(res["pmdec"].to_numpy(dtype=float), nan=0.0)
        g_mag = res["phot_g_mean_mag"].to_numpy(dtype=float)

        cos_dec = np.clip(np.cos(np.deg2rad(g_dec)), 1e-4, 1.0)
        ra_prop = g_ra + (pmra * dt_yr) / (3_600_000.0 * cos_dec)
        dec_prop = np.clip(g_dec + (pmdec * dt_yr) / 3_600_000.0, -90.0, 90.0)

        sc_u = SkyCoord(ra[idx_arr] * u.deg, dec[idx_arr] * u.deg)
        sc_g = SkyCoord(ra_prop * u.deg, dec_prop * u.deg)
        sep_as = sc_u.separation(sc_g).arcsec

        for k in range(len(idx_arr)):
            i = idx_arr[k]
            s = float(sep_as[k])
            if s < best_sep[i]:
                best_sep[i] = s
                gaia_mag[i] = g_mag[k]
            if s <= 5.0:
                matched_5as[i] = True

    # Calibrate instrumental magnitudes: mag_est = zp - 2.5 * log10(flux)
    calib_mask = matched_5as & np.isfinite(gaia_mag) & (flux > 0)
    if np.sum(calib_mask) >= 5:
        zp = float(np.median(gaia_mag[calib_mask] + 2.5 * np.log10(flux[calib_mask])))
        mag_est = zp - 2.5 * np.log10(flux)
    else:
        zp = np.nan
        mag_est = np.full(n, np.nan)

    # Bright-star Gaia match rate for sources brighter than m_lim - 2.0
    if np.isfinite(zp) and np.isfinite(m_lim):
        bright_mask = np.isfinite(mag_est) & (mag_est < (float(m_lim) - 2.0))
        n_bright = int(np.sum(bright_mask))
        n_bright_matched = int(np.sum(bright_mask & matched_5as))
        if n_bright >= 5:
            rate = float(n_bright_matched / n_bright)
            qc_pass = bool(rate >= 0.90)
        elif n >= 20:
            rate = float(np.mean(matched_5as))
            qc_pass = bool(rate >= 0.90)
        else:
            rate = np.nan
            qc_pass = False
    else:
        n_bright = 0
        n_bright_matched = 0
        rate = np.nan
        qc_pass = False

    qc = {
        "zp": zp,
        "n_bright": n_bright,
        "n_bright_matched": n_bright_matched,
        "bright_match_rate": rate,
        "qc_pass": qc_pass,
    }
    return matched_5as, mag_est, qc


def process_pair(
    row: pd.Series,
    exp_meta: dict[str, dict[str, Any]],
    applause_urls: dict[str, str],
    applause_wcs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Download both plates of an out-of-shadow pair, extract, match with PM, and log candidates + QC."""
    t0 = time.time()
    p1, p2 = row["plate1"], row["plate2"]
    e1, e2 = row["exp1"], row["exp2"]
    sa = row["source_archive"]
    applause_wcs = applause_wcs or {}
    meta1 = exp_meta[e1]
    meta2 = exp_meta[e2]

    def _download(pid: str) -> bytes:
        if sa == "DASCH":
            dasch_id = pid.replace("DASCH_", "")
            url = f"http://dc.g-vo.org/dasch/q/getprod/qp/fits-01/{dasch_id}"
        else:
            url = applause_urls[pid]
        for attempt in range(2):
            t_dl = time.monotonic()
            try:
                with requests.get(url, timeout=(30, 120), stream=True) as r:
                    r.raise_for_status()
                    chunks = []
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if time.monotonic() - t_dl > 600.0:
                            raise TimeoutError(f"Download wall-clock timeout (>600s) for {pid}")
                        if chunk:
                            chunks.append(chunk)
                    return b"".join(chunks)
            except Exception:
                if attempt == 1 or (time.monotonic() - t0 > 1000.0):
                    raise
                time.sleep(3.0)
        raise RuntimeError("unreachable")

    b1 = _download(p1)
    ext1 = _extract_plate_sources(b1, override_wcs_str=applause_wcs.get(p1), return_image=True)
    del b1
    b2 = _download(p2)
    ext2 = _extract_plate_sources(b2, override_wcs_str=applause_wcs.get(p2), return_image=True)
    del b2

    if ext1 is None or ext2 is None:
        return {"exp1": e1, "exp2": e2, "source_archive": sa, "instrument": row["instrument"], "status": "extract_failed"}

    df1_all, wcs1, (ny1, nx1), med_r1, rms_grid1, med_rms1, img1 = ext1
    df2_all, wcs2, (ny2, nx2), med_r2, rms_grid2, med_rms2, img2 = ext2

    df1_all, qc1, local_grid1 = calibrate_plate(
        df1_all, wcs1, nx1, ny1, float(meta1["mjd_mid"]), float(meta1["m_lim"]), float(meta1["exptime_sec"]),
        rms_grid=rms_grid1, med_rms=med_rms1,
    )
    df2_all, qc2, local_grid2 = calibrate_plate(
        df2_all, wcs2, nx2, ny2, float(meta2["mjd_mid"]), float(meta2["m_lim"]), float(meta2["exptime_sec"]),
        rms_grid=rms_grid2, med_rms=med_rms2,
    )

    psf1 = df1_all["psf_pass"].to_numpy()
    psf2 = df2_all["psf_pass"].to_numpy()
    df1 = df1_all[psf1].reset_index(drop=True)
    df2 = df2_all[psf2].reset_index(drop=True)

    cat_dir = Path("data/processed/pilot_catalogs")
    cat_dir.mkdir(parents=True, exist_ok=True)
    df1_all.to_parquet(cat_dir / f"{e1}.parquet", index=False)
    df2_all.to_parquet(cat_dir / f"{e2}.parquet", index=False)

    def _build_footprint(eid_i: str, pid_i: str, wcs_i: WCS, nx_i: int, ny_i: int, med_r_i: float,
                         local_mlim_grid: np.ndarray, med_rms_i: float, qc_i: dict[str, Any], meta_i: dict[str, Any]) -> dict[str, Any]:
        mlim_i = float(meta_i["m_lim"])
        corners = wcs_i.pixel_to_world([0, nx_i - 1, nx_i - 1, 0], [0, 0, ny_i - 1, ny_i - 1])
        return {
            "exposure_id": eid_i,
            "plate_id": pid_i,
            "source_archive": sa,
            "instrument": row["instrument"],
            "nx": int(nx_i),
            "ny": int(ny_i),
            "med_r_psf": float(med_r_i),
            "m_lim_archive": mlim_i,
            "m_lim_cal": qc_i["m_lim_cal"],
            "bkg_rms_median": float(med_rms_i),
            "ra_c00": float(corners.ra.deg[0]), "dec_c00": float(corners.dec.deg[0]),
            "ra_c10": float(corners.ra.deg[1]), "dec_c10": float(corners.dec.deg[1]),
            "ra_c11": float(corners.ra.deg[2]), "dec_c11": float(corners.dec.deg[2]),
            "ra_c01": float(corners.ra.deg[3]), "dec_c01": float(corners.dec.deg[3]),
            "wcs_header": wcs_i.to_header_string(),
            "local_mlim_grid_json": json.dumps(np.round(local_mlim_grid, 3).tolist()),
        }

    fp1 = _build_footprint(e1, p1, wcs1, nx1, ny1, med_r1, local_grid1, med_rms1, qc1, meta1)
    fp2 = _build_footprint(e2, p2, wcs2, nx2, ny2, med_r2, local_grid2, med_rms2, qc2, meta2)
    footprints = [fp1, fp2]

    plate_qcs = [
        {"exposure_id": e1, "plate_id": p1, "source_archive": sa, "instrument": row["instrument"],
         "mjd_mid": meta1["mjd_mid"], "m_lim": meta1["m_lim"], "n_src_all": len(df1_all), "n_src": len(df1), "med_r_psf": med_r1, **qc1},
        {"exposure_id": e2, "plate_id": p2, "source_archive": sa, "instrument": row["instrument"],
         "mjd_mid": meta2["mjd_mid"], "m_lim": meta2["m_lim"], "n_src_all": len(df2_all), "n_src": len(df2), "med_r_psf": med_r2, **qc2},
    ]

    def _single_plate_cands(
        src_a: pd.DataFrame, src_b_all: pd.DataFrame, wcs_a: WCS, wcs_b: WCS, ny_b: int, nx_b: int, med_r_b: float,
        local_grid_b: np.ndarray, meta_a: dict[str, Any], meta_b: dict[str, Any], plate_a: str, plate_b: str,
        qc_a_pass: bool, qc_b_pass: bool, img_a: dict[str, np.ndarray], img_b: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        if len(src_a) == 0 or len(src_b_all) == 0:
            return []
        xb, yb = wcs_b.world_to_pixel(SkyCoord(src_a["ra"].to_numpy() * u.deg, src_a["dec"].to_numpy() * u.deg))
        in_b = (xb >= 0.0) & (xb <= float(nx_b - 1)) & (yb >= 0.0) & (yb <= float(ny_b - 1))
        sub_a = src_a[in_b].copy()
        sub_a["xb"] = xb[in_b]
        sub_a["yb"] = yb[in_b]
        sub_a = sub_a.reset_index(drop=True)
        if len(sub_a) == 0:
            return []

        sc_a = SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg)
        sc_b = SkyCoord(src_b_all["ra"].to_numpy() * u.deg, src_b_all["dec"].to_numpy() * u.deg)
        _, sep2d, _ = sc_a.match_to_catalog_sky(sc_b)

        # Bug 1 (catalog): match against ALL raw partner detections <= 15"
        # Bug 2 (pixel): require same-sign peak >= 4*rms, no negative/white core >= -3*rms, no saturated pixels
        pix_ok = (
            (sub_a["peak_snr"].to_numpy() >= 4.0)
            & (sub_a["min3_snr"].to_numpy() >= -3.0)
            & (sub_a["n_sat_pix"].to_numpy() == 0)
        )
        cands_df = sub_a[(sep2d.arcsec > 15.0) & (~sub_a["gaia_hit"].to_numpy()) & pix_ok].reset_index(drop=True)
        if len(cands_df) == 0:
            return []

        # Bug 1 (pixel): partner aperture check at predicted (xb, yb)
        psf_b = med_r_b if (np.isfinite(med_r_b) and med_r_b > 0) else 3.0
        r_ap = max(3, int(round(2.0 * psf_b)))
        b_sub, b_rms = img_b["data_sub"], img_b["rms_img"]
        keep_pix_b = np.ones(len(cands_df), dtype=bool)
        for ic, (xbi, ybi) in enumerate(zip(cands_df["xb"].to_numpy(), cands_df["yb"].to_numpy())):
            ix, iy = int(round(float(xbi))), int(round(float(ybi)))
            x0, x1 = max(0, ix - r_ap), min(nx_b, ix + r_ap + 1)
            y0, y1 = max(0, iy - r_ap), min(ny_b, iy + r_ap + 1)
            if x1 > x0 and y1 > y0:
                patch = b_sub[y0:y1, x0:x1]
                rms_loc = float(np.median(b_rms[y0:y1, x0:x1]))
                ap_snr = abs(float(np.sum(patch))) / (rms_loc * np.sqrt(max(1, patch.size)))
                abs_pk = float(np.max(np.abs(patch))) / rms_loc
                if ap_snr > 3.0 or abs_pk > 4.0:
                    keep_pix_b[ic] = False
        cands_df = cands_df[keep_pix_b].reset_index(drop=True)
        if len(cands_df) == 0:
            return []

        xb_arr = cands_df["xb"].to_numpy(dtype=float)
        yb_arr = cands_df["yb"].to_numpy(dtype=float)
        edge_px = np.minimum(np.minimum(xb_arr, float(nx_b - 1) - xb_arr), np.minimum(yb_arr, float(ny_b - 1) - yb_arr))
        edge_psf = edge_px / psf_b

        ng = local_grid_b.shape[0]
        ix_arr = np.clip((xb_arr / max(nx_b, 1) * ng).astype(int), 0, ng - 1)
        iy_arr = np.clip((yb_arr / max(ny_b, 1) * ng).astype(int), 0, ng - 1)
        partner_local_mlim = local_grid_b[iy_arr, ix_arr]

        mag_est_arr = cands_df["mag_est"].to_numpy(dtype=float)
        mag_margin = np.where(np.isfinite(mag_est_arr), partner_local_mlim - mag_est_arr, np.nan)
        passes_partner = (edge_psf >= 2.0) & np.isfinite(mag_margin) & (mag_margin >= 1.0)

        # Skip logging >500 noise rows for plates that already failed astrometric QC
        if (not qc_a_pass or not qc_b_pass) and len(cands_df) > 500:
            return [
                {"passes_partner_rule": bool(p), "is_asteroid": False, "_skip_save": True}
                for p in passes_partner
            ]

        sc_c = SkyCoord(cands_df["ra"].to_numpy() * u.deg, cands_df["dec"].to_numpy() * u.deg)
        ecl_lats = sc_c.barycentrictrueecliptic.lat.deg

        out_rows = []
        ra_arr = cands_df["ra"].to_numpy(dtype=float)
        dec_arr = cands_df["dec"].to_numpy(dtype=float)
        stamp_dir = Path("data/processed/stamps")
        if qc_a_pass and qc_b_pass and np.any(passes_partner):
            stamp_dir.mkdir(parents=True, exist_ok=True)
        for idx_c in range(len(cands_df)):
            if qc_a_pass and qc_b_pass and passes_partner[idx_c]:
                ca, sa_m, ps_a, hb_a = extract_stamp_60as(
                    img_a["data_sub"], img_a["rms_img"], img_a["sat_mask"], wcs_a, float(ra_arr[idx_c]), float(dec_arr[idx_c])
                )
                cb, sb_m, ps_b, hb_b = extract_stamp_60as(
                    img_b["data_sub"], img_b["rms_img"], img_b["sat_mask"], wcs_b, float(ra_arr[idx_c]), float(dec_arr[idx_c])
                )
                np.savez_compressed(
                    stamp_dir / f"{meta_a['exposure_id']}_{idx_c}.npz",
                    cut_a=ca, sat_a=sa_m, ps_a=np.float32(ps_a), hb_a=np.bool_(hb_a),
                    cut_b=cb, sat_b=sb_m, ps_b=np.float32(ps_b), hb_b=np.bool_(hb_b),
                    ra=np.float64(ra_arr[idx_c]), dec=np.float64(dec_arr[idx_c]),
                    wcs_a_hdr=wcs_a.to_header_string(), wcs_b_hdr=wcs_b.to_header_string(),
                )
            out_rows.append({
                "source_archive": sa,
                "instrument": row["instrument"],
                "exp_a": meta_a["exposure_id"],
                "exp_b": meta_b["exposure_id"],
                "plate_a": plate_a,
                "plate_b": plate_b,
                "ra": float(ra_arr[idx_c]),
                "dec": float(dec_arr[idx_c]),
                "ecliptic_lat_deg": float(ecl_lats[idx_c]),
                "mag_est": float(mag_est_arr[idx_c]),
                "partner_m_lim": float(partner_local_mlim[idx_c]),
                "mag_margin_partner": float(mag_margin[idx_c]),
                "edge_dist_psf": float(edge_psf[idx_c]),
                "passes_partner_rule": bool(passes_partner[idx_c]),
                "plate_a_qc_pass": qc_a_pass,
                "plate_b_qc_pass": qc_b_pass,
                "mjd_mid_a": float(meta_a["mjd_mid"]),
                "exptime_sec_a": float(meta_a["exptime_sec"]),
                "obs_lat_deg_a": float(meta_a["obs_lat_deg"]),
                "obs_lon_deg_a": float(meta_a["obs_lon_deg"]),
                "obs_elev_m_a": float(meta_a["obs_elev_m"]),
                "dt_hours": float(row["dt_hours"]),
                "is_asteroid": False,
                "ast_desig": None,
                "ast_sep_arcsec": np.nan,
                "ast_vmag": np.nan,
                "mpc_ok": False,
            })
        return out_rows

    cands_1 = _single_plate_cands(df1, df2_all, wcs1, wcs2, ny2, nx2, med_r2, local_grid2, meta1, meta2, p1, p2, qc1["qc_pass"], qc2["qc_pass"], img1, img2)
    cands_2 = _single_plate_cands(df2, df1_all, wcs2, wcs1, ny1, nx1, med_r1, local_grid1, meta2, meta1, p2, p1, qc2["qc_pass"], qc1["qc_pass"], img2, img1)
    del img1, img2
    all_cands = cands_1 + cands_2

    c1_raw = len(cands_1)
    c2_raw = len(cands_2)
    c1_rule = sum(1 for c in cands_1 if c["passes_partner_rule"])
    c2_rule = sum(1 for c in cands_2 if c["passes_partner_rule"])

    return {
        "source_archive": sa,
        "instrument": row["instrument"],
        "exp1": e1,
        "exp2": e2,
        "plate1": p1,
        "plate2": p2,
        "dt_hours": float(row["dt_hours"]),
        "fov_deg2": float(row["fov_deg2"]),
        "n_src1": len(df1),
        "n_src2": len(df2),
        "bright_match_1": qc1["bright_match_rate"],
        "bright_match_2": qc2["bright_match_rate"],
        "med_resid_1": qc1["med_resid_after_as"],
        "med_resid_2": qc2["med_resid_after_as"],
        "qc_pass_pair": bool(qc1["qc_pass"] and qc2["qc_pass"]),
        "cands_raw_total": c1_raw + c2_raw,
        "cands_rule_total": c1_rule + c2_rule,
        "elapsed_s": round(time.time() - t0, 1),
        "status": "ok",
        "_plate_qcs": plate_qcs,
        "_footprints": footprints,
        "_cands": all_cands,
    }


def run_pilot(n_pairs: int = 200, max_workers: int = 2, reset: bool = False) -> None:
    sample_path = Path("data/processed/pilot_sample_pairs.csv")
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
    else:
        sample = build_stratified_pilot_sample(n_total=n_pairs)

    sample = sample.head(n_pairs).copy()

    # Re-verify blindness on every single plate before any network call
    exp_df = pd.read_parquet("data/processed/exposures.parquet")
    sh_map = exp_df.set_index("exposure_id")[["shadow_fraction", "shadow_fraction_36000km", "shadow_fraction_20000km"]]
    for col in ["exp1", "exp2"]:
        vals = sh_map.loc[sample[col]].to_numpy()
        assert np.all(vals == 0.0), f"BLINDNESS GUARD TRIGGERED on {col}!"

    used_exps = set(sample["exp1"]).union(set(sample["exp2"]))
    exp_sub = exp_df[exp_df["exposure_id"].isin(used_exps)].copy()
    exp_meta = {
        r["exposure_id"]: r.to_dict()
        for _, r in exp_sub.iterrows()
    }

    app_pids = pd.concat([sample.loc[sample["source_archive"] == "APPLAUSE", "plate1"], sample.loc[sample["source_archive"] == "APPLAUSE", "plate2"]]).unique().tolist()
    applause_urls, applause_wcs = _get_applause_scan_and_wcs_map(app_pids)

    out_path = Path("data/processed/pilot_results.csv")
    qc_path = Path("data/processed/pilot_plate_qc.csv")
    drop_path = Path("data/processed/pilot_dropped_plates.csv")
    fp_path = Path("data/processed/pilot_footprints.parquet")
    cand_path = Path("data/processed/pilot_candidates.parquet")

    if reset:
        for p in [out_path, qc_path, drop_path, fp_path, cand_path]:
            if p.exists():
                p.unlink()
        cat_dir = Path("data/processed/pilot_catalogs")
        if cat_dir.exists():
            for f in cat_dir.glob("*.parquet"):
                f.unlink()

    done_keys = set()
    results: list[dict[str, Any]] = []
    plate_qcs: list[dict[str, Any]] = []
    footprints: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    if out_path.exists() and not reset:
        prev = pd.read_csv(out_path)
        results = prev.to_dict(orient="records")
        done_keys = set(zip(prev["exp1"], prev["exp2"]))
        if qc_path.exists():
            plate_qcs = pd.read_csv(qc_path).to_dict(orient="records")
        if fp_path.exists():
            footprints = pd.read_parquet(fp_path).to_dict(orient="records")
        if cand_path.exists():
            candidates = pd.read_parquet(cand_path).to_dict(orient="records")

    todo = [row for _, row in sample.iterrows() if (row["exp1"], row["exp2"]) not in done_keys]
    import random
    random.Random(42).shuffle(todo)
    print(f"Pilot: {len(sample)} pairs selected, {len(done_keys)} already done, {len(todo)} remaining.")

    def _flush() -> None:
        pd.DataFrame(results).to_csv(out_path, index=False)
        if plate_qcs:
            qc_df = pd.DataFrame(plate_qcs).drop_duplicates(subset=["exposure_id"])
            qc_df.to_csv(qc_path, index=False)
            qc_df[~qc_df["qc_pass"]].to_csv(drop_path, index=False)
        if footprints:
            pd.DataFrame(footprints).drop_duplicates(subset=["exposure_id"]).to_parquet(fp_path, index=False)
        if candidates:
            pd.DataFrame(candidates).to_parquet(cand_path, index=False)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_pair, row, exp_meta, applause_urls, applause_wcs): row for row in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
            except Exception as e:
                r = futs[fut]
                res = {"source_archive": r["source_archive"], "instrument": r["instrument"], "exp1": r["exp1"], "exp2": r["exp2"], "status": f"error: {e}"}
            plate_qcs.extend(res.pop("_plate_qcs", []))
            footprints.extend(res.pop("_footprints", []))
            candidates.extend([c for c in res.pop("_cands", []) if not c.get("_skip_save")])
            results.append(res)
            if k % 5 == 0 or k == len(todo):
                _flush()
                print(
                    f"  [{k}/{len(todo)}] last={res.get('plate1')}/{res.get('plate2')} "
                    f"status={res.get('status')} raw={res.get('cands_raw_total')} rule={res.get('cands_rule_total')} clean={res.get('cands_clean_total')} qc={res.get('qc_pass_pair')}"
                )

    _flush()
    print("Pilot run complete.")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    reset_flag = "--reset" in sys.argv
    run_pilot(n_pairs=n, max_workers=2, reset=reset_flag)
