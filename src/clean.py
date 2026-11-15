"""Quality filtering, DASCH site validation, flash sensitivity, and loss-table logging."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time

from src.fetch import load_config

# MJD of 1957-10-04 00:00:00 UTC (Sputnik 1 launch)
SPUTNIK_MJD = 36115.0

# MJD of 1928-01-01 00:00:00 UTC (Boyden Station move from Arequipa to Bloemfontein)
BOYDEN_MOVE_MJD = 25246.0

# DASCH series -> observatory coordinates (lat_deg, lon_deg, elev_m)
# Cross-checked against Harvard Plate Stacks station history:
# - Northern stations: Cambridge MA (42.381, -71.128) & Oak Ridge MA (42.505, -71.558)
# - Southern stations: Arequipa, Peru (-16.375, -71.525) pre-1928; Bloemfontein, SA (-29.039, +26.405) 1928+
DASCH_NORTH_OBS = (42.505, -71.558, 185.0)
DASCH_AREQUIPA_OBS = (-16.375, -71.525, 2450.0)
DASCH_BLOEMFONTEIN_OBS = (-29.039, 26.405, 1387.0)
DASCH_MARIA_MITCHELL_OBS = (41.281, -70.097, 10.0)
DASCH_LOWELL_OBS = (35.203, -111.665, 2210.0)
DASCH_PALOMAR_OBS = (33.3563, -116.8650, 1712.0)

DASCH_NORTH_SERIES = {
    "ac", "ai", "i", "mc", "rh", "fa", "ir", "c", "ma", "md", "rl", "bm", "j",
    "ka", "ee", "ay", "mb", "h", "ca", "kb", "kd", "kc", "kf", "ke", "er",
    "kh", "kg", "ro", "me", "aco", "an", "dnb",
}
DASCH_SOUTH_SERIES = {
    "b", "mf", "am", "a", "rb", "bi", "ax", "br", "x", "ak", "adh", "pz",
    "bo", "ab", "d",
}

# Documented per-instrument limiting magnitudes (B/pg) for DASCH when per-plate m_lim is unavailable
DASCH_INSTRUMENT_MLIM = {
    "200 inch Hale Telescope": 21.0,
    "24-33 in Jewett Schmidt": 17.5,
    "32-36 inch BakerSchmidt 10 1/2 inch round Armagh-Dunsink-Harvard": 17.5,
    "24-inch Bruce Doublet": 17.5,
    "24-inch Clark Reflector": 17.5,
    "Lowel 40 inch reflector": 17.5,
    "16-inch Metcalf Doublet (Refigured after 3500)": 17.5,
    "13-inch Boyden Refractor": 16.5,
    "12-inch Metcalf Doublet": 16.5,
    "11-inch Draper Refractor": 16.5,
    "10-inch Metcalf Triplet": 16.5,
    "8-inch Bache Doublet": 15.5,
    "8-inch Draper Doublet": 15.5,
    "8-inch Ross Lundin": 15.5,
    "8-inch Brashear Lens": 15.5,
    "7.5-inch Cooke/Clark Refractor at Maria Mitchell Observatory": 15.5,
    "4-inch Cooke Lens": 14.5,
    "4-inch Ross Lundin": 14.5,
    "4-inch Voightlander Lens": 14.5,
    "4-inch Cooke (1-327)": 14.5,
    "3-inch Ross Fecker": 14.0,
    "3 inch Ross-Tessar Lens": 14.0,
    "3-inch Ross": 14.0,
    "3-inch Elmer Ross": 14.0,
    "3 inch Perkin-Zeiss Lens": 14.0,
    "2.8-inch Kodak Aero-Ektar": 13.5,
    "2.6-inch Zeiss-Tessar": 13.5,
    "2.5 inch Cooke Lens": 13.5,
    "2.5 inch Voigtlander (Little Bache or \"Bachito\")": 13.5,
    "2.5-inch Ross Portrait Lens": 13.5,
    "Eastman Aero-Ektar K-24 Lens on a K-19 Barrel": 13.5,
    "Eastman Aero-Ektar K-24 Lens on a K-19 Barrel Formerly KE (1951)": 13.5,
    "Eastman Aero-Ektar K-24 Lens on a K-19 Barrel Formerly KG (1950)": 13.5,
    "Air Force Camera": 13.5,
    "K-19 Air Force Camera": 13.5,
    "KE Camera with Installed Rough Focus": 13.5,
    "1.5-inch Cooke Lenses": 13.0,
    "1.5-inch Ross-Xpress": 13.0,
    "1.5 inch Ross (short focus)": 13.0,
    "1.5 in Cooke \"Long Focus\"": 13.0,
    "1-inch": 13.0,
    "Patrol cameras": 13.0,
    "1 in Cook Lens #832 Series renamed from ac-a": 13.0,
    "New Cooke Lens": 13.0,
}


def _assign_dasch_observatory(series: pd.Series, t_min: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(series)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    elev = np.full(n, np.nan)

    s_vals = series.fillna("").str.lower().to_numpy()
    mjd_vals = pd.to_numeric(t_min, errors="coerce").to_numpy()

    for i, (s, mjd) in enumerate(zip(s_vals, mjd_vals)):
        if s in DASCH_NORTH_SERIES:
            lat[i], lon[i], elev[i] = DASCH_NORTH_OBS
        elif s in DASCH_SOUTH_SERIES:
            if np.isfinite(mjd) and mjd < BOYDEN_MOVE_MJD:
                lat[i], lon[i], elev[i] = DASCH_AREQUIPA_OBS
            else:
                lat[i], lon[i], elev[i] = DASCH_BLOEMFONTEIN_OBS
        elif s == "na":
            lat[i], lon[i], elev[i] = DASCH_MARIA_MITCHELL_OBS
        elif s == "lwla":
            lat[i], lon[i], elev[i] = DASCH_LOWELL_OBS
        elif s == "hale":
            lat[i], lon[i], elev[i] = DASCH_PALOMAR_OBS
    return lat, lon, elev


def _compute_sun_and_pointing_alt_deg(
    mjd_mid: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    elev_m: np.ndarray,
    batch_size: int = 25000,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized solar altitude and plate-center pointing altitude (degrees) at mid-exposure."""
    n = len(mjd_mid)
    sun_alt = np.full(n, np.nan)
    pt_alt = np.full(n, np.nan)
    valid = (
        np.isfinite(mjd_mid)
        & np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & np.isfinite(lat_deg)
        & np.isfinite(lon_deg)
    )
    idx_all = np.where(valid)[0]

    for start in range(0, len(idx_all), batch_size):
        idx = idx_all[start : start + batch_size]
        t = Time(mjd_mid[idx], format="mjd", scale="utc")
        el = np.where(np.isfinite(elev_m[idx]), elev_m[idx], 0.0)
        loc = EarthLocation(lat=lat_deg[idx] * u.deg, lon=lon_deg[idx] * u.deg, height=el * u.m)
        frame = AltAz(obstime=t, location=loc)

        sun_alt[idx] = get_sun(t).transform_to(frame).alt.to_value(u.deg)
        pt_sc = SkyCoord(ra=ra_deg[idx] * u.deg, dec=dec_deg[idx] * u.deg, frame="icrs")
        pt_alt[idx] = pt_sc.transform_to(frame).alt.to_value(u.deg)

    return sun_alt, pt_alt


def load_applause_raw(raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """Join APPLAUSE DR4 exposure, plate, archive, scan summary, solution, and phot_calib."""
    exp = pd.read_parquet(raw_dir / "applause_exposure.parquet")
    plt = pd.read_parquet(raw_dir / "applause_plate.parquet")
    arc = pd.read_parquet(raw_dir / "applause_archive.parquet")
    scn = pd.read_parquet(raw_dir / "applause_scan.parquet")
    sol = pd.read_parquet(raw_dir / "applause_solution.parquet")

    scn_sum = scn.groupby("plate_id", as_index=False).agg(n_scans=("scan_id", "count"))
    scn_sum["has_xy_scans"] = scn_sum["n_scans"] >= 2

    sol_sorted = sol.sort_values(["plate_id", "solution_num", "solution_id"])
    sol_best = sol_sorted.drop_duplicates(subset=["plate_id"], keep="first")[
        ["plate_id", "ra_icrs", "dec_icrs", "fov1", "fov2", "pixel_scale"]
    ].rename(columns={"ra_icrs": "sol_ra", "dec_icrs": "sol_dec"})

    # Per-plate limiting magnitude from applause_dr4.phot_calib
    phot_path = raw_dir / "applause_phot_calib.parquet"
    if phot_path.exists():
        phot = pd.read_parquet(phot_path)
        phot_sum = phot.groupby("plate_id", as_index=False).agg(m_lim=("plate_mag_lim", "median"))
    else:
        phot_sum = pd.DataFrame(columns=["plate_id", "m_lim"])

    df = exp.merge(plt, on=["plate_id", "archive_id"], how="left", suffixes=("", "_plt"))
    df = df.merge(arc[["archive_id", "archive_name"]], on="archive_id", how="left")
    df = df.merge(scn_sum, on="plate_id", how="left")
    df = df.merge(sol_best, on="plate_id", how="left")
    df = df.merge(phot_sum, on="plate_id", how="left")

    jd_start = pd.to_numeric(df["jd_start"], errors="coerce")
    jd_mid = pd.to_numeric(df["jd_mid"], errors="coerce")
    exptime = pd.to_numeric(df["exptime"], errors="coerce")

    mjd_mid = jd_mid - 2400000.5
    mjd_start = np.where(
        np.isfinite(jd_start),
        jd_start - 2400000.5,
        mjd_mid - 0.5 * (exptime / 86400.0),
    )

    ra_deg = np.where(np.isfinite(df["sol_ra"]), df["sol_ra"], pd.to_numeric(df["ra_icrs"], errors="coerce"))
    dec_deg = np.where(np.isfinite(df["sol_dec"]), df["sol_dec"], pd.to_numeric(df["dec_icrs"], errors="coerce"))
    has_wcs = df["sol_ra"].notna() & df["sol_dec"].notna() & (df["fov1"] > 0) & (df["fov2"] > 0)

    inst = df["telescope"].fillna(df["instrument"]).fillna(df["archive_name"]).astype(str)

    # Fill missing m_lim with per-instrument median from phot_calib, then 15.0 fallback
    m_lim = pd.to_numeric(df["m_lim"], errors="coerce")
    inst_med = m_lim.groupby(inst).transform("median")
    m_lim = m_lim.fillna(inst_med).fillna(15.0)
    m_lim_source = np.where(df["m_lim"].notna(), "applause_phot_calib", "instrument_median_or_fallback")

    out = pd.DataFrame(
        {
            "source_archive": "APPLAUSE",
            "sub_archive": df["archive_name"].fillna("Unknown").astype(str),
            "instrument": inst,
            "series": df["series"].fillna("").astype(str),
            "exposure_id": "APPLAUSE_" + df["exposure_id"].astype(str),
            "plate_id": "APPLAUSE_" + df["plate_id"].astype(str),
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "fov_ra_deg": pd.to_numeric(df["fov1"], errors="coerce"),
            "fov_dec_deg": pd.to_numeric(df["fov2"], errors="coerce"),
            "mjd_start": mjd_start,
            "mjd_mid": mjd_mid,
            "exptime_sec": exptime,
            "obs_lat_deg": pd.to_numeric(df["site_latitude"], errors="coerce"),
            "obs_lon_deg": pd.to_numeric(df["site_longitude"], errors="coerce"),
            "obs_elev_m": pd.to_numeric(df["site_elevation"], errors="coerce").fillna(0.0),
            "has_wcs": has_wcs,
            "has_xy_scans": df["has_xy_scans"].fillna(False).astype(bool),
            "multi_exposure_plate": (pd.to_numeric(df["numexp"], errors="coerce").fillna(1) > 1),
            "m_lim": m_lim,
            "m_lim_source": m_lim_source,
        }
    )
    return out


def load_dasch_raw(raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """Load and normalize DASCH narrow_plates and wide_plates."""
    dn = pd.read_parquet(raw_dir / "dasch_narrow_plates.parquet")
    dn["sub_archive"] = "DASCH-narrow"
    dw = pd.read_parquet(raw_dir / "dasch_wide_plates.parquet")
    dw["sub_archive"] = "DASCH-wide"
    df = pd.concat([dn, dw], ignore_index=True)

    series = df["dasch_id"].astype(str).str.extract(r"^([a-zA-Z]+)")[0].fillna("")
    lat, lon, elev = _assign_dasch_observatory(series, df["t_min"])

    t_min = pd.to_numeric(df["t_min"], errors="coerce")
    t_max = pd.to_numeric(df["t_max"], errors="coerce")
    exptime = pd.to_numeric(df["t_exptime"], errors="coerce")

    # Recover missing t_exptime from t_max - t_min where available
    miss_exp = exptime.isna() | (exptime <= 0)
    has_minmax = t_min.notna() & t_max.notna() & (t_max > t_min)
    recovered_mask = miss_exp & has_minmax
    exptime = np.where(recovered_mask, (t_max - t_min) * 86400.0, exptime)
    print(f"[DASCH] Recovered {int(recovered_mask.sum()):,} exposures via t_max - t_min")

    mjd_mid = np.where(
        np.isfinite(t_min) & np.isfinite(t_max) & (t_max >= t_min),
        0.5 * (t_min + t_max),
        t_min + 0.5 * (exptime / 86400.0),
    )

    px_scale = pd.to_numeric(df["s_pixel_scale"], errors="coerce")
    xel1 = pd.to_numeric(df["s_xel1"], errors="coerce")
    xel2 = pd.to_numeric(df["s_xel2"], errors="coerce")
    s_fov = pd.to_numeric(df["s_fov"], errors="coerce")

    fov_w = np.where((xel1 > 0) & (px_scale > 0), xel1 * px_scale / 3600.0, s_fov / np.sqrt(2.0))
    fov_h = np.where((xel2 > 0) & (px_scale > 0), xel2 * px_scale / 3600.0, s_fov / np.sqrt(2.0))

    ra_deg = pd.to_numeric(df["s_ra"], errors="coerce")
    dec_deg = pd.to_numeric(df["s_dec"], errors="coerce")
    has_wcs = (
        (pd.to_numeric(df["calib_level"], errors="coerce") >= 2)
        & np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & (fov_w > 0)
        & (fov_h > 0)
    )

    inst = df["instrument_name"].fillna("Unknown").astype(str)
    m_lim = inst.map(DASCH_INSTRUMENT_MLIM).fillna(14.0).astype(float)

    out = pd.DataFrame(
        {
            "source_archive": "DASCH",
            "sub_archive": df["sub_archive"],
            "instrument": inst,
            "series": series,
            "exposure_id": "DASCH_" + df["dasch_id"].astype(str),
            "plate_id": "DASCH_" + df["dasch_id"].astype(str),
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "fov_ra_deg": fov_w,
            "fov_dec_deg": fov_h,
            "mjd_start": t_min,
            "mjd_mid": mjd_mid,
            "exptime_sec": exptime,
            "obs_lat_deg": lat,
            "obs_lon_deg": lon,
            "obs_elev_m": elev,
            "has_wcs": has_wcs,
            "has_xy_scans": False,
            "multi_exposure_plate": False,
            "m_lim": m_lim,
            "m_lim_source": "dasch_instrument_estimate",
        }
    )
    return out


def load_poss1_raw(raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """
    Load POSS-I VI/25/nposs: use POSS-I-E (red) plates ONLY as the VASCO benchmark.
    ObsE is confirmed as exposure START time in PST (UTC - 8h) via EHA comparison.
    """
    df = pd.read_parquet(raw_dir / "poss1_nposs.parquet")

    valid_coord = df["_RA.icrs"].notna() & df["_DE.icrs"].notna()
    sc = SkyCoord(
        df.loc[valid_coord, "_RA.icrs"].astype(str).tolist(),
        df.loc[valid_coord, "_DE.icrs"].astype(str).tolist(),
        unit=(u.hourangle, u.deg),
        frame="icrs",
    )
    ra_all = np.full(len(df), np.nan)
    dec_all = np.full(len(df), np.nan)
    ra_all[valid_coord.to_numpy()] = sc.ra.deg
    dec_all[valid_coord.to_numpy()] = sc.dec.deg

    rows = []
    for idx, row in df.iterrows():
        obs_date = str(row["Obs"]).strip() if pd.notna(row["Obs"]) else ""
        t_str = str(row["ObsE"]).strip() if pd.notna(row["ObsE"]) else ""
        exptime_min = pd.to_numeric(row["Eexp"], errors="coerce")
        exptime_sec = float(exptime_min) * 60.0 if np.isfinite(exptime_min) else np.nan

        mjd_start = np.nan
        mjd_mid = np.nan
        if obs_date and re.match(r"^\d{4}-\d{2}-\d{2}$", obs_date) and re.match(r"^\d{2}:\d{2}$", t_str):
            hh, mm = map(int, t_str.split(":"))
            t_evening = Time(f"{obs_date}T00:00:00", format="isot", scale="utc")
            day_offset = 1.0 if hh < 12 else 0.0
            t_local = t_evening + (day_offset + (hh + mm / 60.0) / 24.0) * u.day
            t_utc = t_local + 8.0 * u.hour  # PST -> UTC (ObsE is exposure START)
            mjd_start = float(t_utc.mjd)
            if np.isfinite(exptime_sec) and exptime_sec > 0:
                mjd_mid = mjd_start + 0.5 * (exptime_sec / 86400.0)

        rows.append(
            {
                "source_archive": "POSS-I",
                "sub_archive": "POSS-I-E",
                "instrument": "Palomar 48-inch Oschin Schmidt",
                "series": "POSS-I-E",
                "exposure_id": f"POSSI_{row['MLP']}_E",
                "plate_id": f"POSSI_{row['MLP']}_E",
                "ra_deg": ra_all[idx],
                "dec_deg": dec_all[idx],
                "fov_ra_deg": 6.5,
                "fov_dec_deg": 6.5,
                "mjd_start": mjd_start,
                "mjd_mid": mjd_mid,
                "exptime_sec": exptime_sec,
                "obs_lat_deg": 33.3563,
                "obs_lon_deg": -116.8650,
                "obs_elev_m": 1712.0,
                "has_wcs": bool(np.isfinite(ra_all[idx]) and np.isfinite(dec_all[idx])),
                "has_xy_scans": False,
                "multi_exposure_plate": False,
                "m_lim": 20.0,
                "m_lim_source": "poss1_e_canonical",
            }
        )
    return pd.DataFrame(rows)


def apply_quality_filters(
    df: pd.DataFrame,
    sputnik_mjd: float = SPUTNIK_MJD,
    sun_alt_max_deg: float = -12.0,
    pointing_alt_min_deg: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply quality filters sequentially and build loss tables + DASCH series validation table.
    """
    df = df.copy()
    df["f1_valid_time"] = (
        np.isfinite(df["mjd_start"])
        & np.isfinite(df["mjd_mid"])
        & np.isfinite(df["exptime_sec"])
        & (df["exptime_sec"] > 0.0)
        & (df["exptime_sec"] <= 43200.0)
    )
    df["f2_pre_sputnik"] = df["f1_valid_time"] & (df["mjd_mid"] < sputnik_mjd)
    df["f3_has_wcs"] = (
        df["f2_pre_sputnik"]
        & df["has_wcs"]
        & np.isfinite(df["ra_deg"])
        & np.isfinite(df["dec_deg"])
        & (df["fov_ra_deg"] > 0)
        & (df["fov_dec_deg"] > 0)
    )
    df["f4_has_obs_coords"] = df["f3_has_wcs"] & np.isfinite(df["obs_lat_deg"]) & np.isfinite(df["obs_lon_deg"])

    sun_alt = np.full(len(df), np.nan)
    pt_alt = np.full(len(df), np.nan)
    mask_f4 = df["f4_has_obs_coords"].to_numpy()
    if np.any(mask_f4):
        sun_alt[mask_f4], pt_alt[mask_f4] = _compute_sun_and_pointing_alt_deg(
            df.loc[mask_f4, "mjd_mid"].to_numpy(),
            df.loc[mask_f4, "ra_deg"].to_numpy(),
            df.loc[mask_f4, "dec_deg"].to_numpy(),
            df.loc[mask_f4, "obs_lat_deg"].to_numpy(),
            df.loc[mask_f4, "obs_lon_deg"].to_numpy(),
            df.loc[mask_f4, "obs_elev_m"].to_numpy(),
        )
    df["sun_alt_mid_deg"] = sun_alt
    df["pointing_alt_mid_deg"] = pt_alt

    df["f5_sun_below_12"] = df["f4_has_obs_coords"] & np.isfinite(df["sun_alt_mid_deg"]) & (df["sun_alt_mid_deg"] < sun_alt_max_deg)
    df["f6_pointing_above_10"] = df["f5_sun_below_12"] & np.isfinite(df["pointing_alt_mid_deg"]) & (df["pointing_alt_mid_deg"] > pointing_alt_min_deg)

    # Flash sensitivity m_flash = m_lim - 2.5 * log10(exptime / tau)
    exptime = df["exptime_sec"].to_numpy()
    m_lim = df["m_lim"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["m_flash_05s"] = np.where(exptime > 0, m_lim - 2.5 * np.log10(exptime / 0.5), np.nan)
        df["m_flash_01s"] = np.where(exptime > 0, m_lim - 2.5 * np.log10(exptime / 0.1), np.nan)
        df["m_flash_20s"] = np.where(exptime > 0, m_lim - 2.5 * np.log10(exptime / 2.0), np.nan)

    def _agg_loss(grp_cols: list[str]) -> pd.DataFrame:
        g = df.groupby(grp_cols, dropna=False).agg(
            raw_rows=("exposure_id", "count"),
            pass_valid_time=("f1_valid_time", "sum"),
            pass_pre_sputnik=("f2_pre_sputnik", "sum"),
            pass_wcs=("f3_has_wcs", "sum"),
            pass_obs_coords=("f4_has_obs_coords", "sum"),
            pass_sun_alt=("f5_sun_below_12", "sum"),
            pass_pointing_alt=("f6_pointing_above_10", "sum"),
        ).reset_index()
        g["drop_time"] = g["raw_rows"] - g["pass_valid_time"]
        g["drop_post_sputnik"] = g["pass_valid_time"] - g["pass_pre_sputnik"]
        g["drop_no_wcs"] = g["pass_pre_sputnik"] - g["pass_wcs"]
        g["drop_no_obs"] = g["pass_wcs"] - g["pass_obs_coords"]
        g["drop_sun_alt"] = g["pass_obs_coords"] - g["pass_sun_alt"]
        g["drop_pointing_alt"] = g["pass_sun_alt"] - g["pass_pointing_alt"]
        return g.sort_values("pass_pointing_alt", ascending=False)

    loss_archive = _agg_loss(["source_archive", "sub_archive"])
    loss_instrument = _agg_loss(["source_archive", "instrument"])

    # DASCH per-series validation table (pointing + sun check failure rates)
    dasch_f4 = df[(df["source_archive"] == "DASCH") & df["f4_has_obs_coords"]].copy()
    dasch_series_val = (
        dasch_f4.groupby(["series", "instrument"], dropna=False)
        .agg(
            n_f4=("exposure_id", "count"),
            n_fail_sun=("f5_sun_below_12", lambda s: int((~s).sum())),
            n_fail_pointing=("pointing_alt_mid_deg", lambda s: int((s <= pointing_alt_min_deg).sum())),
            n_pass_both=("f6_pointing_above_10", "sum"),
        )
        .reset_index()
    )
    dasch_series_val["fail_sun_pct"] = 100.0 * dasch_series_val["n_fail_sun"] / dasch_series_val["n_f4"]
    dasch_series_val["fail_pointing_pct"] = 100.0 * dasch_series_val["n_fail_pointing"] / dasch_series_val["n_f4"]
    dasch_series_val = dasch_series_val.sort_values("n_f4", ascending=False)

    clean_df = df[df["f6_pointing_above_10"]].drop(
        columns=[
            "f1_valid_time",
            "f2_pre_sputnik",
            "f3_has_wcs",
            "f4_has_obs_coords",
            "f5_sun_below_12",
            "f6_pointing_above_10",
        ]
    ).reset_index(drop=True)

    # Drop whole DASCH series with >10% Sun-check or pointing-check failures (systematic timing/site errors)
    bad_series = set(
        dasch_series_val.loc[
            (dasch_series_val["fail_sun_pct"] > 10.0) | (dasch_series_val["fail_pointing_pct"] > 10.0),
            "series",
        ].astype(str)
    )
    if bad_series:
        drop_series_mask = (clean_df["source_archive"] == "DASCH") & (clean_df["series"].isin(bad_series))
        clean_df = clean_df[~drop_series_mask].reset_index(drop=True)

    return clean_df, loss_archive, loss_instrument, dasch_series_val


def run_cleaning(config_path: str | Path = "config.yaml") -> None:
    cfg = load_config(config_path)
    raw_dir = Path("data/raw")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    sputnik_mjd = float(Time(cfg["cutoffs"]["sputnik_utc"], format="isot", scale="utc").mjd)
    sun_alt_max = float(cfg["cutoffs"]["sun_alt_max_deg"])

    df_app = load_applause_raw(raw_dir)
    df_dasch = load_dasch_raw(raw_dir)
    df_poss = load_poss1_raw(raw_dir)
    df_all = pd.concat([df_app, df_dasch, df_poss], ignore_index=True)

    clean_df, loss_arch, loss_inst, dasch_series_val = apply_quality_filters(
        df_all, sputnik_mjd=sputnik_mjd, sun_alt_max_deg=sun_alt_max, pointing_alt_min_deg=10.0
    )

    clean_df.to_parquet(out_dir / "exposures_clean.parquet", index=False)
    loss_arch.to_csv(out_dir / "loss_table_by_archive.csv", index=False)
    loss_inst.to_csv(out_dir / "loss_table_by_instrument.csv", index=False)
    dasch_series_val.to_csv(out_dir / "dasch_series_validation.csv", index=False)

    print("=== DASCH SERIES SITE VALIDATION (TOP 25) ===")
    print(dasch_series_val.head(25).to_string(index=False))
    print("\n=== DASCH SERIES WITH >10% POINTING OR SUN FAILURES ===")
    bad = dasch_series_val[(dasch_series_val["fail_sun_pct"] > 10.0) | (dasch_series_val["fail_pointing_pct"] > 10.0)]
    print(bad.to_string(index=False))
    print(f"\nTotal kept exposures: {len(clean_df):,}")


if __name__ == "__main__":
    run_cleaning()
