"""Historical-epoch asteroid cross-matching via SkyBoT (one cone search per plate) for Phase 2a/2b."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.imcce import Skybot


def query_skybot_plate(
    ra_center_deg: float,
    dec_center_deg: float,
    radius_deg: float,
    mjd_mid: float,
    obs_code: str = "500",
) -> pd.DataFrame:
    """Run a single SkyBoT cone search for a plate at `mjd_mid`."""
    epoch = Time(float(mjd_mid), format="mjd", scale="utc")
    center = SkyCoord(float(ra_center_deg) * u.deg, float(dec_center_deg) * u.deg)
    for attempt in range(3):
        try:
            tbl = Skybot.cone_search(center, float(radius_deg) * u.deg, epoch, location=obs_code)
            if tbl is None or len(tbl) == 0:
                return pd.DataFrame(columns=["desig", "ra", "dec", "vmag", "motion_as_hr"])
            df = tbl.to_pandas()
            # SkyBoT columns: Name, RA (deg), DEC (deg), V, mv (arcsec/hr or rate)
            ra_col = "RA" if "RA" in df.columns else "ra"
            dec_col = "DEC" if "DEC" in df.columns else "dec"
            name_col = "Name" if "Name" in df.columns else "name"
            v_col = "V" if "V" in df.columns else "vmag"
            # SkyBoT provides 'ra_rate' and 'dec_rate' (arcsec/hr) or 'mv'
            if "ra_rate" in df.columns and "dec_rate" in df.columns:
                mot = np.hypot(df["ra_rate"].to_numpy(dtype=float), df["dec_rate"].to_numpy(dtype=float))
            elif "mv" in df.columns:
                mot = np.abs(df["mv"].to_numpy(dtype=float))
            else:
                mot = np.full(len(df), 40.0)
            return pd.DataFrame({
                "desig": df[name_col].astype(str),
                "ra": df[ra_col].to_numpy(dtype=float),
                "dec": df[dec_col].to_numpy(dtype=float),
                "vmag": pd.to_numeric(df[v_col], errors="coerce"),
                "motion_as_hr": mot,
            })
        except Exception as e:
            if "No solar system object was found" in str(e):
                return pd.DataFrame(columns=["desig", "ra", "dec", "vmag", "motion_as_hr"])
            if attempt == 2:
                return pd.DataFrame(columns=["desig", "ra", "dec", "vmag", "motion_as_hr"])
            time.sleep(2.0 * (attempt + 1))
    return pd.DataFrame(columns=["desig", "ra", "dec", "vmag", "motion_as_hr"])


def crossmatch_candidates_skybot(
    df_cands: pd.DataFrame,
    plate_centers: dict[str, tuple[float, float, float, float]],
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Cross-match candidates against SkyBoT using ONE cone search per unique plate (`exp_a`).
    `plate_centers`: {exposure_id: (ra_center_deg, dec_center_deg, radius_deg, mjd_mid)}
    """
    if len(df_cands) == 0:
        return df_cands.assign(is_asteroid=False, ast_desig=None, ast_sep_arcsec=np.nan, ast_vmag=np.nan)

    df = df_cands.copy().reset_index(drop=True)
    unique_exps = df["exp_a"].unique().tolist()
    skybot_cache: dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for eid in unique_exps:
            ra_c, dec_c, rad_deg, mjd_m = plate_centers[eid]
            futs[ex.submit(query_skybot_plate, ra_c, dec_c, rad_deg, mjd_m)] = eid
        for k, fut in enumerate(as_completed(futs), 1):
            eid = futs[fut]
            skybot_cache[eid] = fut.result()
            if k % 20 == 0 or k == len(unique_exps):
                print(f"  SkyBoT plate queries: {k}/{len(unique_exps)} done")

    is_ast = np.zeros(len(df), dtype=bool)
    desigs: list[str | None] = [None] * len(df)
    seps = np.full(len(df), np.nan)
    vmags = np.full(len(df), np.nan)

    for idx, r in df.iterrows():
        ast_df = skybot_cache.get(r["exp_a"])
        if ast_df is None or len(ast_df) == 0:
            continue
        sc_c = SkyCoord(float(r["ra"]) * u.deg, float(r["dec"]) * u.deg)
        sc_a = SkyCoord(ast_df["ra"].to_numpy() * u.deg, ast_df["dec"].to_numpy() * u.deg)
        sep_as = sc_c.separation(sc_a).arcsec
        best_i = int(np.argmin(sep_as))
        best_sep = float(sep_as[best_i])
        exptime_hr = max(0.0, float(r["exptime_sec_a"]) / 3600.0)
        mot = float(ast_df.iloc[best_i]["motion_as_hr"])
        thresh = float(np.hypot(15.0, 0.5 * mot * exptime_hr)) + 5.0
        if best_sep <= thresh:
            is_ast[idx] = True
            desigs[idx] = str(ast_df.iloc[best_i]["desig"])
            seps[idx] = best_sep
            vmags[idx] = float(ast_df.iloc[best_i]["vmag"])

    df["is_asteroid"] = is_ast
    df["ast_desig"] = desigs
    df["ast_sep_arcsec"] = seps
    df["ast_vmag"] = vmags
    return df
