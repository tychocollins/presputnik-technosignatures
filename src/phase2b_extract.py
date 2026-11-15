"""Phase 2b Staged Plate Extractor & Step 1 QC-Fairness Gate (Frozen Parameters — Zero Changes).

Stages:
  - Stage A: 276 digitized (DASCH + APPLAUSE) Primary partial-shadow exposures
             (Tier A+B, m_flash_05s >= 9.0, 0.05 < shadow_fraction < 0.95).
             Evaluates Step 1 (QC fairness: shadow vs. sunlit by instrument) locally BEFORE
             computing any in-shadow vs. out-of-shadow candidate statistics.
  - Stage B: +227 Strict Paired Tier A (m_flash_05s >= 8.0, 0.05 < shadow_fraction < 0.95)
             exposures + their closest emulsion-matched rapid-repeat partner exposures,
             PLUS +100 instrument-stratified Matched-LST sunlit midnight control exposures
             (|LST_solar - 0h| <= 2h, shadow_fraction == 0 at all altitudes, 5° <= |b_ecl| <= 18°,
             m_flash_05s >= 9.0).
  - Stage C: +514 Remaining Secondary m_flash_05s >= 8.0 partial-shadow exposures and
             m_flash_05s >= 9.0 full-shadow (shadow_fraction >= 0.95) dose-response exposures.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvo
import requests
from astropy import units as u
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord
from scipy.stats import chi2, fisher_exact

from src.asteroids import query_skybot_plate
from src.calibrate import _TimeoutSession, calibrate_plate
from src.pilot import _extract_plate_sources, extract_stamp_60as
from src.tiers import _angular_sep_deg

# Limit Potsdam (www.plate-archive.org) to 1 concurrent FITS stream so it never throttles or drops
_APPLAUSE_DL_SEM = threading.Semaphore(1)
# Serialize heavy SEP 2D float32 array extraction (and hold through stamp cutout on >600 MB Bruce plates)
# so peak RAM stays <= 10 GB on a 16 GB Render Pro Plus instance
_EXTRACT_MEM_LOCK = threading.Lock()


def _get_out_dir() -> Path:
    env_dir = os.environ.get("OUTPUT_DIR")
    if env_dir:
        p = Path(env_dir)
    elif Path("/data").exists() and os.access("/data", os.W_OK):
        p = Path("/data")
    else:
        p = Path("data/processed")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_emulsion_map() -> dict[str, str]:
    ap = pd.read_parquet("data/raw/applause_plate.parquet")
    dn = pd.read_parquet("data/raw/dasch_narrow_plates.parquet")
    dw = pd.read_parquet("data/raw/dasch_wide_plates.parquet")
    ap_map = dict(zip(ap["plate_id"].astype(str), ap["filter"].fillna("").astype(str).str.strip()))
    dasch = pd.concat([dn[["dasch_id", "plate_class"]], dw[["dasch_id", "plate_class"]]])
    dasch_map = dict(zip("DASCH_" + dasch["dasch_id"].astype(str), dasch["plate_class"].fillna("").astype(str).str.strip()))
    out: dict[str, str] = {}
    for k, v in dasch_map.items():
        out[k] = v
    for k, v in ap_map.items():
        out[k] = v
        out[f"APPLAUSE_{k}"] = v
    return out


def _interleave_applause_dasch(df: pd.DataFrame) -> pd.DataFrame:
    """Interleave APPLAUSE and DASCH rows within a stage so workers mostly pull fast AWS S3 DASCH plates while 1 worker streams APPLAUSE."""
    app = df[df["source_archive"] == "APPLAUSE"].reset_index(drop=True)
    das = df[df["source_archive"] != "APPLAUSE"].reset_index(drop=True)
    if len(app) == 0 or len(das) == 0:
        return df.reset_index(drop=True)
    stride = max(1, int(round(len(das) / max(len(app), 1))))
    out_rows = []
    ia, id_ = 0, 0
    while ia < len(app) or id_ < len(das):
        if ia < len(app):
            out_rows.append(app.iloc[ia])
            ia += 1
        for _ in range(stride):
            if id_ < len(das):
                out_rows.append(das.iloc[id_])
                id_ += 1
    return pd.DataFrame(out_rows).reset_index(drop=True)


def build_phase2b_manifest(seed: int = 20261115, force_rebuild: bool = False) -> pd.DataFrame:
    """Deterministic self-contained manifest of Stage A, Stage B, and Stage C exposures (DASCH + APPLAUSE)."""
    manifest_path = Path("data/processed/phase2b_manifest.parquet")
    if manifest_path.exists() and not force_rebuild:
        return pd.read_parquet(manifest_path)

    exp = pd.read_parquet("data/processed/exposures.parquet")
    emul_map = _load_emulsion_map()
    exp["emul"] = exp["plate_id"].astype(str).map(lambda s: emul_map.get(str(s), ""))

    # Embed APPLAUSE scan URLs and scan counts directly so Render needs zero raw parquet files
    scn = pd.read_parquet("data/raw/applause_scan.parquet")
    scn["key"] = "APPLAUSE_" + scn["plate_id"].astype(str)
    scan_counts = scn.groupby("key").size().to_dict()
    scn_first = scn[scn["filename_scan"].notna()].drop_duplicates(subset=["key"], keep="first")
    app_url_map = {
        str(r["key"]): (
            str(r["filename_scan"])
            if str(r["filename_scan"]).startswith("http")
            else f"https://www.plate-archive.org/files/{r['filename_scan']}"
        )
        for _, r in scn_first.iterrows()
    }

    dig = exp[exp["source_archive"].isin(["DASCH", "APPLAUSE"]) & (exp["is_tier_a"] | exp["is_tier_b"])].copy()
    dig = dig.reset_index(drop=True)

    pilot_res = pd.read_csv("data/processed/pilot_results.csv")
    p8 = pd.read_parquet("data/processed/sunlit_m8_plates.parquet")
    already_sunlit = set(pilot_res["exp1"]).union(set(pilot_res["exp2"])).union(set(p8["exposure_id"]))

    # Stage A: Primary partial-shadow m_flash >= 9.0 (0.05 < sf < 0.95)
    stage_a_mask = (
        (dig["m_flash_05s"] >= 9.0)
        & (dig["shadow_fraction"] > 0.05)
        & (dig["shadow_fraction"] < 0.95)
    )
    stage_a_ids = dig.loc[stage_a_mask, "exposure_id"].astype(str).tolist()
    seen = set(stage_a_ids)

    # Strict Paired Tier A (m_flash >= 8.0, 0.05 < sf < 0.95) + closest emulsion-matched rapid-repeat partner
    m8 = dig[dig["m_flash_05s"] >= 8.0].reset_index(drop=True)
    part_m8_set = set(
        m8[(m8["shadow_fraction"] > 0.05) & (m8["shadow_fraction"] < 0.95)]["exposure_id"].astype(str)
    )
    pair_map: dict[str, str] = {}
    for (_, _), grp in m8.groupby(["source_archive", "instrument"], sort=False):
        if len(grp) < 2:
            continue
        mjd = grp["mjd_mid"].to_numpy()
        order = np.argsort(mjd)
        mjd_s = mjd[order]
        ra_s, dec_s = grp["ra_deg"].to_numpy()[order], grp["dec_deg"].to_numpy()[order]
        fov_r = 0.25 * (grp["fov_ra_deg"].to_numpy()[order] + grp["fov_dec_deg"].to_numpy()[order])
        pid_s = grp["plate_id"].astype(str).to_numpy()[order]
        eid_s = grp["exposure_id"].astype(str).to_numpy()[order]
        em_s = grp["emul"].to_numpy()[order]
        for i in range(len(mjd_s)):
            if eid_s[i] not in part_m8_set:
                continue
            lo = int(np.searchsorted(mjd_s, mjd_s[i] - 1.0, side="left"))
            hi = int(np.searchsorted(mjd_s, mjd_s[i] + 1.0, side="right"))
            if hi - lo <= 1:
                continue
            dt = np.abs(mjd_s[lo:hi] - mjd_s[i])
            cm = (dt >= 5.0 / 1440.0) & (dt <= 1.0) & (pid_s[lo:hi] != pid_s[i]) & (em_s[lo:hi] == em_s[i])
            if not np.any(cm):
                continue
            ci = np.arange(lo, hi)[cm]
            dt_c = dt[cm]
            seps = _angular_sep_deg(ra_s[i], dec_s[i], ra_s[ci], dec_s[ci])
            ok = seps <= 0.5 * np.minimum(fov_r[i], fov_r[ci])
            if np.any(ok):
                best_j = ci[ok][int(np.argmin(dt_c[ok]))]
                pair_map[str(eid_s[i])] = str(eid_s[best_j])

    pair_df = pd.DataFrame([{"exp_target": k, "exp_partner": v} for k, v in sorted(pair_map.items())])
    pair_df.to_parquet("data/processed/phase2b_paired_tierA_links.parquet", index=False)

    stage_b_tA_ids: list[str] = []
    for k, v in sorted(pair_map.items()):
        for eid in (k, v):
            if eid not in seen and eid not in already_sunlit:
                stage_b_tA_ids.append(eid)
                seen.add(eid)

    # Matched-LST sunlit midnight control: |LST_solar - 0h| <= 2h, sf == 0 at all 3 altitudes, 5° <= |b_ecl| <= 18°, m_flash >= 9.0
    sc_ecl = SkyCoord(dig["ra_deg"].to_numpy() * u.deg, dig["dec_deg"].to_numpy() * u.deg).transform_to(BarycentricTrueEcliptic())
    b_ecl = np.abs(sc_ecl.lat.deg)
    utc_hr = (dig["mjd_mid"].to_numpy() % 1.0) * 24.0
    loc_sol_hr = (utc_hr + dig["obs_lon_deg"].to_numpy() / 15.0) % 24.0
    dist_mid = np.minimum(loc_sol_hr, 24.0 - loc_sol_hr)
    zero_sf = (
        (dig["shadow_fraction"] == 0.0)
        & (dig["shadow_fraction_36000km"] == 0.0)
        & (dig["shadow_fraction_20000km"] == 0.0)
    )
    lst_pool = dig[
        zero_sf
        & (dig["m_flash_05s"] >= 9.0)
        & (dist_mid <= 2.0)
        & (b_ecl >= 5.0)
        & (b_ecl <= 18.0)
        & (~dig["exposure_id"].isin(already_sunlit | seen))
    ].copy()

    stage_a_inst_counts = dig.loc[stage_a_mask, "instrument"].value_counts()
    stage_a_total = max(int(stage_a_inst_counts.sum()), 1)
    n_lst_target = 100
    lst_parts = []
    for inst, cnt in stage_a_inst_counts.items():
        sub_i = lst_pool[lst_pool["instrument"] == inst]
        if len(sub_i) == 0:
            continue
        n_take = min(len(sub_i), max(2, int(round(n_lst_target * (cnt / stage_a_total)))))
        lst_parts.append(sub_i.sample(n=n_take, random_state=seed))
    lst_picked = pd.concat(lst_parts, ignore_index=True) if lst_parts else pd.DataFrame()
    if len(lst_picked) < n_lst_target:
        rem = lst_pool[~lst_pool["exposure_id"].isin(set(lst_picked.get("exposure_id", [])))].copy()
        if len(rem) > 0:
            extra = rem.sample(n=min(len(rem), n_lst_target - len(lst_picked)), random_state=seed)
            lst_picked = pd.concat([lst_picked, extra], ignore_index=True)
    lst_picked = lst_picked.head(n_lst_target)
    stage_b_lst_ids = lst_picked["exposure_id"].astype(str).tolist()
    seen.update(stage_b_lst_ids)

    # Stage C: Remaining Secondary m_flash >= 8.0 partial-shadow + Full-shadow m_flash >= 9.0 (sf >= 0.95)
    stage_c_mask = (
        (
            (dig["m_flash_05s"] >= 8.0)
            & (dig["shadow_fraction"] > 0.05)
            & (dig["shadow_fraction"] < 0.95)
        )
        | (
            (dig["m_flash_05s"] >= 9.0)
            & (dig["shadow_fraction"] >= 0.95)
        )
    ) & (~dig["exposure_id"].isin(seen | already_sunlit))
    stage_c_ids = dig.loc[stage_c_mask, "exposure_id"].astype(str).tolist()

    dig_idx = dig.set_index("exposure_id")
    stage_frames = []
    for stg, role_pairs in [
        ("A", [("primary_m9_partial", stage_a_ids)]),
        ("B", [("paired_tierA_or_partner", stage_b_tA_ids), ("matched_lst_sunlit_ctrl", stage_b_lst_ids)]),
        ("C", [("secondary_m8_or_fullshadow", stage_c_ids)]),
    ]:
        s_rows = []
        for role, eids in role_pairs:
            for eid in eids:
                r = dig_idx.loc[eid].to_dict()
                pid = str(r["plate_id"])
                r["exposure_id"] = eid
                r["stage"] = stg
                r["phase2b_role"] = role
                r["applause_scan_url"] = app_url_map.get(pid, "")
                r["n_scans"] = int(scan_counts.get(pid, 1))
                s_rows.append(r)
        stage_frames.append(_interleave_applause_dasch(pd.DataFrame(s_rows)))

    manifest = pd.concat(stage_frames, ignore_index=True)
    manifest.to_parquet(manifest_path, index=False)
    print(
        f"Built Phase 2b manifest ({len(manifest)} exposures): "
        f"Stage A={sum(manifest['stage']=='A')}, "
        f"Stage B={sum(manifest['stage']=='B')} (PairedTierA+partners={len(stage_b_tA_ids)}, MatchedLST={len(stage_b_lst_ids)}), "
        f"Stage C={sum(manifest['stage']=='C')}",
        flush=True,
    )
    return manifest


def _fetch_applause_wcs_headers(plate_ids: list[str]) -> dict[str, str]:
    """Query applause_dr4.solution via TAP for header_wcs strings (no local raw parquet required)."""
    raw_pids = sorted({int(pid.replace("APPLAUSE_", "")) for pid in plate_ids if pid.startswith("APPLAUSE_")})
    wcs_map: dict[str, str] = {}
    if not raw_pids:
        return wcs_map
    svc = pyvo.dal.TAPService("https://www.plate-archive.org/tap", session=_TimeoutSession())
    for start in range(0, len(raw_pids), 80):
        chunk = ",".join(str(x) for x in raw_pids[start : start + 80])
        q = (
            f"SELECT plate_id, header_wcs FROM applause_dr4.solution "
            f"WHERE plate_id IN ({chunk}) AND solution_num = 1"
        )
        for attempt in range(4):
            try:
                res = svc.search(q).to_table().to_pandas()
                for _, r in res.iterrows():
                    if pd.notna(r["header_wcs"]):
                        wcs_map[f"APPLAUSE_{int(r['plate_id'])}"] = str(r["header_wcs"])
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"  TAP header_wcs retry {attempt + 1}/3 after {type(exc).__name__}: {exc}", flush=True)
                time.sleep(3.0 * (attempt + 1))
    return wcs_map


def _download_fits_phase2b(pid: str, sa: str, applause_url: str, out_dir: Path | None = None) -> Path:
    """Download FITS to disk temp dir (avoiding Linux /tmp tmpfs RAM); serialize APPLAUSE streams."""
    if sa == "DASCH":
        url = f"http://dc.g-vo.org/dasch/q/getprod/qp/fits-01/{pid.replace('DASCH_', '')}"
        max_attempts = 3
        max_wall_s = 600.0
    else:
        url = applause_url
        max_attempts = 5
        max_wall_s = 1200.0

    tmp_dir = (out_dir / "tmp_fits") if out_dir is not None else None
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def _do_stream() -> Path:
        for attempt in range(max_attempts):
            t0 = time.monotonic()
            tmp_f = tempfile.NamedTemporaryFile(
                suffix=".fits",
                dir=(str(tmp_dir) if tmp_dir is not None else None),
                delete=False,
            )
            tmp_path = Path(tmp_f.name)
            try:
                with requests.get(url, timeout=(30, 180), stream=True) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if time.monotonic() - t0 > max_wall_s:
                            raise TimeoutError(f"Timeout >{max_wall_s}s for {pid}")
                        if chunk:
                            tmp_f.write(chunk)
                tmp_f.close()
                return tmp_path
            except Exception as exc:
                tmp_f.close()
                tmp_path.unlink(missing_ok=True)
                if attempt == max_attempts - 1:
                    raise
                print(f"  Retry {attempt + 1}/{max_attempts - 1} for {pid} after {type(exc).__name__}: {exc}", flush=True)
                time.sleep(4.0 * (attempt + 1))
        raise RuntimeError("unreachable")

    if sa == "APPLAUSE":
        with _APPLAUSE_DL_SEM:
            return _do_stream()
    return _do_stream()


def process_phase2b_plate(
    row: pd.Series,
    applause_wcs: dict[str, str],
    out_dir: Path,
) -> dict[str, Any]:
    """Process one Phase 2b plate with 100% identical frozen extraction, calibration, QC, SkyBoT, and stamp saving."""
    eid = str(row["exposure_id"])
    pid = str(row["plate_id"])
    sa = str(row["source_archive"])
    t0 = time.time()
    print(f"  -> starting {eid} ({pid}, {sa}, stage={row['stage']})...", flush=True)

    fits_path = _download_fits_phase2b(pid, sa, str(row.get("applause_scan_url", "")), out_dir=out_dir)
    sz_mb = fits_path.stat().st_size / 1e6
    print(f"     downloaded {eid} ({sz_mb:.1f} MB in {time.time()-t0:.1f}s), waiting for RAM slot...", flush=True)

    # Hold _EXTRACT_MEM_LOCK across the entire image-in-RAM lifetime so at most ONE plate's pixel array
    # resides in RAM at a time (fits safely inside 8 GB Pro Plus even on 1.13 GB Bruce plates), while the
    # second worker prefetches the next FITS file onto /data/tmp_fits SSD.
    with _EXTRACT_MEM_LOCK:
        try:
            ext = _extract_plate_sources(fits_path, override_wcs_str=applause_wcs.get(pid), return_image=True)
        finally:
            fits_path.unlink(missing_ok=True)
            gc.collect()

        if ext is None:
            gc.collect()
            return {
                "exposure_id": eid,
                "plate_id": pid,
                "source_archive": sa,
                "instrument": str(row["instrument"]),
                "stage": str(row["stage"]),
                "phase2b_role": str(row["phase2b_role"]),
                "status": "extract_failed",
                "qc_pass": False,
                "elapsed_s": round(time.time() - t0, 1),
            }

        df_all, wcs, (ny, nx), med_r, rms_grid, med_rms, img = ext
        print(f"     extracted {eid} shape=({ny},{nx}) n_src={len(df_all)} ({time.time()-t0:.1f}s)", flush=True)
        df_all, qc, local_grid = calibrate_plate(
            df_all,
            wcs,
            nx,
            ny,
            float(row["mjd_mid"]),
            float(row["m_lim"]),
            float(row["exptime_sec"]),
            rms_grid=rms_grid,
            med_rms=med_rms,
        )
        cat_dir = out_dir / "phase2b_catalogs"
        cat_dir.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(cat_dir / f"{eid}.parquet", index=False)

        base_rec: dict[str, Any] = {
            "exposure_id": eid,
            "plate_id": pid,
            "source_archive": sa,
            "instrument": str(row["instrument"]),
            "stage": str(row["stage"]),
            "phase2b_role": str(row["phase2b_role"]),
            "is_tier_a": bool(row["is_tier_a"]),
            "m_flash_05s": float(row["m_flash_05s"]),
            "m_flash_05s_cal": float(qc["m_flash_05s_cal"]),
            "m_lim_cal": float(qc["m_lim_cal"]),
            "phot_band": str(qc.get("phot_band") or ""),
            "sip_order": int(qc.get("sip_order", 0)),
            "bright_match_rate": float(qc["bright_match_rate"]) if np.isfinite(qc["bright_match_rate"]) else np.nan,
            "med_resid_after_as": float(qc["med_resid_after_as"]) if np.isfinite(qc["med_resid_after_as"]) else np.nan,
            "nx": int(nx),
            "ny": int(ny),
            "med_r_psf": float(med_r) if np.isfinite(med_r) and med_r > 0 else 3.0,
            "wcs_header": wcs.to_header_string(),
            "fov_area_deg2": float(row["fov_ra_deg"]) * float(row["fov_dec_deg"]),
            "n_scans": int(row.get("n_scans", 1)),
            "local_mlim_grid_json": json.dumps(np.round(local_grid, 3).tolist()),
            "status": "ok",
            "qc_pass": bool(qc["qc_pass"]),
        }

        if not qc["qc_pass"]:
            del img
            gc.collect()
            base_rec["ast_ra_json"] = "[]"
            base_rec["ast_dec_json"] = "[]"
            base_rec["n_stamps"] = 0
            base_rec["elapsed_s"] = round(time.time() - t0, 1)
            return base_rec

        med_r_val = base_rec["med_r_psf"]
        m_lim_cal = base_rec["m_lim_cal"]
        border_x = max(0.05, 4.0 * med_r_val / max(nx, 1))
        border_y = max(0.05, 4.0 * med_r_val / max(ny, 1))

        mag_all = df_all["mag_est"].to_numpy(dtype=float)
        bright_idx = np.where(np.isfinite(mag_all) & (mag_all < (m_lim_cal - 3.0)))[0]
        if len(bright_idx) > 0:
            sc_bright = SkyCoord(df_all["ra"].to_numpy()[bright_idx] * u.deg, df_all["dec"].to_numpy()[bright_idx] * u.deg)
            r_mask_as = np.clip(20.0 + 8.0 * np.maximum(0.0, (m_lim_cal - 3.0) - mag_all[bright_idx]), 20.0, 60.0)
        else:
            sc_bright = None
            r_mask_as = np.zeros(0)

        morph_ok = (
            df_all["psf_pass"].to_numpy()
            & (~df_all["gaia_hit"].to_numpy())
            & (df_all["ellip"].to_numpy() <= 0.20)
            & (df_all["r_psf"].to_numpy() >= 0.7 * med_r_val)
            & (df_all["r_psf"].to_numpy() <= 1.5 * med_r_val)
            & (df_all["peak_snr"].to_numpy() >= 4.0)
            & (df_all["min3_snr"].to_numpy() >= -3.0)
            & (df_all["n_sat_pix"].to_numpy() == 0)
        )
        sub = df_all[morph_ok].copy().reset_index(drop=True)
        if len(sub) > 0:
            xa, ya = wcs.world_to_pixel(SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg))
            xn, yn = xa / max(nx, 1), ya / max(ny, 1)
            keep = (xn >= border_x) & (xn <= 1.0 - border_x) & (yn >= border_y) & (yn <= 1.0 - border_y)
            if sc_bright is not None and len(sc_bright) > 0:
                sc_k = SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg)
                idx_k, idx_b, sep_kb, _ = sc_bright.search_around_sky(sc_k, 60.0 * u.arcsec)
                for ib, ik, s in zip(idx_b, idx_k, sep_kb.arcsec):
                    if s <= r_mask_as[ib]:
                        keep[ik] = False
            sub = sub[keep].copy().reset_index(drop=True)

        rad_deg = 0.5 * float(np.hypot(row["fov_ra_deg"], row["fov_dec_deg"])) + 0.25
        ast_df = query_skybot_plate(float(row["ra_deg"]), float(row["dec_deg"]), rad_deg, float(row["mjd_mid"]))
        if len(ast_df) > 0:
            ast_ra = [round(float(x), 6) for x in ast_df["ra"].to_numpy(dtype=float)]
            ast_dec = [round(float(x), 6) for x in ast_df["dec"].to_numpy(dtype=float)]
            if len(sub) > 0:
                sc_k = SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg)
                sc_a = SkyCoord(ast_df["ra"].to_numpy() * u.deg, ast_df["dec"].to_numpy() * u.deg)
                _, sep_ast, _ = sc_k.match_to_catalog_sky(sc_a)
                sub = sub[sep_ast.arcsec > 20.0].copy().reset_index(drop=True)
        else:
            ast_ra, ast_dec = [], []

        if len(sub) > 0:
            stamp_dir = out_dir / "phase2b_stamps"
            stamp_dir.mkdir(parents=True, exist_ok=True)
            stamps_snr, stamps_sat, stamps_ps = [], [], []
            for r_c, d_c in zip(sub["ra"].to_numpy(dtype=float), sub["dec"].to_numpy(dtype=float)):
                cs, sm, ps_c, _ = extract_stamp_60as(
                    img["data_sub"], img["rms_img"], img["sat_mask"], wcs, float(r_c), float(d_c)
                )
                stamps_snr.append(cs)
                stamps_sat.append(sm)
                stamps_ps.append(ps_c)
            np.savez_compressed(
                stamp_dir / f"{eid}.npz",
                snr=np.array(stamps_snr, dtype=np.float32),
                sat=np.array(stamps_sat, dtype=bool),
                ps=np.array(stamps_ps, dtype=np.float32),
                ra=sub["ra"].to_numpy(dtype=np.float64),
                dec=sub["dec"].to_numpy(dtype=np.float64),
                wcs_hdr=wcs.to_header_string(),
            )

        del img
        gc.collect()

        base_rec["ast_ra_json"] = json.dumps(ast_ra)
        base_rec["ast_dec_json"] = json.dumps(ast_dec)
        base_rec["n_stamps"] = int(len(sub))
        base_rec["elapsed_s"] = round(time.time() - t0, 1)
        return base_rec


def _package_phase2b_outputs(out_dir: Path, archive_name: str = "phase2b_outputs.tar.gz") -> Path:
    tar_path = out_dir / archive_name
    items = ["phase2b_plates.parquet", "phase2b_catalogs", "phase2b_stamps"]
    with tarfile.open(tar_path, "w:gz") as tf:
        for name in items:
            p = out_dir / name
            if p.exists():
                tf.add(p, arcname=name)
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"\n=== PACKAGED PHASE 2B OUTPUTS: {tar_path} ({size_mb:.1f} MB) ===", flush=True)
    print(f"Download via: scp <render-ssh-host>:{tar_path} ./data/processed/", flush=True)
    return tar_path


def report_step1_qc_fairness(out_dir: Path = Path("data/processed")) -> pd.DataFrame:
    """Step 1 Gate: Compare shadow vs. sunlit QC pass rates stratified by instrument ONLY (zero candidate/shadow inspection)."""
    p2b_path = out_dir / "phase2b_plates.parquet"
    if not p2b_path.exists():
        raise FileNotFoundError(f"Missing {p2b_path}")

    p2b = pd.read_parquet(p2b_path)
    shad_a = p2b[(p2b["stage"] == "A") & (~p2b["status"].astype(str).str.startswith("err:"))].copy()

    qc9 = pd.read_csv("data/processed/pilot_plate_qc.csv")
    p8 = pd.read_parquet("data/processed/sunlit_m8_plates.parquet")
    sun_comb = pd.concat(
        [
            qc9[["plate_id", "instrument", "qc_pass"]],
            p8[~p8["status"].astype(str).str.startswith("err:")][["plate_id", "instrument", "qc_pass"]],
        ],
        ignore_index=True,
    ).drop_duplicates("plate_id")

    insts = sorted(set(shad_a["instrument"]).union(set(sun_comb["instrument"])))
    rows = []
    cmh_num, cmh_var, mh_num, mh_den = 0.0, 0.0, 0.0, 0.0

    for inst in insts:
        sh = shad_a[shad_a["instrument"] == inst]
        su = sun_comb[sun_comb["instrument"] == inst]
        if len(sh) == 0:
            continue
        a = int(sh["qc_pass"].sum())
        b = int(len(sh) - a)
        c = int(su["qc_pass"].sum())
        d = int(len(su) - c)
        n_tot = a + b + c + d
        if len(su) > 0:
            _, p_fish = fisher_exact([[a, b], [c, d]])
            n1, n2, m1, m2 = a + b, c + d, a + c, b + d
            if n_tot > 1 and m1 > 0 and m2 > 0:
                cmh_num += a - (n1 * m1) / n_tot
                cmh_var += (n1 * n2 * m1 * m2) / ((n_tot ** 2) * (n_tot - 1))
                mh_num += (a * d) / n_tot
                mh_den += (b * c) / n_tot
        else:
            p_fish = np.nan
        rows.append({
            "instrument": inst,
            "shadow_pass": a,
            "shadow_total": len(sh),
            "shadow_rate": round(a / len(sh), 4),
            "sunlit_pass": c,
            "sunlit_total": len(su),
            "sunlit_rate": round(c / len(su), 4) if len(su) > 0 else np.nan,
            "diff_pp": round(100.0 * (a / len(sh) - c / len(su)), 2) if len(su) > 0 else np.nan,
            "fisher_p": round(float(p_fish), 4) if np.isfinite(p_fish) else np.nan,
        })

    df_rep = pd.DataFrame(rows).sort_values("shadow_total", ascending=False).reset_index(drop=True)
    cmh_chi2 = (abs(cmh_num) - 0.5) ** 2 / cmh_var if cmh_var > 0 else 0.0
    cmh_p = float(1.0 - chi2.cdf(cmh_chi2, df=1)) if cmh_var > 0 else np.nan
    or_mh = mh_num / mh_den if mh_den > 0 else np.nan

    tot_sh_pass = int(shad_a["qc_pass"].sum())
    tot_sh = int(len(shad_a))
    exp_pass = float(
        sum(
            r["shadow_total"] * r["sunlit_rate"]
            for _, r in df_rep.iterrows()
            if np.isfinite(r["sunlit_rate"])
        )
    )

    print("\n================ STEP 1: QC FAIRNESS REPORT (STAGE A PRIMARY SHADOW vs. SUNLIT) ================")
    print(df_rep.to_string(index=False))
    print("------------------------------------------------------------------------------------------------")
    print(
        f"Total Stage A Shadow QC Pass: {tot_sh_pass} / {tot_sh} ({100.0*tot_sh_pass/max(tot_sh,1):.1f}%) "
        f"| Instrument-Expected from Sunlit Rates: {exp_pass:.1f} ({100.0*exp_pass/max(tot_sh,1):.1f}%)"
    )
    print(f"Stratified Cochran-Mantel-Haenszel OR_MH = {or_mh:.3f}, chi2(1) = {cmh_chi2:.3f}, two-sided p = {cmh_p:.4f}")
    print("================================================================================================\n")
    df_rep.to_csv(out_dir / "phase2b_step1_qc_fairness.csv", index=False)
    return df_rep


def _execute_batch(
    todo: pd.DataFrame,
    applause_wcs: dict[str, str],
    out_dir: Path,
    out_plates_p: Path,
    plate_recs: list[dict[str, Any]],
    max_workers: int,
    label: str,
) -> None:
    if len(todo) == 0:
        return
    flush_lock = threading.Lock()

    def _flush() -> None:
        with flush_lock:
            pd.DataFrame(plate_recs).to_parquet(out_plates_p, index=False)

    if max_workers <= 1:
        for k, (_, r) in enumerate(todo.iterrows(), 1):
            eid = str(r["exposure_id"])
            try:
                rec = process_phase2b_plate(r, applause_wcs, out_dir)
            except Exception as exc:
                print(f"  !! ERROR on {eid}: {type(exc).__name__}: {exc}", flush=True)
                rec = {
                    "exposure_id": eid,
                    "plate_id": str(r["plate_id"]),
                    "source_archive": str(r["source_archive"]),
                    "instrument": str(r["instrument"]),
                    "stage": str(r["stage"]),
                    "phase2b_role": str(r["phase2b_role"]),
                    "status": f"err:{type(exc).__name__}",
                    "qc_pass": False,
                }
            plate_recs.append(rec)
            gc.collect()
            _flush()
            print(
                f"  [{label} {k}/{len(todo)}] done={eid} status={rec.get('status')} qc={rec.get('qc_pass')} "
                f"elapsed={rec.get('elapsed_s', '?')}s",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(process_phase2b_plate, r, applause_wcs, out_dir): r for _, r in todo.iterrows()}
            for k, fut in enumerate(as_completed(futs), 1):
                r = futs[fut]
                eid = str(r["exposure_id"])
                try:
                    rec = fut.result()
                except Exception as exc:
                    print(f"  !! ERROR on {eid}: {type(exc).__name__}: {exc}", flush=True)
                    rec = {
                        "exposure_id": eid,
                        "plate_id": str(r["plate_id"]),
                        "source_archive": str(r["source_archive"]),
                        "instrument": str(r["instrument"]),
                        "stage": str(r["stage"]),
                        "phase2b_role": str(r["phase2b_role"]),
                        "status": f"err:{type(exc).__name__}",
                        "qc_pass": False,
                    }
                with flush_lock:
                    plate_recs.append(rec)
                gc.collect()
                _flush()
                print(
                    f"  [{label} {k}/{len(todo)}] done={eid} status={rec.get('status')} qc={rec.get('qc_pass')} "
                    f"elapsed={rec.get('elapsed_s', '?')}s",
                    flush=True,
                )


def run_stage(stages: list[str], max_workers: int = 2, out_dir: Path | None = None) -> None:
    out_dir = out_dir or _get_out_dir()
    is_cloud = bool(os.environ.get("RENDER")) or str(out_dir) == "/data"
    if is_cloud:
        max_workers = min(max_workers, 2)
    # Clean up any leftover temp FITS files from prior container restarts
    tmp_fits_dir = out_dir / "tmp_fits"
    if tmp_fits_dir.exists():
        for fp in tmp_fits_dir.glob("*.fits"):
            fp.unlink(missing_ok=True)
    manifest = build_phase2b_manifest()
    target = manifest[manifest["stage"].isin(stages)].reset_index(drop=True)

    out_plates_p = out_dir / "phase2b_plates.parquet"
    if out_plates_p.exists():
        done_df = pd.read_parquet(out_plates_p)
        ok_mask = ~done_df["status"].astype(str).str.startswith("err:")
        done_df = done_df[ok_mask].reset_index(drop=True)
        done_ids = set(done_df["exposure_id"].astype(str))
        plate_recs = done_df.to_dict("records")
    else:
        done_ids = set()
        plate_recs = []

    todo_all = target[~target["exposure_id"].astype(str).isin(done_ids)].reset_index(drop=True)
    print(
        f"Phase 2b stages={stages} (out_dir={out_dir}): {len(target)} target exposures, "
        f"{len(target) - len(todo_all)} already done, {len(todo_all)} remaining (max_workers={max_workers}).",
        flush=True,
    )

    if len(todo_all) > 0:
        app_pids = todo_all.loc[todo_all["source_archive"] == "APPLAUSE", "plate_id"].astype(str).tolist()
        applause_wcs = _fetch_applause_wcs_headers(app_pids) if app_pids else {}

        # Execute stage-by-stage in order A -> B -> C so Core (A+B) finishes and packages early before C starts
        for stg in ["A", "B", "C"]:
            if stg not in stages:
                continue
            sub_todo = todo_all[todo_all["stage"] == stg].reset_index(drop=True)
            if len(sub_todo) > 0:
                print(f"\n--- Starting Stage {stg} ({len(sub_todo)} exposures) ---", flush=True)
                _execute_batch(sub_todo, applause_wcs, out_dir, out_plates_p, plate_recs, max_workers, f"Stage-{stg}")
            if stg == "B" and "C" in stages:
                _package_phase2b_outputs(out_dir, "phase2b_core_AB_outputs.tar.gz")

    _package_phase2b_outputs(out_dir, "phase2b_outputs.tar.gz")

    is_cloud = bool(os.environ.get("RENDER")) or str(out_dir) == "/data"
    if not is_cloud and Path("data/processed/pilot_plate_qc.csv").exists() and "A" in stages:
        report_step1_qc_fairness(out_dir)

    if is_cloud:
        print("Render Phase 2b worker complete; idling so persistent disk /data stays accessible for scp.", flush=True)
        while True:
            time.sleep(3600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MAX_WORKERS", "2")))
    parser.add_argument("--step1-only", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    args = parser.parse_args()
    if args.rebuild_manifest:
        build_phase2b_manifest(force_rebuild=True)
    elif args.step1_only:
        report_step1_qc_fairness()
    else:
        run_stage(args.stages, max_workers=args.workers)


if __name__ == "__main__":
    main()
