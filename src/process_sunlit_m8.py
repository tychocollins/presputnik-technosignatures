"""Step 2: Process stratified m_flash >= 8.0 sunlit comparison set (Tier A + matched Tier B) with frozen masks & 30\" de-clustering."""

from __future__ import annotations

import gc
import json
import os
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from src.asteroids import query_skybot_plate
from src.calibrate import calibrate_plate
from src.declump_and_mask import _declump_events
from src.pilot import _extract_plate_sources, _get_applause_scan_and_wcs_map, extract_stamp_60as


def build_sunlit_m8_sample(n_target: int = 400, seed: int = 101) -> pd.DataFrame:
    """Draw a strictly out-of-shadow stratified sample of m_flash >= 8.0 plates matching shadow instruments."""
    exp = pd.read_parquet("data/processed/exposures.parquet")
    pilot_res = pd.read_csv("data/processed/pilot_results.csv")
    already_done = set(pilot_res["exp1"]).union(set(pilot_res["exp2"]))

    zero_mask = (
        (exp["shadow_fraction"] == 0.0)
        & (exp["shadow_fraction_36000km"] == 0.0)
        & (exp["shadow_fraction_20000km"] == 0.0)
    )
    m8_sun = exp[zero_mask & (exp["is_tier_a"] | exp["is_tier_b"]) & (exp["m_flash_05s"] >= 8.0)].copy()
    m8_sun = m8_sun[~m8_sun["exposure_id"].isin(already_done)].copy()

    # Target instruments that pass QC and appear in m>=8 partial-shadow sample
    targets = {
        "10-inch Metcalf Triplet": 110,
        "24-inch Bruce Doublet": 60,
        "8-inch Bache Doublet": 50,
        "16-inch Metcalf Doublet (Refigured after 3500)": 90,
        "8-inch Ross Lundin": 40,
        "12-inch Metcalf Doublet": 20,
        "32-36 inch BakerSchmidt 10 1/2 inch round Armagh-Dunsink-Harvard": 15,
        "24-33 in Jewett Schmidt": 15,
    }

    parts = []
    for inst, n in targets.items():
        sub = m8_sun[m8_sun["instrument"] == inst]
        if len(sub) == 0:
            continue
        # Stratify half 8.0 <= m_flash < 9.0, half m_flash >= 9.0 where available
        s_lo = sub[(sub["m_flash_05s"] >= 8.0) & (sub["m_flash_05s"] < 9.0)]
        s_hi = sub[sub["m_flash_05s"] >= 9.0]
        n_lo = min(len(s_lo), max(n // 2, n - len(s_hi)))
        n_hi = min(len(s_hi), n - n_lo)
        if n_lo > 0:
            parts.append(s_lo.sample(n=n_lo, random_state=seed))
        if n_hi > 0:
            parts.append(s_hi.sample(n=n_hi, random_state=seed))

    sample = pd.concat(parts, ignore_index=True).head(n_target)
    assert (sample["shadow_fraction"] == 0.0).all(), "BLINDNESS VIOLATION!"
    assert (sample["shadow_fraction_36000km"] == 0.0).all(), "BLINDNESS VIOLATION!"
    assert (sample["shadow_fraction_20000km"] == 0.0).all(), "BLINDNESS VIOLATION!"
    sample.to_parquet("data/processed/sunlit_m8_sample.parquet", index=False)
    return sample


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


def _assert_strictly_sunlit(df: pd.DataFrame) -> None:
    """HARD BLINDNESS GUARD: abort immediately if any shadow_fraction > 0 or missing."""
    for col in ("shadow_fraction", "shadow_fraction_36000km", "shadow_fraction_20000km"):
        if col not in df.columns:
            raise SystemExit(f"HARD BLINDNESS GUARD ABORT: missing column {col!r}")
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any() or (vals > 0.0).any():
            n_bad = int((vals.isna() | (vals > 0.0)).sum())
            raise SystemExit(f"HARD BLINDNESS GUARD ABORT: {n_bad} rows have {col} > 0 or NaN!")


def _download_fits_to_file(pid: str, sa: str, applause_urls: dict[str, str]) -> Path:
    if sa == "DASCH":
        # Fresh GET on GAVO each attempt generates a fresh 900s S3 presigned URL
        url = f"http://dc.g-vo.org/dasch/q/getprod/qp/fits-01/{pid.replace('DASCH_', '')}"
    else:
        url = applause_urls[pid]
    for attempt in range(3):
        t0 = time.monotonic()
        tmp_f = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        tmp_path = Path(tmp_f.name)
        try:
            with requests.get(url, timeout=(30, 120), stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if time.monotonic() - t0 > 600.0:
                        raise TimeoutError(f"Timeout >600s for {pid}")
                    if chunk:
                        tmp_f.write(chunk)
            tmp_f.close()
            return tmp_path
        except Exception as exc:
            tmp_f.close()
            tmp_path.unlink(missing_ok=True)
            if attempt == 2:
                raise
            print(f"  Retry {attempt + 1}/2 for {pid} after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def process_single_sunlit_plate(
    row: pd.Series,
    applause_urls: dict[str, str],
    applause_wcs: dict[str, str],
    out_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for col in ("shadow_fraction", "shadow_fraction_36000km", "shadow_fraction_20000km"):
        if float(row[col]) != 0.0:
            raise SystemExit(f"HARD BLINDNESS GUARD ABORT: plate {row['exposure_id']} has {col}={row[col]}")
    out_dir = out_dir or _get_out_dir()
    eid = str(row["exposure_id"])
    pid = str(row["plate_id"])
    sa = str(row["source_archive"])
    t0 = time.time()
    print(f"  -> starting {eid} ({pid}, {sa})...", flush=True)

    fits_path = _download_fits_to_file(pid, sa, applause_urls)
    print(f"     downloaded {eid} ({fits_path.stat().st_size / 1e6:.1f} MB in {time.time()-t0:.1f}s), extracting...", flush=True)
    try:
        ext = _extract_plate_sources(fits_path, override_wcs_str=applause_wcs.get(pid), return_image=True)
    finally:
        fits_path.unlink(missing_ok=True)
        gc.collect()
    if ext is None:
        gc.collect()
        return {"exposure_id": eid, "plate_id": pid, "instrument": row["instrument"], "status": "extract_failed", "qc_pass": False}, []

    df_all, wcs, (ny, nx), med_r, rms_grid, med_rms, img = ext
    print(f"  extracted {eid} shape=({ny},{nx}) n_src={len(df_all)} ({time.time()-t0:.1f}s)", flush=True)
    df_all, qc, local_grid = calibrate_plate(
        df_all, wcs, nx, ny, float(row["mjd_mid"]), float(row["m_lim"]), float(row["exptime_sec"]),
        rms_grid=rms_grid, med_rms=med_rms,
    )
    cat_dir = out_dir / "sunlit_m8_catalogs"
    cat_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(cat_dir / f"{eid}.parquet", index=False)

    if not qc["qc_pass"]:
        del img
        gc.collect()
        return {
            "exposure_id": eid, "plate_id": pid, "instrument": row["instrument"],
            "is_tier_a": bool(row["is_tier_a"]), "m_flash_05s": float(row["m_flash_05s"]),
            "m_flash_05s_cal": float(qc["m_flash_05s_cal"]), "m_lim_cal": float(qc["m_lim_cal"]),
            "status": "ok", "qc_pass": False, "n_events": 0, "elapsed_s": round(time.time() - t0, 1),
        }, []

    # 32x32 fine grid for masked-area subtraction
    g_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32)
    gx_n, gy_n = np.meshgrid(g_pts, g_pts)
    gx_flat, gy_flat = gx_n.ravel(), gy_n.ravel()

    med_r_val = float(med_r) if np.isfinite(med_r) and med_r > 0 else 3.0
    m_lim_cal = float(qc["m_lim_cal"])
    border_x = max(0.05, 4.0 * med_r_val / max(nx, 1))
    border_y = max(0.05, 4.0 * med_r_val / max(ny, 1))
    grid_valid = (gx_flat >= border_x) & (gx_flat <= 1.0 - border_x) & (gy_flat >= border_y) & (gy_flat <= 1.0 - border_y)

    mag_all = df_all["mag_est"].to_numpy(dtype=float)
    bright_idx = np.where(np.isfinite(mag_all) & (mag_all < (m_lim_cal - 3.0)))[0]
    if len(bright_idx) > 0:
        sc_bright = SkyCoord(df_all["ra"].to_numpy()[bright_idx] * u.deg, df_all["dec"].to_numpy()[bright_idx] * u.deg)
        r_mask_as = np.clip(20.0 + 8.0 * np.maximum(0.0, (m_lim_cal - 3.0) - mag_all[bright_idx]), 20.0, 60.0)
        sc_grid = wcs.pixel_to_world(gx_flat * nx, gy_flat * ny)
        idx_g, idx_b, sep_gb, _ = sc_bright.search_around_sky(sc_grid, 60.0 * u.arcsec)
        for ib, ig, s in zip(idx_b, idx_g, sep_gb.arcsec):
            if s <= r_mask_as[ib]:
                grid_valid[ig] = False
    else:
        sc_bright = None
        r_mask_as = np.zeros(0)

    # Candidate filter (frozen morphology + pixel polarity/saturation + edge + bright-star annulus)
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
        xn, yn = xn[keep], yn[keep]
    else:
        xn, yn = np.zeros(0), np.zeros(0)

    # SkyBoT asteroid removal
    if len(sub) > 0:
        rad_deg = 0.5 * float(np.hypot(row["fov_ra_deg"], row["fov_dec_deg"])) + 0.25
        ast_df = query_skybot_plate(float(row["ra_deg"]), float(row["dec_deg"]), rad_deg, float(row["mjd_mid"]))
        if len(ast_df) > 0:
            sc_k = SkyCoord(sub["ra"].to_numpy() * u.deg, sub["dec"].to_numpy() * u.deg)
            sc_a = SkyCoord(ast_df["ra"].to_numpy() * u.deg, ast_df["dec"].to_numpy() * u.deg)
            _, sep_ast, _ = sc_k.match_to_catalog_sky(sc_a)
            ok_ast = sep_ast.arcsec > 20.0
            sub = sub[ok_ast].copy().reset_index(drop=True)
            xn, yn = xn[ok_ast], yn[ok_ast]

    # Save 60" stamps for all surviving candidates while image is in memory
    if len(sub) > 0:
        stamp_dir = out_dir / "sunlit_m8_stamps"
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

    # 30" de-clustering
    ev_xy = _declump_events(sub["ra"].to_numpy(), sub["dec"].to_numpy(), xn, yn, 30.0) if len(sub) > 0 else np.zeros((0, 2))
    ev_rows = [{"exposure_id": eid, "x_norm": float(x), "y_norm": float(y)} for x, y in ev_xy]

    plate_rec = {
        "exposure_id": eid,
        "plate_id": pid,
        "instrument": str(row["instrument"]),
        "is_tier_a": bool(row["is_tier_a"]),
        "m_flash_05s": float(row["m_flash_05s"]),
        "m_flash_05s_cal": float(qc["m_flash_05s_cal"]),
        "m_lim_cal": m_lim_cal,
        "nx": int(nx),
        "ny": int(ny),
        "med_r_psf": float(med_r_val),
        "wcs_header": wcs.to_header_string(),
        "fov_area_deg2": float(row["fov_ra_deg"]) * float(row["fov_dec_deg"]),
        "unmasked_frac": float(np.mean(grid_valid)),
        "valid_grid_32_json": json.dumps(grid_valid.astype(int).tolist()),
        "local_mlim_grid_json": json.dumps(np.round(local_grid, 3).tolist()),
        "status": "ok",
        "qc_pass": True,
        "n_raw_cands": int(len(sub)),
        "n_events": int(len(ev_xy)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    return plate_rec, ev_rows


def _package_outputs(out_dir: Path) -> Path:
    tar_path = out_dir / "sunlit_m8_outputs.tar.gz"
    items = ["sunlit_m8_plates.parquet", "sunlit_m8_events.parquet", "sunlit_m8_catalogs", "sunlit_m8_stamps"]
    with tarfile.open(tar_path, "w:gz") as tf:
        for name in items:
            p = out_dir / name
            if p.exists():
                tf.add(p, arcname=name)
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"\n=== PACKAGED OUTPUTS: {tar_path} ({size_mb:.1f} MB) ===", flush=True)
    print("To download from Render:", flush=True)
    print(f"  1. Open Render Dashboard -> this worker -> Shell (or SSH via `render ssh`).", flush=True)
    print(f"  2. Run `ls -lh {tar_path}` to verify the archive.", flush=True)
    print(f"  3. Copy locally with `scp <render-ssh-host>:{tar_path} ./data/processed/`", flush=True)
    print("     (or serve temporarily / upload to S3 or Google Drive from the shell).", flush=True)
    print("  4. Suspend or delete the Render Background Worker once downloaded.", flush=True)
    return tar_path


def run_sunlit_m8(n_target: int = 400, max_workers: int = 2) -> None:
    out_dir = _get_out_dir()
    sample_path = Path("data/processed/sunlit_m8_sample.parquet")
    if sample_path.exists():
        sample = pd.read_parquet(sample_path)
    else:
        sample = build_sunlit_m8_sample(n_target=n_target)
    _assert_strictly_sunlit(sample)
    print(f"Hard blindness guard passed: {len(sample)} sunlit plates (max shadow_fraction=0.0). Output dir: {out_dir}", flush=True)

    out_plates_p = out_dir / "sunlit_m8_plates.parquet"
    out_events_p = out_dir / "sunlit_m8_events.parquet"
    if out_plates_p.exists():
        done_plates = pd.read_parquet(out_plates_p)
        # Keep completed ('ok' / 'extract_failed') plates; retry any prior transient 'err:*' rows
        ok_mask = ~done_plates["status"].astype(str).str.startswith("err:")
        done_plates = done_plates[ok_mask].reset_index(drop=True)
        done_ids = set(done_plates["exposure_id"])
        plate_recs = done_plates.to_dict("records")
        ev_df = pd.read_parquet(out_events_p) if out_events_p.exists() else pd.DataFrame()
        ev_recs = ev_df[ev_df["exposure_id"].isin(done_ids)].to_dict("records") if len(ev_df) > 0 else []
    else:
        done_ids = set()
        plate_recs, ev_recs = [], []

    todo = sample[~sample["exposure_id"].isin(done_ids)].reset_index(drop=True)
    _assert_strictly_sunlit(todo)
    print(f"Sunlit m_flash>=8.0 set: {len(sample)} total, {len(done_ids)} already done, {len(todo)} remaining (max_workers={max_workers}).", flush=True)

    if len(todo) > 0:
        app_pids = todo.loc[todo["source_archive"] == "APPLAUSE", "plate_id"].astype(str).tolist()
        applause_urls, applause_wcs = _get_applause_scan_and_wcs_map(app_pids) if app_pids else ({}, {})

        def _flush() -> None:
            pd.DataFrame(plate_recs).to_parquet(out_plates_p, index=False)
            pd.DataFrame(ev_recs if ev_recs else [{"exposure_id": "", "x_norm": 0.0, "y_norm": 0.0}]).iloc[: len(ev_recs)].to_parquet(out_events_p, index=False)

        if max_workers <= 1:
            for k, (_, r) in enumerate(todo.iterrows(), 1):
                eid = str(r["exposure_id"])
                try:
                    p_rec, e_list = process_single_sunlit_plate(r, applause_urls, applause_wcs, out_dir)
                    plate_recs.append(p_rec)
                    ev_recs.extend(e_list)
                except Exception as e:
                    print(f"  !! ERROR on {eid}: {type(e).__name__}: {e}", flush=True)
                    plate_recs.append({"exposure_id": eid, "status": f"err:{type(e).__name__}", "qc_pass": False})
                gc.collect()
                _flush()
                last = plate_recs[-1]
                print(f"  [{k}/{len(todo)}] done={eid} status={last.get('status')} qc={last.get('qc_pass')} events={last.get('n_events', 0)} elapsed={last.get('elapsed_s', '?')}s", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(process_single_sunlit_plate, r, applause_urls, applause_wcs, out_dir): r["exposure_id"] for _, r in todo.iterrows()}
                for k, fut in enumerate(as_completed(futs), 1):
                    eid = futs[fut]
                    try:
                        p_rec, e_list = fut.result()
                        plate_recs.append(p_rec)
                        ev_recs.extend(e_list)
                    except Exception as e:
                        print(f"  !! ERROR on {eid}: {type(e).__name__}: {e}", flush=True)
                        plate_recs.append({"exposure_id": eid, "status": f"err:{type(e).__name__}", "qc_pass": False})
                    gc.collect()
                    _flush()
                    last = plate_recs[-1]
                    print(f"  [{k}/{len(todo)}] done={eid} status={last.get('status')} qc={last.get('qc_pass')} events={last.get('n_events', 0)}", flush=True)

    _package_outputs(out_dir)
    if os.environ.get("RENDER"):
        print("Render worker complete; idling so persistent disk stays accessible for download.", flush=True)
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    workers = int(os.environ.get("MAX_WORKERS", "2"))
    run_sunlit_m8(n_target=400, max_workers=workers)
