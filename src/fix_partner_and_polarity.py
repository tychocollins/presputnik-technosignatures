"""Diagnose & fix Bug 1 (partner-presence pixel aperture check) and Bug 2 (polarity & same-sign peak check) on the 150 audit candidates."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import sep
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from pathlib import Path
from src.audit_cutouts import _download_plate, _load_image_and_polarity
from src.pilot import _get_applause_scan_and_wcs_map, extract_stamp_60as


def _check_polarity_robust(img_raw: np.ndarray) -> tuple[np.ndarray, bool, float, float]:
    """
    Determine plate polarity robustly: stars should be POSITIVE after normalization.
    Checks skewness of (p99.5 - p50) vs (p50 - p0.5) on finite non-zero pixels.
    """
    valid = img_raw[np.isfinite(img_raw) & (img_raw != 0)]
    if len(valid) < 100:
        return img_raw, False, 0.0, 0.0
    p05, p50, p995 = np.percentile(valid, [0.5, 50.0, 99.5])
    neg_tail = float(p50 - p05)
    pos_tail = float(p995 - p50)
    inverted = neg_tail > 1.3 * pos_tail
    return (-img_raw if inverted else img_raw), inverted, pos_tail, neg_tail


def run_bugfix_audit() -> None:
    aud = pd.read_csv("data/processed/pilot_audit_150_classified.csv")
    fp = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    key = pd.read_csv("data/processed/blind_human_audit_50_key.csv").set_index("blind_id")

    # 1. Check catalog-level all-source partner match (<= 15") on all 150
    cat_cache = {}
    sep_all_b = np.full(len(aud), np.nan)
    for i, r in aud.iterrows():
        eb = r["exp_b"]
        if eb not in cat_cache:
            cat_cache[eb] = pd.read_parquet(f"data/processed/pilot_catalogs/{eb}.parquet")
        cb = cat_cache[eb]
        sc_r = SkyCoord(r["ra"] * u.deg, r["dec"] * u.deg)
        sc_b = SkyCoord(cb["ra"].to_numpy() * u.deg, cb["dec"].to_numpy() * u.deg)
        sep_all_b[i] = float(np.min(sc_r.separation(sc_b).arcsec))

    aud["sep_b_all"] = sep_all_b
    aud["pass_partner_cat"] = sep_all_b > 15.0

    # 2. Download unique plates to run pixel-level partner aperture check & Plate A same-sign peak check
    unique_exps = sorted(set(aud["exp_a"]).union(set(aud["exp_b"])))
    app_pids = [fp.loc[e, "plate_id"] for e in unique_exps if fp.loc[e, "source_archive"] == "APPLAUSE"]
    app_urls, _ = _get_applause_scan_and_wcs_map(app_pids)

    # Map needed (idx, role) per plate
    needed: dict[str, list[tuple[int, str]]] = {e: [] for e in unique_exps}
    for i, r in aud.iterrows():
        needed[r["exp_a"]].append((i, "A"))
        needed[r["exp_b"]].append((i, "B"))

    n_inverted_plates = 0
    a_peak_snr = np.full(len(aud), np.nan)
    a_min3_snr = np.full(len(aud), np.nan)
    b_ap_snr = np.full(len(aud), np.nan)
    b_abs_peak_snr = np.full(len(aud), np.nan)

    print(f"Checking pixel-level polarity & partner apertures across {len(unique_exps)} plates...", flush=True)

    def _dl_safe(pid: str, sa: str) -> bytes:
        url = f"http://dc.g-vo.org/dasch/q/getprod/qp/fits-01/{pid.replace('DASCH_', '')}" if sa == "DASCH" else app_urls[pid]
        for attempt in range(2):
            t0 = time.monotonic()
            try:
                with requests.get(url, timeout=(20, 60), stream=True) as r:
                    r.raise_for_status()
                    chunks = []
                    for c in r.iter_content(chunk_size=1024 * 1024):
                        if time.monotonic() - t0 > 1800.0:
                            raise TimeoutError(f"Timeout >1800s for {pid}")
                        if c:
                            chunks.append(c)
                    return b"".join(chunks)
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(2.0)
        raise RuntimeError("unreachable")

    def _check_one(eid: str) -> tuple[bool, list[tuple[int, str, float, float]]]:
        fr = fp.loc[eid]
        b = _dl_safe(str(fr["plate_id"]), str(fr["source_archive"]))
        img_raw = _load_image_and_polarity(b)
        del b
        if img_raw is None:
            return False, []
        img, still_inv, _, _ = _check_polarity_robust(img_raw)
        ny, nx = img.shape
        mask = (~np.isfinite(img)) | (img == 0)
        valid_v = img[~mask]
        sat_mask = mask.copy()
        if len(valid_v) > 100:
            v_max, v_min = float(np.max(valid_v)), float(np.min(valid_v))
            if np.sum(valid_v >= v_max - 1e-6) > 10:
                sat_mask |= img >= v_max - 1e-6
            if np.sum(valid_v <= v_min + 1e-6) > 10:
                sat_mask |= img <= v_min + 1e-6
        bkg = sep.Background(np.where(mask, np.nanmedian(img[~mask]) if np.any(~mask) else 0.0, img), mask=mask, bw=128, bh=128, fw=3, fh=3)
        sub_img = img - bkg.back()
        rms_img = np.maximum(bkg.rms(), 1e-6)
        sat_mask |= sub_img < -5.0 * rms_img
        wcs = WCS(fits.Header.fromstring(fr["wcs_header"]))
        med_r = float(fr["med_r_psf"]) if np.isfinite(fr["med_r_psf"]) and fr["med_r_psf"] > 0 else 3.0
        r_ap = max(3, int(round(2.0 * med_r)))

        stamp_dir = Path("data/processed/audit_stamps_150")
        stamp_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for idx, role in needed[eid]:
            r = aud.iloc[idx]
            cs, sm, ps_c, hb_c = extract_stamp_60as(sub_img, rms_img, sat_mask, wcs, float(r["ra"]), float(r["dec"]))
            np.savez_compressed(
                stamp_dir / f"{idx}_{role}.npz",
                snr=cs, sat=sm, ps=np.float32(ps_c), hb=np.bool_(hb_c),
                ra=np.float64(r["ra"]), dec=np.float64(r["dec"]),
            )
            xc, yc = wcs.world_to_pixel(SkyCoord(r["ra"] * u.deg, r["dec"] * u.deg))
            ix, iy = int(round(float(xc))), int(round(float(yc)))
            x0, x1 = max(0, ix - r_ap), min(nx, ix + r_ap + 1)
            y0, y1 = max(0, iy - r_ap), min(ny, iy + r_ap + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            patch = sub_img[y0:y1, x0:x1]
            rms_loc = float(np.median(rms_img[y0:y1, x0:x1]))
            if role == "A":
                p3 = sub_img[max(0, iy - 1):min(ny, iy + 2), max(0, ix - 1):min(nx, ix + 2)]
                n_sat = int(np.sum(sat_mask[max(0, iy - 2):min(ny, iy + 3), max(0, ix - 2):min(nx, ix + 3)]))
                out.append((idx, "A", float(np.max(p3)) / rms_loc, float(np.min(p3)) / rms_loc if n_sat == 0 else -999.0))
            else:
                n_pix = patch.size
                ap_sum = float(np.sum(patch))
                out.append((idx, "B", abs(ap_sum) / (rms_loc * np.sqrt(max(1, n_pix))), float(np.max(np.abs(patch))) / rms_loc))
        del img, sub_img
        return still_inv, out

    dasch_exps = [e for e in unique_exps if fp.loc[e, "source_archive"] == "DASCH"]
    app_exps = [e for e in unique_exps if fp.loc[e, "source_archive"] != "DASCH"]
    done_n = 0

    def _run_pool(exps_sub: list[str], workers: int) -> None:
        nonlocal done_n, n_inverted_plates
        if not exps_sub:
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_check_one, eid): eid for eid in exps_sub}
            for fut in as_completed(futs):
                done_n += 1
                eid = futs[fut]
                try:
                    still_inv, out = fut.result()
                    if still_inv:
                        n_inverted_plates += 1
                    for idx, role, v1, v2 in out:
                        if role == "A":
                            a_peak_snr[idx] = v1
                            a_min3_snr[idx] = v2
                        else:
                            b_ap_snr[idx] = v1
                            b_abs_peak_snr[idx] = v2
                    print(f"  [{done_n}/{len(unique_exps)}] {eid} ok (still_inv={still_inv})", flush=True)
                except Exception as exc:
                    print(f"  [{done_n}/{len(unique_exps)}] WARN {eid} failed: {exc}", flush=True)

    _run_pool(dasch_exps, workers=4)
    _run_pool(app_exps, workers=1)

    aud["a_peak_snr"] = a_peak_snr
    aud["a_min3_snr"] = a_min3_snr
    aud["b_ap_snr"] = b_ap_snr
    aud["b_abs_peak_snr"] = b_abs_peak_snr

    # Frozen fixes:
    # Bug 1 fix: partner cat sep > 15" AND pixel-level partner aperture SNR <= 3.0 AND max|pix| <= 4.0
    aud["pass_bug1_fix"] = aud["pass_partner_cat"] & (aud["b_ap_snr"] <= 3.0) & (aud["b_abs_peak_snr"] <= 4.0)
    # Bug 2 fix: Plate A candidate peak >= 4.0*rms AND no negative/white core (< -3.0*rms)
    aud["pass_bug2_fix"] = (aud["a_peak_snr"] >= 4.0) & (aud["a_min3_snr"] >= -3.0)

    # Full frozen filter v1.1 (morph + bright_nbr + bug1 + bug2)
    aud["pass_frozen_v11"] = (
        (aud["ellip"] <= 0.20)
        & (aud["min_bright_nbr_as"] > 25.0)
        & aud["pass_bug1_fix"]
        & aud["pass_bug2_fix"]
    )
    # Also check r_psf <= 1.5*med_r
    r_ok = np.array([
        0.7 * float(fp.loc[r["exp_a"], "med_r_psf"]) <= r["r_psf"] <= 1.5 * float(fp.loc[r["exp_a"], "med_r_psf"])
        for _, r in aud.iterrows()
    ])
    aud["pass_frozen_v11"] = aud["pass_frozen_v11"] & r_ok

    aud.to_csv("data/processed/pilot_audit_150_bugfixed.csv", index=False)

    print(f"\nPlates with additional polarity inversion caught: {n_inverted_plates} / {len(unique_exps)}")
    print(f"Of 150 audited candidates:")
    print(f"  Removed by Bug 1 (partner all-source <=15\"): {(~aud['pass_partner_cat']).sum()}")
    print(f"  Removed by Bug 1 (partner pixel aperture >3sig or |peak|>4sig): {(aud['pass_partner_cat'] & ~aud['pass_bug1_fix']).sum()}")
    print(f"  Removed by Bug 2 (white/negative core or weak peak): {(~aud['pass_bug2_fix']).sum()}")
    print(f"  Surviving ALL Frozen v1.1 filters: {aud['pass_frozen_v11'].sum()} / 150")
    surv = aud[aud["pass_frozen_v11"]]
    print("\nSurviving breakdown:")
    print(surv["label"].value_counts().to_frame("count").assign(pct=lambda x: (100 * x["count"] / len(surv)).round(1)))

    print("\n--- Diagnostics on the 10 flagged blind-sheet cutouts ---")
    for bid in [1, 12, 13, 14, 21, 28, 29, 35, 37, 42]:
        idx = int(key.loc[bid, "orig_cand_idx"])
        r = aud.iloc[idx]
        print(f"#{bid:2d}: sep_b={r['sep_b_all']:6.2f}\" b_ap={r['b_ap_snr']:5.1f} b_pk={r['b_abs_peak_snr']:5.1f} a_pk={r['a_peak_snr']:5.1f} a_min={r['a_min3_snr']:5.1f} -> kept={bool(r['pass_frozen_v11'])}")


if __name__ == "__main__":
    run_bugfix_audit()
