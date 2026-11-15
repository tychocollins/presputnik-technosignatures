"""Lightweight cutout-only Bug 1 + Bug 2 verification & blind 50-cutout sheet v2 generator.

Uses StarGlass POST /dasch/dr7/cutout (~1 MB) for DASCH and HTTP Range requests (~2 MB)
for APPLAUSE, avoiding full 1 GB plate downloads. Saves 60" stamps to
data/processed/audit_stamps_v2/ so plates are never re-fetched.
"""

from __future__ import annotations

import base64
import gzip
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sep
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from src.pilot import _get_applause_scan_and_wcs_map, extract_stamp_60as


def _normalize_polarity(img_raw: np.ndarray) -> tuple[np.ndarray, bool]:
    valid = img_raw[np.isfinite(img_raw) & (img_raw != 0)]
    if len(valid) < 100:
        return img_raw, False
    p1, p50, p99 = np.percentile(valid, [1.0, 50.0, 99.0])
    img = -img_raw if (p50 - p1) > 1.5 * (p99 - p50) else img_raw.copy()
    valid2 = img[np.isfinite(img) & (img != 0)]
    p05, p50_2, p995 = np.percentile(valid2, [0.5, 50.0, 99.5])
    still_inv = float(p50_2 - p05) > 1.3 * float(p995 - p50_2)
    if still_inv:
        img = -img
    return img, still_inv


def _fetch_dasch_cutout(pid: str, ra: float, dec: float) -> tuple[np.ndarray, WCS] | None:
    plate_short = pid.replace("DASCH_", "")
    payload = {"plate_id": plate_short, "solution_number": 0, "center_ra_deg": float(ra), "center_dec_deg": float(dec)}
    for attempt in range(3):
        try:
            r = requests.post("https://api.starglass.cfa.harvard.edu/public/dasch/dr7/cutout", json=payload, timeout=90)
            r.raise_for_status()
            b64 = r.json()
            if not isinstance(b64, str) or len(b64) < 100:
                return None
            fits_bytes = gzip.decompress(base64.b64decode(b64))
            with fits.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
                for h in hdul:
                    if h.data is not None and h.data.ndim == 2:
                        data = np.ascontiguousarray(h.data, dtype=np.float32)
                        wcs = WCS(h.header, naxis=2)
                        return data, wcs
            return None
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def _fetch_applause_range_cutout(url: str, wcs: WCS, ra: float, dec: float) -> tuple[np.ndarray, WCS] | None:
    try:
        rh = requests.get(url, headers={"Range": "bytes=0-28799"}, timeout=45)
        if rh.status_code not in (200, 206):
            return None
        raw_hdr = rh.content
        end_idx = raw_hdr.find(b"END     ")
        if end_idx < 0:
            return None
        data_offset = ((end_idx // 2880) + 1) * 2880
        hdr = fits.Header.fromstring(raw_hdr[:data_offset].decode("ascii", errors="ignore"))
        nx = int(hdr["NAXIS1"])
        ny = int(hdr["NAXIS2"])
        bitpix = int(hdr["BITPIX"])
        bzero = float(hdr.get("BZERO", 0.0))
        bscale = float(hdr.get("BSCALE", 1.0))
        dtype_map = {16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8", 8: "u1"}
        if bitpix not in dtype_map:
            return None
        dt = np.dtype(dtype_map[bitpix])
        bpp = dt.itemsize

        scales = wcs.proj_plane_pixel_scales()
        ps = float(np.mean([scales[0].to_value(u.arcsec), scales[1].to_value(u.arcsec)]))
        half_rows = max(45, min(120, int(round(45.0 / max(ps, 0.5)))))

        xc, yc = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        iy = int(round(float(yc)))
        y0 = max(0, iy - half_rows)
        y1 = min(ny, iy + half_rows + 1)
        if y1 <= y0:
            return None

        b0 = data_offset + y0 * nx * bpp
        b1 = data_offset + y1 * nx * bpp - 1
        rd = requests.get(url, headers={"Range": f"bytes={b0}-{b1}"}, timeout=120)
        if rd.status_code not in (200, 206):
            return None
        arr = np.frombuffer(rd.content, dtype=dt).reshape((y1 - y0, nx)).astype(np.float32) * bscale + bzero
        # Shift WCS CRPIX2 by -y0 so coordinates map directly to the strip
        wcs_sub = wcs.deepcopy()
        wcs_sub.wcs.crpix[1] -= y0
        return np.ascontiguousarray(arr), wcs_sub
    except Exception:
        return None


def _measure_stamp(
    img_raw: np.ndarray, wcs: WCS, ra: float, dec: float, med_r: float, role: str
) -> tuple[np.ndarray, np.ndarray, float, bool, float, float, bool]:
    img, still_inv = _normalize_polarity(img_raw)
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

    bw = min(128, max(32, nx // 4))
    bh = min(128, max(32, ny // 4))
    fill_val = float(np.nanmedian(valid_v)) if len(valid_v) > 0 else 0.0
    bkg = sep.Background(np.where(mask, fill_val, img), mask=mask, bw=bw, bh=bh, fw=3, fh=3)
    sub_img = img - bkg.back()
    rms_img = np.maximum(bkg.rms(), 1e-6)
    sat_mask |= sub_img < -5.0 * rms_img

    snr_stamp, sat_stamp, ps, hb = extract_stamp_60as(sub_img, rms_img, sat_mask, wcs, ra, dec, box_arcsec=60.0)
    xc, yc = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    ix, iy = int(round(float(xc))), int(round(float(yc)))
    r_ap = max(3, int(round(2.0 * (med_r if np.isfinite(med_r) and med_r > 0 else 3.0))))
    x0, x1 = max(0, ix - r_ap), min(nx, ix + r_ap + 1)
    y0, y1 = max(0, iy - r_ap), min(ny, iy + r_ap + 1)
    if x1 <= x0 or y1 <= y0:
        return snr_stamp, sat_stamp, ps, hb, np.nan, np.nan, still_inv

    rms_loc = float(np.median(rms_img[y0:y1, x0:x1]))
    if role == "A":
        p3 = sub_img[max(0, iy - 1):min(ny, iy + 2), max(0, ix - 1):min(nx, ix + 2)]
        n_sat = int(np.sum(sat_mask[max(0, iy - 2):min(ny, iy + 3), max(0, ix - 2):min(nx, ix + 3)]))
        v1 = float(np.max(p3)) / rms_loc
        v2 = (float(np.min(p3)) / rms_loc) if n_sat == 0 else -999.0
    else:
        patch = sub_img[y0:y1, x0:x1]
        v1 = abs(float(np.sum(patch))) / (rms_loc * np.sqrt(max(1, patch.size)))
        v2 = float(np.max(np.abs(patch))) / rms_loc
    return snr_stamp, sat_stamp, ps, hb, v1, v2, still_inv


def run_lightweight_audit() -> None:
    aud = pd.read_csv("data/processed/pilot_audit_150_classified.csv")
    fp = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    key = pd.read_csv("data/processed/blind_human_audit_50_key.csv").set_index("blind_id")

    # 1. Catalog-level all-source partner match on all 150
    cat_cache: dict[str, pd.DataFrame] = {}
    sep_b = np.full(len(aud), np.nan)
    for eb, grp in aud.groupby("exp_b", sort=False):
        if eb not in cat_cache:
            cat_cache[eb] = pd.read_parquet(f"data/processed/pilot_catalogs/{eb}.parquet")
        cb = cat_cache[eb]
        sc_a = SkyCoord(grp["ra"].to_numpy() * u.deg, grp["dec"].to_numpy() * u.deg)
        sc_b = SkyCoord(cb["ra"].to_numpy() * u.deg, cb["dec"].to_numpy() * u.deg)
        _, d2d, _ = sc_a.match_to_catalog_sky(sc_b)
        sep_b[aud.index.get_indexer(grp.index)] = d2d.arcsec
    aud["sep_b_all"] = sep_b
    aud["pass_partner_cat"] = aud["sep_b_all"] > 15.0

    med_r_a = np.array([float(fp.loc[e, "med_r_psf"]) for e in aud["exp_a"]])
    pre_ok = (
        (aud["ellip"] <= 0.20)
        & (aud["r_psf"] >= 0.7 * med_r_a)
        & (aud["r_psf"] <= 1.5 * med_r_a)
        & (aud["min_bright_nbr_as"] > 25.0)
        & aud["pass_partner_cat"]
    )
    flagged_idxs = {int(key.loc[bid, "orig_cand_idx"]) for bid in [1, 12, 13, 14, 21, 28, 29, 35, 37, 42]}
    target_idxs = sorted(set(np.where(pre_ok)[0].tolist()).union(flagged_idxs))
    print(f"Pre-pixel survivors: {pre_ok.sum()}/150; fetching cutouts for {len(target_idxs)} candidates (including all 10 flagged cutouts)...", flush=True)

    # Also build extra pool from pilot_candidates so we have >= 50 bug-fixed survivors for blind_human_audit_50_v2
    cand = pd.read_parquet("data/processed/pilot_candidates.parquet")
    clean = cand[cand["passes_partner_rule"] & cand["plate_a_qc_pass"] & cand["plate_b_qc_pass"] & (~cand["is_asteroid"])].copy()
    # Match clean candidates to pilot_catalogs to get ellip, r_psf, sep_b_all
    extra_rows = []
    rng = np.random.default_rng(2026)
    clean_shuf = clean.sample(frac=1.0, random_state=2026).reset_index(drop=True)
    aud_coords = SkyCoord(aud["ra"].to_numpy() * u.deg, aud["dec"].to_numpy() * u.deg)
    for _, cr in clean_shuf.iterrows():
        if len(extra_rows) >= 70:
            break
        ea, eb = cr["exp_a"], cr["exp_b"]
        if ea not in cat_cache:
            cat_cache[ea] = pd.read_parquet(f"data/processed/pilot_catalogs/{ea}.parquet")
        if eb not in cat_cache:
            cat_cache[eb] = pd.read_parquet(f"data/processed/pilot_catalogs/{eb}.parquet")
        ca, cb = cat_cache[ea], cat_cache[eb]
        sc_c = SkyCoord(cr["ra"] * u.deg, cr["dec"] * u.deg)
        if np.min(sc_c.separation(aud_coords).arcsec) < 5.0:
            continue
        ia = int(np.argmin(sc_c.separation(SkyCoord(ca["ra"].to_numpy() * u.deg, ca["dec"].to_numpy() * u.deg)).arcsec))
        mr = float(fp.loc[ea, "med_r_psf"])
        if not (ca.loc[ia, "ellip"] <= 0.20 and 0.7 * mr <= ca.loc[ia, "r_psf"] <= 1.5 * mr):
            continue
        sb = float(np.min(sc_c.separation(SkyCoord(cb["ra"].to_numpy() * u.deg, cb["dec"].to_numpy() * u.deg)).arcsec))
        if sb <= 15.0:
            continue
        extra_rows.append({"exp_a": ea, "exp_b": eb, "ra": float(cr["ra"]), "dec": float(cr["dec"]), "med_r_a": mr, "med_r_b": float(fp.loc[eb, "med_r_psf"]), "source": "extra"})

    # Combine target_idxs from aud + extra_rows
    jobs = []
    for idx in target_idxs:
        r = aud.iloc[idx]
        jobs.append({
            "key": f"aud_{idx}", "orig_idx": idx, "exp_a": r["exp_a"], "exp_b": r["exp_b"],
            "ra": float(r["ra"]), "dec": float(r["dec"]), "med_r_a": float(fp.loc[r["exp_a"], "med_r_psf"]),
            "med_r_b": float(fp.loc[r["exp_b"], "med_r_psf"]), "is_pre_ok": bool(pre_ok.iloc[idx]),
        })
    for j, er in enumerate(extra_rows):
        jobs.append({"key": f"ext_{j}", "orig_idx": -1, **er, "is_pre_ok": True})

    app_pids = sorted({
        fp.loc[e, "plate_id"]
        for job in jobs for e in (job["exp_a"], job["exp_b"])
        if fp.loc[e, "source_archive"] == "APPLAUSE"
    })
    app_urls, _ = _get_applause_scan_and_wcs_map(app_pids) if app_pids else ({}, {})

    stamp_dir = Path("data/processed/audit_stamps_v2")
    stamp_dir.mkdir(parents=True, exist_ok=True)

    def _process_job(job: dict) -> dict:
        out_p = stamp_dir / f"{job['key']}.npz"
        if out_p.exists():
            z = np.load(out_p)
            return {**job, "a_pk": float(z["a_pk"]), "a_min": float(z["a_min"]), "b_ap": float(z["b_ap"]), "b_pk": float(z["b_pk"]), "ok": True}

        res_pair = {}
        for role, eid, mr in [("A", job["exp_a"], job["med_r_a"]), ("B", job["exp_b"], job["med_r_b"])]:
            fr = fp.loc[eid]
            pid, sa = str(fr["plate_id"]), str(fr["source_archive"])
            if sa == "DASCH":
                fetched = _fetch_dasch_cutout(pid, job["ra"], job["dec"])
            else:
                wcs_full = WCS(fits.Header.fromstring(fr["wcs_header"]))
                fetched = _fetch_applause_range_cutout(app_urls[pid], wcs_full, job["ra"], job["dec"])
            if fetched is None:
                return {**job, "a_pk": np.nan, "a_min": np.nan, "b_ap": np.nan, "b_pk": np.nan, "ok": False}
            img_raw, wcs_c = fetched
            res_pair[role] = _measure_stamp(img_raw, wcs_c, job["ra"], job["dec"], mr, role)

        ca, sa_m, ps_a, hb_a, a_pk, a_min, _ = res_pair["A"]
        cb, sb_m, ps_b, hb_b, b_ap, b_pk, _ = res_pair["B"]
        np.savez_compressed(
            out_p,
            cut_a=ca, sat_a=sa_m, ps_a=np.float32(ps_a),
            cut_b=cb, sat_b=sb_m, ps_b=np.float32(ps_b),
            a_pk=np.float32(a_pk), a_min=np.float32(a_min),
            b_ap=np.float32(b_ap), b_pk=np.float32(b_pk),
        )
        return {**job, "a_pk": a_pk, "a_min": a_min, "b_ap": b_ap, "b_pk": b_pk, "ok": True}

    print(f"Fetching {len(jobs)} stamp pairs ({len(target_idxs)} audit + {len(extra_rows)} extra) with 2 workers...", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_process_job, j): j["key"] for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if k % 5 == 0 or k == len(jobs):
                n_ok = sum(1 for x in results if x["ok"])
                print(f"  [{k}/{len(jobs)}] ok={n_ok}/{k}", flush=True)

    res_df = pd.DataFrame(results)
    res_df["pass_bug1"] = (res_df["b_ap"] <= 3.0) & (res_df["b_pk"] <= 4.0)
    res_df["pass_bug2"] = (res_df["a_pk"] >= 4.0) & (res_df["a_min"] >= -3.0)
    res_df["pass_all"] = res_df["ok"] & res_df["is_pre_ok"] & res_df["pass_bug1"] & res_df["pass_bug2"]
    res_df.to_csv("data/processed/audit_v2_results.csv", index=False)

    aud_res = res_df[res_df["orig_idx"] >= 0].set_index("orig_idx")
    pre_df = aud_res[aud_res["is_pre_ok"] & aud_res["ok"]]
    print(f"\n=== Audit 150 pre-pixel survivors measured: {len(pre_df)} ===")
    print(f"  Removed by Bug 1 pixel check (b_ap>3 or b_pk>4): {(~pre_df['pass_bug1']).sum()}")
    print(f"  Removed by Bug 2 pixel check (white/sat/weak):   {(pre_df['pass_bug1'] & ~pre_df['pass_bug2']).sum()}")
    print(f"  Surviving ALL v1.1 filters (from 150):            {pre_df['pass_all'].sum()}")

    print("\n--- Diagnostics on the 10 flagged blind-sheet cutouts ---")
    for bid in [1, 12, 13, 14, 21, 28, 29, 35, 37, 42]:
        idx = int(key.loc[bid, "orig_cand_idx"])
        r = aud_res.loc[idx]
        sb = float(aud.loc[idx, "sep_b_all"])
        print(f"#{bid:2d}: sep_b={sb:6.2f}\" b_ap={r['b_ap']:5.1f} b_pk={r['b_pk']:5.1f} a_pk={r['a_pk']:5.1f} a_min={r['a_min']:6.1f} -> kept={bool(r['pass_all'])}")

    # Generate blind_human_audit_50_v2.png from up to 50 surviving stamp pairs
    surv = res_df[res_df["pass_all"]].copy()
    print(f"\nTotal bug-fixed surviving stamp pairs available for blind sheet v2: {len(surv)}")
    n_show = min(50, len(surv))
    chosen = surv.sample(n=n_show, random_state=42).reset_index(drop=True)
    chosen["blind_id"] = np.arange(1, n_show + 1)
    chosen.to_csv("data/processed/blind_human_audit_50_v2_key.csv", index=False)

    fig, axes = plt.subplots(10, 10, figsize=(20, 22), dpi=130)
    for row_i in range(10):
        for col_pair in range(5):
            k_i = row_i * 5 + col_pair
            ax_a = axes[row_i, 2 * col_pair]
            ax_b = axes[row_i, 2 * col_pair + 1]
            ax_a.axis("off")
            ax_b.axis("off")
            if k_i >= n_show:
                continue
            crow = chosen.iloc[k_i]
            z = np.load(stamp_dir / f"{crow['key']}.npz")
            for ax, cut_k, sat_k, tag in [(ax_a, "cut_a", "sat_a", f"#{k_i+1} A"), (ax_b, "cut_b", "sat_b", "B")]:
                img_s = np.array(z[cut_k], dtype=float)
                sat_s = np.array(z[sat_k], dtype=bool)
                # Display dark-stars-on-white (gray_r) with saturated pixels masked cleanly
                disp = np.where(sat_s, np.nan, np.clip(img_s, -3.0, 12.0))
                cmap = plt.get_cmap("gray_r").copy()
                cmap.set_bad(color="#d9534f")
                ax.imshow(disp, origin="lower", cmap=cmap, vmin=-2.0, vmax=10.0)
                cy, cx = np.array(disp.shape) / 2.0
                ax.plot(cx, cy, "+", color="#00aa00", markersize=8, markeredgewidth=1.0)
                ax.set_title(tag, fontsize=7, pad=1)
    plt.tight_layout(pad=0.3)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("reports/figures/blind_human_audit_50_v2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reports/figures/blind_human_audit_50_v2.png ({n_show} pairs) and data/processed/blind_human_audit_50_v2_key.csv")


if __name__ == "__main__":
    run_lightweight_audit()
