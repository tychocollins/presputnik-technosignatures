"""Step 1: Render 60-arcsec cutout pairs and audit 150 random clean pilot candidates."""

from __future__ import annotations

import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import zoom

from src.pilot import _get_applause_scan_and_wcs_map


def _download_plate(pid: str, sa: str, applause_urls: dict[str, str]) -> bytes:
    if sa == "DASCH":
        dasch_id = pid.replace("DASCH_", "")
        url = f"http://dc.g-vo.org/dasch/q/getprod/qp/fits-01/{dasch_id}"
    else:
        url = applause_urls[pid]
    for attempt in range(2):
        try:
            with requests.get(url, timeout=(30, 120), stream=True) as r:
                r.raise_for_status()
                chunks = [c for c in r.iter_content(chunk_size=1024 * 1024) if c]
                return b"".join(chunks)
        except Exception:
            if attempt == 1:
                raise
            time.sleep(3.0)
    raise RuntimeError("unreachable")


def _load_image_and_polarity(fits_bytes: bytes) -> np.ndarray | None:
    with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
        tmp.write(fits_bytes)
        tmp.flush()
        with fits.open(tmp.name, memmap=False) as hdul:
            for h in hdul:
                if h.data is not None and h.data.ndim == 2:
                    data = np.ascontiguousarray(h.data, dtype=np.float32)
                    valid = data[np.isfinite(data) & (data != 0)]
                    if len(valid) > 100:
                        p1, p50, p99 = np.percentile(valid, [1, 50, 99])
                        if (p50 - p1) > 1.5 * (p99 - p50):
                            data = -data
                    return data
    return None


def _extract_cutout(
    img: np.ndarray, wcs: WCS, ra_deg: float, dec_deg: float, box_arcsec: float = 60.0
) -> tuple[np.ndarray, float, bool]:
    """Return (64x64 normalized cutout, pixel_scale_arcsec, hits_border)."""
    ny, nx = img.shape
    sc = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    xc, yc = wcs.world_to_pixel(sc)
    xc, yc = float(xc), float(yc)

    # Estimate local pixel scale (arcsec/pix)
    sc_dx = wcs.pixel_to_world(xc + 1.0, yc)
    pix_scale = float(sc.separation(sc_dx).arcsec)
    if not (0.2 <= pix_scale <= 20.0):
        pix_scale = 1.5

    half_pix = max(12, min(80, int(round(0.5 * box_arcsec / pix_scale))))
    x0, x1 = int(round(xc)) - half_pix, int(round(xc)) + half_pix + 1
    y0, y1 = int(round(yc)) - half_pix, int(round(yc)) + half_pix + 1
    hits_border = (x0 < 0 or y0 < 0 or x1 > nx or y1 > ny)

    pad_val = float(np.nanmedian(img[max(0, y0):min(ny, y1), max(0, x0):min(nx, x1)])) if (x1 > 0 and y1 > 0 and x0 < nx and y0 < ny) else 0.0
    patch = np.full((2 * half_pix + 1, 2 * half_pix + 1), pad_val, dtype=np.float32)

    sx0, sx1 = max(0, x0), min(nx, x1)
    sy0, sy1 = max(0, y0), min(ny, y1)
    if sx1 > sx0 and sy1 > sy0:
        patch[(sy0 - y0):(sy1 - y0), (sx0 - x0):(sx1 - x0)] = img[sy0:sy1, sx0:sx1]

    # Resample to 64x64 for uniform contact-sheet & feature extraction
    scale = 64.0 / patch.shape[0]
    cut64 = zoom(patch, scale, order=1)
    return cut64[:64, :64], pix_scale, hits_border


def _classify_cutout(
    cut_a: np.ndarray,
    cut_b: np.ndarray,
    cand_row: pd.Series,
    cat_a: pd.DataFrame,
    hits_border: bool,
) -> tuple[str, dict[str, float]]:
    """Classify candidate into: edge, bright-star halo or spike, scanner artifact, blend, plate defect, stellar-looking unexplained."""
    med_a = float(np.nanmedian(cut_a))
    mad_a = float(np.nanmedian(np.abs(cut_a - med_a))) * 1.4826 + 1e-6
    norm_a = (cut_a - med_a) / mad_a

    # Central 9x9 around (32, 32)
    cy, cx = 32, 32
    center_9 = norm_a[cy - 4:cy + 5, cx - 4:cx + 5]
    center_3 = norm_a[cy - 1:cy + 2, cx - 1:cx + 2]
    max_c = float(np.nanmax(center_3))
    sum_3 = float(np.nansum(np.maximum(center_3, 0.0))) + 1e-6
    sharp_ratio = max_c / sum_3  # >0.55 => 1-2 pixel defect/cosmic

    # Stripe / scanner artifact: strong 1D row or column excess across full 64px
    row_med = np.nanmedian(norm_a, axis=1)
    col_med = np.nanmedian(norm_a, axis=0)
    stripe_score = float(max(np.nanmax(np.abs(row_med)), np.nanmax(np.abs(col_med))))

    # Bright neighbor in catalog within 35 arcsec
    sc_c = SkyCoord(float(cand_row["ra"]) * u.deg, float(cand_row["dec"]) * u.deg)
    sc_cat = SkyCoord(cat_a["ra"].to_numpy() * u.deg, cat_a["dec"].to_numpy() * u.deg)
    seps = sc_c.separation(sc_cat).arcsec
    nbr_mask = (seps > 3.0) & (seps < 35.0)
    bright_nbr = False
    min_nbr_sep = 99.0
    if np.any(nbr_mask):
        sub_cat = cat_a[nbr_mask]
        sub_seps = seps[nbr_mask]
        brighter = sub_cat["mag_est"].to_numpy() < (float(cand_row["mag_est"]) - 2.5)
        if np.any(brighter):
            bright_nbr = True
            min_nbr_sep = float(np.min(sub_seps[brighter]))

    # Blend check: secondary peak in outer ring (5..15 px from center) or high ellipticity in catalog
    ring_mask = np.ones((64, 64), dtype=bool)
    yy, xx = np.ogrid[:64, :64]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring = norm_a[(r2 >= 25) & (r2 <= 225)]
    outer_peak = float(np.nanmax(ring)) if len(ring) else 0.0

    # Match catalog row for ellipticity / r_psf
    idx_self = int(np.argmin(seps))
    ellip = float(cat_a.iloc[idx_self]["ellip"]) if "ellip" in cat_a.columns else 0.0
    r_psf = float(cat_a.iloc[idx_self]["r_psf"]) if "r_psf" in cat_a.columns else 2.0

    edge_dist = float(cand_row["edge_dist_psf"])

    # Hierarchical classification
    if hits_border or edge_dist < 2.5:
        label = "edge"
    elif bright_nbr or float(np.nanpercentile(norm_a, 99.5)) > 25.0 and float(np.nanmedian(norm_a[r2 <= 100])) > 2.5:
        label = "bright-star halo or spike"
    elif stripe_score > 2.8:
        label = "scanner artifact"
    elif sharp_ratio > 0.52 or max_c < 3.0 or float(np.nanmin(center_9)) < -4.5:
        label = "plate defect"
    elif ellip > 0.28 or outer_peak > 0.6 * max_c:
        label = "blend"
    else:
        label = "stellar-looking unexplained"

    return label, {
        "sharp_ratio": sharp_ratio,
        "stripe_score": stripe_score,
        "max_snr_cut": max_c,
        "ellip": ellip,
        "r_psf": r_psf,
        "min_bright_nbr_as": min_nbr_sep,
    }


def run_audit() -> None:
    samp = pd.read_parquet("data/processed/pilot_audit_sample_150.parquet")
    exp = pd.read_parquet("data/processed/exposures.parquet")
    zero_ids = set(
        exp.loc[
            (exp["shadow_fraction"] == 0.0)
            & (exp["shadow_fraction_36000km"] == 0.0)
            & (exp["shadow_fraction_20000km"] == 0.0),
            "exposure_id",
        ]
    )
    assert samp["exp_a"].isin(zero_ids).all() and samp["exp_b"].isin(zero_ids).all(), "BLINDNESS VIOLATION!"

    fp = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    unique_exps = sorted(set(samp["exp_a"]).union(set(samp["exp_b"])))
    applause_pids = [fp.loc[e, "plate_id"] for e in unique_exps if fp.loc[e, "source_archive"] == "APPLAUSE"]
    applause_urls, _ = _get_applause_scan_and_wcs_map(applause_pids)

    # Map exp_id -> list of (cand_idx, role)
    needed: dict[str, list[tuple[int, str]]] = {e: [] for e in unique_exps}
    for i, r in samp.iterrows():
        needed[r["exp_a"]].append((i, "A"))
        needed[r["exp_b"]].append((i, "B"))

    cutouts_a: dict[int, tuple[np.ndarray, float, bool]] = {}
    cutouts_b: dict[int, np.ndarray] = {}

    def _fetch_and_cut(eid: str) -> tuple[str, list[tuple[int, str, np.ndarray, float, bool]]]:
        fr = fp.loc[eid]
        pid = str(fr["plate_id"])
        sa = str(fr["source_archive"])
        b = _download_plate(pid, sa, applause_urls)
        img = _load_image_and_polarity(b)
        del b
        if img is None:
            return eid, []
        wcs = WCS(fits.Header.fromstring(fr["wcs_header"]))
        out = []
        for idx, role in needed[eid]:
            row = samp.iloc[idx]
            cut, ps, hb = _extract_cutout(img, wcs, float(row["ra"]), float(row["dec"]), box_arcsec=60.0)
            out.append((idx, role, cut, ps, hb))
        del img
        return eid, out

    print(f"Downloading & cutting {len(unique_exps)} unique out-of-shadow plates for 150 candidates...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_fetch_and_cut, eid) for eid in unique_exps]
        for k, fut in enumerate(as_completed(futs), 1):
            eid, res_list = fut.result()
            for idx, role, cut, ps, hb in res_list:
                if role == "A":
                    cutouts_a[idx] = (cut, ps, hb)
                else:
                    cutouts_b[idx] = cut
            if k % 10 == 0 or k == len(unique_exps):
                print(f"  Plates cut: {k}/{len(unique_exps)}")

    # Classify
    cat_cache: dict[str, pd.DataFrame] = {}
    records = []
    for idx, row in samp.iterrows():
        ea = row["exp_a"]
        if ea not in cat_cache:
            cat_cache[ea] = pd.read_parquet(f"data/processed/pilot_catalogs/{ea}.parquet")
        cut_a, ps, hb = cutouts_a[idx]
        cut_b = cutouts_b[idx]
        label, metrics = _classify_cutout(cut_a, cut_b, row, cat_cache[ea], hb)
        records.append({
            "cand_idx": idx,
            "instrument": row["instrument"],
            "exp_a": ea,
            "exp_b": row["exp_b"],
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "mag_est": float(row["mag_est"]),
            "partner_m_lim": float(row["partner_m_lim"]),
            "label": label,
            **metrics,
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv("data/processed/pilot_audit_150_classified.csv", index=False)

    print("\n--- CANDIDATE AUDIT SUMMARY (N=150) ---")
    print(df_out["label"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
    ct = pd.crosstab(df_out["instrument"], df_out["label"], normalize="index").mul(100).round(1)
    print("\n--- FRACTIONS BY INSTRUMENT (%) ---")
    print(ct.to_string())

    # Save 3 contact sheets (50 pairs per sheet: 10 rows x 5 pairs = 10x10 panels)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    for sheet_idx in range(3):
        sub_idxs = list(range(sheet_idx * 50, min(len(samp), (sheet_idx + 1) * 50)))
        fig, axes = plt.subplots(10, 10, figsize=(20, 22))
        for p, idx in enumerate(sub_idxs):
            r = p // 5
            c = (p % 5) * 2
            cut_a, _, _ = cutouts_a[idx]
            cut_b = cutouts_b[idx]
            rec = df_out.iloc[idx]
            for j, (cut, tag) in enumerate([(cut_a, "A"), (cut_b, "B")]):
                ax = axes[r, c + j]
                med = np.nanmedian(cut)
                mad = np.nanmedian(np.abs(cut - med)) * 1.4826 + 1e-6
                ax.imshow(cut, cmap="gray_r", vmin=med - 2 * mad, vmax=med + 6 * mad, origin="lower")
                ax.plot(32, 32, "+", color="red", markersize=5, alpha=0.7)
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    short_inst = rec["instrument"][:14]
                    short_lbl = rec["label"][:12]
                    ax.set_title(f"#{idx} {short_inst}\n{short_lbl} m={rec['mag_est']:.1f}", fontsize=6.5)
                else:
                    ax.set_title(f"Partner (lim={rec['partner_m_lim']:.1f})", fontsize=6.5)
        plt.tight_layout()
        out_png = f"reports/figures/pilot_audit_contact_sheet_{sheet_idx + 1}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved {out_png}")


if __name__ == "__main__":
    run_audit()
