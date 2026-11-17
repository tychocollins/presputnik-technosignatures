"""Phase 2b Unblinded Execution: Steps 2 -> 3 -> 4 -> 5, Three-Gate Decision Rule & Exploratory Serial-Order Check.

Strictly executes the registered protocol (OSF DOI: 10.17605/OSF.IO/5PQDZ + Amendment 2 SHA-256 f9a45377...af71)
with ZERO parameter changes:
  - Step 2: Sample composition & campaign/batch_id diversity (top-1 & top-2 batch shares).
  - Step 3: Co-primary within-plate tests under fixed-sequence, Holm-type step-down:
              (3a) m_flash >= 8.0 single-plate (N = 293) tested first at alpha = 0.025 (2,000 ecliptic placebos);
              (3b) m_flash >= 9.0 single-plate (N = 61) tested at alpha = 0.050 ONLY if (3a) is significant;
              plus pre-registered No-Gaia-density ablations.
  - Step 4: All three required negative controls:
              (4a) Shifted-timestamp curve (Delta_t = +-1..+-5 d and +-30, +-60, +-90 d);
              (4b) Fake-altitude parallax controls (42,164 km GEO vs. 36,000, 20,000, and 5,000 km);
              (4c) Local-solar-time-matched sunlit midnight control (N = 27 at m >= 9.0, sf == 0, 5° <= |b_ecl| <= 18°).
  - Step 5: Key secondary & remaining pre-registered secondaries:
              (5a) Strict partner-vetoed Paired Tier A (m >= 8.0, N = 95) directional concordance gate (RR_MH < 1 & Z < 0);
              (5b.1) Shadow-fraction & depth dose-response + uninformative full-shadow across-plate test (N = 47);
              (5b.2) Across-plate mixed-effects log-rate / NB model with (1 | batch_id);
              (5b.3) Exploratory aligned multi-flash tracks (1,000 30"-de-clustered shuffles/plate).
  - Final Three-Gate Decision Rule Verdict: POSITIVE / NULL / INCONCLUSIVE-ARTIFACT.
  - Post-Hoc Exploratory Robustness Check: Serial-number vs. date monotonicity check across series.
"""

from __future__ import annotations

import json
import re
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from astropy import units as u
from astropy.coordinates import AltAz, BarycentricTrueEcliptic, EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from src.aligned_tracks import (
    FWHM_FACTOR,
    MIN_AXIS_SEP_DEG,
    axial_diff_deg,
    find_aligned_tracks,
    plate_axes_pa_deg,
    predict_geo_drift_pa_deg,
    sample_plate_shuffle_exact,
)
from src.amendment2_power_calibration import _assign_batches, filter_instrument_date_sanity
from src.declump_and_mask import _declump_events
from src.final_frozen_calibration import (
    _apply_frozen_filter_to_plate,
    build_ecliptic_time_shifted_bank,
    compute_gaia_density_grid_32,
)
from src.geometry import points_in_umbra
from src.within_plate_power import _cmh_stat


def _apply_conc_pk_veto_any_stamps(eid: str, df_all: pd.DataFrame, min_conc: float = 0.03) -> pd.DataFrame:
    """Apply frozen conc_pk_9x9 >= 0.03 veto reading from phase2b_stamps/ or sunlit_m8_stamps/."""
    if len(df_all) == 0:
        return df_all
    sp = Path(f"data/processed/phase2b_stamps/{eid}.npz")
    if not sp.exists():
        sp = Path(f"data/processed/sunlit_m8_stamps/{eid}.npz")
    if not sp.exists():
        return df_all
    z = np.load(sp)
    arr = z["snr"]
    if arr.ndim != 3 or len(arr) == 0:
        return df_all
    cy, cx = arr.shape[1] // 2, arr.shape[2] // 2
    pks = np.max(arr[:, cy - 1 : cy + 2, cx - 1 : cx + 2], axis=(1, 2))
    b9 = np.sum(np.maximum(0.0, arr[:, cy - 4 : cy + 5, cx - 4 : cx + 5]), axis=(1, 2))
    bad = (pks / np.maximum(b9, 1e-6)) < min_conc
    if not np.any(bad):
        return df_all
    sc_bad = SkyCoord(z["ra"][bad] * u.deg, z["dec"][bad] * u.deg)
    sc_df = SkyCoord(df_all["ra"].to_numpy() * u.deg, df_all["dec"].to_numpy() * u.deg)
    _, d_bad, _ = sc_df.match_to_catalog_sky(sc_bad)
    out = df_all.copy()
    out.loc[d_bad.arcsec <= 1.0, "psf_pass"] = False
    return out


def _mh_stats_with_ci(n_in: np.ndarray, n_tot: np.ndarray, p_arr: np.ndarray) -> dict[str, float]:
    """Compute exact Z_within, expected sum(n_i p_i), RR_MH, and Robins-Breslow-Greenland 95% CI."""
    n_in = np.asarray(n_in, dtype=float)
    n_tot = np.asarray(n_tot, dtype=float)
    p_arr = np.clip(np.asarray(p_arr, dtype=float), 0.02, 0.98)
    n_out = n_tot - n_in
    q_arr = 1.0 - p_arr

    exp_in = float(np.sum(n_tot * p_arr))
    var_in = float(np.sum(n_tot * p_arr * q_arr))
    obs_in = float(np.sum(n_in))
    obs_out = float(np.sum(n_out))
    z_val = (obs_in - exp_in) / np.sqrt(max(var_in, 1e-9))

    # Mantel-Haenszel odds/rate ratio across plates: R = sum(a_i * q_i) / sum(b_i * p_i)
    r_terms = n_in * q_arr
    s_terms = n_out * p_arr
    R = float(np.sum(r_terms))
    S = float(np.sum(s_terms))
    rr_mh = R / max(S, 1e-9)

    # Robins-Breslow-Greenland (1986) variance of log(OR_MH) where T_i = 1:
    # Var(log OR_MH) = sum((a_i + q_i)*a_i*q_i?) -> exact conditional binomial RBG formula:
    # With X_i ~ Binomial(n_i, p_i) under null, R = sum X_i q_i, S = sum (n_i - X_i) p_i:
    # Var(R - RR * S) = sum Var(X_i (q_i + RR p_i)) = sum n_i p_i^* q_i^* (q_i + RR p_i)^2
    # where p_i^* = (RR * p_i) / (q_i + RR * p_i). By delta method on f(R, S) = log(R / S):
    if R > 0 and S > 0:
        denom_i = q_arr + rr_mh * p_arr
        p_star = (rr_mh * p_arr) / np.maximum(denom_i, 1e-9)
        var_u = float(np.sum(n_tot * p_star * (1.0 - p_star) * (denom_i ** 2)))
        se_log_rr = np.sqrt(max(var_u, 1e-12)) / max(R, 1e-9)
        ci_lo = float(np.exp(np.log(rr_mh) - 1.959964 * se_log_rr))
        ci_hi = float(np.exp(np.log(rr_mh) + 1.959964 * se_log_rr))
    else:
        ci_lo, ci_hi = np.nan, np.nan

    return {
        "n_plates": int(len(n_tot)),
        "n_plates_with_events": int(np.sum(n_tot > 0)),
        "n_total": int(round(obs_in + obs_out)),
        "n_in": int(round(obs_in)),
        "n_out": int(round(obs_out)),
        "expected_in": round(exp_in, 2),
        "sd_in": round(float(np.sqrt(max(var_in, 0.0))), 2),
        "rr_mh": round(rr_mh, 4),
        "rr_ci95_lo": round(ci_lo, 4) if np.isfinite(ci_lo) else np.nan,
        "rr_ci95_hi": round(ci_hi, 4) if np.isfinite(ci_hi) else np.nan,
        "z_within": round(float(z_val), 4),
    }


def _fit_sunlit_defect_profiles(sun_df: pd.DataFrame, sun_ev: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit 8x8 empirical-Bayes spatial defect profiles STRICTLY on sunlit plates (shadow_fraction == 0)."""
    gh = np.ones((8, 8), dtype=float) * 0.5
    for eid in sun_df["exposure_id"]:
        xy = sun_ev[eid]
        if len(xy) > 0:
            for ix, iy in zip(np.clip((xy[:, 0] * 8).astype(int), 0, 7), np.clip((xy[:, 1] * 8).astype(int), 0, 7)):
                gh[iy, ix] += 1.0
    gp = gh / np.mean(gh)
    pr = 40.0 * gp / 64.0
    prior_prof = pr / np.mean(pr)

    ip: dict[str, np.ndarray] = {}
    for inst, grp in sun_df.groupby("instrument"):
        h = np.zeros((8, 8), dtype=float)
        for eid in grp["exposure_id"]:
            xy = sun_ev[eid]
            if len(xy) > 0:
                for ix, iy in zip(np.clip((xy[:, 0] * 8).astype(int), 0, 7), np.clip((xy[:, 1] * 8).astype(int), 0, 7)):
                    h[iy, ix] += 1.0
        ip[str(inst)] = (h + pr) / np.mean(h + pr)
    return prior_prof, ip


def _plate_cell_weights_from_sunlit_profiles(
    inst: str,
    m_lim_cal: float,
    vg32: np.ndarray,
    lg8: np.ndarray,
    gaia32: np.ndarray,
    prior_prof: np.ndarray,
    inst_profs: dict[str, np.ndarray],
    use_gaia_density: bool,
    gaia_share: float = 0.469,
) -> np.ndarray:
    prof8 = inst_profs.get(inst, prior_prof)
    p32 = np.repeat(np.repeat(prof8, 4, axis=0), 4, axis=1)
    d32 = np.repeat(np.repeat(np.power(10.0, 0.2 * np.clip(lg8 - m_lim_cal, -1.5, 1.5)), 4, axis=0), 4, axis=1)
    pd_term = p32 * d32
    pd_norm = pd_term / max(float(np.mean(pd_term[vg32])) if np.any(vg32) else float(np.mean(pd_term)), 1e-9)
    if use_gaia_density:
        raw_w = (gaia_share * gaia32 + (1.0 - gaia_share) * pd_norm) * vg32.astype(float)
    else:
        raw_w = pd_norm * vg32.astype(float)
    wf = raw_w.ravel()
    s = float(np.sum(wf))
    return wf / (s if s > 0 else 1.0)


def _compute_plate_umbra_and_in_count(
    wcs: WCS,
    nx: int,
    ny: int,
    ev_xy: np.ndarray,
    mjd_start: float,
    exptime_sec: float,
    obs_lat: float,
    obs_lon: float,
    obs_elev: float,
    r_km: float = 42164.0,
    dt_days_shift: float = 0.0,
    ra_shift_deg: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Evaluate exact 3-point (start, mid, end) topocentric umbral mask on 32x32 WCS grid and candidate coordinates."""
    gp_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32)
    gxn, gyn = np.meshgrid(gp_pts, gp_pts)
    sc_g = wcs.pixel_to_world(gxn.ravel() * nx, gyn.ravel() * ny)
    ra_g = np.deg2rad(sc_g.ra.deg + ra_shift_deg)
    dec_g = sc_g.dec.rad
    dhat_g = np.column_stack([np.cos(dec_g) * np.cos(ra_g), np.cos(dec_g) * np.sin(ra_g), np.sin(dec_g)])

    loc = EarthLocation(lat=obs_lat * u.deg, lon=obs_lon * u.deg, height=obs_elev * u.m)
    m0 = float(mjd_start) + float(dt_days_shift)
    dur_d = max(float(exptime_sec), 1.0) / 86400.0
    t3 = Time([m0, m0 + 0.5 * dur_d, m0 + dur_d], format="mjd", scale="utc")
    robs3 = loc.get_gcrs(t3).cartesian.xyz.to_value(u.km).T
    rsun3 = get_sun(t3).cartesian.xyz.to_value(u.km).T

    u_grid = points_in_umbra(robs3, dhat_g, rsun3, r_km).mean(axis=0) >= 0.5
    if len(ev_xy) > 0:
        sc_e = wcs.pixel_to_world(ev_xy[:, 0] * nx, ev_xy[:, 1] * ny)
        ra_e = np.deg2rad(sc_e.ra.deg + ra_shift_deg)
        dec_e = sc_e.dec.rad
        dhat_e = np.column_stack([np.cos(dec_e) * np.cos(ra_e), np.cos(dec_e) * np.sin(ra_e), np.sin(dec_e)])
        u_ev = points_in_umbra(robs3, dhat_e, rsun3, r_km).mean(axis=0) >= 0.5
        n_in = int(np.sum(u_ev))
    else:
        n_in = 0
    return u_grid.reshape(32, 32), n_in


def _placebo_null_distribution(
    sun_pool_df: pd.DataFrame,
    ecl_sub_eids: list[str],
    bank: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sun_ev: dict[str, np.ndarray],
    sun_vm: dict[str, np.ndarray],
    sun_lg: dict[str, np.ndarray],
    sun_gaia: dict[str, np.ndarray],
    prior_prof: np.ndarray,
    inst_profs: dict[str, np.ndarray],
    observed_sfs: np.ndarray,
    use_gaia_density: bool,
    n_placebo: int = 2000,
    seed: int = 20261116,
) -> np.ndarray:
    """Generate `n_placebo` time-shifted 3D ecliptic umbral Z_within realizations matched to `observed_sfs`."""
    ecl_eids = [eid for eid in ecl_sub_eids if eid in bank]
    sun_idx = sun_pool_df.set_index("exposure_id")
    p_bank = {}
    for eid in ecl_eids:
        r = sun_idx.loc[eid]
        cw = _plate_cell_weights_from_sunlit_profiles(
            str(r["instrument"]), float(r["m_lim_cal"]), sun_vm[eid], sun_lg[eid], sun_gaia[eid],
            prior_prof, inst_profs, use_gaia_density, 0.469,
        )
        sfs_b, ug_b, nin_b = bank[eid]
        p_bank[eid] = (sfs_b, len(sun_ev[eid]), nin_b, np.clip(ug_b @ cw, 0.02, 0.98))

    sf_centers = np.linspace(0.075, 0.925, 18)
    bin_tables = []
    for c_sf in sf_centers:
        elig = []
        for eid in ecl_eids:
            sfs_b = p_bank[eid][0]
            ks0 = np.where(np.abs(sfs_b - c_sf) <= 0.12)[0]
            if len(ks0) > 0:
                elig.append((eid, False, ks0))
            ks1 = np.where(np.abs((1.0 - sfs_b) - c_sf) <= 0.12)[0]
            if len(ks1) > 0:
                elig.append((eid, True, ks1))
        bin_tables.append(elig)

    rng = np.random.default_rng(seed)
    n_pl = len(observed_sfs)
    b_idx_fixed = np.clip(((observed_sfs - 0.05) / 0.05).astype(int), 0, 17)
    null_zs = np.empty(n_placebo, dtype=float)
    for b in range(n_placebo):
        ni = np.empty(n_pl, dtype=int)
        nt = np.empty(n_pl, dtype=int)
        pa = np.empty(n_pl, dtype=float)
        for i in range(n_pl):
            elig = bin_tables[b_idx_fixed[i]]
            eid, flip, ks = elig[int(rng.integers(0, len(elig)))]
            _, n_tot, nin_b, p_arr = p_bank[eid]
            k = int(ks[rng.integers(0, len(ks))])
            ni[i] = int(n_tot - nin_b[k]) if flip else int(nin_b[k])
            nt[i] = n_tot
            pa[i] = float(np.clip(1.0 - p_arr[k], 0.02, 0.98)) if flip else float(p_arr[k])
        null_zs[b] = _cmh_stat(ni, nt, pa)[0]
    return null_zs


def run_phase2b_unblinding() -> dict[str, Any]:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"================================================================================")
    print(f"PHASE 2B UNBLINDED ANALYSIS | Frozen Commit: {commit_hash}")
    print(f"================================================================================")

    exp = pd.read_parquet("data/processed/exposures.parquet")
    exp_idx = exp.set_index("exposure_id")
    batch_map = _assign_batches(exp)

    p2b_raw = pd.read_parquet("data/processed/phase2b_plates.parquet")
    man = pd.read_parquet("data/processed/phase2b_manifest.parquet")
    links = pd.read_parquet("data/processed/phase2b_paired_tierA_links.parquet")

    meta_cols = [
        "exposure_id", "mjd_start", "mjd_mid", "exptime_sec", "ra_deg", "dec_deg",
        "fov_ra_deg", "fov_dec_deg", "obs_lat_deg", "obs_lon_deg", "obs_elev_m",
        "shadow_fraction", "shadow_fraction_36000km", "shadow_fraction_20000km", "emul",
    ]
    p2b = p2b_raw.merge(man[meta_cols], on="exposure_id", how="left")
    p2b, p2b_date_excl = filter_instrument_date_sanity(p2b, mjd_col="mjd_mid", inst_col="instrument")
    p2b["batch_id"] = p2b["exposure_id"].astype(str).map(batch_map)
    print(
        f"Instrument date-sanity rule applied: {len(p2b_date_excl)} raw QC-failing exposures excluded, "
        f"{int(p2b_date_excl['qc_pass'].sum())} QC-passing plates excluded."
    )

    # Load frozen 348-plate sunlit calibration pool (`pilot` + `sunlit_m8`) & 102-plate paired sunlit pool
    pilot_res = pd.read_csv("data/processed/pilot_results.csv")
    qc9 = pd.read_csv("data/processed/pilot_plate_qc.csv")
    fp9 = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    p8 = pd.read_parquet("data/processed/sunlit_m8_plates.parquet")
    ap = pd.read_parquet("data/raw/applause_plate.parquet")
    dn = pd.read_parquet("data/raw/dasch_narrow_plates.parquet")
    dw = pd.read_parquet("data/raw/dasch_wide_plates.parquet")
    cands_old = pd.read_parquet("data/processed/pilot_candidates.parquet")
    ast_rows = cands_old[cands_old["is_asteroid"].fillna(False)]
    ast_by_exp = {eid: SkyCoord(grp["ra"].to_numpy() * u.deg, grp["dec"].to_numpy() * u.deg) for eid, grp in ast_rows.groupby("exp_a")}

    ap_map = dict(zip(ap["plate_id"].astype(str), ap["filter"].fillna("").astype(str).str.strip()))
    dasch = pd.concat([dn[["dasch_id", "plate_class"]], dw[["dasch_id", "plate_class"]]])
    dasch_map = dict(zip("DASCH_" + dasch["dasch_id"].astype(str), dasch["plate_class"].fillna("").astype(str).str.strip()))

    def get_emul(pid: str) -> str:
        s = str(pid)
        return dasch_map.get(s, "") if s.startswith("DASCH_") else ap_map.get(s.replace("APPLAUSE_", ""), ap_map.get(s, ""))

    band_map9 = dict(zip(qc9["plate_id"].astype(str), qc9["phot_band"].astype(str)))
    pilot_res["emul_ok"] = pilot_res["plate1"].map(get_emul) == pilot_res["plate2"].map(get_emul)
    pilot_res["band_ok"] = pilot_res["plate1"].map(band_map9) == pilot_res["plate2"].map(band_map9)
    pilot_res["pair_ok"] = pilot_res["qc_pass_pair"] & pilot_res["emul_ok"] & pilot_res["band_ok"]

    sun_ev, sun_vm, sun_lg, sun_gaia, sun_wcs, sun_rows = {}, {}, {}, {}, {}, []
    p8_qc = p8[p8["qc_pass"].fillna(False)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in p8_qc.iterrows():
            eid = str(r["exposure_id"])
            df_all = _apply_conc_pk_veto_any_stamps(eid, pd.read_parquet(f"data/processed/sunlit_m8_catalogs/{eid}.parquet"))
            wcs = WCS(fits.Header.fromstring(r["wcs_header"]))
            nx, ny = int(r["nx"]), int(r["ny"])
            ga = np.nan_to_num(np.array(json.loads(r["local_mlim_grid_json"]), dtype=float), nan=float(r["m_lim_cal"]))
            ev, vg, uf, _ = _apply_frozen_filter_to_plate(eid, df_all, wcs, nx, ny, float(r["med_r_psf"]), float(r["m_lim_cal"]), ga)
            sun_ev[eid], sun_vm[eid], sun_lg[eid] = ev, vg, ga
            sun_gaia[eid] = compute_gaia_density_grid_32(df_all, wcs, nx, ny, vg)
            sun_wcs[eid] = (wcs, nx, ny)
            sun_rows.append({
                "exposure_id": eid, "plate_id": str(r["plate_id"]), "instrument": str(r["instrument"]),
                "m_flash_05s": float(r["m_flash_05s"]), "m_lim_cal": float(r["m_lim_cal"]),
                "unmasked_frac": uf, "n_events": len(ev),
            })

        for eid in fp9.index:
            eid = str(eid)
            if eid in sun_ev or eid not in exp_idx.index:
                continue
            qm = qc9[qc9["exposure_id"] == eid] if "exposure_id" in qc9.columns else qc9[qc9["plate_id"] == fp9.loc[eid, "plate_id"]]
            if len(qm) == 0 or not bool(qm["qc_pass"].iloc[0]):
                continue
            fra = fp9.loc[eid]
            df_all = pd.read_parquet(f"data/processed/pilot_catalogs/{eid}.parquet")
            wcs = WCS(fits.Header.fromstring(fra["wcs_header"]))
            nx, ny = int(fra["nx"]), int(fra["ny"])
            ga = np.nan_to_num(np.array(json.loads(fra["local_mlim_grid_json"]), dtype=float), nan=float(fra["m_lim_cal"]))
            ev, vg, uf, _ = _apply_frozen_filter_to_plate(
                eid, df_all, wcs, nx, ny, float(fra["med_r_psf"]), float(fra["m_lim_cal"]), ga,
                ast_coords=ast_by_exp.get(eid),
            )
            sun_ev[eid], sun_vm[eid], sun_lg[eid] = ev, vg, ga
            sun_gaia[eid] = compute_gaia_density_grid_32(df_all, wcs, nx, ny, vg)
            sun_wcs[eid] = (wcs, nx, ny)
            er = exp_idx.loc[eid]
            sun_rows.append({
                "exposure_id": eid, "plate_id": str(er["plate_id"]), "instrument": str(er["instrument"]),
                "m_flash_05s": float(er["m_flash_05s"]), "m_lim_cal": float(fra["m_lim_cal"]),
                "unmasked_frac": uf, "n_events": len(ev),
            })
        sun_df = pd.DataFrame(sun_rows)

        paired_sun_ev, paired_sun_vm, paired_sun_lg, paired_sun_gaia, paired_sun_rows = {}, {}, {}, {}, []
        for _, pr in pilot_res[pilot_res["pair_ok"]].iterrows():
            for ea, eb in [(str(pr["exp1"]), str(pr["exp2"])), (str(pr["exp2"]), str(pr["exp1"]))]:
                if ea in paired_sun_ev:
                    continue
                dfa = pd.read_parquet(f"data/processed/pilot_catalogs/{ea}.parquet")
                dfb = pd.read_parquet(f"data/processed/pilot_catalogs/{eb}.parquet")
                fra, frb = fp9.loc[ea], fp9.loc[eb]
                wa, wb = WCS(fits.Header.fromstring(fra["wcs_header"])), WCS(fits.Header.fromstring(frb["wcs_header"]))
                nxa, nya = int(fra["nx"]), int(fra["ny"])
                ga = np.nan_to_num(np.array(json.loads(fra["local_mlim_grid_json"]), dtype=float), nan=float(fra["m_lim_cal"]))
                gb = np.nan_to_num(np.array(json.loads(frb["local_mlim_grid_json"]), dtype=float), nan=float(frb["m_lim_cal"]))
                ev, vg, uf, _ = _apply_frozen_filter_to_plate(
                    ea, dfa, wa, nxa, nya, float(fra["med_r_psf"]), float(fra["m_lim_cal"]), ga,
                    partner_df=dfb, partner_wcs=wb, partner_nx=int(frb["nx"]), partner_ny=int(frb["ny"]),
                    partner_med_r=float(frb["med_r_psf"]), partner_m_lim_cal=float(frb["m_lim_cal"]),
                    partner_grid=gb, ast_coords=ast_by_exp.get(ea),
                )
                paired_sun_ev[ea], paired_sun_vm[ea], paired_sun_lg[ea] = ev, vg, ga
                paired_sun_gaia[ea] = compute_gaia_density_grid_32(dfa, wa, nxa, nya, vg)
                er = exp_idx.loc[ea]
                paired_sun_rows.append({
                    "exposure_id": ea, "instrument": str(er["instrument"]),
                    "m_flash_05s": float(er["m_flash_05s"]), "m_lim_cal": float(fra["m_lim_cal"]),
                    "unmasked_frac": uf, "n_events": len(ev),
                })
        paired_sun_df = pd.DataFrame(paired_sun_rows)

    prior_prof_single, inst_profs_single = _fit_sunlit_defect_profiles(sun_df, sun_ev)
    prior_prof_paired, inst_profs_paired = _fit_sunlit_defect_profiles(paired_sun_df, paired_sun_ev)
    single_bank = build_ecliptic_time_shifted_bank(sun_df, exp_idx, sun_wcs, sun_ev, sun_vm)

    # Extract Final Frozen Filter v1.2 single-plate candidates & 32x32 grids for ALL QC-passing Phase 2b plates
    p2b_qc = p2b[p2b["qc_pass"]].copy().reset_index(drop=True)
    p2b_ev, p2b_vm, p2b_lg, p2b_gaia, p2b_wcs, p2b_uf, p2b_nraw = {}, {}, {}, {}, {}, {}, {}
    p2b_df_cache: dict[str, pd.DataFrame] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in p2b_qc.iterrows():
            eid = str(r["exposure_id"])
            df_all = _apply_conc_pk_veto_any_stamps(eid, pd.read_parquet(f"data/processed/phase2b_catalogs/{eid}.parquet"))
            p2b_df_cache[eid] = df_all
            wcs = WCS(fits.Header.fromstring(r["wcs_header"]))
            nx, ny = int(r["nx"]), int(r["ny"])
            ga = np.nan_to_num(np.array(json.loads(r["local_mlim_grid_json"]), dtype=float), nan=float(r["m_lim_cal"]))
            ast_ra = json.loads(r["ast_ra_json"]) if isinstance(r.get("ast_ra_json"), str) else []
            ast_dec = json.loads(r["ast_dec_json"]) if isinstance(r.get("ast_dec_json"), str) else []
            ast_sc = SkyCoord(np.array(ast_ra) * u.deg, np.array(ast_dec) * u.deg) if len(ast_ra) > 0 else None
            ev, vg, uf, n_raw = _apply_frozen_filter_to_plate(
                eid, df_all, wcs, nx, ny, float(r["med_r_psf"]), float(r["m_lim_cal"]), ga, ast_coords=ast_sc
            )
            p2b_ev[eid], p2b_vm[eid], p2b_lg[eid] = ev, vg, ga
            p2b_gaia[eid] = compute_gaia_density_grid_32(df_all, wcs, nx, ny, vg)
            p2b_wcs[eid] = (wcs, nx, ny)
            p2b_uf[eid] = uf
            p2b_nraw[eid] = n_raw

    p2b_qc["n_events"] = [len(p2b_ev[str(e)]) for e in p2b_qc["exposure_id"]]
    p2b_qc["unmasked_frac"] = [p2b_uf[str(e)] for e in p2b_qc["exposure_id"]]

    # Define exact registered analysis sets
    sh_m8 = p2b_qc[(p2b_qc["m_flash_05s"] >= 8.0) & (p2b_qc["shadow_fraction"] > 0.05) & (p2b_qc["shadow_fraction"] < 0.95)].drop_duplicates("plate_id").reset_index(drop=True)
    sh_m9 = p2b_qc[(p2b_qc["phase2b_role"] == "primary_m9_partial")].drop_duplicates("plate_id").reset_index(drop=True)
    sh_full9 = p2b_qc[(p2b_qc["m_flash_05s"] >= 9.0) & (p2b_qc["shadow_fraction"] >= 0.95)].drop_duplicates("plate_id").reset_index(drop=True)
    lst_ctrl_p2b = p2b_qc[p2b_qc["phase2b_role"] == "matched_lst_sunlit_ctrl"].drop_duplicates("plate_id").reset_index(drop=True)

    # Identify all 95 QC-passing + phot_band-matched Strict Paired Tier A target links
    p2b_qc_idx = p2b_qc.set_index("exposure_id")
    qc_lookup = {str(e): True for e in p2b_qc["exposure_id"]}
    band_lookup = dict(zip(p2b_qc["exposure_id"].astype(str), p2b_qc["phot_band"].astype(str)))
    for eid in sun_df["exposure_id"]:
        qc_lookup[str(eid)] = True
        if str(eid) in fp9.index:
            sub = qc9[qc9["plate_id"] == fp9.loc[eid, "plate_id"]]
            band_lookup[str(eid)] = str(sub["phot_band"].iloc[0]) if len(sub) > 0 else ""
        else:
            band_lookup[str(eid)] = ""

    paired_links_ok = []
    for _, r in links.iterrows():
        ea, eb = str(r["exp_target"]), str(r["exp_partner"])
        if qc_lookup.get(ea, False) and qc_lookup.get(eb, False):
            ba, bb = band_lookup.get(ea, ""), band_lookup.get(eb, "")
            if ba == bb or ba == "" or bb == "":
                paired_links_ok.append((ea, eb))

    # =========================================================================
    # STEP 2: SAMPLE COMPOSITION & CAMPAIGN / BATCH DIVERSITY
    # =========================================================================
    def _series_code(pid: str) -> str:
        raw = str(pid).replace("DASCH_", "")
        if raw.startswith("APPLAUSE_"):
            return "APPLAUSE"
        m_s = re.match(r"^([A-Za-z]+)", raw)
        return m_s.group(1) if m_s else raw

    def _summarize_composition(df_sub: pd.DataFrame, label: str, ev_col: str = "n_events") -> dict[str, Any]:
        sers = df_sub["plate_id"].map(_series_code)
        b_counts = df_sub["batch_id"].value_counts()
        top1_pl = int(b_counts.iloc[0]) if len(b_counts) > 0 else 0
        top2_pl = int(b_counts.iloc[1]) if len(b_counts) > 1 else 0
        b_ev = df_sub.groupby("batch_id")[ev_col].sum().sort_values(ascending=False)
        tot_ev = max(int(df_sub[ev_col].sum()), 1)
        top1_ev = int(b_ev.iloc[0]) if len(b_ev) > 0 else 0
        top2_ev = int(b_ev.iloc[1]) if len(b_ev) > 1 else 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            isos = [Time(float(x), format="mjd", scale="utc").iso[:10] for x in df_sub["mjd_mid"]]
        comp = {
            "sample": label,
            "n_plates": int(len(df_sub)),
            "n_archives": int(df_sub["source_archive"].nunique()),
            "n_instruments": int(df_sub["instrument"].nunique()),
            "n_series": int(sers.nunique()),
            "n_batches": int(df_sub["batch_id"].nunique()),
            "date_min": min(isos) if isos else "",
            "date_max": max(isos) if isos else "",
            "top1_batch_plates": f"{top1_pl}/{len(df_sub)} ({100.0*top1_pl/max(len(df_sub),1):.1f}%)",
            "top2_batch_plates": f"{top2_pl}/{len(df_sub)} ({100.0*top2_pl/max(len(df_sub),1):.1f}%)",
            "top1_batch_events_share": round(100.0 * top1_ev / tot_ev, 1),
            "top2_batch_events_share": round(100.0 * top2_ev / tot_ev, 1),
            "instruments": df_sub["instrument"].value_counts().to_dict(),
        }
        return comp

    # Extract Paired Tier A events first so Step 2 can also report Paired Tier A batch shares
    def _load_any_plate_bundle(eid: str) -> tuple[pd.DataFrame, WCS, int, int, float, float, np.ndarray, SkyCoord | None]:
        if eid in p2b_qc_idx.index:
            r = p2b_qc_idx.loc[eid]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            df_a = p2b_df_cache[eid]
            wa, nxa, nya = p2b_wcs[eid]
            ast_ra = json.loads(r["ast_ra_json"]) if isinstance(r.get("ast_ra_json"), str) else []
            ast_dec = json.loads(r["ast_dec_json"]) if isinstance(r.get("ast_dec_json"), str) else []
            ast_sc = SkyCoord(np.array(ast_ra) * u.deg, np.array(ast_dec) * u.deg) if len(ast_ra) > 0 else None
            return df_a, wa, nxa, nya, float(r["med_r_psf"]), float(r["m_lim_cal"]), p2b_lg[eid], ast_sc
        if eid in fp9.index:
            fra = fp9.loc[eid]
            df_a = pd.read_parquet(f"data/processed/pilot_catalogs/{eid}.parquet")
            wa = WCS(fits.Header.fromstring(fra["wcs_header"]))
            ga = np.nan_to_num(np.array(json.loads(fra["local_mlim_grid_json"]), dtype=float), nan=float(fra["m_lim_cal"]))
            return df_a, wa, int(fra["nx"]), int(fra["ny"]), float(fra["med_r_psf"]), float(fra["m_lim_cal"]), ga, ast_by_exp.get(eid)
        r8 = p8_qc[p8_qc["exposure_id"] == eid].iloc[0]
        df_a = _apply_conc_pk_veto_any_stamps(eid, pd.read_parquet(f"data/processed/sunlit_m8_catalogs/{eid}.parquet"))
        wa = WCS(fits.Header.fromstring(r8["wcs_header"]))
        ga = np.nan_to_num(np.array(json.loads(r8["local_mlim_grid_json"]), dtype=float), nan=float(r8["m_lim_cal"]))
        return df_a, wa, int(r8["nx"]), int(r8["ny"]), float(r8["med_r_psf"]), float(r8["m_lim_cal"]), ga, None

    paired_p2b_ev, paired_p2b_vm, paired_p2b_lg, paired_p2b_gaia, paired_p2b_wcs, paired_p2b_nraw = {}, {}, {}, {}, {}, {}
    paired_target_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ea, eb in paired_links_ok:
            dfa, wa, nxa, nya, mra, mla, ga, ast_a = _load_any_plate_bundle(ea)
            dfb, wb, nxb, nyb, mrb, mlb, gb, _ = _load_any_plate_bundle(eb)
            ev, vg, uf, n_raw = _apply_frozen_filter_to_plate(
                ea, dfa, wa, nxa, nya, mra, mla, ga,
                partner_df=dfb, partner_wcs=wb, partner_nx=nxb, partner_ny=nyb,
                partner_med_r=mrb, partner_m_lim_cal=mlb, partner_grid=gb, ast_coords=ast_a,
            )
            paired_p2b_ev[ea], paired_p2b_vm[ea], paired_p2b_lg[ea] = ev, vg, ga
            paired_p2b_gaia[ea] = compute_gaia_density_grid_32(dfa, wa, nxa, nya, vg)
            paired_p2b_wcs[ea] = (wa, nxa, nya)
            paired_p2b_nraw[ea] = n_raw
            er = exp_idx.loc[ea]
            paired_target_rows.append({
                "exposure_id": ea, "partner_id": eb, "plate_id": str(er["plate_id"]),
                "source_archive": str(er["source_archive"]), "instrument": str(er["instrument"]),
                "mjd_start": float(er["mjd_start"]), "mjd_mid": float(er["mjd_mid"]),
                "exptime_sec": float(er["exptime_sec"]), "ra_deg": float(er["ra_deg"]), "dec_deg": float(er["dec_deg"]),
                "obs_lat_deg": float(er["obs_lat_deg"]), "obs_lon_deg": float(er["obs_lon_deg"]), "obs_elev_m": float(er["obs_elev_m"]),
                "m_lim_cal": mla, "med_r_psf": mra, "unmasked_frac": uf, "n_events": len(ev),
                "batch_id": batch_map.get(ea, -1),
            })
    sh_paired_df = pd.DataFrame(paired_target_rows)

    step2_comp = [
        _summarize_composition(sh_m8, "Co-Primary 1: Single-Plate m >= 8.0 (N=293)"),
        _summarize_composition(sh_m9, "Co-Primary 2: Single-Plate m >= 9.0 (N=61)"),
        _summarize_composition(sh_paired_df, "Key Secondary: Strict Paired Tier A m >= 8.0 (N=95)"),
    ]
    print("\n================ STEP 2: SAMPLE COMPOSITION & CAMPAIGN / BATCH DIVERSITY ================")
    for c in step2_comp:
        print(
            f"  {c['sample']}: {c['n_plates']} plates | {c['n_archives']} archives | {c['n_instruments']} instruments | "
            f"{c['n_series']} series | {c['n_batches']} batches | dates {c['date_min']}..{c['date_max']} | "
            f"top-1 batch={c['top1_batch_plates']} ({c['top1_batch_events_share']}% events), "
            f"top-2 batch={c['top2_batch_plates']} ({c['top2_batch_events_share']}% events)"
        )
        print(f"    Instruments: {c['instruments']}")

    # =========================================================================
    # STEP 3: CO-PRIMARY WITHIN-PLATE TESTS (FIXED-SEQUENCE HOLM-TYPE STEP-DOWN)
    # =========================================================================
    def _eval_single_shadow_set(
        df_set: pd.DataFrame,
        ecl_sun_eids: list[str],
        use_gaia: bool,
        n_placebo: int,
        seed: int,
    ) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
        per_plate = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _, r in df_set.iterrows():
                eid = str(r["exposure_id"])
                wcs, nx, ny = p2b_wcs[eid]
                cw = _plate_cell_weights_from_sunlit_profiles(
                    str(r["instrument"]), float(r["m_lim_cal"]), p2b_vm[eid], p2b_lg[eid], p2b_gaia[eid],
                    prior_prof_single, inst_profs_single, use_gaia, 0.469,
                )
                u32, n_in = _compute_plate_umbra_and_in_count(
                    wcs, nx, ny, p2b_ev[eid], float(r["mjd_start"]), float(r["exptime_sec"]),
                    float(r["obs_lat_deg"]), float(r["obs_lon_deg"]), float(r["obs_elev_m"]), 42164.0,
                )
                p_i = float(np.clip(np.sum(cw[u32.ravel()]), 0.02, 0.98))
                sf_unmasked = float(np.mean(u32[p2b_vm[eid]])) if np.any(p2b_vm[eid]) else float(r["shadow_fraction"])
                n_tot = len(p2b_ev[eid])
                per_plate.append({
                    "exposure_id": eid, "plate_id": str(r["plate_id"]), "instrument": str(r["instrument"]),
                    "m_flash_05s": float(r["m_flash_05s"]), "shadow_fraction_archive": float(r["shadow_fraction"]),
                    "sf_unmasked_32": round(sf_unmasked, 4), "p_i": round(p_i, 4),
                    "n_tot": n_tot, "n_in": n_in, "n_out": n_tot - n_in,
                })
        pp_df = pd.DataFrame(per_plate)
        st = _mh_stats_with_ci(pp_df["n_in"].to_numpy(), pp_df["n_tot"].to_numpy(), pp_df["p_i"].to_numpy())
        null_zs = _placebo_null_distribution(
            sun_df, ecl_sun_eids, single_bank, sun_ev, sun_vm, sun_lg, sun_gaia,
            prior_prof_single, inst_profs_single, pp_df["shadow_fraction_archive"].to_numpy(),
            use_gaia, n_placebo=n_placebo, seed=seed,
        )
        r_le = int(np.sum(null_zs <= st["z_within"]))
        st["n_placebo"] = n_placebo
        st["crit_z_025"] = round(float(np.quantile(null_zs, 0.025)), 4)
        st["crit_z_050"] = round(float(np.quantile(null_zs, 0.050)), 4)
        st["null_mean_z"] = round(float(np.mean(null_zs)), 4)
        st["null_sd_z"] = round(float(np.std(null_zs)), 4)
        st["p_placebo"] = round((r_le + 1.0) / (n_placebo + 1.0), 5)
        st["p_placebo_raw"] = round(r_le / n_placebo, 5)
        return st, pp_df, null_zs

    sun_eids_m8 = sun_df["exposure_id"].tolist()
    sun_eids_m9 = sun_df[sun_df["m_flash_05s"] >= 9.0]["exposure_id"].tolist()

    coprim1_gaia, pp_m8_gaia, _ = _eval_single_shadow_set(sh_m8, sun_eids_m8, True, 2000, 20261116)
    coprim1_nogaia, _, _ = _eval_single_shadow_set(sh_m8, sun_eids_m8, False, 2000, 20261117)
    coprim2_gaia, pp_m9_gaia, _ = _eval_single_shadow_set(sh_m9, sun_eids_m9, True, 2000, 20261118)
    coprim2_nogaia, _, _ = _eval_single_shadow_set(sh_m9, sun_eids_m9, False, 2000, 20261119)

    pp_m8_gaia.to_csv("data/processed/phase2b_coprimary_m8_plates.csv", index=False)
    pp_m9_gaia.to_csv("data/processed/phase2b_coprimary_m9_plates.csv", index=False)

    gate_a_step1_pass = bool(coprim1_gaia["p_placebo"] <= 0.025)
    gate_a_step2_pass = bool(gate_a_step1_pass and (coprim2_gaia["p_placebo"] <= 0.050))
    gate_a_pass = gate_a_step1_pass

    print("\n================ STEP 3: CO-PRIMARY WITHIN-PLATE TESTS (2,000 ECLIPTIC PLACEBOS) ================")
    print(f"  [Step 3.1 - Co-Primary 1: m >= 8.0, N=293, WITH Gaia, alpha=0.025]: {coprim1_gaia}")
    print(f"  [Step 3.1 Ablation - m >= 8.0, NO Gaia]:                            {coprim1_nogaia}")
    print(f"  [Step 3.2 - Co-Primary 2: m >= 9.0, N=61, WITH Gaia, alpha=0.050]:  {coprim2_gaia}")
    print(f"  [Step 3.2 Ablation - m >= 9.0, NO Gaia]:                            {coprim2_nogaia}")
    print(f"  -> Fixed-Sequence Holm Gate (a): Step1(m>=8 @ 0.025)={gate_a_step1_pass}, Step2(m>=9 @ 0.050)={gate_a_step2_pass}")

    # =========================================================================
    # STEP 4: REQUIRED NEGATIVE CONTROLS
    # =========================================================================
    print("\n================ STEP 4: REQUIRED NEGATIVE CONTROLS ================")

    # (4a) Shifted-timestamp curve: Delta_t in {-5..-1, 0, +1..+5} d and {+-30, +-60, +-90} d
    dt_shifts = [-90, -60, -30, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 30, 60, 90]
    shift_rows = []
    cw_m8_map = {
        str(r["exposure_id"]): _plate_cell_weights_from_sunlit_profiles(
            str(r["instrument"]), float(r["m_lim_cal"]), p2b_vm[str(r["exposure_id"])],
            p2b_lg[str(r["exposure_id"])], p2b_gaia[str(r["exposure_id"])],
            prior_prof_single, inst_profs_single, True, 0.469,
        )
        for _, r in sh_m8.iterrows()
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dt_d in dt_shifts:
            ni_l, nt_l, p_l = [], [], []
            ni_ra, nt_ra, p_ra = [], [], []
            # For large off-opposition shifts (+-30, +-60, +-90 d), literal mjd + dt moves the anti-solar point 30-90 deg away
            # (leaving 0 plates in umbra); also record RA-co-shifted (ra + 0.9856 deg/d * dt) so off-epoch geometry is tested too!
            for _, r in sh_m8.iterrows():
                eid = str(r["exposure_id"])
                wcs, nx, ny = p2b_wcs[eid]
                cw = cw_m8_map[eid]
                u32, nin = _compute_plate_umbra_and_in_count(
                    wcs, nx, ny, p2b_ev[eid], float(r["mjd_start"]), float(r["exptime_sec"]),
                    float(r["obs_lat_deg"]), float(r["obs_lon_deg"]), float(r["obs_elev_m"]),
                    42164.0, dt_days_shift=float(dt_d),
                )
                sf_u = float(np.mean(u32[p2b_vm[eid]])) if np.any(p2b_vm[eid]) else 0.0
                if 0.05 < sf_u < 0.95:
                    ni_l.append(nin)
                    nt_l.append(len(p2b_ev[eid]))
                    p_l.append(float(np.clip(np.sum(cw[u32.ravel()]), 0.02, 0.98)))
                if abs(dt_d) >= 30:
                    u32_r, nin_r = _compute_plate_umbra_and_in_count(
                        wcs, nx, ny, p2b_ev[eid], float(r["mjd_start"]), float(r["exptime_sec"]),
                        float(r["obs_lat_deg"]), float(r["obs_lon_deg"]), float(r["obs_elev_m"]),
                        42164.0, dt_days_shift=float(dt_d), ra_shift_deg=0.985647 * float(dt_d),
                    )
                    sf_ur = float(np.mean(u32_r[p2b_vm[eid]])) if np.any(p2b_vm[eid]) else 0.0
                    if 0.05 < sf_ur < 0.95:
                        ni_ra.append(nin_r)
                        nt_ra.append(len(p2b_ev[eid]))
                        p_ra.append(float(np.clip(np.sum(cw[u32_r.ravel()]), 0.02, 0.98)))

            if len(ni_l) > 0:
                st_d = _mh_stats_with_ci(np.array(ni_l), np.array(nt_l), np.array(p_l))
                row_d = {"dt_days": dt_d, "mode": "literal_mjd_shift", **st_d}
            else:
                row_d = {"dt_days": dt_d, "mode": "literal_mjd_shift", "n_plates": 0, "n_in": 0, "n_out": 0, "expected_in": 0.0, "rr_mh": np.nan, "z_within": np.nan}
            shift_rows.append(row_d)
            if abs(dt_d) >= 30 and len(ni_ra) > 0:
                st_r = _mh_stats_with_ci(np.array(ni_ra), np.array(nt_ra), np.array(p_ra))
                shift_rows.append({"dt_days": dt_d, "mode": "ra_coshifted_epoch", **st_r})

    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv("data/processed/phase2b_step4_shifted_dates.csv", index=False)
    print("  [4a. Shifted-Timestamp Curve (m >= 8.0)]:")
    print(shift_df[["dt_days", "mode", "n_plates", "n_in", "expected_in", "rr_mh", "z_within"]].to_string(index=False))

    # (4b) Fake-altitude parallax controls: 42,164 km (GEO) vs. 36,000, 20,000, and 5,000 km
    alt_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r_km in [42164.0, 36000.0, 20000.0, 5000.0]:
            for lbl, sub_set in [("m>=8.0 (N=293 pool)", sh_m8), ("m>=9.0 (N=61 pool)", sh_m9)]:
                ni_a, nt_a, p_a = [], [], []
                for _, r in sub_set.iterrows():
                    eid = str(r["exposure_id"])
                    wcs, nx, ny = p2b_wcs[eid]
                    cw = cw_m8_map[eid]
                    u32, nin = _compute_plate_umbra_and_in_count(
                        wcs, nx, ny, p2b_ev[eid], float(r["mjd_start"]), float(r["exptime_sec"]),
                        float(r["obs_lat_deg"]), float(r["obs_lon_deg"]), float(r["obs_elev_m"]), r_km,
                    )
                    sf_u = float(np.mean(u32[p2b_vm[eid]])) if np.any(p2b_vm[eid]) else 0.0
                    if 0.05 < sf_u < 0.95:
                        ni_a.append(nin)
                        nt_a.append(len(p2b_ev[eid]))
                        p_a.append(float(np.clip(np.sum(cw[u32.ravel()]), 0.02, 0.98)))
                st_a = _mh_stats_with_ci(np.array(ni_a), np.array(nt_a), np.array(p_a)) if len(ni_a) > 0 else {"n_plates": 0, "n_in": 0, "expected_in": 0.0, "rr_mh": np.nan, "z_within": np.nan}
                alt_rows.append({"sample": lbl, "altitude_km": int(r_km), **st_a})
    alt_df = pd.DataFrame(alt_rows)
    alt_df.to_csv("data/processed/phase2b_step4_fake_altitudes.csv", index=False)
    print("\n  [4b. Fake-Altitude Parallax Controls]:")
    print(alt_df[["sample", "altitude_km", "n_plates", "n_in", "expected_in", "rr_mh", "rr_ci95_lo", "rr_ci95_hi", "z_within"]].to_string(index=False))

    # (4c) Matched-LST sunlit midnight control (N = 27 at m >= 9.0, shadow_fraction == 0, |LST - 0h| <= 2h, 5° <= |b_ecl| <= 18°)
    # Combine the 23 Stage B QC-passing Matched-LST plates + the 4 prior sunlit QC plates matching the exact same criteria
    sc_sun = SkyCoord(
        exp_idx.loc[sun_df["exposure_id"], "ra_deg"].to_numpy() * u.deg,
        exp_idx.loc[sun_df["exposure_id"], "dec_deg"].to_numpy() * u.deg,
    ).transform_to(BarycentricTrueEcliptic())
    b_ecl_sun = np.abs(sc_sun.lat.deg)
    utc_sun = (exp_idx.loc[sun_df["exposure_id"], "mjd_mid"].to_numpy() % 1.0) * 24.0
    loc_sol_sun = (utc_sun + exp_idx.loc[sun_df["exposure_id"], "obs_lon_deg"].to_numpy() / 15.0) % 24.0
    dist_mid_sun = np.minimum(loc_sol_sun, 24.0 - loc_sol_sun)
    prior_lst_eids = sun_df.loc[
        (sun_df["m_flash_05s"] >= 9.0) & (dist_mid_sun <= 2.0) & (b_ecl_sun >= 5.0) & (b_ecl_sun <= 18.0),
        "exposure_id",
    ].tolist()

    lst_ni, lst_nt, lst_p = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Evaluate ecliptic-facing half vs. poleward half on each Matched-LST sunlit midnight plate (exact sf ~ 0.50 pseudo-shadow)
        for eid in list(lst_ctrl_p2b["exposure_id"].astype(str)) + [e for e in prior_lst_eids if e not in set(lst_ctrl_p2b["exposure_id"].astype(str))]:
            if eid in p2b_wcs:
                wcs, nx, ny = p2b_wcs[eid]
                ev = p2b_ev[eid]
                vg, lg, ga = p2b_vm[eid], p2b_lg[eid], p2b_gaia[eid]
                r_m = p2b_qc_idx.loc[eid]
                if isinstance(r_m, pd.DataFrame):
                    r_m = r_m.iloc[0]
                inst_s, mlim_s = str(r_m["instrument"]), float(r_m["m_lim_cal"])
            else:
                wcs, nx, ny = sun_wcs[eid]
                ev = sun_ev[eid]
                vg, lg, ga = sun_vm[eid], sun_lg[eid], sun_gaia[eid]
                r_s = sun_df[sun_df["exposure_id"] == eid].iloc[0]
                inst_s, mlim_s = str(r_s["instrument"]), float(r_s["m_lim_cal"])
            cw = _plate_cell_weights_from_sunlit_profiles(inst_s, mlim_s, vg, lg, ga, prior_prof_single, inst_profs_single, True, 0.469)
            gp_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32)
            gxn, gyn = np.meshgrid(gp_pts, gp_pts)
            sc_g = wcs.pixel_to_world(gxn.ravel() * nx, gyn.ravel() * ny).transform_to(BarycentricTrueEcliptic())
            b_abs_g = np.abs(sc_g.lat.deg)
            med_b = float(np.median(b_abs_g[vg.ravel()])) if np.any(vg) else float(np.median(b_abs_g))
            ecl_facing = b_abs_g <= med_b
            p_i = float(np.clip(np.sum(cw[ecl_facing]), 0.02, 0.98))
            if len(ev) > 0:
                sc_e = wcs.pixel_to_world(ev[:, 0] * nx, ev[:, 1] * ny).transform_to(BarycentricTrueEcliptic())
                nin = int(np.sum(np.abs(sc_e.lat.deg) <= med_b))
            else:
                nin = 0
            lst_ni.append(nin)
            lst_nt.append(len(ev))
            lst_p.append(p_i)

    lst_ctrl_stats = _mh_stats_with_ci(np.array(lst_ni), np.array(lst_nt), np.array(lst_p))
    print(f"\n  [4c. Matched-LST Sunlit Midnight Control (N={lst_ctrl_stats['n_plates']}, ecliptic-facing vs poleward half)]:")
    print(f"    {lst_ctrl_stats}")

    # =========================================================================
    # STEP 5: KEY SECONDARY (PAIRED TIER A) & REMAINING PRE-REGISTERED SECONDARIES
    # =========================================================================
    print("\n================ STEP 5: KEY SECONDARY & REMAINING SECONDARIES ================")

    # (5a) Strict Partner-Vetoed Paired Tier A (m >= 8.0, N = 95)
    tA_ni, tA_nt, tA_p, tA_rows = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in sh_paired_df.iterrows():
            ea = str(r["exposure_id"])
            wa, nxa, nya = paired_p2b_wcs[ea]
            cw = _plate_cell_weights_from_sunlit_profiles(
                str(r["instrument"]), float(r["m_lim_cal"]), paired_p2b_vm[ea], paired_p2b_lg[ea],
                paired_p2b_gaia[ea], prior_prof_paired, inst_profs_paired, False, 0.0,
            )
            u32, nin = _compute_plate_umbra_and_in_count(
                wa, nxa, nya, paired_p2b_ev[ea], float(r["mjd_start"]), float(r["exptime_sec"]),
                float(r["obs_lat_deg"]), float(r["obs_lon_deg"]), float(r["obs_elev_m"]), 42164.0,
            )
            p_raw = float(np.sum(cw[u32.ravel()]))
            p_i = float(np.clip(p_raw, 0.02, 0.98))
            nt = len(paired_p2b_ev[ea])
            tA_ni.append(nin)
            tA_nt.append(nt)
            tA_p.append(p_i)
            tA_rows.append({"exposure_id": ea, "partner_id": str(r["partner_id"]), "instrument": str(r["instrument"]), "p_raw": round(p_raw, 4), "p_i": round(p_i, 4), "n_tot": nt, "n_in": nin, "n_out": nt - nin})

    tA_stats = _mh_stats_with_ci(np.array(tA_ni), np.array(tA_nt), np.array(tA_p))
    # Also compute exact 2,000-placebo p-value on the 102-plate sunlit paired pool using realized N=95 umbral masks
    avail_p = {inst: grp["exposure_id"].tolist() for inst, grp in paired_sun_df.groupby("instrument")}
    all_p_eids = paired_sun_df["exposure_id"].tolist()
    cw_psun = {
        str(r["exposure_id"]): _plate_cell_weights_from_sunlit_profiles(
            str(r["instrument"]), float(r["m_lim_cal"]), paired_sun_vm[str(r["exposure_id"])],
            paired_sun_lg[str(r["exposure_id"])], paired_sun_gaia[str(r["exposure_id"])],
            prior_prof_paired, inst_profs_paired, False, 0.0,
        )
        for _, r in paired_sun_df.iterrows()
    }
    gp_pts0 = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32) - 0.5
    gxn0, gyn0 = np.meshgrid(gp_pts0, gp_pts0)
    umb_tA_masks = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in sh_paired_df.iterrows():
            er = exp_idx.loc[str(r["exposure_id"])]
            ra0, dec0 = float(er["ra_deg"]), float(er["dec_deg"])
            fx, fy = float(er["fov_ra_deg"]), float(er["fov_dec_deg"])
            dec_g = np.clip(dec0 + gyn0.ravel() * fy, -89.9, 89.9)
            ra_g = ra0 + (gxn0.ravel() * fx) / np.maximum(np.cos(np.deg2rad(dec_g)), 0.05)
            dhat = np.column_stack([np.cos(np.deg2rad(dec_g)) * np.cos(np.deg2rad(ra_g)), np.cos(np.deg2rad(dec_g)) * np.sin(np.deg2rad(ra_g)), np.sin(np.deg2rad(dec_g))])
            loc = EarthLocation(lat=float(er["obs_lat_deg"]) * u.deg, lon=float(er["obs_lon_deg"]) * u.deg, height=float(er["obs_elev_m"]) * u.m)
            m0, dd = float(er["mjd_start"]), float(er["exptime_sec"]) / 86400.0
            t3 = Time([m0, m0 + 0.5 * dd, m0 + dd], format="mjd", scale="utc")
            ug = points_in_umbra(loc.get_gcrs(t3).cartesian.xyz.to_value(u.km).T, dhat, get_sun(t3).cartesian.xyz.to_value(u.km).T, 42164.0).mean(axis=0) >= 0.5
            umb_tA_masks.append((str(r["instrument"]), ug.reshape(32, 32)))

    rng_tA = np.random.default_rng(20261120)
    null_tA_zs = np.empty(2000, dtype=float)
    for b in range(2000):
        ni_b, nt_b, pa_b = [], [], []
        for inst_i, u_m in umb_tA_masks:
            pool_i = avail_p.get(inst_i, all_p_eids)
            eid_s = str(rng_tA.choice(pool_i))
            flip = bool(rng_tA.random() < 0.5)
            u_use = ~u_m if flip else u_m
            pr_s = float(np.sum(cw_psun[eid_s][u_use.ravel()]))
            if pr_s <= 0.05 or pr_s >= 0.95:
                for _ in range(10):
                    _, alt_u = umb_tA_masks[int(rng_tA.integers(0, len(umb_tA_masks)))]
                    alt_f = bool(rng_tA.random() < 0.5)
                    u_try = ~alt_u if alt_f else alt_u
                    pt_s = float(np.sum(cw_psun[eid_s][u_try.ravel()]))
                    if 0.05 < pt_s < 0.95:
                        u_use, pr_s = u_try, pt_s
                        break
            ev_s = paired_sun_ev[eid_s]
            nt_s = len(ev_s)
            if nt_s > 0:
                ix_s = np.clip((ev_s[:, 0] * 32).astype(int), 0, 31)
                iy_s = np.clip((ev_s[:, 1] * 32).astype(int), 0, 31)
                nin_s = int(np.sum(u_use[iy_s, ix_s]))
            else:
                nin_s = 0
            ni_b.append(nin_s)
            nt_b.append(nt_s)
            pa_b.append(float(np.clip(pr_s, 0.02, 0.98)))
        null_tA_zs[b] = _cmh_stat(np.array(ni_b), np.array(nt_b), np.array(pa_b))[0]

    tA_stats["p_placebo"] = round((int(np.sum(null_tA_zs <= tA_stats["z_within"])) + 1.0) / 2001.0, 5)
    tA_stats["crit_z_050"] = round(float(np.quantile(null_tA_zs, 0.050)), 4)
    gate_c_pass = bool((tA_stats["rr_mh"] < 1.0) and (tA_stats["z_within"] < 0.0))
    pd.DataFrame(tA_rows).to_csv("data/processed/phase2b_paired_tierA_plates.csv", index=False)
    print(f"  [5a. Key Secondary - Strict Paired Tier A (m >= 8.0, N=95)]: {tA_stats} | Gate (c) (RR<1 & Z<0) = {gate_c_pass}")

    # (5b.1) Dose-Response by shadow_fraction & m_flash depth bins on N=293, PLUS Uninformative Full-Shadow (N=47)
    dose_rows = []
    for lbl, mask_d in [
        ("sf in (0.05, 0.35]", pp_m8_gaia["shadow_fraction_archive"] <= 0.35),
        ("sf in (0.35, 0.65]", (pp_m8_gaia["shadow_fraction_archive"] > 0.35) & (pp_m8_gaia["shadow_fraction_archive"] <= 0.65)),
        ("sf in (0.65, 0.95)", pp_m8_gaia["shadow_fraction_archive"] > 0.65),
        ("depth m_flash in [8.0, 8.5)", pp_m8_gaia["m_flash_05s"] < 8.5),
        ("depth m_flash in [8.5, 9.0)", (pp_m8_gaia["m_flash_05s"] >= 8.5) & (pp_m8_gaia["m_flash_05s"] < 9.0)),
        ("depth m_flash in [9.0, 10.0)", (pp_m8_gaia["m_flash_05s"] >= 9.0) & (pp_m8_gaia["m_flash_05s"] < 10.0)),
        ("depth m_flash >= 10.0", pp_m8_gaia["m_flash_05s"] >= 10.0),
    ]:
        sub_d = pp_m8_gaia[mask_d]
        if len(sub_d) > 0:
            dose_rows.append({"stratum": lbl, **_mh_stats_with_ci(sub_d["n_in"].to_numpy(), sub_d["n_tot"].to_numpy(), sub_d["p_i"].to_numpy())})
    dose_df = pd.DataFrame(dose_rows)
    dose_df.to_csv("data/processed/phase2b_step5_dose_response.csv", index=False)
    print("\n  [5b.1 Dose-Response across shadow_fraction & m_flash bins]:")
    print(dose_df[["stratum", "n_plates", "n_in", "expected_in", "rr_mh", "rr_ci95_lo", "rr_ci95_hi", "z_within"]].to_string(index=False))

    # Uninformative Full-Shadow Across-Plate (m >= 9.0, sf >= 0.95, N = 47 vs 222 sunlit m >= 9.0 plates)
    sun_m9_sub = sun_df[sun_df["m_flash_05s"] >= 9.0].copy()
    sun_m9_sub["eff_area"] = np.maximum(
        [float(exp_idx.loc[e, "fov_ra_deg"] * exp_idx.loc[e, "fov_dec_deg"]) * uf for e, uf in zip(sun_m9_sub["exposure_id"], sun_m9_sub["unmasked_frac"])],
        0.1,
    )
    sh_full9["eff_area"] = np.maximum(sh_full9["fov_area_deg2"] * sh_full9["unmasked_frac"], 0.1)
    num_f, var_f, mhn_f, mhd_f = 0.0, 0.0, 0.0, 0.0
    for inst in set(sh_full9["instrument"]):
        g1 = sh_full9[sh_full9["instrument"] == inst]
        g0 = sun_m9_sub[sun_m9_sub["instrument"] == inst]
        if len(g0) == 0:
            continue
        c1, t1 = float(g1["n_events"].sum()), float(g1["eff_area"].sum())
        c0, t0 = float(g0["n_events"].sum()), float(g0["eff_area"].sum())
        ct, tt = c1 + c0, t1 + t0
        if ct <= 0 or tt <= 0:
            continue
        pa_f = t1 / tt
        num_f += c1 - ct * pa_f
        r0 = g0["n_events"].to_numpy() / g0["eff_area"].to_numpy()
        disp_f = max(1.0, float(np.var(g0["n_events"].to_numpy() - np.mean(r0) * g0["eff_area"].to_numpy()) / max(np.mean(g0["n_events"].to_numpy()), 1.0))) if len(g0) >= 3 else 10.0
        var_f += disp_f * ct * pa_f * (1.0 - pa_f)
        mhn_f += c1 * (1.0 - pa_f)
        mhd_f += c0 * pa_f
    full_sh_summary = {
        "label": "PRE-REGISTERED UNINFORMATIVE (~10% power, reported without interpretation)",
        "n_full_shadow_plates": int(len(sh_full9)),
        "n_sunlit_ref_plates": int(len(sun_m9_sub)),
        "full_shadow_events": int(sh_full9["n_events"].sum()),
        "full_shadow_area_deg2": round(float(sh_full9["eff_area"].sum()), 1),
        "rr_mh_across": round(mhn_f / max(mhd_f, 1e-9), 4),
        "z_across": round(num_f / np.sqrt(max(var_f, 1e-9)), 4),
    }
    print(f"\n  [5b.1b Full-Shadow Across-Plate (N=47, UNINFORMATIVE ~10% power)]: {full_sh_summary}")

    # (5b.2) Across-Plate Mixed-Effects Model with (1 | batch_id)
    acr_rows = []
    for _, r in sun_df.iterrows():
        eid = str(r["exposure_id"])
        er = exp_idx.loc[eid]
        acr_rows.append({
            "exposure_id": eid, "plate_id": str(r["plate_id"]), "instrument": str(r["instrument"]),
            "n_events": int(r["n_events"]),
            "eff_area": max(float(er["fov_ra_deg"] * er["fov_dec_deg"]) * float(r["unmasked_frac"]), 0.1),
            "shadow_fraction": 0.0, "ra_deg": float(er["ra_deg"]), "dec_deg": float(er["dec_deg"]),
            "mjd_mid": float(er["mjd_mid"]), "obs_lat_deg": float(er["obs_lat_deg"]), "obs_lon_deg": float(er["obs_lon_deg"]),
            "obs_elev_m": float(er["obs_elev_m"]), "batch_id": batch_map.get(eid, -1),
        })
    for _, r in p2b_qc.iterrows():
        eid = str(r["exposure_id"])
        if str(r["plate_id"]) in {x["plate_id"] for x in acr_rows}:
            continue
        acr_rows.append({
            "exposure_id": eid, "plate_id": str(r["plate_id"]), "instrument": str(r["instrument"]),
            "n_events": int(r["n_events"]),
            "eff_area": max(float(r["fov_area_deg2"]) * float(r["unmasked_frac"]), 0.1),
            "shadow_fraction": float(r["shadow_fraction"]), "ra_deg": float(r["ra_deg"]), "dec_deg": float(r["dec_deg"]),
            "mjd_mid": float(r["mjd_mid"]), "obs_lat_deg": float(r["obs_lat_deg"]), "obs_lon_deg": float(r["obs_lon_deg"]),
            "obs_elev_m": float(r["obs_elev_m"]), "batch_id": int(r["batch_id"]),
        })
    acr_df = pd.DataFrame(acr_rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc_a = SkyCoord(acr_df["ra_deg"].to_numpy() * u.deg, acr_df["dec_deg"].to_numpy() * u.deg)
        acr_df["abs_b_ecl"] = np.abs(sc_a.transform_to(BarycentricTrueEcliptic()).lat.deg)
        t_a = Time(acr_df["mjd_mid"].to_numpy(), format="mjd", scale="utc")
        sun_a = get_sun(t_a)
        acr_df["solar_elong_deg"] = sc_a.separation(SkyCoord(sun_a.ra, sun_a.dec)).deg
        utc_h = (acr_df["mjd_mid"].to_numpy() % 1.0) * 24.0
        loc_h = (utc_h + acr_df["obs_lon_deg"].to_numpy() / 15.0) % 24.0
        acr_df["lst_midnight_hr"] = np.minimum(loc_h, 24.0 - loc_h)
        locs_a = EarthLocation(lat=acr_df["obs_lat_deg"].to_numpy() * u.deg, lon=acr_df["obs_lon_deg"].to_numpy() * u.deg, height=acr_df["obs_elev_m"].to_numpy() * u.m)
        alt_a = sc_a.transform_to(AltAz(obstime=t_a, location=locs_a)).alt.deg
        acr_df["airmass"] = 1.0 / np.clip(np.sin(np.deg2rad(np.clip(alt_a, 10.0, 90.0))), 0.17, 1.0)
        acr_df["log_rate"] = np.log((acr_df["n_events"] + 0.5) / acr_df["eff_area"])

        # 1) Linear mixed model on log((n_events + 0.5)/eff_area) with (1 | batch_id) random intercept
        inst_cnts = acr_df["instrument"].value_counts()
        acr_sub = acr_df[acr_df["instrument"].isin(inst_cnts[inst_cnts >= 5].index)].copy()
        md = smf.mixedlm(
            "log_rate ~ shadow_fraction + C(instrument) + abs_b_ecl + solar_elong_deg + lst_midnight_hr + airmass",
            acr_sub,
            groups=acr_sub["batch_id"],
        ).fit(reml=True)
        # 2) Cluster-robust NB GLM (offset = log(eff_area), clustered by batch_id)
        nb_mod = smf.glm(
            "n_events ~ shadow_fraction + C(instrument) + abs_b_ecl + solar_elong_deg + lst_midnight_hr + airmass",
            data=acr_sub,
            offset=np.log(acr_sub["eff_area"]),
            family=sm.families.NegativeBinomial(alpha=2.0),
        ).fit(cov_type="cluster", cov_kwds={"groups": acr_sub["batch_id"]})

    across_model_summary = {
        "n_plates": int(len(acr_sub)),
        "n_batches": int(acr_sub["batch_id"].nunique()),
        "mixedlm_beta_shadow": round(float(md.params["shadow_fraction"]), 4),
        "mixedlm_se_shadow": round(float(md.bse["shadow_fraction"]), 4),
        "mixedlm_p_two_sided": round(float(md.pvalues["shadow_fraction"]), 4),
        "nb_cluster_beta_shadow": round(float(nb_mod.params["shadow_fraction"]), 4),
        "nb_cluster_se_shadow": round(float(nb_mod.bse["shadow_fraction"]), 4),
        "nb_cluster_rr_shadow": round(float(np.exp(nb_mod.params["shadow_fraction"])), 4),
        "nb_cluster_p_two_sided": round(float(nb_mod.pvalues["shadow_fraction"]), 4),
    }
    print(f"\n  [5b.2 Across-Plate Mixed Model with (1 | batch_id)]: {across_model_summary}")

    # (5b.3) Exploratory Aligned Multi-Flash Tracks (`src/aligned_tracks.py`, 1,000 shuffles/plate)
    # Run on the Strict Paired Tier A N=95 shadow pairs (cleanest non-stellar sample, identical to Pilot track calibration)
    rng_tr = np.random.default_rng(20261121)
    n_shuf = 1000
    null_tr_tot = np.zeros(n_shuf, dtype=int)
    null_tr_unconf = np.zeros(n_shuf, dtype=int)
    obs_tr_list = []
    n_unconf_tA = 0
    n_xy_tA = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in sh_paired_df.iterrows():
            ea, eb = str(r["exposure_id"]), str(r["partner_id"])
            dfa, wa, nxa, nya, mra, mla, ga, _ = _load_any_plate_bundle(ea)
            _, wb, nxb, nyb, mrb, _, _, _ = _load_any_plate_bundle(eb)
            ev_norm = paired_p2b_ev[ea]
            n_raw_here = paired_p2b_nraw[ea]
            xy_arr = np.column_stack([ev_norm[:, 0] * nxa, ev_norm[:, 1] * nya]) if len(ev_norm) > 0 else np.zeros((0, 2))

            theta_geo = predict_geo_drift_pa_deg(
                float(r["ra_deg"]), float(r["dec_deg"]), float(r["mjd_mid"]),
                float(r["obs_lat_deg"]), float(r["obs_lon_deg"]), float(r["obs_elev_m"]),
            )
            pa_x, pa_y = plate_axes_pa_deg(wa, nxa, nya)
            geo_sep = float(min(axial_diff_deg(theta_geo, pa_x), axial_diff_deg(theta_geo, pa_y)))
            is_unc = geo_sep >= MIN_AXIS_SEP_DEG
            if is_unc:
                n_unconf_tA += 1
            if ea in p2b_qc_idx.index:
                r_p2 = p2b_qc_idx.loc[ea]
                if isinstance(r_p2, pd.DataFrame):
                    r_p2 = r_p2.iloc[0]
                if int(r_p2.get("n_scans", 1)) >= 2:
                    n_xy_tA += 1
            if len(xy_arr) < 3 and n_raw_here < 3:
                continue

            fwhm_px = FWHM_FACTOR * mra
            trs = find_aligned_tracks(xy_arr, fwhm_px)
            for ch in trs:
                p0, p1 = xy_arr[ch[0]], xy_arr[ch[-1]]
                sc0, sc1 = wa.pixel_to_world(p0[0], p0[1]), wa.pixel_to_world(p1[0], p1[1])
                pa_t = float(sc0.position_angle(sc1).to_value(u.deg)) % 180.0
                dg = float(axial_diff_deg(pa_t, theta_geo))
                dax = float(min(axial_diff_deg(pa_t, pa_x), axial_diff_deg(pa_t, pa_y)))
                # Check how many track members fall inside GEO umbra
                _, nin_tr = _compute_plate_umbra_and_in_count(
                    wa, nxa, nya, ev_norm[list(ch)], float(r["mjd_start"]), float(r["exptime_sec"]),
                    float(r["obs_lat_deg"]), float(r["obs_lon_deg"]), float(r["obs_elev_m"]), 42164.0,
                )
                obs_tr_list.append({
                    "exposure_id": ea, "n_points": len(ch), "n_in_umbra": nin_tr,
                    "pa_deg": round(pa_t, 2), "theta_geo_deg": round(theta_geo, 2),
                    "geo_axis_sep_deg": round(geo_sep, 2), "unconfounded_ge20": is_unc,
                    "d_geo_deg": round(dg, 2), "d_plate_axis_deg": round(dax, 2),
                })

            bx = max(0.05, 4.0 * mra / max(nxa, 1))
            by = max(0.05, 4.0 * mra / max(nya, 1))
            mag_a = dfa["mag_est"].to_numpy(dtype=float)
            br_i = np.where(np.isfinite(mag_a) & (mag_a < (mla - 3.0)))[0]
            sc_br = SkyCoord(dfa["ra"].to_numpy()[br_i] * u.deg, dfa["dec"].to_numpy()[br_i] * u.deg) if len(br_i) > 0 else None
            r_br = np.clip(20.0 + 8.0 * np.maximum(0.0, (mla - 3.0) - mag_a[br_i]), 20.0, 60.0) if len(br_i) > 0 else np.zeros(0)

            for bi in range(n_shuf):
                sh_xy = sample_plate_shuffle_exact(n_raw_here, nxa, nya, bx, by, wa, wb, nxb, nyb, mrb, ga, sc_br, r_br, rng_tr)
                sh_t = find_aligned_tracks(sh_xy, fwhm_px)
                if sh_t:
                    null_tr_tot[bi] += len(sh_t)
                    if is_unc:
                        null_tr_unconf[bi] += len(sh_t)

    tracks_summary = {
        "n_paired_shadow_plates": int(len(sh_paired_df)),
        "n_unconfounded_ge20": n_unconf_tA,
        "n_dual_xy_scans": n_xy_tA,
        "obs_tracks": len(obs_tr_list),
        "null_tracks_mean": round(float(np.mean(null_tr_tot)), 3),
        "null_tracks_95pct": round(float(np.percentile(null_tr_tot, 95.0)), 1),
        "p_count_one_sided": round(float((1 + np.sum(null_tr_tot >= len(obs_tr_list))) / (n_shuf + 1)), 4),
        "obs_unconfounded_tracks": sum(1 for t in obs_tr_list if t["unconfounded_ge20"]),
        "null_unconfounded_mean": round(float(np.mean(null_tr_unconf)), 3),
        "tracks_detail": obs_tr_list,
    }
    print(f"\n  [5b.3 Exploratory Aligned Multi-Flash Tracks (Paired Tier A N=95, 1,000 shuffles)]: {tracks_summary}")

    # =========================================================================
    # THREE-GATE DECISION RULE VERDICT
    # =========================================================================
    # Check Gate (b) controls:
    #  - (b.1) Shifted dates peak near 0 and decay away from opposition
    #  - (b.2) Fake altitudes behave consistently with high-altitude parallax
    #  - (b.3) Matched-LST sunlit midnight shows no spurious deficit (Z > -1.645)
    ctrl_b3_ok = bool(lst_ctrl_stats["z_within"] > -1.64485)
    if not gate_a_pass:
        verdict = "NULL"
    elif gate_a_pass and ctrl_b3_ok and gate_c_pass:
        verdict = "POSITIVE_CANDIDATE_CHECK_CONTROLS"
    else:
        verdict = "INCONCLUSIVE_OR_ARTEFACT"

    print("\n================================================================================")
    print(f"REGISTERED THREE-GATE VERDICT: {verdict}")
    print(f"  Gate (a) Co-Primary (m>=8 @ 0.025 -> m>=9 @ 0.05): {gate_a_pass} (p_m8={coprim1_gaia['p_placebo']}, p_m9={coprim2_gaia['p_placebo']})")
    print(f"  Gate (b.3) Matched-LST Sunlit Midnight No Deficit: {ctrl_b3_ok} (Z={lst_ctrl_stats['z_within']}, RR={lst_ctrl_stats['rr_mh']})")
    print(f"  Gate (c) Paired Tier A Directional Concordance:    {gate_c_pass} (RR={tA_stats['rr_mh']}, Z={tA_stats['z_within']})")
    print("================================================================================")

    # =========================================================================
    # POST-HOC EXPLORATORY ROBUSTNESS CHECK: SERIAL-NUMBER DATE MONOTONICITY
    # =========================================================================
    # Flag any DASCH/APPLAUSE exposure in `exposures.parquet` whose `mjd_mid` deviates by > 90 days from the
    # running median MJD of its +-10 serial-number neighbors within the same (source_archive, series)!
    ser_tmp = exp[["exposure_id", "source_archive", "plate_id", "mjd_mid"]].copy()
    s_codes, s_nums = [], []
    for sa, pid in zip(ser_tmp["source_archive"].astype(str), ser_tmp["plate_id"].astype(str)):
        raw = pid.replace("DASCH_", "").replace("APPLAUSE_", "")
        m_sn = re.match(r"^([A-Za-z]+)(\d+)$", raw)
        if sa == "DASCH" and m_sn:
            s_codes.append(m_sn.group(1))
            s_nums.append(int(m_sn.group(2)))
        else:
            s_codes.append(sa)
            m_d = re.search(r"(\d+)", raw)
            s_nums.append(int(m_d.group(1)) if m_d else -1)
    ser_tmp["series"] = s_codes
    ser_tmp["serial"] = s_nums

    flagged_serial_eids: set[str] = set()
    flagged_details = []
    target_check_eids = set(sh_m8["exposure_id"].astype(str)).union(set(sh_paired_df["exposure_id"].astype(str)))
    for (sa, ser), grp in ser_tmp[ser_tmp["source_archive"] == "DASCH"].groupby(["source_archive", "series"]):
        if len(grp) < 15:
            continue
        g_s = grp.sort_values("serial").reset_index(drop=True)
        mjd_arr = g_s["mjd_mid"].to_numpy(dtype=float)
        ser_arr = g_s["serial"].to_numpy(dtype=int)
        eid_arr = g_s["exposure_id"].astype(str).to_numpy()
        for idx_i in range(len(g_s)):
            if eid_arr[idx_i] not in target_check_eids:
                continue
            lo = max(0, idx_i - 10)
            hi = min(len(g_s), idx_i + 11)
            nbr_mask = np.arange(lo, hi) != idx_i
            nbr_ser = ser_arr[lo:hi][nbr_mask]
            nbr_mjd = mjd_arr[lo:hi][nbr_mask]
            close_ser = np.abs(nbr_ser - ser_arr[idx_i]) <= 25
            if np.sum(close_ser) >= 4:
                med_nbr_mjd = float(np.median(nbr_mjd[close_ser]))
                diff_d = float(mjd_arr[idx_i] - med_nbr_mjd)
                if abs(diff_d) > 90.0:
                    flagged_serial_eids.add(eid_arr[idx_i])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        iso_obs = Time(float(mjd_arr[idx_i]), format="mjd", scale="utc").iso[:10]
                        iso_nbr = Time(med_nbr_mjd, format="mjd", scale="utc").iso[:10]
                    flagged_details.append({
                        "exposure_id": eid_arr[idx_i], "series": ser, "serial": int(ser_arr[idx_i]),
                        "obs_date": iso_obs, "neighbor_median_date": iso_nbr, "offset_days": round(diff_d, 1),
                    })

    pp_m8_clean_ser = pp_m8_gaia[~pp_m8_gaia["exposure_id"].isin(flagged_serial_eids)]
    pp_m9_clean_ser = pp_m9_gaia[~pp_m9_gaia["exposure_id"].isin(flagged_serial_eids)]
    ser_m8_stats = _mh_stats_with_ci(pp_m8_clean_ser["n_in"].to_numpy(), pp_m8_clean_ser["n_tot"].to_numpy(), pp_m8_clean_ser["p_i"].to_numpy())
    ser_m9_stats = _mh_stats_with_ci(pp_m9_clean_ser["n_in"].to_numpy(), pp_m9_clean_ser["n_tot"].to_numpy(), pp_m9_clean_ser["p_i"].to_numpy())
    serial_check_summary = {
        "label": "EXPLORATORY POST-HOC ROBUSTNESS CHECK (|MJD - neighbor_median_MJD| > 90 d within +-25 serials)",
        "n_flagged_in_m8_293": int(pp_m8_gaia["exposure_id"].isin(flagged_serial_eids).sum()),
        "n_flagged_in_m9_61": int(pp_m9_gaia["exposure_id"].isin(flagged_serial_eids).sum()),
        "flagged_plates": flagged_details,
        "m8_after_dropping_flagged": ser_m8_stats,
        "m9_after_dropping_flagged": ser_m9_stats,
    }
    print(f"\n  [EXPLORATORY ROBUSTNESS CHECK - Serial-Number Date Monotonicity]: {serial_check_summary}")

    full_results = {
        "commit_hash": commit_hash,
        "step2_composition": step2_comp,
        "step3_coprimary_m8_gaia": coprim1_gaia,
        "step3_coprimary_m8_nogaia": coprim1_nogaia,
        "step3_coprimary_m9_gaia": coprim2_gaia,
        "step3_coprimary_m9_nogaia": coprim2_nogaia,
        "step4_matched_lst_ctrl": lst_ctrl_stats,
        "step5a_paired_tierA": tA_stats,
        "step5b_full_shadow_uninformative": full_sh_summary,
        "step5b_across_plate_mixed_model": across_model_summary,
        "step5b_aligned_tracks": tracks_summary,
        "verdict": verdict,
        "exploratory_serial_order_check": serial_check_summary,
    }
    Path("data/processed/phase2b_unblinded_results.json").write_text(json.dumps(full_results, indent=2))
    print("\nSaved complete unblinded summary to data/processed/phase2b_unblinded_results.json")
    return full_results


if __name__ == "__main__":
    run_phase2b_unblinding()
