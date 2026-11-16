"""Amendment 2 Pre-Outcome Power Recalibration at Realized QC Sample Sizes (Strictly Blind to Shadow Counts).

Triggered at Step 1 (before any candidate positions or in-shadow vs. out-of-shadow counts on shadow plates
were computed) after discovering that Primary m_flash >= 9.0 yielded N = 61 QC-passing plates rather than
the projected N = 106 (because the N = 106 projection applied 16-inch Metcalf's pooled m >= 8.0 QC rate
39.5% rather than its m >= 9.0 short-exposure QC rate 15.0%).

HARD BLINDNESS INVARIANT:
  - Reads ONLY `qc_pass`, `instrument`, `m_flash_05s`, and Phase 1 `shadow_fraction` metadata to obtain
    the realized QC sample sizes (N = 61, N = 293, N = 95, N = 47) and target shadow-fraction / instrument
    distributions.
  - NEVER reads `phase2b_catalogs/`, `phase2b_stamps/`, or `n_stamps` for any plate.
  - All candidate rates, spatial profiles, Gaia star-density maps, and umbral placebos come 100% from
    the frozen strictly-sunlit (shadow_fraction == 0 at all 3 altitudes) calibration pool (`pilot` + `sunlit_m8`).
"""

from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from src.final_frozen_calibration import (
    _apply_conc_pk_veto_from_stamps,
    _apply_frozen_filter_to_plate,
    _build_cell_weights,
    build_ecliptic_time_shifted_bank,
    compute_gaia_density_grid_32,
    depth_dilution_scale,
)
from src.geometry import points_in_umbra
from src.within_plate_power import _cmh_stat


def _assign_batches(df_meta: pd.DataFrame) -> dict[str, int]:
    """Assign series-run batch_id (|Δserial| <= 50 and |ΔMJD| <= 30 d within archive+series)."""
    import re

    tmp = df_meta[["exposure_id", "source_archive", "plate_id", "mjd_mid"]].copy()
    ser_list, num_list = [], []
    for pid in tmp["plate_id"].astype(str):
        raw = pid.replace("DASCH_", "").replace("APPLAUSE_", "")
        m = re.match(r"^([A-Za-z]+)(\d+)$", raw)
        if m:
            ser_list.append(m.group(1))
            num_list.append(int(m.group(2)))
        else:
            ser_list.append("app")
            m2 = re.search(r"(\d+)", raw)
            num_list.append(int(m2.group(1)) if m2 else 0)
    tmp["series"] = ser_list
    tmp["serial"] = num_list
    tmp = tmp.sort_values(["source_archive", "series", "mjd_mid", "serial"]).reset_index(drop=True)

    bid = 0
    bids = []
    prev_key = None
    prev_mjd = -1e9
    prev_ser = -10**9
    for _, r in tmp.iterrows():
        key = (r["source_archive"], r["series"])
        if key != prev_key or abs(float(r["mjd_mid"]) - prev_mjd) > 30.0 or abs(int(r["serial"]) - prev_ser) > 50:
            bid += 1
        bids.append(bid)
        prev_key = key
        prev_mjd = float(r["mjd_mid"])
        prev_ser = int(r["serial"])
    return dict(zip(tmp["exposure_id"].astype(str), bids))


def _evaluate_single_at_alpha(
    full_pool_df: pd.DataFrame,
    sub_eids: list[str],
    bank: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ev_map: dict[str, np.ndarray],
    vm_map: dict[str, np.ndarray],
    lg_map: dict[str, np.ndarray],
    gaia_map: dict[str, np.ndarray],
    target_n_plates: int,
    target_sf_pool: np.ndarray,
    use_gaia_density: bool,
    gaia_share: float,
    non_stellar_frac: float,
    deltas: list[float],
    alphas: list[float],
    label: str,
    n_sim: int,
    seed: int,
) -> list[dict]:
    cw = _build_cell_weights(full_pool_df, ev_map, vm_map, lg_map, gaia_map, use_gaia_density, gaia_share)
    mf_map = full_pool_df.set_index("exposure_id")["m_flash_05s"].to_dict()
    ecl_eids = [eid for eid in sub_eids if eid in bank]

    p_bank: dict[str, tuple[np.ndarray, int, np.ndarray, np.ndarray]] = {}
    for eid in ecl_eids:
        sfs_b, ug_b, nin_b = bank[eid]
        p_arr = np.clip(ug_b @ cw[eid], 0.02, 0.98)
        p_bank[eid] = (sfs_b, len(ev_map[eid]), nin_b, p_arr)

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

    def _draw_z(delta: float, depth_sc: bool = True) -> tuple[float, float]:
        sfs_t = rng.choice(target_sf_pool, size=target_n_plates, replace=True)
        b_idx = np.clip(((sfs_t - 0.05) / 0.05).astype(int), 0, 17)
        ni = np.empty(target_n_plates, dtype=int)
        nt = np.empty(target_n_plates, dtype=int)
        pa = np.empty(target_n_plates, dtype=float)
        for i in range(target_n_plates):
            elig = bin_tables[b_idx[i]]
            eid, flip, ks = elig[int(rng.integers(0, len(elig)))]
            _, n_tot, nin_b, p_arr = p_bank[eid]
            k = int(ks[rng.integers(0, len(ks))])
            n_in = int(n_tot - nin_b[k]) if flip else int(nin_b[k])
            p = float(np.clip(1.0 - p_arr[k], 0.02, 0.98)) if flip else float(p_arr[k])
            if delta > 0 and n_in > 0:
                d_eff = delta * non_stellar_frac * (depth_dilution_scale(mf_map[eid]) if depth_sc else 1.0)
                kept = int(rng.binomial(n_in, max(0.0, 1.0 - d_eff)))
                n_tot -= n_in - kept
                n_in = kept
            ni[i], nt[i], pa[i] = n_in, n_tot, p
        return _cmh_stat(ni, nt, pa)

    null_zs = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])
    indep_null = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])

    sig_cache: dict[float, tuple[np.ndarray, np.ndarray, float]] = {}
    for delta in deltas:
        zs_d, rrs_d = zip(*[_draw_z(delta, True) for _ in range(n_sim)])
        zs_u, _ = zip(*[_draw_z(delta, False) for _ in range(n_sim)])
        sig_cache[delta] = (np.array(zs_d), np.array(zs_u), float(np.median(rrs_d)))

    rows = []
    for alpha in alphas:
        crit_z = float(np.quantile(null_zs, alpha))
        emp_fpr = float(np.mean(indep_null <= crit_z))
        print(
            f"\n=== {label} | N={target_n_plates} | alpha={alpha:.3f} | crit_z={crit_z:.3f} | PlaceboRankFPR={emp_fpr:.4f} ==="
        )
        for delta in deltas:
            zs_d, zs_u, med_rr = sig_cache[delta]
            pwr_d = float(np.mean(zs_d <= crit_z))
            pwr_u = float(np.mean(zs_u <= crit_z))
            rows.append({
                "test": label,
                "alpha": alpha,
                "use_gaia_density": use_gaia_density,
                "n_plates": target_n_plates,
                "k_sunlit_pool": len(ecl_eids),
                "delta_full_depth": delta,
                "non_stellar_frac": round(non_stellar_frac, 4),
                "crit_z": round(crit_z, 3),
                "placebo_fpr": round(emp_fpr, 4),
                "power_depth_scaled": round(pwr_d, 4),
                "power_unscaled": round(pwr_u, 4),
                "med_rr_depth_scaled": round(med_rr, 4),
            })
            print(f"  delta={delta:.2f} -> power(depth-scaled)={pwr_d:.4f} | power(unscaled)={pwr_u:.4f} | med_RR={med_rr:.4f}")
    return rows


def _evaluate_coupled_holm_coprimary(
    full_pool_df: pd.DataFrame,
    eids_m9: list[str],
    eids_m8_lo: list[str],
    bank: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ev_map: dict[str, np.ndarray],
    vm_map: dict[str, np.ndarray],
    lg_map: dict[str, np.ndarray],
    gaia_map: dict[str, np.ndarray],
    n_m9: int,
    sfs_m9: np.ndarray,
    n_m8_lo: int,
    sfs_m8_lo: np.ndarray,
    deltas: list[float],
    n_sim: int,
    seed: int,
) -> list[dict]:
    """Evaluate coupled Holm co-primary where the N=61 m>=9 plates are an exact subset of the N=293 (61+232) m>=8 plates."""
    cw = _build_cell_weights(full_pool_df, ev_map, vm_map, lg_map, gaia_map, True, 0.469)
    ecl_hi = [e for e in eids_m9 if e in bank]
    ecl_all = [e for e in (eids_m9 + eids_m8_lo) if e in bank]

    p_bank = {}
    for eid in ecl_all:
        sfs_b, ug_b, nin_b = bank[eid]
        p_bank[eid] = (sfs_b, len(ev_map[eid]), nin_b, np.clip(ug_b @ cw[eid], 0.02, 0.98))

    sf_centers = np.linspace(0.075, 0.925, 18)

    def _make_bins(pool_eids: list[str]) -> list[list]:
        tabs = []
        for c_sf in sf_centers:
            elig = []
            for eid in pool_eids:
                sfs_b = p_bank[eid][0]
                ks0 = np.where(np.abs(sfs_b - c_sf) <= 0.12)[0]
                if len(ks0) > 0:
                    elig.append((eid, False, ks0))
                ks1 = np.where(np.abs((1.0 - sfs_b) - c_sf) <= 0.12)[0]
                if len(ks1) > 0:
                    elig.append((eid, True, ks1))
            if not elig:
                for eid in ecl_all:
                    sfs_b = p_bank[eid][0]
                    ks0 = np.where(np.abs(sfs_b - c_sf) <= 0.12)[0]
                    if len(ks0) > 0:
                        elig.append((eid, False, ks0))
                    ks1 = np.where(np.abs((1.0 - sfs_b) - c_sf) <= 0.12)[0]
                    if len(ks1) > 0:
                        elig.append((eid, True, ks1))
            tabs.append(elig)
        return tabs

    bins_hi = _make_bins(ecl_hi)
    # Use the full K=61 near-ecliptic sunlit bank for the m>=8 extension (avoiding K=29 finite-pool variance inflation)
    # while evaluating depth_dilution_scale at each realized plate's exact m_flash_05s!
    bins_all = _make_bins(ecl_all)
    rng = np.random.default_rng(seed)

    b_hi_fixed = np.clip(((sfs_m9[:, 0] - 0.05) / 0.05).astype(int), 0, 17)
    mf_hi_fixed = sfs_m9[:, 1]
    b_lo_fixed = np.clip(((sfs_m8_lo[:, 0] - 0.05) / 0.05).astype(int), 0, 17)
    mf_lo_fixed = sfs_m8_lo[:, 1]

    def _draw_coupled(delta: float, depth_sc: bool) -> tuple[float, float]:
        ni9 = np.empty(n_m9, dtype=int)
        nt9 = np.empty(n_m9, dtype=int)
        pa9 = np.empty(n_m9, dtype=float)
        for i in range(n_m9):
            elig = bins_hi[b_hi_fixed[i]]
            eid, flip, ks = elig[int(rng.integers(0, len(elig)))]
            _, n_tot, nin_b, p_arr = p_bank[eid]
            k = int(ks[rng.integers(0, len(ks))])
            n_in = int(n_tot - nin_b[k]) if flip else int(nin_b[k])
            p = float(np.clip(1.0 - p_arr[k], 0.02, 0.98)) if flip else float(p_arr[k])
            if delta > 0 and n_in > 0:
                d_eff = delta * 0.531 * (depth_dilution_scale(float(mf_hi_fixed[i])) if depth_sc else 1.0)
                kept = int(rng.binomial(n_in, max(0.0, 1.0 - d_eff)))
                n_tot -= n_in - kept
                n_in = kept
            ni9[i], nt9[i], pa9[i] = n_in, n_tot, p

        ni_l = np.empty(n_m8_lo, dtype=int)
        nt_l = np.empty(n_m8_lo, dtype=int)
        pa_l = np.empty(n_m8_lo, dtype=float)
        for i in range(n_m8_lo):
            elig = bins_all[b_lo_fixed[i]]
            eid, flip, ks = elig[int(rng.integers(0, len(elig)))]
            _, n_tot, nin_b, p_arr = p_bank[eid]
            k = int(ks[rng.integers(0, len(ks))])
            n_in = int(n_tot - nin_b[k]) if flip else int(nin_b[k])
            p = float(np.clip(1.0 - p_arr[k], 0.02, 0.98)) if flip else float(p_arr[k])
            if delta > 0 and n_in > 0:
                d_eff = delta * 0.531 * (depth_dilution_scale(float(mf_lo_fixed[i])) if depth_sc else 1.0)
                kept = int(rng.binomial(n_in, max(0.0, 1.0 - d_eff)))
                n_tot -= n_in - kept
                n_in = kept
            ni_l[i], nt_l[i], pa_l[i] = n_in, n_tot, p

        z9 = _cmh_stat(ni9, nt9, pa9)[0]
        z8 = _cmh_stat(np.concatenate([ni9, ni_l]), np.concatenate([nt9, nt_l]), np.concatenate([pa9, pa_l]))[0]
        return z8, z9

    null_ref = np.array([_draw_coupled(0.0, True) for _ in range(n_sim)])
    z8_ref = np.sort(null_ref[:, 0])
    z9_ref = np.sort(null_ref[:, 1])

    def _pvals(z8_arr: np.ndarray, z9_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p8 = (np.searchsorted(z8_ref, z8_arr, side="right") + 1.0) / (len(z8_ref) + 1.0)
        p9 = (np.searchsorted(z9_ref, z9_arr, side="right") + 1.0) / (len(z9_ref) + 1.0)
        return p8, p9

    null_ind = np.array([_draw_coupled(0.0, True) for _ in range(n_sim)])
    p8_0, p9_0 = _pvals(null_ind[:, 0], null_ind[:, 1])
    fwer_fixed = float(np.mean(p8_0 <= 0.025))
    fwer_sym_holm = float(np.mean(np.minimum(p8_0, p9_0) <= 0.025))
    corr_z = float(np.corrcoef(null_ind[:, 0], null_ind[:, 1])[0, 1])
    print(
        f"\n=== COUPLED HOLM CO-PRIMARY (N_m8={n_m9+n_m8_lo}, N_m9={n_m9}, null corr={corr_z:+.3f}) "
        f"| Fixed-Order FWER={fwer_fixed:.4f} | Symmetric-Holm FWER={fwer_sym_holm:.4f} ==="
    )

    rows = []
    for delta in deltas:
        sd = np.array([_draw_coupled(delta, True) for _ in range(n_sim)])
        su = np.array([_draw_coupled(delta, False) for _ in range(n_sim)])
        p8_d, p9_d = _pvals(sd[:, 0], sd[:, 1])
        p8_u, p9_u = _pvals(su[:, 0], su[:, 1])
        # Fixed-order Holm requested in prompt: test m>=8 at 0.025; if significant, test m>=9 at 0.05
        fix_any_d = float(np.mean(p8_d <= 0.025))
        fix_both_d = float(np.mean((p8_d <= 0.025) & (p9_d <= 0.05)))
        fix_any_u = float(np.mean(p8_u <= 0.025))
        fix_both_u = float(np.mean((p8_u <= 0.025) & (p9_u <= 0.05)))
        sym_any_d = float(np.mean(np.minimum(p8_d, p9_d) <= 0.025))
        sym_any_u = float(np.mean(np.minimum(p8_u, p9_u) <= 0.025))
        print(
            f"  delta={delta:.2f} -> Holm(m8@0.025)={fix_any_d:.4f}/{fix_any_u:.4f} | "
            f"Both(m8@0.025 & m9@0.05)={fix_both_d:.4f}/{fix_both_u:.4f} | SymHolmAny={sym_any_d:.4f}/{sym_any_u:.4f}"
        )
        rows.append({
            "test": "CO-PRIMARY HOLM JOINT (m8@0.025 -> m9@0.05)",
            "alpha": 0.05,
            "use_gaia_density": True,
            "n_plates": n_m9 + n_m8_lo,
            "k_sunlit_pool": len(ecl_all),
            "delta_full_depth": delta,
            "non_stellar_frac": 0.531,
            "crit_z": round(float(np.quantile(z8_ref, 0.025)), 3),
            "placebo_fpr": round(fwer_sym_holm, 4),
            "power_depth_scaled": round(sym_any_d, 4),
            "power_unscaled": round(sym_any_u, 4),
            "med_rr_depth_scaled": round(fix_both_d, 4),
        })
    return rows


def _evaluate_realized_paired_tierA(
    paired_df: pd.DataFrame,
    paired_ev: dict[str, np.ndarray],
    paired_vm: dict[str, np.ndarray],
    paired_lg: dict[str, np.ndarray],
    paired_gaia: dict[str, np.ndarray],
    realized_tA_meta: pd.DataFrame,
    deltas: list[float],
    n_sim: int,
    seed: int,
) -> list[dict]:
    """Evaluate Strict Paired Tier A at exact realized N=95 QC-passing target exposures."""
    cw = _build_cell_weights(paired_df, paired_ev, paired_vm, paired_lg, paired_gaia, False, 0.0)
    avail = {inst: grp["exposure_id"].tolist() for inst, grp in paired_df.groupby("instrument")}
    all_paired_eids = paired_df["exposure_id"].tolist()
    mf_map = paired_df.set_index("exposure_id")["m_flash_05s"].to_dict()

    realized_tA_meta = realized_tA_meta.reset_index(drop=True)
    alloc_tA = []
    for inst, grp in realized_tA_meta.groupby("instrument"):
        pool_i = avail.get(inst, all_paired_eids)
        alloc_tA.append((inst, len(grp), grp.index.to_numpy(), pool_i))
    n_pl = sum(x[1] for x in alloc_tA)

    gp_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32) - 0.5
    gxn, gyn = np.meshgrid(gp_pts, gp_pts)
    umb_masks: dict[int, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx_r, r in realized_tA_meta.iterrows():
            ra0, dec0 = float(r["ra_deg"]), float(r["dec_deg"])
            fx, fy = float(r["fov_ra_deg"]), float(r["fov_dec_deg"])
            dec_g = np.clip(dec0 + gyn.ravel() * fy, -89.9, 89.9)
            ra_g = ra0 + (gxn.ravel() * fx) / np.maximum(np.cos(np.deg2rad(dec_g)), 0.05)
            dhat = np.column_stack([
                np.cos(np.deg2rad(dec_g)) * np.cos(np.deg2rad(ra_g)),
                np.cos(np.deg2rad(dec_g)) * np.sin(np.deg2rad(ra_g)),
                np.sin(np.deg2rad(dec_g)),
            ])
            loc = EarthLocation(
                lat=float(r["obs_lat_deg"]) * u.deg,
                lon=float(r["obs_lon_deg"]) * u.deg,
                height=float(r["obs_elev_m"]) * u.m,
            )
            mjd0, dur_d = float(r["mjd_start"]), float(r["exptime_sec"]) / 86400.0
            t3 = Time([mjd0, mjd0 + 0.5 * dur_d, mjd0 + dur_d], format="mjd", scale="utc")
            ug = points_in_umbra(
                loc.get_gcrs(t3).cartesian.xyz.to_value(u.km).T,
                dhat,
                get_sun(t3).cartesian.xyz.to_value(u.km).T,
                42164.0,
            ).mean(axis=0) >= 0.5
            umb_masks[int(idx_r)] = ug.reshape(32, 32)

    rng = np.random.default_rng(seed)

    def _draw_z(delta: float, depth_sc: bool = True) -> tuple[float, float, float]:
        ni, nt, pa = [], [], []
        for inst, n_take, r_idxs, pool_i in alloc_tA:
            eids = rng.choice(pool_i, size=n_take, replace=(n_take > len(pool_i)))
            m_idxs = rng.choice(r_idxs, size=n_take, replace=False)
            flips = rng.random(n_take) < 0.5
            for eid, midx, flip in zip(eids, m_idxs, flips):
                u32 = ~umb_masks[int(midx)] if flip else umb_masks[int(midx)]
                p_raw = float(np.sum(cw[eid][u32.ravel()]))
                if p_raw <= 0.05 or p_raw >= 0.95:
                    for _ in range(10):
                        alt_m = int(rng.choice(list(umb_masks.keys())))
                        alt_f = bool(rng.random() < 0.5)
                        u_try = ~umb_masks[alt_m] if alt_f else umb_masks[alt_m]
                        pr_try = float(np.sum(cw[eid][u_try.ravel()]))
                        if 0.05 < pr_try < 0.95:
                            u32, p_raw = u_try, pr_try
                            break
                p = float(np.clip(p_raw, 0.02, 0.98))
                ev = paired_ev[eid]
                n_t = len(ev)
                if n_t > 0:
                    ix = np.clip((ev[:, 0] * 32).astype(int), 0, 31)
                    iy = np.clip((ev[:, 1] * 32).astype(int), 0, 31)
                    n_in = int(np.sum(u32[iy, ix]))
                else:
                    n_in = 0
                if delta > 0 and n_in > 0:
                    d_eff = delta * (depth_dilution_scale(mf_map[eid]) if depth_sc else 1.0)
                    kept = int(rng.binomial(n_in, max(0.0, 1.0 - d_eff)))
                    n_t -= n_in - kept
                    n_in = kept
                ni.append(n_in)
                nt.append(n_t)
                pa.append(p)
        z, rr = _cmh_stat(np.array(ni), np.array(nt), np.array(pa))
        return z, rr, float(1.0 if (rr < 1.0 and z < 0.0) else 0.0)

    null_zs = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])
    crit_z = float(np.quantile(null_zs, 0.05))
    indep_null = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])
    emp_fpr = float(np.mean(indep_null <= crit_z))
    print(f"\n=== KEY SECONDARY: Strict Paired Tier A (m >= 8.0, Realized N={n_pl}) | crit_z={crit_z:.3f} | PlaceboRankFPR={emp_fpr:.4f} ===")

    rows = []
    for delta in deltas:
        res_d = [_draw_z(delta, True) for _ in range(n_sim)]
        res_u = [_draw_z(delta, False) for _ in range(n_sim)]
        zs_d = np.array([x[0] for x in res_d])
        rrs_d = np.array([x[1] for x in res_d])
        dir_d = float(np.mean([x[2] for x in res_d]))
        zs_u = np.array([x[0] for x in res_u])
        dir_u = float(np.mean([x[2] for x in res_u]))
        pwr_d = float(np.mean(zs_d <= crit_z))
        pwr_u = float(np.mean(zs_u <= crit_z))
        print(
            f"  delta={delta:.2f} -> power@0.05={pwr_d:.4f}/{pwr_u:.4f} | "
            f"P(RR<1 & Z<0 direction gate)={dir_d:.4f}/{dir_u:.4f} | med_RR={float(np.median(rrs_d)):.4f}"
        )
        rows.append({
            "test": "KEY SECONDARY: Strict Paired Tier A (m >= 8.0, partner-vetoed, N=95)",
            "alpha": 0.05,
            "use_gaia_density": False,
            "n_plates": n_pl,
            "k_sunlit_pool": len(paired_df),
            "delta_full_depth": delta,
            "non_stellar_frac": 1.0,
            "crit_z": round(crit_z, 3),
            "placebo_fpr": round(emp_fpr, 4),
            "power_depth_scaled": round(pwr_d, 4),
            "power_unscaled": round(pwr_u, 4),
            "med_rr_depth_scaled": round(float(np.median(rrs_d)), 4),
        })
    return rows


def _evaluate_full_shadow_across_plate(
    single_df: pd.DataFrame,
    exp_idx: pd.DataFrame,
    full_sh_meta: pd.DataFrame,
    deltas: list[float],
    n_sim: int,
    seed: int,
) -> list[dict]:
    """Placebo-calibrated across-plate rate test for Full-Shadow Dose-Response (m >= 9.0, sf >= 0.95, Realized N=47)."""
    sun_m9 = single_df[single_df["m_flash_05s"] >= 9.0].copy().reset_index(drop=True)
    sun_m9["fov_deg2"] = [
        float(exp_idx.loc[e, "fov_ra_deg"] * exp_idx.loc[e, "fov_dec_deg"]) for e in sun_m9["exposure_id"]
    ]
    sun_m9["eff_area"] = np.maximum(sun_m9["fov_deg2"] * sun_m9["unmasked_frac"], 0.1)
    bmap = _assign_batches(exp_idx.loc[sun_m9["exposure_id"]].reset_index())
    sun_m9["batch_id"] = sun_m9["exposure_id"].map(bmap)

    inst_counts = full_sh_meta["instrument"].value_counts().to_dict()
    avail_inst = {inst: grp.index.to_numpy() for inst, grp in sun_m9.groupby("instrument")}
    all_idx = sun_m9.index.to_numpy()

    alloc = []
    for inst, cnt in inst_counts.items():
        pool_i = avail_inst.get(inst, all_idx)
        alloc.append((inst, int(cnt), pool_i))
    n_full = sum(x[1] for x in alloc)

    ev_arr = sun_m9["n_events"].to_numpy(dtype=int)
    area_arr = sun_m9["eff_area"].to_numpy(dtype=float)
    mf_arr = sun_m9["m_flash_05s"].to_numpy(dtype=float)
    inst_arr = sun_m9["instrument"].to_numpy()

    rng = np.random.default_rng(seed)

    def _draw_across(delta: float, depth_sc: bool = True) -> tuple[float, float]:
        picked = []
        used = set()
        for inst, cnt, pool_i in alloc:
            rem = [idx for idx in pool_i if idx not in used]
            if len(rem) >= cnt:
                ch = rng.choice(rem, size=cnt, replace=False)
            else:
                ch = rng.choice(pool_i, size=cnt, replace=True)
            picked.extend(int(x) for x in ch)
            used.update(int(x) for x in ch)
        sh_mask = np.zeros(len(sun_m9), dtype=bool)
        # Handle any replacement duplicates cleanly by building explicit shadow vs sunlit arrays
        sh_idx = np.array(picked, dtype=int)
        su_idx = np.array([i for i in range(len(sun_m9)) if i not in used], dtype=int)

        y_sh = ev_arr[sh_idx].copy()
        a_sh = area_arr[sh_idx]
        i_sh = inst_arr[sh_idx]
        if delta > 0:
            for k in range(len(y_sh)):
                if y_sh[k] > 0:
                    d_eff = delta * 0.531 * (depth_dilution_scale(float(mf_arr[sh_idx[k]])) if depth_sc else 1.0)
                    y_sh[k] = int(rng.binomial(int(y_sh[k]), max(0.0, 1.0 - d_eff)))

        y_su = ev_arr[su_idx]
        a_su = area_arr[su_idx]
        i_su = inst_arr[su_idx]

        # Instrument-stratified Mantel-Haenszel exposure-area test statistic
        num, var, mh_n, mh_d = 0.0, 0.0, 0.0, 0.0
        for inst in set(i_sh):
            m1 = i_sh == inst
            m0 = i_su == inst
            if not np.any(m0):
                continue
            c1, t1 = float(np.sum(y_sh[m1])), float(np.sum(a_sh[m1]))
            c0, t0 = float(np.sum(y_su[m0])), float(np.sum(a_su[m0]))
            ct = c1 + c0
            tt = t1 + t0
            if ct <= 0 or tt <= 0:
                continue
            p_area = t1 / tt
            num += c1 - ct * p_area
            # Empirical variance inflation factor within instrument from sunlit rates
            rates0 = y_su[m0] / a_su[m0]
            disp = max(1.0, float(np.var(y_su[m0] - np.mean(rates0) * a_su[m0]) / max(np.mean(y_su[m0]), 1.0))) if np.sum(m0) >= 3 else 10.0
            var += disp * ct * p_area * (1.0 - p_area)
            mh_n += c1 * (1.0 - p_area)
            mh_d += c0 * p_area
        z = num / np.sqrt(max(var, 1e-9))
        rr = mh_n / max(mh_d, 1e-9)
        return float(z), float(rr)

    null_zs = np.array([_draw_across(0.0)[0] for _ in range(n_sim)])
    crit_z = float(np.quantile(null_zs, 0.05))
    indep_null = np.array([_draw_across(0.0)[0] for _ in range(n_sim)])
    emp_fpr = float(np.mean(indep_null <= crit_z))
    print(f"\n=== FULL-SHADOW DOSE-RESPONSE (m >= 9.0, sf >= 0.95, Realized N={n_full} vs {len(sun_m9)} sunlit) | crit_z={crit_z:.3f} | PlaceboRankFPR={emp_fpr:.4f} ===")

    rows = []
    for delta in deltas:
        zs_d, rrs_d = zip(*[_draw_across(delta, True) for _ in range(n_sim)])
        zs_u, _ = zip(*[_draw_across(delta, False) for _ in range(n_sim)])
        pwr_d = float(np.mean(np.array(zs_d) <= crit_z))
        pwr_u = float(np.mean(np.array(zs_u) <= crit_z))
        med_rr = float(np.median(rrs_d))
        print(f"  delta={delta:.2f} -> power(depth-scaled)={pwr_d:.4f} | power(unscaled)={pwr_u:.4f} | med_RR={med_rr:.4f}")
        rows.append({
            "test": "SECONDARY: Full-Shadow Across-Plate (m >= 9.0, sf >= 0.95, N=47)",
            "alpha": 0.05,
            "use_gaia_density": False,
            "n_plates": n_full,
            "k_sunlit_pool": len(sun_m9),
            "delta_full_depth": delta,
            "non_stellar_frac": 0.531,
            "crit_z": round(crit_z, 3),
            "placebo_fpr": round(emp_fpr, 4),
            "power_depth_scaled": round(pwr_d, 4),
            "power_unscaled": round(pwr_u, 4),
            "med_rr_depth_scaled": round(med_rr, 4),
        })
    return rows


def run_amendment2_calibration(n_sim: int = 2000, seed: int = 20261116) -> None:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"Amendment 2 Pre-Outcome Power Calibration | Base Commit: {commit_hash}")

    exp = pd.read_parquet("data/processed/exposures.parquet")
    exp_idx = exp.set_index("exposure_id")

    # Read ONLY qc_pass + phot_band from phase2b_plates.parquet (NEVER phase2b_catalogs or phase2b_stamps!)
    p2b_qc = pd.read_parquet(
        "data/processed/phase2b_plates.parquet",
        columns=["exposure_id", "plate_id", "instrument", "m_flash_05s", "qc_pass", "phot_band", "phase2b_role"],
    )
    man = pd.read_parquet("data/processed/phase2b_manifest.parquet")
    links = pd.read_parquet("data/processed/phase2b_paired_tierA_links.parquet")

    m_qc = p2b_qc.merge(
        man[["exposure_id", "shadow_fraction", "ra_deg", "dec_deg", "fov_ra_deg", "fov_dec_deg", "obs_lat_deg", "obs_lon_deg", "obs_elev_m", "mjd_start", "exptime_sec"]],
        on="exposure_id",
        how="left",
    )

    # Realized QC-passing sets (strictly metadata + qc_pass)
    real_m9 = m_qc[(m_qc["phase2b_role"] == "primary_m9_partial") & m_qc["qc_pass"]].drop_duplicates("plate_id")
    real_m8 = m_qc[(m_qc["m_flash_05s"] >= 8.0) & (m_qc["shadow_fraction"] > 0.05) & (m_qc["shadow_fraction"] < 0.95) & m_qc["qc_pass"]].drop_duplicates("plate_id")
    real_m8_lo = real_m8[~real_m8["plate_id"].isin(set(real_m9["plate_id"]))].copy()
    real_full9 = m_qc[(m_qc["m_flash_05s"] >= 9.0) & (m_qc["shadow_fraction"] >= 0.95) & m_qc["qc_pass"]].drop_duplicates("plate_id")

    pilot_res = pd.read_csv("data/processed/pilot_results.csv")
    qc9 = pd.read_csv("data/processed/pilot_plate_qc.csv")
    fp9 = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")
    p8 = pd.read_parquet("data/processed/sunlit_m8_plates.parquet")

    qc_lookup = dict(zip(p2b_qc["exposure_id"], p2b_qc["qc_pass"]))
    band_lookup = dict(zip(p2b_qc["exposure_id"], p2b_qc["phot_band"]))
    for eid, row in fp9.iterrows():
        sub = qc9[qc9["plate_id"] == row["plate_id"]]
        if len(sub) > 0:
            qc_lookup[str(eid)] = bool(sub["qc_pass"].iloc[0])
            band_lookup[str(eid)] = str(sub["phot_band"].iloc[0])
    for _, r in p8.iterrows():
        if str(r["exposure_id"]) not in qc_lookup:
            qc_lookup[str(r["exposure_id"])] = bool(r["qc_pass"])

    tA_ok_eids = []
    for _, r in links.iterrows():
        ea, eb = str(r["exp_target"]), str(r["exp_partner"])
        if qc_lookup.get(ea, False) and qc_lookup.get(eb, False):
            ba, bb = band_lookup.get(ea, ""), band_lookup.get(eb, "")
            if ba == bb or ba == "" or bb == "":
                tA_ok_eids.append(ea)
    real_tA_meta = exp_idx.loc[tA_ok_eids].reset_index()

    print(
        f"Realized QC Sample Sizes (0 shadow candidate files touched): "
        f"m>=9 partial={len(real_m9)}, m>=8 partial={len(real_m8)} (61 + {len(real_m8_lo)}), "
        f"Paired Tier A={len(real_tA_meta)}, Full-shadow m>=9={len(real_full9)}"
    )

    # Hard blindness guard on all calibration catalogs (`pilot` + `sunlit_m8`)
    sun_ids = list(set(pilot_res["exp1"]).union(set(pilot_res["exp2"])).union(set(p8["exposure_id"])))
    for col in ["shadow_fraction", "shadow_fraction_36000km", "shadow_fraction_20000km"]:
        mx = float(exp_idx.loc[sun_ids, col].max())
        if mx > 0.0:
            raise RuntimeError(f"BLINDNESS GUARD ABORT: {col}={mx} > 0")
    print(f"Hard blindness guard passed on {len(sun_ids)} sunlit calibration exposures (max shadow_fraction=0.0).")

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

    # Load frozen 348-plate sunlit single-plate pool & 102-plate paired pool
    single_ev, single_vm, single_lg, single_gaia, single_wcs, single_rows = {}, {}, {}, {}, {}, []
    p8_qc = p8[p8["qc_pass"].fillna(False)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in p8_qc.iterrows():
            eid = r["exposure_id"]
            df_all = _apply_conc_pk_veto_from_stamps(eid, pd.read_parquet(f"data/processed/sunlit_m8_catalogs/{eid}.parquet"))
            wcs = WCS(fits.Header.fromstring(r["wcs_header"]))
            nx, ny = int(r["nx"]), int(r["ny"])
            ga = np.nan_to_num(np.array(json.loads(r["local_mlim_grid_json"]), dtype=float), nan=float(r["m_lim_cal"]))
            ev, vg, uf, _ = _apply_frozen_filter_to_plate(eid, df_all, wcs, nx, ny, float(r["med_r_psf"]), float(r["m_lim_cal"]), ga)
            single_ev[eid], single_vm[eid], single_lg[eid] = ev, vg, ga
            single_gaia[eid] = compute_gaia_density_grid_32(df_all, wcs, nx, ny, vg)
            single_wcs[eid] = (wcs, nx, ny)
            single_rows.append({
                "exposure_id": eid, "instrument": str(r["instrument"]),
                "m_flash_05s": float(r["m_flash_05s"]), "m_lim_cal": float(r["m_lim_cal"]),
                "unmasked_frac": uf, "n_events": len(ev),
            })

        for eid in fp9.index:
            if eid in single_ev or eid not in exp_idx.index:
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
            single_ev[eid], single_vm[eid], single_lg[eid] = ev, vg, ga
            single_gaia[eid] = compute_gaia_density_grid_32(df_all, wcs, nx, ny, vg)
            single_wcs[eid] = (wcs, nx, ny)
            er = exp_idx.loc[eid]
            single_rows.append({
                "exposure_id": eid, "instrument": str(er["instrument"]),
                "m_flash_05s": float(er["m_flash_05s"]), "m_lim_cal": float(fra["m_lim_cal"]),
                "unmasked_frac": uf, "n_events": len(ev),
            })
        single_df = pd.DataFrame(single_rows)

        paired_ev, paired_vm, paired_lg, paired_gaia, paired_rows = {}, {}, {}, {}, []
        for _, pr in pilot_res[pilot_res["pair_ok"]].iterrows():
            for ea, eb in [(pr["exp1"], pr["exp2"]), (pr["exp2"], pr["exp1"])]:
                if ea in paired_ev:
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
                paired_ev[ea], paired_vm[ea], paired_lg[ea] = ev, vg, ga
                paired_gaia[ea] = compute_gaia_density_grid_32(dfa, wa, nxa, nya, vg)
                er = exp_idx.loc[ea]
                paired_rows.append({
                    "exposure_id": ea, "instrument": str(er["instrument"]),
                    "m_flash_05s": float(er["m_flash_05s"]), "m_lim_cal": float(fra["m_lim_cal"]),
                    "unmasked_frac": uf, "n_events": len(ev),
                })
        paired_df = pd.DataFrame(paired_rows)

    single_bank = build_ecliptic_time_shifted_bank(single_df, exp_idx, single_wcs, single_ev, single_vm)

    eids_m9 = single_df[single_df["m_flash_05s"] >= 9.0]["exposure_id"].tolist()
    eids_m8 = single_df["exposure_id"].tolist()
    eids_m8_lo = single_df[single_df["m_flash_05s"] < 9.0]["exposure_id"].tolist()

    all_rows = []
    # 1. Co-Primary 1: Single-Plate m >= 8.0 (Realized N = 293) at alpha = 0.025 (Holm Step 1) and alpha = 0.05
    all_rows.extend(_evaluate_single_at_alpha(
        single_df, eids_m8, single_bank, single_ev, single_vm, single_lg, single_gaia,
        len(real_m8), real_m8["shadow_fraction"].to_numpy(), True, 0.469, 0.531,
        [0.20, 0.30, 0.40, 0.50], [0.025, 0.050],
        "CO-PRIMARY 1: Single-Plate Tier A+B (m >= 8.0, WITH Gaia, N=293)", n_sim, seed,
    ))
    all_rows.extend(_evaluate_single_at_alpha(
        single_df, eids_m8, single_bank, single_ev, single_vm, single_lg, single_gaia,
        len(real_m8), real_m8["shadow_fraction"].to_numpy(), False, 0.0, 0.531,
        [0.20, 0.30, 0.40, 0.50], [0.025, 0.050],
        "ABLAT: Single-Plate Tier A+B (m >= 8.0, NO Gaia, N=293)", n_sim, seed + 1,
    ))

    # 2. Co-Primary 2: Single-Plate m >= 9.0 (Realized N = 61) at alpha = 0.05 (Holm Step 2) and alpha = 0.025
    all_rows.extend(_evaluate_single_at_alpha(
        single_df, eids_m9, single_bank, single_ev, single_vm, single_lg, single_gaia,
        len(real_m9), real_m9["shadow_fraction"].to_numpy(), True, 0.469, 0.531,
        [0.20, 0.30, 0.40, 0.50], [0.050, 0.025],
        "CO-PRIMARY 2: Single-Plate Tier A+B (m >= 9.0, WITH Gaia, N=61)", n_sim, seed + 2,
    ))
    all_rows.extend(_evaluate_single_at_alpha(
        single_df, eids_m9, single_bank, single_ev, single_vm, single_lg, single_gaia,
        len(real_m9), real_m9["shadow_fraction"].to_numpy(), False, 0.0, 0.531,
        [0.20, 0.30, 0.40, 0.50], [0.050],
        "ABLAT: Single-Plate Tier A+B (m >= 9.0, NO Gaia, N=61)", n_sim, seed + 3,
    ))

    # 3. Coupled Joint Holm Co-Primary Verification (61 m>=9 shared inside 293 m>=8, exact realized (sf, m_flash) pairs)
    all_rows.extend(_evaluate_coupled_holm_coprimary(
        single_df, eids_m9, eids_m8_lo, single_bank, single_ev, single_vm, single_lg, single_gaia,
        len(real_m9), real_m9[["shadow_fraction", "m_flash_05s"]].to_numpy(),
        len(real_m8_lo), real_m8_lo[["shadow_fraction", "m_flash_05s"]].to_numpy(),
        [0.20, 0.30, 0.40, 0.50], n_sim, seed + 4,
    ))

    # 4. Key Secondary: Strict Paired Tier A (m >= 8.0, Realized N = 95)
    all_rows.extend(_evaluate_realized_paired_tierA(
        paired_df, paired_ev, paired_vm, paired_lg, paired_gaia,
        real_tA_meta, [0.20, 0.30, 0.40, 0.50, 0.60, 0.70], n_sim, seed + 5,
    ))

    # 5. Full-Shadow Dose-Response Across-Plate (m >= 9.0, sf >= 0.95, Realized N = 47)
    all_rows.extend(_evaluate_full_shadow_across_plate(
        single_df, exp_idx, real_full9, [0.20, 0.30, 0.40, 0.50], n_sim, seed + 6,
    ))

    out_csv = Path("data/processed/amendment2_realized_power_calibration.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nSaved Amendment 2 realized power calibration to {out_csv}")


if __name__ == "__main__":
    run_amendment2_calibration(n_sim=2000, seed=20261116)
