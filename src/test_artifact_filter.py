"""Step 5: Apply frozen cutout-audit artifact filter to saved pilot catalogs and re-measure alpha_hat and within-plate power."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from astropy import units as u
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord, get_sun
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from src.within_plate_power import _cmh_stat, _placebo_shadow_mask_and_share


def rebuild_filtered_plate_data() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    res = pd.read_csv("data/processed/pilot_results.csv")
    qc_pairs = res[res["qc_pass_pair"]].copy()
    exp = pd.read_parquet("data/processed/exposures.parquet").set_index("exposure_id")
    fp = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")

    # Load existing asteroid flags per (exp_a, ra, dec)
    cands_old = pd.read_parquet("data/processed/pilot_candidates.parquet")
    ast_rows = cands_old[cands_old["is_asteroid"].fillna(False)]
    ast_by_exp = {
        eid: SkyCoord(grp["ra"].to_numpy() * u.deg, grp["dec"].to_numpy() * u.deg)
        for eid, grp in ast_rows.groupby("exp_a")
    }

    cat_cache: dict[str, pd.DataFrame] = {}
    wcs_cache: dict[str, WCS] = {}
    grid_cache: dict[str, np.ndarray] = {}

    def _get_plate(eid: str):
        if eid not in cat_cache:
            cat_cache[eid] = pd.read_parquet(f"data/processed/pilot_catalogs/{eid}.parquet")
            fr = fp.loc[eid]
            wcs_cache[eid] = WCS(fits.Header.fromstring(fr["wcs_header"]))
            g = np.array(json.loads(fr["local_mlim_grid_json"]), dtype=float)
            grid_cache[eid] = np.nan_to_num(g, nan=float(fr["m_lim_cal"]))
        return cat_cache[eid], wcs_cache[eid], grid_cache[eid], fp.loc[eid]

    cand_xy: dict[str, np.ndarray] = {}
    plate_counts: dict[str, int] = {}

    for _, pr in qc_pairs.iterrows():
        e1, e2 = pr["exp1"], pr["exp2"]
        for ea, eb in [(e1, e2), (e2, e1)]:
            if ea in cand_xy:
                continue
            df_a_all, wcs_a, _, fr_a = _get_plate(ea)
            df_b_all, wcs_b, grid_b, fr_b = _get_plate(eb)

            # Strict morphology filter on Plate A (from cutout audit)
            med_r_a = float(fr_a["med_r_psf"])
            morph_ok = (
                df_a_all["psf_pass"].to_numpy()
                & (~df_a_all["gaia_hit"].to_numpy())
                & (df_a_all["ellip"].to_numpy() <= 0.25)
                & (df_a_all["r_psf"].to_numpy() >= 0.7 * med_r_a)
                & (df_a_all["r_psf"].to_numpy() <= 1.8 * med_r_a)
            )
            sub_a = df_a_all[morph_ok].copy().reset_index(drop=True)
            if len(sub_a) == 0:
                cand_xy[ea] = np.zeros((0, 2))
                plate_counts[ea] = 0
                continue

            # Partner footprint + ALL-source match within 15"
            nx_b, ny_b = int(fr_b["nx"]), int(fr_b["ny"])
            xb, yb = wcs_b.world_to_pixel(SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg))
            in_b = (xb >= 0.0) & (xb <= float(nx_b - 1)) & (yb >= 0.0) & (yb <= float(ny_b - 1))
            sub_a = sub_a[in_b].copy().reset_index(drop=True)
            xb, yb = xb[in_b], yb[in_b]
            if len(sub_a) == 0:
                cand_xy[ea] = np.zeros((0, 2))
                plate_counts[ea] = 0
                continue

            med_r_b = float(fr_b["med_r_psf"]) if np.isfinite(fr_b["med_r_psf"]) and fr_b["med_r_psf"] > 0 else 3.0
            edge_px = np.minimum(np.minimum(xb, float(nx_b - 1) - xb), np.minimum(yb, float(ny_b - 1) - yb))
            edge_psf = edge_px / med_r_b

            ng = grid_b.shape[0]
            ix_b = np.clip((xb / max(nx_b, 1) * ng).astype(int), 0, ng - 1)
            iy_b = np.clip((yb / max(ny_b, 1) * ng).astype(int), 0, ng - 1)
            local_lim_b = grid_b[iy_b, ix_b]
            mag_margin = local_lim_b - sub_a["mag_est"].to_numpy(dtype=float)

            sc_sub = SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg)
            sc_b_all = SkyCoord(df_b_all["ra"].to_numpy() * u.deg, df_b_all["dec"].to_numpy() * u.deg)
            _, sep_b, _ = sc_sub.match_to_catalog_sky(sc_b_all)

            keep = (sep_b.arcsec > 15.0) & (edge_psf >= 4.0) & (mag_margin >= 1.0)
            sub_a = sub_a[keep].copy().reset_index(drop=True)

            # Bright-star halo/spike proximity veto: any source on Plate A > 2.5 mag brighter within 25"
            if len(sub_a) > 0:
                sc_k = SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg)
                sc_all_a = SkyCoord(df_a_all["ra"].to_numpy() * u.deg, df_a_all["dec"].to_numpy() * u.deg)
                idx_a, idx_k, _, _ = sc_k.search_around_sky(sc_all_a, 25.0 * u.arcsec)
                bright_veto = np.zeros(len(sub_a), dtype=bool)
                mag_all_a = df_a_all["mag_est"].to_numpy(dtype=float)
                mag_k = sub_a["mag_est"].to_numpy(dtype=float)
                for ia, ik in zip(idx_a, idx_k):
                    if np.isfinite(mag_all_a[ia]) and np.isfinite(mag_k[ik]) and mag_all_a[ia] < (mag_k[ik] - 2.5):
                        bright_veto[ik] = True
                sub_a = sub_a[~bright_veto].copy().reset_index(drop=True)

            # Remove known SkyBoT asteroids
            if len(sub_a) > 0 and ea in ast_by_exp:
                sc_k = SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg)
                _, sep_ast, _ = sc_k.match_to_catalog_sky(ast_by_exp[ea])
                sub_a = sub_a[sep_ast.arcsec > 5.0].copy().reset_index(drop=True)

            nx_a, ny_a = int(fr_a["nx"]), int(fr_a["ny"])
            if len(sub_a) > 0:
                xa, ya = wcs_a.world_to_pixel(SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg))
                xn = np.clip(xa / max(nx_a, 1), 0.0, 1.0)
                yn = np.clip(ya / max(ny_a, 1), 0.0, 1.0)
                in_act = (xn >= 0.05) & (xn <= 0.95) & (yn >= 0.05) & (yn <= 0.95)
                cand_xy[ea] = np.column_stack([xn[in_act], yn[in_act]])
            else:
                cand_xy[ea] = np.zeros((0, 2))
            plate_counts[ea] = len(cand_xy[ea])

    # Build plates_df
    rows = []
    for eid in sorted(cand_xy.keys()):
        er = exp.loc[eid]
        fr = fp.loc[eid]
        rows.append({
            "exposure_id": eid,
            "instrument": str(er["instrument"]),
            "ra_deg": float(er["ra_deg"]),
            "dec_deg": float(er["dec_deg"]),
            "mjd_mid": float(er["mjd_mid"]),
            "obs_lon_deg": float(er["obs_lon_deg"]),
            "pointing_alt_mid_deg": float(er["pointing_alt_mid_deg"]),
            "eff_area_deg2": 0.81 * float(er["fov_ra_deg"]) * float(er["fov_dec_deg"]),
            "m_lim_cal": float(fr["m_lim_cal"]),
            "n_clean": plate_counts[eid],
        })
    plates_df = pd.DataFrame(rows)

    # Spatial defect profile (fit on sunlit plates only)
    inst_profiles: dict[str, np.ndarray] = {}
    global_hist = np.ones((8, 8), dtype=float) * 0.5
    for eid, xy in cand_xy.items():
        if len(xy) > 0:
            gx = np.clip((xy[:, 0] * 8).astype(int), 0, 7)
            gy = np.clip((xy[:, 1] * 8).astype(int), 0, 7)
            for ix, iy in zip(gx, gy):
                global_hist[iy, ix] += 1.0
    global_prof = global_hist / np.mean(global_hist)

    for inst, grp in plates_df.groupby("instrument"):
        h = np.zeros((8, 8), dtype=float)
        for eid in grp["exposure_id"]:
            xy = cand_xy[eid]
            if len(xy) > 0:
                gx = np.clip((xy[:, 0] * 8).astype(int), 0, 7)
                gy = np.clip((xy[:, 1] * 8).astype(int), 0, 7)
                for ix, iy in zip(gx, gy):
                    h[iy, ix] += 1.0
        prior = 40.0 * global_prof / 64.0
        inst_profiles[inst] = (h + prior) / np.mean(h + prior)

    cell_weights: dict[str, np.ndarray] = {}
    for _, r in plates_df.iterrows():
        eid = r["exposure_id"]
        prof = inst_profiles[r["instrument"]]
        d_m = np.clip(grid_cache[eid] - r["m_lim_cal"], -1.5, 1.5)
        w_tot = prof * np.power(10.0, 0.2 * d_m)
        cell_weights[eid] = w_tot / np.sum(w_tot)

    return plates_df, cand_xy, cell_weights, plate_counts


def main() -> None:
    plates_df, cand_xy, cell_weights, _ = rebuild_filtered_plate_data()
    print(f"Filtered clean candidates across {len(plates_df)} QC-passed plates: {plates_df['n_clean'].sum()}")
    print(plates_df.groupby("instrument")["n_clean"].agg(["count", "sum", "mean", "std"]).round(2).to_string())

    # Re-measure alpha_hat on filtered counts
    inst_n = plates_df["instrument"].value_counts()
    keep = set(inst_n[inst_n >= 3].index)
    df_nb = plates_df[plates_df["instrument"].isin(keep)].reset_index(drop=True)
    sc = SkyCoord(df_nb["ra_deg"].to_numpy() * u.deg, df_nb["dec_deg"].to_numpy() * u.deg, frame="icrs")
    df_nb["abs_b_ecl"] = np.abs(sc.transform_to(BarycentricTrueEcliptic()).lat.deg)
    df_nb["solar_elong_deg"] = sc.separation(get_sun(Time(df_nb["mjd_mid"].to_numpy(), format="mjd", scale="utc"))).deg
    ut_hr = (df_nb["mjd_mid"].to_numpy() % 1.0) * 24.0
    lst_hr = np.mod(ut_hr + df_nb["obs_lon_deg"].to_numpy() / 15.0, 24.0)
    df_nb["lst_midnight_hr"] = np.mod(lst_hr + 12.0, 24.0) - 12.0
    df_nb["airmass"] = 1.0 / np.sin(np.deg2rad(np.clip(df_nb["pointing_alt_mid_deg"].to_numpy(), 10.0, 90.0)))

    X0 = pd.DataFrame({"const": 1.0}, index=df_nb.index)
    for col in ("abs_b_ecl", "solar_elong_deg", "lst_midnight_hr", "airmass"):
        v = df_nb[col].to_numpy(dtype=float)
        X0[col] = (v - np.mean(v)) / max(float(np.std(v)), 1e-9)
    X0 = pd.concat([X0, pd.get_dummies(df_nb["instrument"], prefix="inst", drop_first=True, dtype=float)], axis=1)
    y0 = df_nb["n_clean"].to_numpy(dtype=float)
    off0 = np.log(df_nb["eff_area_deg2"].to_numpy(dtype=float))
    pois = sm.GLM(y0, X0, family=sm.families.Poisson(), offset=off0).fit()
    mu0 = np.maximum(pois.fittedvalues, 1e-6)

    def _nb_nll(alpha: float) -> float:
        r = 1.0 / alpha
        return -float(np.sum(
            gammaln(y0 + r) - gammaln(r) - gammaln(y0 + 1.0)
            + r * np.log(r / (r + mu0)) + y0 * np.log(mu0 / (r + mu0))
        ))

    alpha_after = float(minimize_scalar(_nb_nll, bounds=(0.01, 20.0), method="bounded").x)
    print(f"\nNB overdispersion alpha_hat AFTER cutout-audit filter = {alpha_after:.4f} (was 2.6717; Poisson chi2/df = {pois.pearson_chi2 / pois.df_resid:.2f})")

    # Re-run within-plate 1000-placebo power study for Tier A and Tier A+B
    exp = pd.read_parquet("data/processed/exposures.parquet")
    qc = pd.read_csv("data/processed/pilot_plate_qc.csv").drop_duplicates("plate_id")
    inst_qc = qc.groupby("instrument")["qc_pass"].mean().to_dict()
    avail_by_inst = {inst: grp["exposure_id"].tolist() for inst, grp in plates_df.groupby("instrument")}
    rng = np.random.default_rng(42)

    all_rows = []
    for scale_ab, label in [(False, "Tier A (m_flash >= 9.0)"), (True, "Tier A+B (m_flash >= 9.0)")]:
        mask = (exp["is_tier_a"] | exp["is_tier_b"]) if scale_ab else exp["is_tier_a"]
        mask = mask & (exp["m_flash_05s"] >= 9.0)
        part = exp[mask & (exp["shadow_fraction"] > 0.05) & (exp["shadow_fraction"] < 0.95)]
        part_by_inst = part.groupby("instrument")["shadow_fraction"].agg(list).to_dict()
        alloc = []
        for inst, sf_list in part_by_inst.items():
            n_take = int(round(len(sf_list) * inst_qc.get(inst, 0.0)))
            if n_take > 0 and inst in avail_by_inst and len(avail_by_inst[inst]) > 0:
                alloc.append((inst, n_take, np.array(sf_list, dtype=float)))

        def _draw_z(delta: float) -> tuple[float, float]:
            ni, nt, pa = [], [], []
            for inst, n_take, sf_pool in alloc:
                eids = rng.choice(avail_by_inst[inst], size=n_take, replace=(n_take > len(avail_by_inst[inst])))
                sfs = rng.choice(sf_pool, size=n_take, replace=True)
                for eid, sf in zip(eids, sfs):
                    n_in, n_tot, p = _placebo_shadow_mask_and_share(cand_xy[eid], cell_weights[eid], float(sf), rng)
                    if delta > 0 and n_in > 0:
                        kept = int(rng.binomial(n_in, 1.0 - delta))
                        n_tot = n_tot - (n_in - kept)
                        n_in = kept
                    ni.append(n_in)
                    nt.append(n_tot)
                    pa.append(p)
            return _cmh_stat(np.array(ni), np.array(nt), np.array(pa))

        null_zs = np.array([_draw_z(0.0)[0] for _ in range(1000)])
        crit_z = float(np.quantile(null_zs, 0.05))
        emp_fpr = float(np.mean(np.array([_draw_z(0.0)[0] for _ in range(1000)]) <= crit_z))
        print(f"\n=== {label} (filtered, N_plates={sum(x[1] for x in alloc)}) | crit_z={crit_z:.3f} | FPR={emp_fpr:.4f} ===")
        for delta in [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:
            zs, rrs = [], []
            for _ in range(1000):
                z, rr = _draw_z(delta)
                zs.append(z)
                rrs.append(rr)
            pwr = float(np.mean(np.array(zs) <= crit_z))
            all_rows.append({"tier": label, "delta": delta, "power": round(pwr, 4), "med_rr": round(float(np.median(rrs)), 4)})
            print(f"  delta={delta:.2f} -> power={pwr:.4f}, med_RR={np.median(rrs):.4f}")

    pd.DataFrame(all_rows).to_csv("data/processed/within_plate_power_filtered.csv", index=False)


if __name__ == "__main__":
    main()
