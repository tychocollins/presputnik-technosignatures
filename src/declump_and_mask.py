"""Amendment 1 Steps 1-3: Bright-star & edge masking with area subtraction, 30\"/60\" de-clustering, alpha_hat, and 1000-placebo power."""

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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import gammaln

from src.test_artifact_filter import rebuild_filtered_plate_data
from src.within_plate_power import _cmh_stat


def _declump_events(ra: np.ndarray, dec: np.ndarray, x_norm: np.ndarray, y_norm: np.ndarray, radius_arcsec: float) -> np.ndarray:
    """Merge candidates within `radius_arcsec` into single events (connected-component centroids)."""
    n = len(ra)
    if n <= 1:
        return np.column_stack([x_norm, y_norm]) if n == 1 else np.zeros((0, 2))
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    i1, i2, _, _ = sc.search_around_sky(sc, radius_arcsec * u.arcsec)
    adj = csr_matrix((np.ones(len(i1), dtype=int), (i1, i2)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    out = np.zeros((n_comp, 2), dtype=float)
    for c in range(n_comp):
        m = labels == c
        out[c, 0] = float(np.mean(x_norm[m]))
        out[c, 1] = float(np.mean(y_norm[m]))
    return out


def build_masked_declumped_data(cluster_arcsec: float = 30.0):
    res = pd.read_csv("data/processed/pilot_results.csv")
    qc_pairs = res[res["qc_pass_pair"]].copy()
    exp = pd.read_parquet("data/processed/exposures.parquet").set_index("exposure_id")
    fp = pd.read_parquet("data/processed/pilot_footprints.parquet").set_index("exposure_id")

    cands_old = pd.read_parquet("data/processed/pilot_candidates.parquet")
    ast_rows = cands_old[cands_old["is_asteroid"].fillna(False)]
    ast_by_exp = {
        eid: SkyCoord(grp["ra"].to_numpy() * u.deg, grp["dec"].to_numpy() * u.deg)
        for eid, grp in ast_rows.groupby("exp_a")
    }

    cat_cache, wcs_cache, grid_cache = {}, {}, {}

    def _get_plate(eid: str):
        if eid not in cat_cache:
            cat_cache[eid] = pd.read_parquet(f"data/processed/pilot_catalogs/{eid}.parquet")
            fr = fp.loc[eid]
            wcs_cache[eid] = WCS(fits.Header.fromstring(fr["wcs_header"]))
            g = np.array(json.loads(fr["local_mlim_grid_json"]), dtype=float)
            grid_cache[eid] = np.nan_to_num(g, nan=float(fr["m_lim_cal"]))
        return cat_cache[eid], wcs_cache[eid], grid_cache[eid], fp.loc[eid]

    event_xy: dict[str, np.ndarray] = {}
    valid_grid_masks: dict[str, np.ndarray] = {}
    plate_rows = []

    # 32x32 fine grid over [0, 1]^2 for exact masked-area subtraction
    g_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32)
    gx_n, gy_n = np.meshgrid(g_pts, g_pts)
    gx_flat, gy_flat = gx_n.ravel(), gy_n.ravel()

    for _, pr in qc_pairs.iterrows():
        e1, e2 = pr["exp1"], pr["exp2"]
        for ea, eb in [(e1, e2), (e2, e1)]:
            if ea in event_xy:
                continue
            df_a_all, wcs_a, _, fr_a = _get_plate(ea)
            df_b_all, wcs_b, grid_b, fr_b = _get_plate(eb)

            nx_a, ny_a = int(fr_a["nx"]), int(fr_a["ny"])
            med_r_a = float(fr_a["med_r_psf"]) if np.isfinite(fr_a["med_r_psf"]) and fr_a["med_r_psf"] > 0 else 3.0
            m_lim_a = float(fr_a["m_lim_cal"])

            # Bright stars on Plate A for annulus masking: mag_est < m_lim_a - 3.0
            mag_all_a = df_a_all["mag_est"].to_numpy(dtype=float)
            bright_idx = np.where(np.isfinite(mag_all_a) & (mag_all_a < (m_lim_a - 3.0)))[0]
            if len(bright_idx) > 0:
                sc_bright = SkyCoord(df_a_all["ra"].to_numpy()[bright_idx] * u.deg, df_a_all["dec"].to_numpy()[bright_idx] * u.deg)
                r_mask_as = np.clip(20.0 + 8.0 * np.maximum(0.0, (m_lim_a - 3.0) - mag_all_a[bright_idx]), 20.0, 60.0)
            else:
                sc_bright = None
                r_mask_as = np.zeros(0)

            # Edge mask in normalized coords: max(0.05, 4.0 * med_r_a / min(nx, ny))
            border_x = max(0.05, 4.0 * med_r_a / max(nx_a, 1))
            border_y = max(0.05, 4.0 * med_r_a / max(ny_a, 1))
            grid_valid = (gx_flat >= border_x) & (gx_flat <= 1.0 - border_x) & (gy_flat >= border_y) & (gy_flat <= 1.0 - border_y)

            # Subtract bright-star annuli from 32x32 grid area
            if sc_bright is not None and len(sc_bright) > 0:
                sc_grid = wcs_a.pixel_to_world(gx_flat * nx_a, gy_flat * ny_a)
                idx_g, idx_b, sep_gb, _ = sc_bright.search_around_sky(sc_grid, 60.0 * u.arcsec)
                for ib, ig, s in zip(idx_b, idx_g, sep_gb.arcsec):
                    if s <= r_mask_as[ib]:
                        grid_valid[ig] = False
            valid_grid_masks[ea] = grid_valid.reshape(32, 32)

            # Apply morphology + partner-all + edge + bright-star mask to Plate A candidates
            morph_ok = (
                df_a_all["psf_pass"].to_numpy()
                & (~df_a_all["gaia_hit"].to_numpy())
                & (df_a_all["ellip"].to_numpy() <= 0.25)
                & (df_a_all["r_psf"].to_numpy() >= 0.7 * med_r_a)
                & (df_a_all["r_psf"].to_numpy() <= 1.8 * med_r_a)
            )
            sub_a = df_a_all[morph_ok].copy().reset_index(drop=True)
            if len(sub_a) > 0:
                nx_b, ny_b = int(fr_b["nx"]), int(fr_b["ny"])
                xb, yb = wcs_b.world_to_pixel(SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg))
                in_b = (xb >= 0.0) & (xb <= float(nx_b - 1)) & (yb >= 0.0) & (yb <= float(ny_b - 1))
                sub_a = sub_a[in_b].copy().reset_index(drop=True)
                xb, yb = xb[in_b], yb[in_b]

            if len(sub_a) > 0:
                med_r_b = float(fr_b["med_r_psf"]) if np.isfinite(fr_b["med_r_psf"]) and fr_b["med_r_psf"] > 0 else 3.0
                edge_px_b = np.minimum(np.minimum(xb, float(nx_b - 1) - xb), np.minimum(yb, float(ny_b - 1) - yb))
                edge_psf_b = edge_px_b / med_r_b

                ng = grid_b.shape[0]
                ix_b = np.clip((xb / max(nx_b, 1) * ng).astype(int), 0, ng - 1)
                iy_b = np.clip((yb / max(ny_b, 1) * ng).astype(int), 0, ng - 1)
                mag_margin = grid_b[iy_b, ix_b] - sub_a["mag_est"].to_numpy(dtype=float)

                sc_sub = SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg)
                sc_b_all = SkyCoord(df_b_all["ra"].to_numpy() * u.deg, df_b_all["dec"].to_numpy() * u.deg)
                _, sep_b, _ = sc_sub.match_to_catalog_sky(sc_b_all)
                sub_a = sub_a[(sep_b.arcsec > 15.0) & (edge_psf_b >= 4.0) & (mag_margin >= 1.0)].copy().reset_index(drop=True)

            # Plate A edge + bright-star annulus mask
            if len(sub_a) > 0:
                xa, ya = wcs_a.world_to_pixel(SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg))
                xn = xa / max(nx_a, 1)
                yn = ya / max(ny_a, 1)
                keep_a = (xn >= border_x) & (xn <= 1.0 - border_x) & (yn >= border_y) & (yn <= 1.0 - border_y)
                if sc_bright is not None and len(sc_bright) > 0:
                    sc_k = SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg)
                    idx_k, idx_b, sep_kb, _ = sc_bright.search_around_sky(sc_k, 60.0 * u.arcsec)
                    for ib, ik, s in zip(idx_b, idx_k, sep_kb.arcsec):
                        if s <= r_mask_as[ib]:
                            keep_a[ik] = False
                sub_a = sub_a[keep_a].copy().reset_index(drop=True)
                xn, yn = xn[keep_a], yn[keep_a]
            else:
                xn, yn = np.zeros(0), np.zeros(0)

            # Asteroid removal
            if len(sub_a) > 0 and ea in ast_by_exp:
                sc_k = SkyCoord(sub_a["ra"].to_numpy() * u.deg, sub_a["dec"].to_numpy() * u.deg)
                _, sep_ast, _ = sc_k.match_to_catalog_sky(ast_by_exp[ea])
                ok_ast = sep_ast.arcsec > 5.0
                sub_a = sub_a[ok_ast].copy().reset_index(drop=True)
                xn, yn = xn[ok_ast], yn[ok_ast]

            # De-cluster into events within `cluster_arcsec`
            if len(sub_a) > 0:
                ev_xy = _declump_events(sub_a["ra"].to_numpy(), sub_a["dec"].to_numpy(), xn, yn, cluster_arcsec)
            else:
                ev_xy = np.zeros((0, 2))
            event_xy[ea] = ev_xy

            er = exp.loc[ea]
            unmasked_frac = float(np.mean(valid_grid_masks[ea]))
            plate_rows.append({
                "exposure_id": ea,
                "instrument": str(er["instrument"]),
                "ra_deg": float(er["ra_deg"]),
                "dec_deg": float(er["dec_deg"]),
                "mjd_mid": float(er["mjd_mid"]),
                "obs_lon_deg": float(er["obs_lon_deg"]),
                "pointing_alt_mid_deg": float(er["pointing_alt_mid_deg"]),
                "eff_area_deg2": max(0.05, unmasked_frac) * float(er["fov_ra_deg"]) * float(er["fov_dec_deg"]),
                "m_lim_cal": m_lim_a,
                "n_events": len(ev_xy),
            })

    plates_df = pd.DataFrame(plate_rows)

    # Spatial defect profile on 8x8 -> upsampled to 32x32, multiplied by valid_grid_masks & local depth
    global_hist = np.ones((8, 8), dtype=float) * 0.5
    for xy in event_xy.values():
        if len(xy) > 0:
            gx = np.clip((xy[:, 0] * 8).astype(int), 0, 7)
            gy = np.clip((xy[:, 1] * 8).astype(int), 0, 7)
            for ix, iy in zip(gx, gy):
                global_hist[iy, ix] += 1.0
    global_prof = global_hist / np.mean(global_hist)

    inst_profs = {}
    for inst, grp in plates_df.groupby("instrument"):
        h = np.zeros((8, 8), dtype=float)
        for eid in grp["exposure_id"]:
            xy = event_xy[eid]
            if len(xy) > 0:
                gx = np.clip((xy[:, 0] * 8).astype(int), 0, 7)
                gy = np.clip((xy[:, 1] * 8).astype(int), 0, 7)
                for ix, iy in zip(gx, gy):
                    h[iy, ix] += 1.0
        prior = 40.0 * global_prof / 64.0
        inst_profs[inst] = (h + prior) / np.mean(h + prior)

    cell_weights_32: dict[str, np.ndarray] = {}
    for _, r in plates_df.iterrows():
        eid = r["exposure_id"]
        prof8 = inst_profs[r["instrument"]]
        prof32 = np.repeat(np.repeat(prof8, 4, axis=0), 4, axis=1)
        d_m = np.clip(grid_cache[eid] - r["m_lim_cal"], -1.5, 1.5)
        depth32 = np.repeat(np.repeat(np.power(10.0, 0.2 * d_m), 4, axis=0), 4, axis=1)
        w32 = prof32 * depth32 * valid_grid_masks[eid].astype(float)
        s = float(np.sum(w32))
        cell_weights_32[eid] = w32 / (s if s > 0 else 1.0)

    return plates_df, event_xy, cell_weights_32


def evaluate_declumped(cluster_arcsec: float, n_sim: int = 1000, seed: int = 42):
    plates_df, event_xy, cell_weights_32 = build_masked_declumped_data(cluster_arcsec=cluster_arcsec)

    # Measure alpha_hat
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
    y0 = df_nb["n_events"].to_numpy(dtype=float)
    off0 = np.log(df_nb["eff_area_deg2"].to_numpy(dtype=float))
    pois = sm.GLM(y0, X0, family=sm.families.Poisson(), offset=off0).fit()
    mu0 = np.maximum(pois.fittedvalues, 1e-6)

    def _nb_nll(alpha: float) -> float:
        r = 1.0 / alpha
        return -float(np.sum(gammaln(y0 + r) - gammaln(r) - gammaln(y0 + 1.0) + r * np.log(r / (r + mu0)) + y0 * np.log(mu0 / (r + mu0))))

    alpha_hat = float(minimize_scalar(_nb_nll, bounds=(0.01, 20.0), method="bounded").x)
    print(f"\n================ CLUSTER RADIUS = {cluster_arcsec:.0f}\" ================")
    print(f"Total events across {len(plates_df)} plates: {plates_df['n_events'].sum()} | alpha_hat = {alpha_hat:.4f} | Poisson chi2/df = {pois.pearson_chi2 / pois.df_resid:.2f}")

    exp = pd.read_parquet("data/processed/exposures.parquet")
    qc = pd.read_csv("data/processed/pilot_plate_qc.csv").drop_duplicates("plate_id")
    inst_qc = qc.groupby("instrument")["qc_pass"].mean().to_dict()
    avail_by_inst = {inst: grp["exposure_id"].tolist() for inst, grp in plates_df.groupby("instrument")}
    rng = np.random.default_rng(seed)

    g_pts = np.linspace(0.5 / 32.0, 1.0 - 0.5 / 32.0, 32) - 0.5
    gx, gy = np.meshgrid(g_pts, g_pts)

    rows = []
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
                    theta = rng.uniform(0.0, 2.0 * np.pi)
                    c, s = np.cos(theta), np.sin(theta)
                    proj_g = c * gx + s * gy
                    w32 = cell_weights_32[eid]
                    valid_proj = proj_g[w32 > 0]
                    thresh = float(np.quantile(valid_proj, float(sf))) if len(valid_proj) else 0.0
                    p = float(np.clip(np.sum(w32[proj_g <= thresh]), 0.02, 0.98))
                    xy = event_xy[eid]
                    n_tot = len(xy)
                    if n_tot > 0:
                        proj_c = c * (xy[:, 0] - 0.5) + s * (xy[:, 1] - 0.5)
                        n_in = int(np.sum(proj_c <= thresh))
                    else:
                        n_in = 0
                    if delta > 0 and n_in > 0:
                        kept = int(rng.binomial(n_in, 1.0 - delta))
                        n_tot = n_tot - (n_in - kept)
                        n_in = kept
                    ni.append(n_in); nt.append(n_tot); pa.append(p)
            return _cmh_stat(np.array(ni), np.array(nt), np.array(pa))

        null_zs = np.array([_draw_z(0.0)[0] for _ in range(n_sim)])
        crit_z = float(np.quantile(null_zs, 0.05))
        emp_fpr = float(np.mean(np.array([_draw_z(0.0)[0] for _ in range(n_sim)]) <= crit_z))
        print(f"\n--- {label} (N_plates={sum(x[1] for x in alloc)}) | crit_z={crit_z:.3f} | Placebo FPR={emp_fpr:.4f} ---")
        for delta in [0.20, 0.30, 0.40, 0.50]:
            zs, rrs = [], []
            for _ in range(n_sim):
                z, rr = _draw_z(delta)
                zs.append(z); rrs.append(rr)
            pwr = float(np.mean(np.array(zs) <= crit_z))
            rows.append({
                "cluster_arcsec": cluster_arcsec, "tier": label, "delta": delta,
                "crit_z": round(crit_z, 3), "fpr": round(emp_fpr, 4),
                "power": round(pwr, 4), "med_rr": round(float(np.median(rrs)), 4),
            })
            print(f"  delta={delta:.2f} -> power={pwr:.4f}, med_RR={np.median(rrs):.4f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d30 = evaluate_declumped(30.0, n_sim=1000, seed=42)
    d60 = evaluate_declumped(60.0, n_sim=1000, seed=42)
    pd.concat([d30, d60], ignore_index=True).to_csv("data/processed/declumped_power_calibration.csv", index=False)
