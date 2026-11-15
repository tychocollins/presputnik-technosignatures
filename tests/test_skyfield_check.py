"""Independent Skyfield Earth-umbra geometry check on 50 exposures (oversampling boundary)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skyfield.api import Loader, wgs84

R_EARTH_KM = 6378.137
R_SUN_KM = 695700.0


def skyfield_shadow_fraction(
    ra_deg: float,
    dec_deg: float,
    width_deg: float,
    height_deg: float,
    mjd_start: float,
    exptime_sec: float,
    obs_lat_deg: float,
    obs_lon_deg: float,
    obs_elev_m: float,
    eph,
    ts,
    r_geo_km: float = 42164.0,
) -> float:
    """Compute shadow_fraction independently using Skyfield + DE421."""
    earth = eph["earth"]
    sun = eph["sun"]

    # Time grid
    if exptime_sec <= 600.0:
        mjds = np.array([mjd_start, mjd_start + 0.5 * exptime_sec / 86400.0, mjd_start + exptime_sec / 86400.0])
    else:
        n_t = min(25, max(3, int(np.ceil(exptime_sec / 600.0)) + 1))
        mjds = np.linspace(mjd_start, mjd_start + exptime_sec / 86400.0, n_t)

    # Skyfield Time from MJD (JD = MJD + 2400000.5)
    t = ts.ut1_jd(mjds + 2400000.5)

    obs = wgs84.latlon(obs_lat_deg, obs_lon_deg, elevation_m=obs_elev_m)
    # Geocentric observatory and Sun vectors in ICRF (km)
    r_obs = obs.at(t).position.km.T            # (T, 3)
    r_sun = (sun - earth).at(t).position.km.T   # (T, 3)

    # Tangent plane ray grid
    nx = int(np.clip(np.ceil(width_deg / 0.25), 5, 15))
    ny = int(np.clip(np.ceil(height_deg / 0.25), 5, 15))
    xi_g, eta_g = np.meshgrid(
        np.deg2rad(np.linspace(-0.5 * width_deg, 0.5 * width_deg, nx)),
        np.deg2rad(np.linspace(-0.5 * height_deg, 0.5 * height_deg, ny)),
        indexing="xy",
    )
    xi = xi_g.ravel()
    eta = eta_g.ravel()

    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    u0 = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    e_ra = np.array([-np.sin(ra), np.cos(ra), 0.0])
    e_dec = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
    vecs = u0[None, :] + np.tan(xi)[:, None] * e_ra[None, :] + np.tan(eta)[:, None] * e_dec[None, :]
    d_hat = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # (N, 3)

    # Solve ray-sphere intersection at r_geo_km
    b = r_obs @ d_hat.T  # (T, N)
    c = np.sum(r_obs ** 2, axis=1, keepdims=True) - r_geo_km ** 2
    disc = b ** 2 - c
    s = -b + np.sqrt(np.maximum(disc, 0.0))
    valid = (disc >= 0.0) & (s > 0.0)

    r_pt = r_obs[:, None, :] + s[:, :, None] * d_hat[None, :, :]  # (T, N, 3)
    d_sun = np.linalg.norm(r_sun, axis=1, keepdims=True)
    u_sun = r_sun / d_sun

    z = -np.einsum("tni,ti->tn", r_pt, u_sun)
    r_perp = np.sqrt(np.maximum(0.0, np.sum(r_pt ** 2, axis=2) - z ** 2))

    sin_a = (R_SUN_KM - R_EARTH_KM) / d_sun
    cos_a = np.sqrt(1.0 - sin_a ** 2)
    r_umbra = (R_EARTH_KM / cos_a) - z * (sin_a / cos_a)

    in_umbra = valid & (z > 0.0) & (r_umbra > 0.0) & (r_perp <= r_umbra)
    return float(np.mean(in_umbra))


def main() -> None:
    load = Loader("data/raw")
    eph = load("de421.bsp")
    ts = load.timescale()

    df = pd.read_parquet("data/processed/exposures.parquet")
    df_de = df[df["mjd_start"] >= 15020.0]  # DE421 valid from 1900-01-01
    boundary = df_de[(df_de["shadow_fraction"] > 0.1) & (df_de["shadow_fraction"] < 0.9)].sample(n=35, random_state=42)
    other = df_de[(df_de["shadow_fraction"] <= 0.1) | (df_de["shadow_fraction"] >= 0.9)].sample(n=15, random_state=42)
    sample = pd.concat([boundary, other], ignore_index=True)

    diffs = []
    for _, r in sample.iterrows():
        sf = skyfield_shadow_fraction(
            ra_deg=float(r["ra_deg"]),
            dec_deg=float(r["dec_deg"]),
            width_deg=float(r["fov_ra_deg"]),
            height_deg=float(r["fov_dec_deg"]),
            mjd_start=float(r["mjd_start"]),
            exptime_sec=float(r["exptime_sec"]),
            obs_lat_deg=float(r["obs_lat_deg"]),
            obs_lon_deg=float(r["obs_lon_deg"]),
            obs_elev_m=float(r["obs_elev_m"]),
            eph=eph,
            ts=ts,
        )
        diffs.append(abs(sf - float(r["shadow_fraction"])))

    diffs_arr = np.array(diffs)
    print(f"Skyfield vs Pipeline (N=50: 35 boundary, 15 non-boundary):")
    print(f"  Median |diff|: {np.median(diffs_arr):.6f}")
    print(f"  Max    |diff|: {np.max(diffs_arr):.6f}")
    assert np.median(diffs_arr) < 0.01, f"Median diff too large: {np.median(diffs_arr)}"
    assert np.max(diffs_arr) < 0.05, f"Max diff too large: {np.max(diffs_arr)}"
    print("Skyfield independent geometry check PASSED.")


if __name__ == "__main__":
    main()
