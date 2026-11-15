"""Vectorized topocentric Earth-umbra shadow geometry."""

from __future__ import annotations

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, get_sun
from astropy.time import Time

R_EARTH_KM = float(const.R_earth.to_value(u.km))  # 6378.1 km
R_SUN_KM = float(const.R_sun.to_value(u.km))      # 695700.0 km


def make_tangent_plane_rays(
    ra_deg: float,
    dec_deg: float,
    width_deg: float,
    height_deg: float,
    step_deg: float = 0.25,
    min_samples: int = 5,
    max_samples: int = 15,
) -> np.ndarray:
    """Return (N, 3) unit vectors in ICRS/GCRS covering a rectangular field via TAN projection."""
    nx = int(np.clip(np.ceil(width_deg / step_deg), min_samples, max_samples))
    ny = int(np.clip(np.ceil(height_deg / step_deg), min_samples, max_samples))

    xi_deg = np.linspace(-0.5 * width_deg, 0.5 * width_deg, nx)
    eta_deg = np.linspace(-0.5 * height_deg, 0.5 * height_deg, ny)
    xi_grid, eta_grid = np.meshgrid(np.deg2rad(xi_deg), np.deg2rad(eta_deg), indexing="xy")

    xi = xi_grid.ravel()
    eta = eta_grid.ravel()

    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    cos_ra, sin_ra = np.cos(ra), np.sin(ra)
    cos_dec, sin_dec = np.cos(dec), np.sin(dec)

    u0 = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])
    e_ra = np.array([-sin_ra, cos_ra, 0.0])
    e_dec = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec])

    vecs = (
        u0[None, :]
        + np.tan(xi)[:, None] * e_ra[None, :]
        + np.tan(eta)[:, None] * e_dec[None, :]
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def make_time_grid(
    mjd_start: float,
    duration_sec: float,
    step_min: float = 10.0,
) -> Time:
    """Return Time array: start, mid, end, plus intermediate steps for exposures > step_min."""
    dur_days = duration_sec / 86400.0
    if duration_sec <= step_min * 60.0:
        mjds = np.array([mjd_start, mjd_start + 0.5 * dur_days, mjd_start + dur_days])
    else:
        n_steps = max(3, int(np.ceil(duration_sec / (step_min * 60.0))) + 1)
        mjds = np.linspace(mjd_start, mjd_start + dur_days, n_steps)
    return Time(mjds, format="mjd", scale="utc")


def points_in_umbra(
    r_obs_km: np.ndarray,
    d_hat: np.ndarray,
    r_sun_km: np.ndarray,
    r_target_km: float = 42164.0,
) -> np.ndarray:
    """
    Vectorized test of whether topocentric rays hit Earth's umbra at geocentric radius r_target_km.

    Parameters
    ----------
    r_obs_km : (T, 3) array of observatory GCRS positions in km.
    d_hat : (N, 3) array of unit sky-direction vectors.
    r_sun_km : (T, 3) array of geocentric Sun positions in km.
    r_target_km : float, geocentric sphere radius in km (e.g. 42164 for GEO).

    Returns
    -------
    in_umbra : (T, N) boolean array.
    """
    # b = r_obs · d_hat -> shape (T, N)
    b = np.einsum("ti,ni->tn", r_obs_km, d_hat)
    r_obs_sq = np.sum(r_obs_km ** 2, axis=1, keepdims=True)  # (T, 1)
    c = r_obs_sq - r_target_km ** 2

    disc = b ** 2 - c
    valid = disc >= 0.0
    s = np.where(valid, -b + np.sqrt(np.maximum(disc, 0.0)), -1.0)
    valid &= s > 0.0

    # r_pt shape (T, N, 3)
    r_pt = r_obs_km[:, None, :] + s[:, :, None] * d_hat[None, :, :]

    # Sun unit vector and distance
    d_sun = np.linalg.norm(r_sun_km, axis=1, keepdims=True)  # (T, 1)
    u_sun = r_sun_km / d_sun  # (T, 3)

    # Anti-solar projection z = -(r_pt · u_sun) -> (T, N)
    z = -np.einsum("tni,ti->tn", r_pt, u_sun)

    # Perpendicular distance to shadow axis
    r_pt_sq = np.sum(r_pt ** 2, axis=2)
    r_perp_sq = np.maximum(0.0, r_pt_sq - z ** 2)
    r_perp = np.sqrt(r_perp_sq)

    # Exact umbral cone tangent to Earth and Sun spheres
    # sin(alpha_u) = (R_S - R_E) / d_sun
    sin_alpha = (R_SUN_KM - R_EARTH_KM) / d_sun  # (T, 1)
    tan_alpha = sin_alpha / np.sqrt(1.0 - sin_alpha ** 2)
    # Umbral radius at anti-solar distance z: R_E / cos(alpha_u) - z * tan(alpha_u)
    cos_alpha = np.sqrt(1.0 - sin_alpha ** 2)
    r_umbra = (R_EARTH_KM / cos_alpha) - z * tan_alpha

    return valid & (z > 0.0) & (r_umbra > 0.0) & (r_perp <= r_umbra)


def compute_exposure_shadow(
    ra_deg: float,
    dec_deg: float,
    width_deg: float,
    height_deg: float,
    mjd_start: float,
    duration_sec: float,
    obs_lat_deg: float,
    obs_lon_deg: float,
    obs_elev_m: float = 0.0,
    altitudes_km: tuple[float, ...] = (42164.0,),
    step_deg: float = 0.25,
    min_samples: int = 5,
    time_step_min: float = 10.0,
) -> dict[float, float]:
    """Compute shadow_fraction for one exposure at each requested geocentric radius."""
    d_hat = make_tangent_plane_rays(
        ra_deg, dec_deg, width_deg, height_deg, step_deg=step_deg, min_samples=min_samples
    )
    times = make_time_grid(mjd_start, duration_sec, step_min=time_step_min)

    loc = EarthLocation(
        lat=obs_lat_deg * u.deg,
        lon=obs_lon_deg * u.deg,
        height=obs_elev_m * u.m,
    )
    r_obs_km = loc.get_gcrs(times).cartesian.xyz.to_value(u.km).T  # (T, 3)
    r_sun_km = get_sun(times).cartesian.xyz.to_value(u.km).T       # (T, 3)

    out: dict[float, float] = {}
    for r_km in altitudes_km:
        mask = points_in_umbra(r_obs_km, d_hat, r_sun_km, r_target_km=r_km)
        out[r_km] = float(np.mean(mask))
    return out


def compute_batch_shadow(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    fov_ra_deg: np.ndarray,
    fov_dec_deg: np.ndarray,
    mjd_start: np.ndarray,
    exptime_sec: np.ndarray,
    obs_lat_deg: np.ndarray,
    obs_lon_deg: np.ndarray,
    obs_elev_m: np.ndarray,
    altitudes_km: tuple[float, ...] = (42164.0, 36000.0, 20000.0),
    step_deg: float = 0.25,
    min_samples: int = 5,
    time_step_min: float = 10.0,
    batch_size: int = 5000,
) -> dict[float, np.ndarray]:
    """
    Fast batched computation of shadow_fraction for N exposures.
    Batches astropy get_gcrs() and get_sun() calls, then runs vectorized numpy umbra geometry.
    """
    n = len(ra_deg)
    results = {r_km: np.zeros(n, dtype=np.float32) for r_km in altitudes_km}

    # Precompute number of time steps per exposure
    n_steps = np.where(
        exptime_sec <= time_step_min * 60.0,
        3,
        np.maximum(3, np.ceil(exptime_sec / (time_step_min * 60.0)).astype(int) + 1),
    )
    # Cap time steps at 25 (e.g. 4h exposure at 10m step) for memory safety
    n_steps = np.minimum(n_steps, 25)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        # Build flat time and location arrays for this batch
        mjds_list = []
        lats_list = []
        lons_list = []
        elevs_list = []
        offsets = [0]
        for i in range(start, end):
            k = int(n_steps[i])
            dur_days = exptime_sec[i] / 86400.0
            mjds_list.append(np.linspace(mjd_start[i], mjd_start[i] + dur_days, k))
            lats_list.append(np.full(k, obs_lat_deg[i]))
            lons_list.append(np.full(k, obs_lon_deg[i]))
            elevs_list.append(np.full(k, obs_elev_m[i]))
            offsets.append(offsets[-1] + k)

        flat_mjds = np.concatenate(mjds_list)
        flat_lats = np.concatenate(lats_list)
        flat_lons = np.concatenate(lons_list)
        flat_elevs = np.concatenate(elevs_list)

        times = Time(flat_mjds, format="mjd", scale="utc")
        locs = EarthLocation(lat=flat_lats * u.deg, lon=flat_lons * u.deg, height=flat_elevs * u.m)
        r_obs_all = locs.get_gcrs(times).cartesian.xyz.to_value(u.km).T
        r_sun_all = get_sun(times).cartesian.xyz.to_value(u.km).T

        for local_i, i in enumerate(range(start, end)):
            a, b = offsets[local_i], offsets[local_i + 1]
            r_obs_i = r_obs_all[a:b]
            r_sun_i = r_sun_all[a:b]

            # Exact geometric pre-filter: at r >= 20,000 km, topocentric umbra + parallax
            # is always within 30 deg of the geocentric anti-solar vector (-u_sun).
            # If plate center is > (30 + 0.5 * diag_deg) from -u_sun, shadow_fraction is 0.
            ra_r = np.deg2rad(float(ra_deg[i]))
            dec_r = np.deg2rad(float(dec_deg[i]))
            u0 = np.array([np.cos(dec_r) * np.cos(ra_r), np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)])
            u_sun_mid = r_sun_i[len(r_sun_i) // 2]
            u_sun_mid = u_sun_mid / np.linalg.norm(u_sun_mid)
            cos_anti = float(np.dot(u0, -u_sun_mid))
            diag_half = 0.5 * np.hypot(float(fov_ra_deg[i]), float(fov_dec_deg[i]))
            if cos_anti < np.cos(np.deg2rad(min(179.0, 32.0 + diag_half))):
                continue

            d_hat_i = make_tangent_plane_rays(
                float(ra_deg[i]),
                float(dec_deg[i]),
                float(fov_ra_deg[i]),
                float(fov_dec_deg[i]),
                step_deg=step_deg,
                min_samples=min_samples,
            )
            for r_km in altitudes_km:
                mask = points_in_umbra(r_obs_i, d_hat_i, r_sun_i, r_target_km=r_km)
                results[r_km][i] = float(np.mean(mask))

    return results

