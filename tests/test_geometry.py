"""Validation tests for topocentric Earth-umbra shadow geometry."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, get_body, get_sun
from astropy.time import Time

from src.geometry import (
    R_EARTH_KM,
    compute_exposure_shadow,
    make_tangent_plane_rays,
    points_in_umbra,
)


def test_lunar_eclipses_1950s():
    """
    Known 1950s total lunar eclipses:
    - 1950-09-26 04:16:40 UTC (visible Americas/Atlantic)
    - 1953-01-29 23:47:20 UTC (visible Europe/Africa/Americas)
    - 1956-11-18 06:47:40 UTC (visible Americas)
    At totality, Moon's topocentric ray at lunar distance must be inside Earth's umbra;
    6 hours earlier it must be outside.
    """
    cases = [
        ("1950-09-26T04:16:40", 33.3563, -116.8650),  # Palomar
        ("1953-01-29T23:47:20", 53.4800, 10.2400),    # Hamburg
        ("1956-11-18T06:47:40", 42.5050, -71.5580),   # Oak Ridge / Harvard
    ]
    for iso_utc, lat, lon in cases:
        t_tot = Time(iso_utc, format="isot", scale="utc")
        t_early = t_tot - 6.0 * u.hour

        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=0.0 * u.m)

        # At totality
        r_obs_tot = loc.get_gcrs(t_tot).cartesian.xyz.to_value(u.km).reshape(1, 3)
        r_sun_tot = get_sun(t_tot).cartesian.xyz.to_value(u.km).reshape(1, 3)
        moon_tot = get_body("moon", t_tot, location=loc)
        moon_gcrs_tot = get_body("moon", t_tot).cartesian.xyz.to_value(u.km)
        r_moon_km = float(np.linalg.norm(moon_gcrs_tot))

        d_moon_tot = moon_tot.gcrs.cartesian.xyz.to_value(u.km)
        d_hat_tot = (d_moon_tot / np.linalg.norm(d_moon_tot)).reshape(1, 3)
        in_tot = points_in_umbra(r_obs_tot, d_hat_tot, r_sun_tot, r_target_km=r_moon_km)
        assert bool(in_tot[0, 0]), f"Expected Moon in umbra at {iso_utc}"

        # 6 hours earlier
        r_obs_e = loc.get_gcrs(t_early).cartesian.xyz.to_value(u.km).reshape(1, 3)
        r_sun_e = get_sun(t_early).cartesian.xyz.to_value(u.km).reshape(1, 3)
        moon_e = get_body("moon", t_early, location=loc)
        moon_gcrs_e = get_body("moon", t_early).cartesian.xyz.to_value(u.km)
        r_moon_e_km = float(np.linalg.norm(moon_gcrs_e))

        d_moon_e = moon_e.gcrs.cartesian.xyz.to_value(u.km)
        d_hat_e = (d_moon_e / np.linalg.norm(d_moon_e)).reshape(1, 3)
        in_early = points_in_umbra(r_obs_e, d_hat_e, r_sun_e, r_target_km=r_moon_e_km)
        assert not bool(in_early[0, 0]), f"Expected Moon outside umbra 6h before {iso_utc}"


def test_antisolar_parallax_at_geo():
    """
    At local midnight near an equinox (e.g. Palomar, lat ~33.36N, lon ~-116.86E),
    the geocentric shadow center at r_GEO = 42,164 km is at the anti-solar point,
    while the topocentric direction to that 3D point has a parallax offset of
    ~arcsin(R_E * sin(lat) / r_GEO) in declination (~4.8 deg).
    Verify that the true topocentric direction to the anti-solar GEO point IS in umbra
    and matches the analytic parallax formula.
    """
    t = Time("1955-03-21T07:48:00", format="isot", scale="utc")  # ~local midnight at lon -117
    loc = EarthLocation(lat=33.3563 * u.deg, lon=-116.8650 * u.deg, height=1712.0 * u.m)

    r_obs = loc.get_gcrs(t).cartesian.xyz.to_value(u.km)
    r_sun = get_sun(t).cartesian.xyz.to_value(u.km)
    u_sun = r_sun / np.linalg.norm(r_sun)

    r_geo = 42164.0
    # Geocentric shadow center at r_GEO
    p_shadow = -r_geo * u_sun
    # Topocentric vector from observatory to shadow center
    v_topo = p_shadow - r_obs
    d_hat_shadow = v_topo / np.linalg.norm(v_topo)

    # Check it is in umbra
    in_u = points_in_umbra(
        r_obs.reshape(1, 3),
        d_hat_shadow.reshape(1, 3),
        r_sun.reshape(1, 3),
        r_target_km=r_geo,
    )
    assert bool(in_u[0, 0])

    # Analytic parallax angle between geocentric anti-solar (-u_sun) and topocentric d_hat_shadow
    cos_ang = np.clip(np.dot(-u_sun, d_hat_shadow), -1.0, 1.0)
    ang_deg = float(np.rad2deg(np.arccos(cos_ang)))
    # At local midnight, observatory is roughly at angle ~lat from anti-solar axis
    # sin(parallax) ~ |r_obs x (-u_sun)| / |v_topo|
    perp_obs = np.linalg.norm(np.cross(r_obs, -u_sun))
    expected_ang_deg = float(np.rad2deg(np.arcsin(perp_obs / np.linalg.norm(v_topo))))
    assert abs(ang_deg - expected_ang_deg) < 0.05
    assert 3.0 < ang_deg < 9.0  # Non-trivial topocentric parallax at GEO (~4-8 deg)


def test_solar_direction_never_in_shadow():
    """A field pointing directly at the Sun is never in shadow."""
    t = Time("1954-06-15T20:00:00", format="isot", scale="utc")
    sun_sc = get_sun(t)
    res = compute_exposure_shadow(
        ra_deg=float(sun_sc.ra.deg),
        dec_deg=float(sun_sc.dec.deg),
        width_deg=6.0,
        height_deg=6.0,
        mjd_start=float(t.mjd),
        duration_sec=600.0,
        obs_lat_deg=33.3563,
        obs_lon_deg=-116.8650,
        obs_elev_m=1712.0,
        altitudes_km=(42164.0, 36000.0, 20000.0),
    )
    for r_km, frac in res.items():
        assert frac == 0.0, f"Sun direction had shadow_fraction={frac} at {r_km} km"


def test_footprint_bounds_and_extremes():
    """
    shadow_fraction is in [0, 1]; tiny field centered on topocentric GEO shadow center is 1.0,
    and tiny field 30 deg away is 0.0.
    """
    t = Time("1955-03-21T07:48:00", format="isot", scale="utc")
    loc = EarthLocation(lat=33.3563 * u.deg, lon=-116.8650 * u.deg, height=1712.0 * u.m)
    r_obs = loc.get_gcrs(t).cartesian.xyz.to_value(u.km)
    r_sun = get_sun(t).cartesian.xyz.to_value(u.km)
    u_sun = r_sun / np.linalg.norm(r_sun)

    p_shadow = -42164.0 * u_sun
    v_topo = p_shadow - r_obs
    d_hat = v_topo / np.linalg.norm(v_topo)
    ra_center = float(np.rad2deg(np.arctan2(d_hat[1], d_hat[0])) % 360.0)
    dec_center = float(np.rad2deg(np.arcsin(d_hat[2])))

    # Tiny 0.5x0.5 deg field well inside the ~17 deg GEO shadow
    res_in = compute_exposure_shadow(
        ra_deg=ra_center,
        dec_deg=dec_center,
        width_deg=0.5,
        height_deg=0.5,
        mjd_start=float(t.mjd),
        duration_sec=300.0,
        obs_lat_deg=33.3563,
        obs_lon_deg=-116.8650,
        obs_elev_m=1712.0,
        altitudes_km=(42164.0,),
    )
    assert res_in[42164.0] == 1.0

    # Tiny field 30 deg away in RA
    res_out = compute_exposure_shadow(
        ra_deg=(ra_center + 30.0) % 360.0,
        dec_deg=dec_center,
        width_deg=0.5,
        height_deg=0.5,
        mjd_start=float(t.mjd),
        duration_sec=300.0,
        obs_lat_deg=33.3563,
        obs_lon_deg=-116.8650,
        obs_elev_m=1712.0,
        altitudes_km=(42164.0,),
    )
    assert res_out[42164.0] == 0.0
