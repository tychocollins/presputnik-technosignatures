"""Schema discovery and cached metadata fetching for APPLAUSE DR4, DASCH, and POSS-I."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyvo
import requests
import yaml
from astroquery.vizier import Vizier


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_tap_schema(
    tap_url: str,
    name: str,
    table_filter: str | None = None,
    cache_dir: Path = Path("data/raw"),
) -> dict[str, Any]:
    """Query TAP_SCHEMA.tables and TAP_SCHEMA.columns for a TAP service."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file = cache_dir / f"schema_{name}.json"

    service = pyvo.dal.TAPService(tap_url)

    q_tables = "SELECT schema_name, table_name, table_type, description FROM TAP_SCHEMA.tables"
    if table_filter:
        q_tables += f" WHERE {table_filter}"
    tables_df = service.search(q_tables).to_table().to_pandas()

    q_cols = (
        "SELECT table_name, column_name, datatype, unit, ucd, description "
        "FROM TAP_SCHEMA.columns"
    )
    cols_df = service.search(q_cols).to_table().to_pandas()

    if table_filter and "table_name" in tables_df.columns:
        valid_tables = set(tables_df["table_name"].astype(str))
        cols_df = cols_df[cols_df["table_name"].astype(str).isin(valid_tables)]

    result: dict[str, Any] = {
        "tap_url": tap_url,
        "tables": tables_df.to_dict(orient="records"),
        "columns_by_table": {},
    }
    for tbl, grp in cols_df.groupby("table_name"):
        result["columns_by_table"][str(tbl)] = grp.drop(columns=["table_name"]).to_dict(
            orient="records"
        )

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def discover_poss1_vizier(
    catalog_id: str = "VI/25",
    cache_dir: Path = Path("data/raw"),
    vizier_server: str = "vizier.cfa.harvard.edu",
) -> dict[str, Any]:
    """Inspect VizieR catalog VI/25 tables and columns."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file = cache_dir / "schema_poss1_vizier.json"

    v = Vizier(row_limit=5, columns=["**"], vizier_server=vizier_server)
    tables = v.get_catalogs(catalog_id)

    info: dict[str, Any] = {
        "catalog_id": catalog_id,
        "vizier_server": vizier_server,
        "tables": {},
    }
    for tname in tables.keys():
        tbl = tables[tname]
        cols = []
        for cname in tbl.colnames:
            col = tbl[cname]
            cols.append(
                {
                    "name": cname,
                    "dtype": str(col.dtype),
                    "unit": str(col.unit) if col.unit is not None else None,
                    "description": str(col.description) if col.description is not None else None,
                }
            )
        info["tables"][tname] = {
            "num_sample_rows": len(tbl),
            "columns": cols,
            "sample": tbl.to_pandas().head(2).to_dict(orient="records"),
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)

    return info


def _tap_query_async(service: pyvo.dal.TAPService, query: str, maxrec: int = 500000) -> pd.DataFrame:
    """Run an async TAP query and return a pandas DataFrame."""
    job = service.submit_job(query, maxrec=maxrec)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=300)
    if job.phase != "COMPLETED":
        raise RuntimeError(f"TAP job failed with phase={job.phase}: {job.query}")
    res = job.fetch_result()
    df = res.to_table().to_pandas()
    job.delete()
    return df


def fetch_applause(
    tap_url: str,
    cache_dir: Path = Path("data/raw"),
    rate_limit: float = 1.0,
    force: bool = False,
) -> dict[str, Path]:
    """Fetch and cache APPLAUSE DR4 metadata tables."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "archive": cache_dir / "applause_archive.parquet",
        "plate": cache_dir / "applause_plate.parquet",
        "exposure": cache_dir / "applause_exposure.parquet",
        "scan": cache_dir / "applause_scan.parquet",
        "solution": cache_dir / "applause_solution.parquet",
    }
    if not force and all(p.exists() for p in paths.values()):
        return paths

    svc = pyvo.dal.TAPService(tap_url)

    queries = {
        "archive": "SELECT archive_id, archive_name, institute, num_plates, num_scans FROM applause_dr4.archive",
        "plate": (
            "SELECT plate_id, archive_id, plate_num, series, plate_format, numexp, "
            "observatory, site_name, site_longitude, site_latitude, site_elevation, "
            "telescope, ota_name, instrument, ota_scale, filter, method_code "
            "FROM applause_dr4.plate"
        ),
        "exposure": (
            "SELECT exposure_id, plate_id, archive_id, exposure_num, object_name, "
            "ra_icrs, dec_icrs, flag_coord, ut_start, ut_mid, ut_end, "
            "year_start, year_mid, year_end, jd_start, jd_mid, jd_end, "
            "exptime, flag_time, num_sub, method_code "
            "FROM applause_dr4.exposure "
            "WHERE year_mid < 1958 OR year_mid IS NULL"
        ),
        "scan": (
            "SELECT scan_id, plate_id, archive_id, filename_scan, naxis1, naxis2, "
            "scanner, scan_res1, scan_res2, pixel_size1, pixel_size2, scan_notes "
            "FROM applause_dr4.scan"
        ),
        "solution": (
            "SELECT solution_id, scan_id, plate_id, archive_id, solution_num, "
            "ra_icrs, dec_icrs, fov1, fov2, pixel_scale, rotation_angle, "
            "ncp_on_plate, scp_on_plate, num_xmatch, num_gaia_edr3 "
            "FROM applause_dr4.solution"
        ),
    }

    for key, q in queries.items():
        out_p = paths[key]
        if out_p.exists() and not force:
            continue
        print(f"[APPLAUSE] Fetching {key}...")
        df = _tap_query_async(svc, q, maxrec=500000)
        # Convert any object/bytes columns cleanly
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).replace({"None": None, "nan": None, "<NA>": None})
        df.to_parquet(out_p, index=False)
        print(f"[APPLAUSE] Saved {len(df)} rows -> {out_p}")
        time.sleep(rate_limit)

    return paths


def fetch_dasch(
    tap_url: str,
    cache_dir: Path = Path("data/raw"),
    rate_limit: float = 1.0,
    force: bool = False,
) -> dict[str, Path]:
    """Fetch and cache DASCH narrow_plates and wide_plates (pre-1958 or null t_min)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "narrow": cache_dir / "dasch_narrow_plates.parquet",
        "wide": cache_dir / "dasch_wide_plates.parquet",
    }
    if not force and all(p.exists() for p in paths.values()):
        return paths

    svc = pyvo.dal.TAPService(tap_url)
    cols = (
        "dasch_id, obs_id, obs_collection, calib_level, s_ra, s_dec, s_fov, "
        "t_min, t_max, t_exptime, facility_name, instrument_name, "
        "s_xel1, s_xel2, s_pixel_scale, plate_class"
    )
    for key, table_name in [("narrow", "dasch.narrow_plates"), ("wide", "dasch.wide_plates")]:
        out_p = paths[key]
        if out_p.exists() and not force:
            continue
        q = f"SELECT {cols} FROM {table_name} WHERE t_min < 36116.0 OR t_min IS NULL"
        print(f"[DASCH] Fetching {table_name}...")
        df = _tap_query_async(svc, q, maxrec=500000)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).replace({"None": None, "nan": None, "<NA>": None})
        df.to_parquet(out_p, index=False)
        print(f"[DASCH] Saved {len(df)} rows -> {out_p}")
        time.sleep(rate_limit)

    return paths


def fetch_poss1(
    catalog_id: str = "VI/25",
    cache_dir: Path = Path("data/raw"),
    vizier_server: str = "vizier.cfa.harvard.edu",
    force: bool = False,
) -> Path:
    """Fetch and cache full POSS-I VI/25/nposs plate list."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_p = cache_dir / "poss1_nposs.parquet"
    if out_p.exists() and not force:
        return out_p

    print(f"[POSS-I] Fetching {catalog_id} from {vizier_server}...")
    v = Vizier(row_limit=-1, columns=["**"], vizier_server=vizier_server)
    tables = v.get_catalogs(catalog_id)
    df = tables[0].to_pandas()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).replace({"None": None, "nan": None, "<NA>": None})
    df.to_parquet(out_p, index=False)
    print(f"[POSS-I] Saved {len(df)} rows -> {out_p}")
    return out_p


def fetch_all(config_path: str | Path = "config.yaml", force: bool = False) -> None:
    cfg = load_config(config_path)
    cache_dir = Path("data/raw")
    rate_limit = float(cfg["data_sources"].get("rate_limit_seconds", 1.0))

    fetch_applause(
        cfg["data_sources"]["applause_tap_url"],
        cache_dir=cache_dir,
        rate_limit=rate_limit,
        force=force,
    )
    fetch_dasch(
        cfg["data_sources"]["dasch_tap_url"],
        cache_dir=cache_dir,
        rate_limit=rate_limit,
        force=force,
    )
    fetch_poss1(
        cfg["data_sources"]["poss1_vizier_catalog"],
        cache_dir=cache_dir,
        force=force,
    )


if __name__ == "__main__":
    fetch_all()
