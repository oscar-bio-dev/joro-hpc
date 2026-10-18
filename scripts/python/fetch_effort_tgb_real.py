#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/fetch_effort_tgb_real.py
Esfuerzo Target-Group Background (TGB) REAL para `cell_annual_observations.effort_araneae`.

Fuente: GBIF Maps API (densidad de ocurrencias, sin autenticación) para el orden *Araneae*
(taxonKey = 1496), años 2014-2024, con el bounding box de la malla `grid_5km`.
Cada tesela MVT (zoom 8, EPSG:4326) se pide con `bin=square&squareSize=8` (cuadrados de
~1.1 km) y se agrega a la celda de 5 km que contiene el centro del cuadrado (PostGIS).

Subcomandos:
  fetch   Descarga los cuadrados de conteo por año -> CSV reanudable en SCRATCH.
  ingest  Respalda el esfuerzo sintético previo y escribe el esfuerzo real en la BD.

Reemplaza el valor constante 238 (fondo) y el valor invertido 3 (celdas con presencia) del
esfuerzo sintético. Los datos de presencia (y_joro_detected, joro_count) NO se modifican.
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mapbox_vector_tile as mvt
import psycopg2
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("joro.effort")

DB_URL = "postgresql://joro_admin:REDACTED_DB_PASSWORD@localhost:5433/joro_db"
TAXON_KEY = 1496  # Araneae (orden)
YEARS = list(range(2014, 2025))
ZOOM = 8
SQUARE_PX = 8
OUT_DIR = Path("/mnt/hpc_scratch/joro-hpc/effort")
TILE_URL = "https://api.gbif.org/v2/map/occurrence/density/{z}/{x}/{y}.mvt"


def grid_bbox_wgs84() -> tuple[float, float, float, float]:
    with psycopg2.connect(DB_URL) as con, con.cursor() as cur:
        cur.execute("SELECT min(centroid_lon), min(centroid_lat), max(centroid_lon), max(centroid_lat) FROM grid_5km")
        lo0, la0, lo1, la1 = cur.fetchone()
    return lo0 - 0.1, la0 - 0.1, lo1 + 0.1, la1 + 0.1


def tile_range(bbox: tuple[float, float, float, float]) -> tuple[range, range, float]:
    deg = 180.0 / (2 ** ZOOM)
    lo0, la0, lo1, la1 = bbox
    xs = range(int((lo0 + 180.0) // deg), int((lo1 + 180.0) // deg) + 1)
    ys = range(int((90.0 - la1) // deg), int((90.0 - la0) // deg) + 1)
    return xs, ys, deg


def fetch_tile(session: requests.Session, year: int, tx: int, ty: int, deg: float) -> list[tuple[float, float, int]]:
    params = {
        "srs": "EPSG:4326",
        "taxonKey": TAXON_KEY,
        "year": f"{year},{year}",
        "bin": "square",
        "squareSize": SQUARE_PX,
    }
    for attempt in range(6):
        try:
            r = session.get(TILE_URL.format(z=ZOOM, x=tx, y=ty), params=params, timeout=60)
            if r.status_code == 204 or not r.content:
                return []
            r.raise_for_status()
            break
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"tesela {year}/{tx}/{ty} falló tras reintentos")

    lon0 = tx * deg - 180.0
    lat_top = 90.0 - ty * deg
    out: list[tuple[float, float, int]] = []
    layers = mvt.decode(r.content, default_options={"y_coord_down": True})
    for layer in layers.values():
        ext = float(layer.get("extent", 4096))
        for feat in layer["features"]:
            ring = feat["geometry"]["coordinates"][0]
            cx = sum(p[0] for p in ring[:4]) / 4.0
            cy = sum(p[1] for p in ring[:4]) / 4.0
            out.append((lon0 + cx / ext * deg, lat_top - cy / ext * deg, int(feat["properties"]["total"])))
    return out


def cmd_fetch(args: argparse.Namespace) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xs, ys, deg = tile_range(grid_bbox_wgs84())
    log.info("Teselas por año: %d x %d = %d (zoom %d, %.3f°/tesela)", len(xs), len(ys), len(xs) * len(ys), ZOOM, deg)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers))
    for year in args.years:
        path = OUT_DIR / f"gbif_araneae_squares_{year}.csv"
        if path.exists() and not args.force:
            log.info("%d: ya existe (%s), se omite", year, path.name)
            continue
        t0 = time.time()
        rows: list[tuple[float, float, int]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(fetch_tile, session, year, tx, ty, deg) for tx in xs for ty in ys]
            for fut in as_completed(futs):
                rows.extend(fut.result())
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["lon", "lat", "total"])
            w.writerows(rows)
        tmp.rename(path)
        log.info("%d: %d cuadrados, %d registros Araneae, %.0f s", year, len(rows), sum(r[2] for r in rows), time.time() - t0)
    return 0


def cmd_ingest(_: argparse.Namespace) -> int:
    con = psycopg2.connect(DB_URL)
    cur = con.cursor()
    cur.execute("SELECT to_regclass('public.cell_annual_observations_effort_placeholder_backup')")
    if cur.fetchone()[0] is None:
        log.info("Respaldando esfuerzo sintético previo -> cell_annual_observations_effort_placeholder_backup")
        cur.execute("CREATE TABLE cell_annual_observations_effort_placeholder_backup AS "
                    "SELECT cell_id, year, effort_araneae FROM cell_annual_observations")
    cur.execute("DROP TABLE IF EXISTS tmp_gbif_squares")
    cur.execute("CREATE TEMP TABLE tmp_gbif_squares (year smallint, lon double precision, lat double precision, total integer)")
    for year in YEARS:
        path = OUT_DIR / f"gbif_araneae_squares_{year}.csv"
        if not path.exists():
            log.error("Falta %s: ejecute primero `fetch`", path)
            return 1
        buf = io.StringIO()
        with open(path) as fh:
            next(fh)
            for line in fh:
                buf.write(f"{year},{line}")
        buf.seek(0)
        cur.copy_expert("COPY tmp_gbif_squares (year, lon, lat, total) FROM STDIN WITH (FORMAT csv)", buf)
    cur.execute("ANALYZE tmp_gbif_squares")
    log.info("Asignando cuadrados a celdas de 5 km (PostGIS)...")
    cur.execute("""
        CREATE TEMP TABLE tmp_effort AS
        SELECT g.cell_id, s.year, sum(s.total)::integer AS n
        FROM tmp_gbif_squares s
        JOIN grid_5km g
          ON ST_Contains(g.geom, ST_Transform(ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326), 5070))
        GROUP BY g.cell_id, s.year
    """)
    cur.execute("""
        UPDATE cell_annual_observations o
           SET effort_araneae = GREATEST(COALESCE(e.n, 0), o.joro_count, 0)
          FROM (SELECT o2.cell_id, o2.year, t.n
                  FROM cell_annual_observations o2
                  LEFT JOIN tmp_effort t ON t.cell_id = o2.cell_id AND t.year = o2.year) e
         WHERE o.cell_id = e.cell_id AND o.year = e.year
    """)
    log.info("Filas de esfuerzo actualizadas: %d", cur.rowcount)
    con.commit()
    cur.execute("""
        SELECT year, count(*), count(DISTINCT effort_araneae), min(effort_araneae),
               percentile_disc(0.5) WITHIN GROUP (ORDER BY effort_araneae), max(effort_araneae),
               sum((effort_araneae = 0)::int), sum(y_joro_detected)
        FROM cell_annual_observations GROUP BY year ORDER BY year
    """)
    print("year  filas  distintos  min  mediana  max  celdas_con_0  presencias")
    for r in cur.fetchall():
        print(*r, sep="  ")
    con.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--years", type=int, nargs="+", default=YEARS)
    f.add_argument("--workers", type=int, default=12)
    f.add_argument("--force", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    i = sub.add_parser("ingest")
    i.set_defaults(fn=cmd_ingest)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
