#!/usr/bin/env python3
"""
JORO-HPC · scripts/python/fetch_gbif_araneae_exact.py
Conteo EXACTO de ocurrencias georreferenciadas de Araneae (taxonKey = 1496) por celda 5 km y año.

Reemplaza la aproximación por rejilla de densidad (Maps API) de `fetch_effort_tgb_real.py`.
Fuente: GBIF Occurrence Search API (pública, sin autenticación): se paginan TODOS los registros
con coordenadas y sin problemas geoespaciales (hasCoordinate=true, hasGeospatialIssue=false) dentro del
bounding box de la malla, año a año (2014-2024). Como la API limita offset+limit a 100,000, el bounding box se
divide recursivamente hasta que cada pieza tenga <= 90,000 registros. Los registros se deduplican por `key` y se
valida que el total coincida con el `count` de GBIF para el año completo.

Subcomandos:
  fetch   Pagina GBIF -> CSV reanudable por año en SCRATCH (key, lon, lat)
          (la API pública resultó impracticable: ~40 s por página de 300 registros; 2014 tardó 26 min)
  fetch-opendata  Mismo conteo EXACTO leyendo el snapshot Parquet de GBIF en AWS Open Data con DuckDB
          (s3://gbif-open-data-us-east-1/occurrence/<fecha>/; order = 'Araneae' == taxonKey 1496; mismos filtros
          hasCoordinate / sin problemas geoespaciales). Valida contra el CSV de la API para 2014.
  ingest  Asigna cada registro a su celda (centroide por punto en la celda EPSG:5070) y escribe
          effort_araneae = GREATEST(conteo, joro_count) en cell_annual_observations
          (respaldo previo: cell_annual_observations_effort_density_backup)
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psycopg2
import requests
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
from build_real_covariates import CELL, DB_URL, grid_geometry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("joro.gbif.exact")

API = "https://api.gbif.org/v1/occurrence/search"
TAXON_KEY = 1496
YEARS = list(range(2014, 2025))
OUT_DIR = Path("/mnt/hpc_scratch/joro-hpc/effort")
BBOX = (-95.0, 25.0, -70.0, 45.0)  # lon_min, lat_min, lon_max, lat_max (dominio de la malla)
PAGE = 300
MAX_PER_QUERY = 3_000  # cajas pequeñas: evita la paginación profunda (offset alto), que en GBIF es muy lenta


def _get(session: requests.Session, params: dict) -> dict:
    for attempt in range(8):
        try:
            r = session.get(API, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            log.warning("HTTP %s (intento %d)", r.status_code, attempt + 1)
        except (requests.RequestException, ValueError) as exc:
            log.warning("error de red/JSON (intento %d): %s", attempt + 1, exc)
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GBIF no respondió tras reintentos: {params}")


def _params(year: int, bbox: tuple[float, float, float, float], **extra) -> dict:
    lo0, la0, lo1, la1 = bbox
    return {"taxonKey": TAXON_KEY, "year": year, "hasCoordinate": "true", "hasGeospatialIssue": "false",
            "decimalLongitude": f"{lo0},{lo1}", "decimalLatitude": f"{la0},{la1}", **extra}


def split_boxes(session, year, bbox) -> list[tuple[tuple, int]]:
    n = _get(session, _params(year, bbox, limit=0))["count"]
    if n <= MAX_PER_QUERY:
        return [(bbox, n)] if n > 0 else []
    lo0, la0, lo1, la1 = bbox
    if (lo1 - lo0) >= (la1 - la0):
        mid = (lo0 + lo1) / 2.0
        halves = [(lo0, la0, mid, la1), (mid, la0, lo1, la1)]
    else:
        mid = (la0 + la1) / 2.0
        halves = [(lo0, la0, lo1, mid), (lo0, mid, lo1, la1)]
    return split_boxes(session, year, halves[0]) + split_boxes(session, year, halves[1])


def fetch_page(session, year, bbox, offset):
    data = _get(session, _params(year, bbox, limit=PAGE, offset=offset))
    return [(r["key"], r["decimalLongitude"], r["decimalLatitude"]) for r in data["results"]
            if "decimalLongitude" in r and "decimalLatitude" in r]


def cmd_fetch(args) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers))
    for year in args.years:
        path = OUT_DIR / f"gbif_araneae_exact_{year}.csv"
        if path.exists() and not args.force:
            log.info("%d: ya existe, se omite", year)
            continue
        t0 = time.time()
        total = _get(session, _params(year, BBOX, limit=0))["count"]
        boxes = split_boxes(session, year, BBOX)
        assert sum(n for _, n in boxes) >= total, f"{year}: la partición no cubre el total de GBIF"  # >=: registros exactamente sobre el borde entre cajas se deduplican por key
        log.info("%d: %d registros en %d cajas", year, total, len(boxes))
        jobs = [(b, off) for b, n in boxes for off in range(0, n, PAGE)]
        recs: dict[int, tuple[float, float]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for page in pool.map(lambda j: fetch_page(session, year, j[0], j[1]), jobs):
                for key, lon, lat in page:
                    recs[key] = (lon, lat)
        if len(recs) != total:
            raise RuntimeError(f"{year}: {len(recs)} registros únicos obtenidos vs {total} esperados")
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "lon", "lat"])
            for key, (lon, lat) in recs.items():
                w.writerow([key, f"{lon:.6f}", f"{lat:.6f}"])
        tmp.rename(path)
        log.info("%d: %d registros exactos guardados (%.0f s, %d páginas)", year, len(recs), time.time() - t0, len(jobs))
    return 0


def cmd_fetch_opendata(args) -> int:
    """Snapshot GBIF en AWS Open Data (Parquet) consultado con DuckDB: mismos filtros que la API."""
    import duckdb
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lon0, lat0, lon1, lat1 = BBOX
    src = f"s3://gbif-open-data-us-east-1/occurrence/{args.snapshot}/occurrence.parquet/*"
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-east-1'; "
                f"SET threads={args.threads};")
    bad = "ZERO_COORDINATE|COORDINATE_OUT_OF_RANGE|COORDINATE_INVALID|COUNTRY_COORDINATE_MISMATCH"
    t0 = time.time()
    log.info("Consultando snapshot %s (Araneae, bbox, %d-%d)...", args.snapshot, min(YEARS), max(YEARS))
    df = con.execute(f"""
        SELECT gbifid, year, decimallongitude AS lon, decimallatitude AS lat
        FROM read_parquet('{src}')
        WHERE "order" = 'Araneae' AND kingdom = 'Animalia'
          AND year BETWEEN {min(YEARS)} AND {max(YEARS)}
          AND decimallatitude BETWEEN {lat0} AND {lat1}
          AND decimallongitude BETWEEN {lon0} AND {lon1}
          AND NOT regexp_matches(coalesce(CAST(issue AS VARCHAR), ''), '{bad}')
    """).df()
    log.info("Consulta completada: %d registros en %.0f s", len(df), time.time() - t0)
    for year in YEARS:
        sub = df[df["year"] == year].drop_duplicates("gbifid")
        path = OUT_DIR / f"gbif_araneae_exact_{year}.csv"
        if year == 2014 and path.exists() and not path.with_suffix(".api.csv").exists():
            api_keys = set(np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0,), dtype=np.int64, ndmin=1).tolist())
            snap_keys = set(sub["gbifid"].astype(np.int64).tolist())
            inter = len(api_keys & snap_keys)
            log.info("Validación 2014: API %d | snapshot %d | comunes %d (%.2f%% de la API)",
                     len(api_keys), len(snap_keys), inter, 100.0 * inter / max(1, len(api_keys)))
            path.rename(path.with_suffix(".api.csv"))
        tmp = path.with_suffix(".tmp")
        sub[["gbifid", "lon", "lat"]].rename(columns={"gbifid": "key"}).to_csv(tmp, index=False, float_format="%.6f")
        tmp.rename(path)
        log.info("%d: %d registros exactos guardados", year, len(sub))
    return 0


def cmd_ingest(args) -> int:
    gx0, gy0, nx, ny = grid_geometry()
    to5070 = Transformer.from_crs(4326, 5070, always_xy=True)
    rows = []  # (cell_id, year, n)
    totals = {}
    for year in args.years:
        path = OUT_DIR / f"gbif_araneae_exact_{year}.csv"
        if not path.exists():
            log.error("Falta %s: ejecute primero `fetch`", path)
            return 1
        arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2), ndmin=2)
        x, y = to5070.transform(arr[:, 0], arr[:, 1])
        ix = np.floor((np.asarray(x) - gx0) / CELL).astype(int)
        iy = np.floor((np.asarray(y) - gy0) / CELL).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        cid = 1 + ix[ok] * ny + iy[ok]
        u, c = np.unique(cid, return_counts=True)
        rows += [(int(a), year, int(b)) for a, b in zip(u, c)]
        totals[year] = (len(arr), int(ok.sum()))
        log.info("%d: %d registros, %d dentro de la malla, %d celdas con registros", year, len(arr), ok.sum(), len(u))

    con = psycopg2.connect(DB_URL)
    cur = con.cursor()
    cur.execute("SELECT to_regclass('public.cell_annual_observations_effort_density_backup')")
    if cur.fetchone()[0] is None:
        cur.execute("CREATE TABLE cell_annual_observations_effort_density_backup AS "
                    "SELECT cell_id, year, effort_araneae FROM cell_annual_observations")
    cur.execute("CREATE TEMP TABLE tmp_effort (cell_id integer, year smallint, n integer)")
    buf = io.StringIO()
    for r in rows:
        buf.write(f"{r[0]},{r[1]},{r[2]}\n")
    buf.seek(0)
    cur.copy_expert("COPY tmp_effort FROM STDIN WITH (FORMAT csv)", buf)
    cur.execute("""
        UPDATE cell_annual_observations o
           SET effort_araneae = GREATEST(COALESCE(e.n, 0), o.joro_count, 0)
          FROM (SELECT o2.cell_id, o2.year, t.n
                  FROM cell_annual_observations o2
                  LEFT JOIN tmp_effort t ON t.cell_id = o2.cell_id AND t.year = o2.year) e
         WHERE o.cell_id = e.cell_id AND o.year = e.year AND o.year = ANY(%s)
    """, (list(args.years),))
    log.info("Filas de esfuerzo actualizadas: %d", cur.rowcount)
    con.commit()
    cur.execute("""
        SELECT year, count(*), sum(effort_araneae), max(effort_araneae), sum((effort_araneae > 0)::int),
               count(DISTINCT effort_araneae)
        FROM cell_annual_observations WHERE year = ANY(%s) GROUP BY year ORDER BY year
    """, (list(args.years),))
    print("year  celdas  sum_esfuerzo  max  celdas>0  distintos   (no se leen etiquetas de presencia)")
    for r in cur.fetchall():
        print(*r, sep="  ")
    con.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--years", type=int, nargs="+", default=YEARS)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--force", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    o = sub.add_parser("fetch-opendata")
    o.add_argument("--snapshot", default="2026-10-01")
    o.add_argument("--threads", type=int, default=8)
    o.set_defaults(fn=cmd_fetch_opendata)
    i = sub.add_parser("ingest")
    i.add_argument("--years", type=int, nargs="+", default=YEARS,
                   help="años a escribir (2014-2019 para calibración; 2020-2024 solo se prepara para la Fase 5)")
    i.set_defaults(fn=cmd_ingest)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
