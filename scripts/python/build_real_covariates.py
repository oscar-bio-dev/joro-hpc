#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/build_real_covariates.py
Sustituye las covariables CONSTANTES de `grid_5km` (z_imp = z_borde = z_dosel = 0.5,
pop_density_h = 15.0, road_density_r = 1.2) por capas espacialmente explícitas derivadas de
datos públicos reales del este de EE. UU., estandarizadas (media 0, desviación estándar 1)
sobre las 256,020 celdas del dominio.

Fuentes (todas reales):
  * NLCD 2019 Land Cover   (MRLC WCS, 250 m remuestreo por vecino más cercano, EPSG:5070)
  * NLCD 2019 Impervious   (MRLC WCS, idem)
  * U.S. Census TIGER/Line 2019 Primary Roads (interestatales y carreteras primarias)

Capas (por celda de 5 km, 20 x 20 píxeles de 250 m):
  z_imp          fracción impermeable media (%)                          -> log1p -> z
  z_dosel        fracción de cobertura forestal (NLCD 41, 42, 43, 90)    -> z
  z_borde        densidad de TRANSICIONES de cobertura: fracción de píxeles con >=1 vecino (4-conexo)
                 de otra clase amplia (urbano, bosque, matorral/pastizal, cultivo, agua, humedal, suelo
                 desnudo); incluye el borde de bosque pero no crece con la cantidad de bosque  -> z
  pop_density_h  PROXY de densidad humana: clases NLCD "Developed" con densidades típicas por
                 clase (NO es dato censal)                               -> log1p -> z
  road_density_r km de carretera primaria por km^2 (TIGER)               -> log1p(10 x) -> z

El script respalda los valores previos en `grid_5km_covariates_placeholder_backup` y guarda
las capas físicas (sin estandarizar) en `grid_5km_covariates_raw` para trazabilidad.
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psycopg2
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("joro.covariates")

Image.MAX_IMAGE_PIXELS = None
DB_URL = "postgresql://joro_admin:REDACTED_DB_PASSWORD@localhost:5433/joro_db"
WCS = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
LAYERS = {
    "lc": "mrlc_download__NLCD_2019_Land_Cover_L48",
    "imp": "mrlc_download__NLCD_2019_Impervious_L48",
}
NLCD_X = (-2493045.0, 2342655.0)  # extensión de la cobertura (EPSG:5070)
NLCD_Y = (177285.0, 3310005.0)
PIX = 250.0
CELL = 5000.0
PPC = int(CELL / PIX)  # 20 píxeles por lado de celda
TILE_PX = 2000
RAW_DIR = Path("/mnt/hpc_scratch/joro-hpc/nlcd_raw")
ROADS_URL = "https://www2.census.gov/geo/tiger/TIGER2019/PRIMARYROADS/tl_2019_us_primaryroads.zip"

FOREST = (41, 42, 43, 90)
# Clases amplias de cobertura para detectar transiciones (borde)
BROAD_GROUP = {11: 1, 21: 2, 22: 2, 23: 2, 24: 2, 31: 3, 41: 4, 42: 4, 43: 4, 90: 4, 95: 5,
               52: 6, 71: 6, 81: 7, 82: 7}
# Densidad humana típica [personas/km^2] por clase NLCD "Developed" (proxy, no censal)
DEV_DENSITY = {21: 30.0, 22: 700.0, 23: 2500.0, 24: 7000.0}


def grid_geometry() -> tuple[float, float, int, int]:
    with psycopg2.connect(DB_URL) as con, con.cursor() as cur:
        cur.execute("SELECT min(ST_XMin(geom)), min(ST_YMin(geom)), count(DISTINCT ST_XMin(geom)), "
                    "count(DISTINCT ST_YMin(geom)), count(*) FROM grid_5km")
        x0, y0, nx, ny, n = cur.fetchone()
    assert nx * ny == n, "la malla no es rectangular completa"
    return float(x0), float(y0), int(nx), int(ny)


def fetch_tile(layer: str, x0: float, x1: float, y0: float, y1: float, path: Path) -> None:
    if path.exists():
        return
    w, h = int(round((x1 - x0) / PIX)), int(round((y1 - y0) / PIX))
    params = {
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
        "coverageId": LAYERS[layer], "format": "image/tiff",
        "subset": [f"X({x0},{x1})", f"Y({y0},{y1})"], "scalesize": f"i({w}),j({h})",
    }
    for attempt in range(6):
        try:
            r = requests.get(WCS, params=params, timeout=900)
            if r.status_code == 200 and "tif" in r.headers.get("content-type", ""):
                path.write_bytes(r.content)
                return
            log.warning("WCS %s %s: HTTP %s %s", layer, path.name, r.status_code, r.text[:120])
        except requests.RequestException as exc:
            log.warning("WCS %s %s: %s", layer, path.name, exc)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"No se pudo descargar {path.name}")


def load_layer(layer: str, gx0: float, gy0: float, nx: int, ny: int, workers: int) -> np.ndarray:
    """Devuelve el ráster [filas (sur->norte), columnas (oeste->este)] uint8 sobre la malla; 255 = sin dato."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    W, H = nx * PPC, ny * PPC
    out = np.full((H, W), 255, dtype=np.uint8)
    jobs = []
    for ty in range(0, H, TILE_PX):
        for tx in range(0, W, TILE_PX):
            tw, th = min(TILE_PX, W - tx), min(TILE_PX, H - ty)
            x0, x1 = gx0 + tx * PIX, gx0 + (tx + tw) * PIX
            y0, y1 = gy0 + ty * PIX, gy0 + (ty + th) * PIX
            cx0, cx1 = max(x0, NLCD_X[0]), min(x1, NLCD_X[1])
            cy0, cy1 = max(y0, NLCD_Y[0]), min(y1, NLCD_Y[1])
            if cx1 - cx0 < PIX or cy1 - cy0 < PIX:
                continue
            # alinear los recortes a la rejilla de 250 m de la malla
            cx0 = gx0 + np.ceil((cx0 - gx0) / PIX) * PIX
            cx1 = gx0 + np.floor((cx1 - gx0) / PIX) * PIX
            cy0 = gy0 + np.ceil((cy0 - gy0) / PIX) * PIX
            cy1 = gy0 + np.floor((cy1 - gy0) / PIX) * PIX
            path = RAW_DIR / f"{layer}_{int(cx0)}_{int(cy0)}_{int(cx1)}_{int(cy1)}.tif"
            jobs.append((tx, ty, cx0, cx1, cy0, cy1, path))
    log.info("%s: %d teselas WCS", layer, len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda j: fetch_tile(layer, j[2], j[3], j[4], j[5], j[6]), jobs))
    for tx, ty, cx0, cx1, cy0, cy1, path in jobs:
        arr = np.array(Image.open(path))
        arr = arr[::-1]  # la imagen viene de norte a sur; se invierte a sur -> norte
        c0 = int(round((cx0 - gx0) / PIX))
        r0 = int(round((cy0 - gy0) / PIX))
        out[r0:r0 + arr.shape[0], c0:c0 + arr.shape[1]] = arr
    return out


def block_mean(a: np.ndarray, ny: int, nx: int) -> np.ndarray:
    return a.reshape(ny, PPC, nx, PPC).mean(axis=(1, 3))


def transition_edge(group: np.ndarray) -> np.ndarray:
    """Píxeles con al menos un vecino 4-conexo de otra clase amplia (los sin dato no cuentan)."""
    p = np.pad(group, 1, mode="edge")
    edge = np.zeros(group.shape, dtype=bool)
    for nb in (p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]):
        edge |= (nb != group) & (nb != 0) & (group != 0)
    return edge


def road_density(gx0: float, gy0: float, nx: int, ny: int) -> np.ndarray:
    import geopandas as gpd

    log.info("Descargando TIGER/Line Primary Roads 2019...")
    r = requests.get(ROADS_URL, timeout=900)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    tmp = RAW_DIR / "tiger_primaryroads"
    tmp.mkdir(parents=True, exist_ok=True)
    zf.extractall(tmp)
    roads = gpd.read_file(next(tmp.glob("*.shp"))).to_crs(5070)
    counts = np.zeros((ny, nx), dtype=np.float64)
    step = 250.0
    for geom in roads.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for line in parts:
            n = max(2, int(line.length // step) + 1)
            dists = np.linspace(0.0, line.length, n)
            pts = np.array([line.interpolate(d).coords[0] for d in dists])
            ix = np.floor((pts[:, 0] - gx0) / CELL).astype(int)
            iy = np.floor((pts[:, 1] - gy0) / CELL).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            seg_len = line.length / max(1, n - 1)
            np.add.at(counts, (iy[ok], ix[ok]), seg_len)
    return counts / 1000.0 / 25.0  # km de carretera por km^2


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="calcula y reporta sin escribir en la BD")
    args = ap.parse_args()

    gx0, gy0, nx, ny = grid_geometry()
    log.info("Malla: x0=%.0f y0=%.0f, %d columnas x %d filas (cell_id = 1 + ix*%d + iy)", gx0, gy0, nx, ny, ny)

    lc = load_layer("lc", gx0, gy0, nx, ny, args.workers)
    imp = load_layer("imp", gx0, gy0, nx, ny, args.workers)
    valid = (lc != 255) & (lc != 0)
    log.info("Píxeles con dato NLCD: %.1f %%", 100.0 * valid.mean())

    imp_f = np.where((imp <= 100) & valid, imp, 0).astype(np.float32)
    forest = np.isin(lc, FOREST) & valid
    group = np.zeros(lc.shape, dtype=np.uint8)
    for cls, g in BROAD_GROUP.items():
        group[lc == cls] = g
    edge = transition_edge(group)
    dens = np.zeros(lc.shape, dtype=np.float32)
    for cls, d in DEV_DENSITY.items():
        dens[lc == cls] = d

    raw = {
        "imp_pct": block_mean(imp_f, ny, nx),
        "forest_frac": block_mean(forest.astype(np.float32), ny, nx),
        "edge_frac": block_mean(edge.astype(np.float32), ny, nx),
        "pop_proxy": block_mean(dens, ny, nx),
        "valid_frac": block_mean(valid.astype(np.float32), ny, nx),
    }
    raw["road_km_km2"] = road_density(gx0, gy0, nx, ny)

    # arrays [iy, ix] -> vector por cell_id = 1 + ix*ny + iy  (ix mayor => transponer a [ix, iy])
    flat = {k: v.T.reshape(-1) for k, v in raw.items()}
    cell_ids = np.arange(1, nx * ny + 1)

    z = {
        "z_imp": standardize(np.log1p(flat["imp_pct"])),
        "z_dosel": standardize(flat["forest_frac"]),
        "z_borde": standardize(flat["edge_frac"]),
        "pop_density_h": standardize(np.log1p(flat["pop_proxy"])),
        "road_density_r": standardize(np.log1p(10.0 * flat["road_km_km2"])),
    }
    names = list(z)
    corr = np.corrcoef(np.vstack([z[k] for k in names]))
    log.info("Correlación entre covariables estandarizadas:\n%s",
             "\n".join(f"{n:>15s} " + " ".join(f"{c:6.2f}" for c in row) for n, row in zip(names, corr)))
    for k in names:
        log.info("%-15s media %.4f  sd %.4f  min %.2f  max %.2f", k, z[k].mean(), z[k].std(ddof=1), z[k].min(), z[k].max())

    if args.dry_run:
        return 0

    con = psycopg2.connect(DB_URL)
    cur = con.cursor()
    cur.execute("SELECT to_regclass('public.grid_5km_covariates_placeholder_backup')")
    if cur.fetchone()[0] is None:
        log.info("Respaldando covariables previas -> grid_5km_covariates_placeholder_backup")
        cur.execute("CREATE TABLE grid_5km_covariates_placeholder_backup AS "
                    "SELECT cell_id, z_imp, z_borde, z_dosel, pop_density_h, road_density_r FROM grid_5km")
    cur.execute("DROP TABLE IF EXISTS grid_5km_covariates_raw")
    cur.execute("CREATE TABLE grid_5km_covariates_raw (cell_id integer PRIMARY KEY, imp_pct real, forest_frac real, "
                "edge_frac real, pop_proxy real, road_km_km2 real, valid_frac real)")
    buf = io.StringIO()
    for i, cid in enumerate(cell_ids):
        buf.write(f"{cid},{flat['imp_pct'][i]:.5f},{flat['forest_frac'][i]:.5f},{flat['edge_frac'][i]:.5f},"
                  f"{flat['pop_proxy'][i]:.4f},{flat['road_km_km2'][i]:.5f},{flat['valid_frac'][i]:.4f}\n")
    buf.seek(0)
    cur.copy_expert("COPY grid_5km_covariates_raw FROM STDIN WITH (FORMAT csv)", buf)
    cur.execute("COMMENT ON TABLE grid_5km_covariates_raw IS "
                "'Capas físicas sin estandarizar: NLCD 2019 (imp, bosque, borde), proxy poblacional por clase NLCD Developed, TIGER 2019 primary roads'")

    cur.execute("CREATE TEMP TABLE tmp_z (cell_id integer, z_imp real, z_borde real, z_dosel real, pop real, road real)")
    buf = io.StringIO()
    for i, cid in enumerate(cell_ids):
        buf.write(f"{cid},{z['z_imp'][i]:.6f},{z['z_borde'][i]:.6f},{z['z_dosel'][i]:.6f},"
                  f"{z['pop_density_h'][i]:.6f},{z['road_density_r'][i]:.6f}\n")
    buf.seek(0)
    cur.copy_expert("COPY tmp_z FROM STDIN WITH (FORMAT csv)", buf)
    cur.execute("UPDATE grid_5km g SET z_imp = t.z_imp, z_borde = t.z_borde, z_dosel = t.z_dosel, "
                "pop_density_h = t.pop, road_density_r = t.road FROM tmp_z t WHERE g.cell_id = t.cell_id")
    log.info("grid_5km actualizada: %d filas", cur.rowcount)
    con.commit()

    cur.execute("SELECT avg(z_imp), stddev(z_imp), avg(z_borde), stddev(z_borde), avg(z_dosel), stddev(z_dosel), "
                "stddev(pop_density_h), stddev(road_density_r) FROM grid_5km")
    print("SQL  avg/sd  z_imp, z_borde, z_dosel | sd pop, sd road:", [round(v, 4) for v in cur.fetchone()])
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
