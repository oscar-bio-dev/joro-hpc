#!/usr/bin/env python3
"""
JORO-HPC · scripts/python/build_census_population.py
Población censal REAL en grid_5km (reemplaza el proxy por clase NLCD Developed).

Fuente: US Census Bureau, 2020 Census Redistricting Data (PL 94-171), archivo de encabezado geográfico
`{st}geo2020.pl` (delimitado por '|'; registros SUMLEV=750 = bloques censales) con los campos POP100 e
INTPTLAT/INTPTLON (punto interno del bloque). Se descarga un zip por estado del dominio [-95,-70] x [25,45].
(La capa TIGER2020PL TABBLOCK20 NO trae población en sus atributos, por eso se usa el encabezado PL94.)

Método: cada bloque aporta su POP100 a la celda 5 km (EPSG:5070) que contiene su punto interno.
  densidad = población de la celda / 25 km^2 ;  pop_density_h = estandarizar( log1p(densidad) )
(media 0, desviación estándar 1 sobre las 256,020 celdas, igual que las demás covariables).

Limitaciones: censo 2020 aplicado a toda la ventana 2014-2019 (no hay serie anual en esta capa);
la asignación por punto interno es una aproximación en bloques mayores que 5 km (rurales).
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psycopg2
import requests
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
from build_real_covariates import CELL, DB_URL, grid_geometry, standardize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("joro.census")

RAW = Path("/mnt/hpc_scratch/joro-hpc/census_raw")
URL = ("https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/"
       "{name}/{st}2020.pl.zip")
# Estados con territorio dentro (o en el borde) del dominio [-95,-70] x [25,45]
STATES = {
    "01": ("AL", "Alabama"), "05": ("AR", "Arkansas"), "09": ("CT", "Connecticut"), "10": ("DE", "Delaware"),
    "11": ("DC", "District_of_Columbia"), "12": ("FL", "Florida"), "13": ("GA", "Georgia"), "17": ("IL", "Illinois"),
    "18": ("IN", "Indiana"), "19": ("IA", "Iowa"), "20": ("KS", "Kansas"), "21": ("KY", "Kentucky"),
    "22": ("LA", "Louisiana"), "23": ("ME", "Maine"), "24": ("MD", "Maryland"), "25": ("MA", "Massachusetts"),
    "26": ("MI", "Michigan"), "27": ("MN", "Minnesota"), "28": ("MS", "Mississippi"), "29": ("MO", "Missouri"),
    "31": ("NE", "Nebraska"), "33": ("NH", "New_Hampshire"), "34": ("NJ", "New_Jersey"), "36": ("NY", "New_York"),
    "37": ("NC", "North_Carolina"), "39": ("OH", "Ohio"), "40": ("OK", "Oklahoma"), "42": ("PA", "Pennsylvania"),
    "44": ("RI", "Rhode_Island"), "45": ("SC", "South_Carolina"), "47": ("TN", "Tennessee"), "48": ("TX", "Texas"),
    "50": ("VT", "Vermont"), "51": ("VA", "Virginia"), "54": ("WV", "West_Virginia"), "55": ("WI", "Wisconsin"),
}


def download(fips: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    st, name = STATES[fips]
    zpath = RAW / f"{st.lower()}2020.pl.zip"
    if zpath.exists() and zipfile.is_zipfile(zpath):
        return zpath
    tmp = zpath.with_suffix(".part")
    for attempt in range(5):
        try:
            with requests.get(URL.format(name=name, st=st.lower()), stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.rename(zpath)
            return zpath
        except requests.RequestException as exc:
            log.warning("%s intento %d: %s", fips, attempt + 1, exc)
    raise RuntimeError(f"no se pudo descargar el estado {fips}")


def read_blocks(zpath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lon, lat, pop) de los bloques (SUMLEV 750) del encabezado geográfico PL94 del estado.

    Se indexa desde el final del registro (…|POP100|HU100|INTPTLAT|INTPTLON|LSADC|PARTFLAG|UGA), que es
    estable aunque el número de campos intermedios cambie.
    """
    lon, lat, pop = [], [], []
    with zipfile.ZipFile(zpath) as z:
        member = next(n for n in z.namelist() if n.endswith("geo2020.pl"))
        with z.open(member) as fh:
            for raw in io.TextIOWrapper(fh, encoding="latin-1"):
                f = raw.rstrip("\r\n").split("|")
                if len(f) < 10 or f[2] != "750":
                    continue
                pop.append(int(f[-7]))
                lat.append(float(f[-5]))
                lon.append(float(f[-4]))
    return np.array(lon), np.array(lat), np.array(pop, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gx0, gy0, nx, ny = grid_geometry()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        zips = dict(zip(STATES, pool.map(download, STATES)))

    to5070 = Transformer.from_crs(4326, 5070, always_xy=True)
    counts = np.zeros((nx, ny), dtype=np.float64)  # [ix, iy] -> cell_id = 1 + ix*ny + iy
    total_all, total_in = 0, 0
    for fips, zpath in zips.items():
        lon, lat, pop = read_blocks(zpath)
        x, y = to5070.transform(lon, lat)
        ix = np.floor((np.asarray(x) - gx0) / CELL).astype(int)
        iy = np.floor((np.asarray(y) - gy0) / CELL).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.add.at(counts, (ix[ok], iy[ok]), pop[ok])
        total_all += int(pop.sum())
        total_in += int(pop[ok].sum())
        log.info("%s %s: %d bloques, pob. total %s, dentro de la malla %s", fips, STATES[fips][0], len(pop),
                 f"{int(pop.sum()):,}", f"{int(pop[ok].sum()):,}")
    log.info("Población dentro de la malla: %s de %s en los estados descargados", f"{total_in:,}", f"{total_all:,}")

    flat = counts.reshape(-1)                      # índice = ix*ny + iy  == cell_id - 1
    dens = flat / (CELL / 1000.0) ** 2             # personas / km^2
    z = standardize(np.log1p(dens))
    log.info("celdas con población > 0: %d de %d | densidad máx %.0f /km2 | z: media %.4f sd %.4f",
             int((flat > 0).sum()), flat.size, dens.max(), z.mean(), z.std(ddof=1))
    if args.dry_run:
        return 0

    con = psycopg2.connect(DB_URL)
    cur = con.cursor()
    cur.execute("SELECT to_regclass('public.grid_5km_pop_proxy_backup')")
    if cur.fetchone()[0] is None:
        cur.execute("CREATE TABLE grid_5km_pop_proxy_backup AS SELECT cell_id, pop_density_h FROM grid_5km")
    cur.execute("ALTER TABLE grid_5km_covariates_raw ADD COLUMN IF NOT EXISTS pop_count real")
    cur.execute("ALTER TABLE grid_5km_covariates_raw ADD COLUMN IF NOT EXISTS pop_census_km2 real")
    cur.execute("CREATE TEMP TABLE tmp_pop (cell_id integer, pop_count real, dens real, z real)")
    buf = io.StringIO()
    for i in range(flat.size):
        buf.write(f"{i + 1},{flat[i]:.1f},{dens[i]:.4f},{z[i]:.6f}\n")
    buf.seek(0)
    cur.copy_expert("COPY tmp_pop FROM STDIN WITH (FORMAT csv)", buf)
    cur.execute("UPDATE grid_5km_covariates_raw r SET pop_count = t.pop_count, pop_census_km2 = t.dens "
                "FROM tmp_pop t WHERE r.cell_id = t.cell_id")
    cur.execute("UPDATE grid_5km g SET pop_density_h = t.z FROM tmp_pop t WHERE g.cell_id = t.cell_id")
    log.info("grid_5km.pop_density_h actualizada: %d filas", cur.rowcount)
    con.commit()
    cur.execute("SELECT avg(pop_density_h), stddev(pop_density_h) FROM grid_5km")
    print("SQL  avg/sd pop_density_h:", [round(v, 6) for v in cur.fetchone()])
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
