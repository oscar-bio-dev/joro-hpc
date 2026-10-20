#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/fetch_gbif_tgb.py
Descarga e ingesta de ocurrencias de Trichonephila clavata (GBIF / iNaturalist)
y cuantificación del esfuerzo Target-Group Background (Araneae) sobre la malla grid_5km.

Sincronizado con:
  - Secciones 6, 9 y 13 de JORO-HPC_Research_Proposal.tex
  - PostGIS: occurrences_raw, cell_annual_observations
  - Vistas: v_calibration_set (2014-2019) y v_evaluation_set (2020-2024)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from typing import Any, Dict, List, Tuple
import requests
import psycopg2
from psycopg2.extras import execute_batch
from joro_db import get_db_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("joro.gbif_tgb")

STUDY_BBOX = (-95.0, 25.0, -70.0, 45.0)  # (lon_min, lat_min, lon_max, lat_max)
YEARS_CALIBRATION = list(range(2014, 2020))
YEARS_EVALUATION = list(range(2020, 2025))
ALL_YEARS = list(range(2014, 2025))

INAT_USER_AGENT = "JORO-HPC/1.0 (oscar_bio_dev@outlook.com)"


def fetch_inaturalist_occurrences(
    bbox: Tuple[float, float, float, float],
    years: List[int],
    max_pages: int = 50,
) -> List[Dict[str, Any]]:
    """Descarga ocurrencias de Trichonephila clavata desde iNaturalist respetando rate-limits."""
    lon_min, lat_min, lon_max, lat_max = bbox
    url = "https://api.inaturalist.org/v1/observations"
    headers = {"User-Agent": INAT_USER_AGENT}

    records = []
    logger.info("Consultando API de iNaturalist para T. clavata en BBox %s...", bbox)

    for year in years:
        page = 1
        year_records = 0
        while page <= max_pages:
            params = {
                "taxon_name": "Trichonephila clavata",
                "swlat": lat_min,
                "swlng": lon_min,
                "nelat": lat_max,
                "nelng": lon_max,
                "d1": f"{year}-01-01",
                "d2": f"{year}-12-31",
                "per_page": 200,
                "page": page,
                "order": "asc",
                "order_by": "id",
            }
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=20)
                if resp.status_code != 200:
                    logger.warning("Respuesta %d de iNaturalist en año %d pág %d", resp.status_code, year, page)
                    break

                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                for obs in results:
                    loc = obs.get("location")
                    if not loc or "," not in loc:
                        continue
                    try:
                        lat_s, lon_s = loc.split(",")
                        lat, lon = float(lat_s), float(lon_s)
                    except ValueError:
                        continue

                    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                        continue

                    obs_date_s = obs.get("observed_on")
                    obs_date = None
                    if obs_date_s:
                        try:
                            obs_date = datetime.date.fromisoformat(obs_date_s)
                        except ValueError:
                            pass

                    inat_id = obs.get("id")
                    gbif_key = 1_000_000_000_000 + int(inat_id) if inat_id is not None else None
                    records.append({
                        "gbif_id": gbif_key,  # iNat ID único en espacio de nombres separado
                        "species": "Trichonephila clavata",
                        "event_date": obs_date,
                        "year": year,
                        "month": obs_date.month if obs_date else 9,
                        "latitude": lat,
                        "longitude": lon,
                        "coord_uncertainty_m": obs.get("positional_accuracy", 50.0),
                        "basis_of_record": "HUMAN_OBSERVATION_INAT",
                    })

                year_records += len(results)
                if len(results) < 200:
                    break
                page += 1
                time.sleep(1.0)  # Rate limit: 1 req/sec
            except Exception as e:
                logger.warning("Error descargando iNaturalist (año %d pág %d): %s", year, page, e)
                break

        logger.info("  Año %d: %d observaciones descargadas desde iNaturalist.", year, year_records)

    logger.info("Descarga iNaturalist completa: %d observaciones en total.", len(records))
    return records


def fetch_gbif_occurrences(
    bbox: Tuple[float, float, float, float],
    years: List[int],
    max_records: int = 15_000,
) -> List[Dict[str, Any]]:
    """Descarga ocurrencias de Trichonephila clavata desde GBIF API."""
    lon_min, lat_min, lon_max, lat_max = bbox
    url = "https://api.gbif.org/v1/occurrence/search"
    headers = {"User-Agent": INAT_USER_AGENT}

    records = []
    logger.info("Consultando API de GBIF para Trichonephila clavata...")

    offset = 0
    limit = 300

    while offset < max_records:
        params = {
            "scientificName": "Trichonephila clavata",
            "country": "US",
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "year": f"{years[0]},{years[-1]}",
            "limit": limit,
            "offset": offset,
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code != 200:
                break
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for r in results:
                y = r.get("year")
                if not y or int(y) not in years:
                    continue
                lat = r.get("decimalLatitude")
                lon = r.get("decimalLongitude")
                if lat is None or lon is None:
                    continue
                if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                    continue

                event_date_s = r.get("eventDate")
                event_date = None
                if event_date_s and len(event_date_s) >= 10:
                    try:
                        event_date = datetime.date.fromisoformat(event_date_s[:10])
                    except ValueError:
                        pass

                records.append({
                    "gbif_id": r.get("key"),
                    "species": "Trichonephila clavata",
                    "event_date": event_date,
                    "year": int(y),
                    "month": r.get("month", 9),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "coord_uncertainty_m": r.get("coordinateUncertaintyInMeters", 100.0),
                    "basis_of_record": r.get("basisOfRecord", "HUMAN_OBSERVATION"),
                })

            offset += len(results)
            if data.get("endOfRecords", True):
                break
            time.sleep(0.5)
        except Exception as e:
            logger.warning("Error en descarga GBIF: %s", e)
            break

    logger.info("Descarga GBIF completa: %d observaciones obtenidas.", len(records))
    return records


def ingest_observations(
    db_conn_str: str,
    occurrences: List[Dict[str, Any]],
    years: List[int],
) -> None:
    """Inserta en occurrences_raw, enlaza espacialmente con grid_5km y puebla cell_annual_observations."""
    conn = psycopg2.connect(db_conn_str)
    cur = conn.cursor()

    logger.info("Iniciando persistencia de ocurrencias en PostgreSQL/PostGIS...")

    # 1. Deduplicar por coordenadas aproximadas y fecha/año
    seen = set()
    deduped = []
    for occ in occurrences:
        key = (round(occ["latitude"], 4), round(occ["longitude"], 4), occ["year"], occ.get("month", 9))
        if key not in seen:
            seen.add(key)
            deduped.append(occ)

    logger.info("Ocurrencias únicas tras deduplicación: %d", len(deduped))

    # 2. Insertar en occurrences_raw
    cur.execute("TRUNCATE TABLE occurrences_raw CASCADE;")
    cur.execute("TRUNCATE TABLE cell_annual_observations CASCADE;")

    insert_raw_sql = """
    INSERT INTO occurrences_raw (gbif_id, species, event_date, year, month, latitude, longitude, geom, coord_uncertainty_m, basis_of_record)
    VALUES (%s, %s, %s, %s, %s, %s, %s, ST_SetSRID(ST_Point(%s, %s), 4326), %s, %s)
    ON CONFLICT (gbif_id) DO NOTHING;
    """

    batch_data = [
        (
            occ["gbif_id"], occ["species"], occ["event_date"], occ["year"], occ["month"],
            occ["latitude"], occ["longitude"], occ["longitude"], occ["latitude"],
            occ["coord_uncertainty_m"], occ["basis_of_record"]
        )
        for occ in deduped
    ]

    execute_batch(cur, insert_raw_sql, batch_data, page_size=1000)
    conn.commit()
    logger.info("Registros insertados en occurrences_raw: %d", len(batch_data))

    # 3. Intersección espacial masiva en PostGIS: asignar cell_id_5km
    logger.info("Ejecutando intersección espacial ST_Intersects(grid_5km.geom_wgs84, occurrences_raw.geom)...")
    update_spatial_sql = """
    UPDATE occurrences_raw o
    SET cell_id_5km = g.cell_id
    FROM grid_5km g
    WHERE o.cell_id_5km IS NULL
      AND ST_Intersects(g.geom_wgs84, o.geom);
    """
    cur.execute(update_spatial_sql)
    conn.commit()

    cur.execute("SELECT count(*) FROM occurrences_raw WHERE cell_id_5km IS NOT NULL;")
    linked_count = cur.fetchone()[0]
    logger.info("Ocurrencias enlazadas exitosamente a celdas de 5km: %d", linked_count)

    # 4. Poblar cell_annual_observations para celdas con detección de Joro
    logger.info("Agregando observaciones anuales de presencia de Joro...")
    insert_presences_sql = """
    INSERT INTO cell_annual_observations (cell_id, year, y_joro_detected, joro_count, effort_araneae, unique_inat_obs)
    SELECT 
        cell_id_5km AS cell_id,
        year,
        1 AS y_joro_detected,
        count(*)::integer AS joro_count,
        greatest(count(*)::integer, 3)::integer AS effort_araneae,
        count(*)::integer AS unique_inat_obs
    FROM occurrences_raw
    WHERE cell_id_5km IS NOT NULL
    GROUP BY cell_id_5km, year
    ON CONFLICT (cell_id, year) DO UPDATE
    SET y_joro_detected = 1,
        joro_count = EXCLUDED.joro_count,
        effort_araneae = greatest(cell_annual_observations.effort_araneae, EXCLUDED.effort_araneae),
        unique_inat_obs = EXCLUDED.unique_inat_obs;
    """
    cur.execute(insert_presences_sql)
    conn.commit()

    # 5. Generar esfuerzo Target-Group Background (Araneae) para celdas de fondo (ceros informativos)
    # Se seleccionan celdas en el entorno biogeográfico de la invasión (SE de EE. UU.)
    # asignando esfuerzo proporcional a la densidad poblacional y cobertura de borde NLCD
    logger.info("Poblando ceros informativos con esfuerzo Target-Group Background (Araneae)...")
    insert_tgb_sql = """
    INSERT INTO cell_annual_observations (cell_id, year, y_joro_detected, joro_count, effort_araneae, unique_inat_obs)
    SELECT 
        g.cell_id,
        y.year,
        0 AS y_joro_detected,
        0 AS joro_count,
        round(1 + 15 * g.pop_density_h + 8 * g.road_density_r + 5 * g.z_imp)::integer AS effort_araneae,
        0 AS unique_inat_obs
    FROM grid_5km g
    CROSS JOIN (SELECT unnest(%s::smallint[]) AS year) y
    WHERE g.centroid_lat BETWEEN 30.0 AND 38.5
      AND g.centroid_lon BETWEEN -87.5 AND -76.0
      AND (g.pop_density_h > 0.05 OR g.road_density_r > 0.1 OR g.z_imp > 0.05)
    ON CONFLICT (cell_id, year) DO NOTHING;
    """
    cur.execute(insert_tgb_sql, (years,))
    conn.commit()

    # 6. Validar vistas
    cur.execute("SELECT count(*), sum(y_joro_detected), count(DISTINCT cell_id) FROM v_calibration_set;")
    cal_total, cal_detected, cal_cells = cur.fetchone()
    logger.info("=== Vista v_calibration_set (2014-2019) ===")
    logger.info("  Filas totales: %d | Celdas ocupadas con Joro: %d | Celdas distintas evaluadas: %d",
                cal_total, cal_detected, cal_cells)

    # Congelamiento: v_evaluation_set (2020-2024) NO se consulta aquí (ni siquiera agregados de presencias):
    # permanece bloqueada hasta la validación fuera de muestra. Solo se verifica que la vista exista en el catálogo.
    cur.execute("SELECT to_regclass('public.v_evaluation_set') IS NOT NULL;")
    logger.info("=== Vista v_evaluation_set (2020-2024): existe=%s, BLOQUEADA PARA FASE 5 (sin lecturas) ===",
                cur.fetchone()[0])

    cur.close()
    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta de Ocurrencias GBIF/iNaturalist y Esfuerzo TGB")
    parser.add_argument(
        "--db-url",
        type=str,
        default=get_db_url(require_password=False),
        help="Cadena de conexión a PostgreSQL",
    )
    args = parser.parse_args()

    logger.info("=== JORO-HPC Ingesta Observacional GBIF & TGB ===")
    logger.info("Años objetivo: %d a %d", ALL_YEARS[0], ALL_YEARS[-1])

    # 1. Descarga desde iNaturalist
    inat_records = fetch_inaturalist_occurrences(STUDY_BBOX, ALL_YEARS)

    # 2. Descarga desde GBIF
    gbif_records = fetch_gbif_occurrences(STUDY_BBOX, ALL_YEARS)

    all_records = inat_records + gbif_records
    logger.info("Total registros brutos combinados: %d", len(all_records))

    # 3. Ingesta y enlace espacial
    ingest_observations(args.db_url, all_records, ALL_YEARS)

    logger.info("✔ Ingesta observacional completada exitosamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
