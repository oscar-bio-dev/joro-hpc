#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/fetch_gbif_tgb.py
Descarga de ocurrencias de Trichonephila clavata y del grupo objetivo (Araneae)
para construir la matriz de detección y esfuerzo Target-Group Background (TGB).
"""

from __future__ import annotations

import logging
from typing import List, Tuple
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine, text

logger = logging.getLogger("joro.gbif")

JORO_TAXON_KEY = 5167542      # Trichonephila clavata
ARANEAE_TAXON_KEY = 1496      # Orden Araneae (grupo objetivo para esfuerzo TGB)


def download_gbif_paginated(
    taxon_key: int,
    bbox: Tuple[float, float, float, float],
    years: List[int],
    max_records: int = 25_000,
) -> pd.DataFrame:
    """Descarga ocurrencias desde GBIF mediante paginación con pygbif."""
    try:
        from pygbif import occurrences as occ
    except ImportError:
        logger.error("Librería 'pygbif' no instalada. Ejecute: pip install pygbif")
        return pd.DataFrame()

    lon_min, lat_min, lon_max, lat_max = bbox
    geometry_wkt = f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, {lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"

    all_records = []
    page_size = 300
    offset = 0

    logger.info("Iniciando descarga paginada para Taxon %d en BBox...", taxon_key)

    while offset < max_records:
        try:
            res = occ.search(
                taxonKey=taxon_key,
                geometry=geometry_wkt,
                hasCoordinate=True,
                limit=page_size,
                offset=offset,
            )
            results = res.get("results", [])
            if not results:
                break

            for r in results:
                year = r.get("year")
                if year and int(year) in years:
                    all_records.append({
                        "gbif_id": r.get("key"),
                        "species": r.get("species", "Trichonephila clavata"),
                        "year": int(year),
                        "month": r.get("month"),
                        "latitude": r.get("decimalLatitude"),
                        "longitude": r.get("decimalLongitude"),
                        "coord_uncertainty_m": r.get("coordinateUncertaintyInMeters", 1000.0),
                        "basis_of_record": r.get("basisOfRecord"),
                    })

            offset += len(results)
            if res.get("endOfRecords", True):
                break
        except Exception as e:
            logger.warning("Error durante descarga GBIF (offset %d): %s", offset, e)
            break

    df = pd.DataFrame(all_records)
    logger.info("Descarga finalizada para Taxon %d: %d registros filtrados.", taxon_key, len(df))
    return df


def fetch_all_occurrences(
    db_url: str,
    bbox: Tuple[float, float, float, float],
    years: List[int],
    limit: int = 50_000,
) -> None:
    """Descarga ocurrencias de GBIF y puebla occurrences_raw y cell_annual_observations."""
    engine = create_engine(db_url)

    # 1. Descargar registros de T. clavata
    df_joro = download_gbif_paginated(JORO_TAXON_KEY, bbox, years, max_records=limit)

    # 2. Descargar registros del grupo objetivo (Araneae) para cuantificar esfuerzo observacional
    df_araneae = download_gbif_paginated(ARANEAE_TAXON_KEY, bbox, years, max_records=limit)

    if df_joro.empty and df_araneae.empty:
        logger.warning("No se obtuvieron registros (sin conexión o API indisponible).")
        return

    # 3. Convertir a GeoDataFrames con geometría WGS84
    if not df_joro.empty:
        geometry_joro = [Point(xy) for xy in zip(df_joro["longitude"], df_joro["latitude"])]
        gdf_joro = gpd.GeoDataFrame(df_joro, geometry=geometry_joro, crs="EPSG:4326")

        try:
            # Subir ocurrencias crudas a occurrences_raw
            logger.info("Insertando %d registros en occurrences_raw...", len(gdf_joro))
            gdf_joro.to_postgis("occurrences_raw", engine, if_exists="append", index=False)
        except Exception as e:
            logger.warning("No se pudo escribir en occurrences_raw: %s", e)

    # 4. Asignación espacial a celdas de 5 km e inserción en cell_annual_observations
    logger.info("Calculando matriz de esfuerzo y detección en malla 5x5 km...")
    try:
        # Cargar malla existente
        grid_gdf = gpd.read_postgis("SELECT cell_id, geom_wgs84 as geometry FROM grid_5km;", engine, geom_col="geometry")
        
        # Intersección espacial
        if not df_joro.empty:
            joined_joro = gpd.sjoin(gdf_joro, grid_gdf, how="inner", predicate="intersects")
            agg_joro = joined_joro.groupby(["cell_id", "year"]).size().reset_index(name="joro_count")
            agg_joro["y_joro_detected"] = 1
        else:
            agg_joro = pd.DataFrame(columns=["cell_id", "year", "joro_count", "y_joro_detected"])

        # Esfuerzo Araneae
        if not df_araneae.empty:
            geom_araneae = [Point(xy) for xy in zip(df_araneae["longitude"], df_araneae["latitude"])]
            gdf_araneae = gpd.GeoDataFrame(df_araneae, geometry=geom_araneae, crs="EPSG:4326")
            joined_araneae = gpd.sjoin(gdf_araneae, grid_gdf, how="inner", predicate="intersects")
            agg_araneae = joined_araneae.groupby(["cell_id", "year"]).size().reset_index(name="effort_araneae")
        else:
            agg_araneae = pd.DataFrame(columns=["cell_id", "year", "effort_araneae"])

        # Unión y persistencia
        obs_matrix = pd.merge(agg_joro, agg_araneae, on=["cell_id", "year"], how="outer").fillna(0)
        obs_matrix["y_joro_detected"] = obs_matrix["y_joro_detected"].astype(int)
        obs_matrix["joro_count"] = obs_matrix["joro_count"].astype(int)
        obs_matrix["effort_araneae"] = obs_matrix["effort_araneae"].astype(int)

        logger.info("Guardando %d filas en cell_annual_observations...", len(obs_matrix))
        obs_matrix.to_sql("cell_annual_observations", engine, if_exists="append", index=False)
        logger.info("cell_annual_observations actualizada exitosamente.")

    except Exception as e:
        logger.warning("Fallo en la intersección espacial con grid_5km: %s", e)
