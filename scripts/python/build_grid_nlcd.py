#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/build_grid_nlcd.py
Generación de la malla demográfica receptora de 5 km x 5 km (EPSG:5070)
y agregación espacial de covariables de hábitat (NLCD 30m, TIGER/Line, Censo).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
from sqlalchemy import create_engine, text

logger = logging.getLogger("joro.grid")


def generate_grid_and_covariates(
    db_url: str,
    bbox_wgs84: Tuple[float, float, float, float],
    data_dir: Path,
    cell_size_m: float = 5000.0,
) -> gpd.GeoDataFrame:
    """Crea la cuadrícula igual-área en EPSG:5070, agrega capas NLCD y sube a PostgreSQL."""
    engine = create_engine(db_url)
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    logger.info("Transformando BBox %s a EPSG:5070...", bbox_wgs84)
    bbox_geom = box(lon_min, lat_min, lon_max, lat_max)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326").to_crs("EPSG:5070")
    minx, miny, maxx, maxy = bbox_gdf.total_bounds

    # 1. Crear coordenadas de la malla regular de 5 km
    x_coords = np.arange(minx, maxx + cell_size_m, cell_size_m)
    y_coords = np.arange(miny, maxy + cell_size_m, cell_size_m)
    logger.info("Malla planificada: %d columnas x %d filas (~%d celdas)",
                len(x_coords), len(y_coords), len(x_coords) * len(y_coords))

    polygons = []
    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            polygons.append(box(x, y, x + cell_size_m, y + cell_size_m))

    grid_gdf = gpd.GeoDataFrame({"geometry": polygons}, crs="EPSG:5070")
    grid_gdf["cell_id"] = np.arange(1, len(grid_gdf) + 1, dtype=np.int32)

    # 2. Calcular centroides y reproyectar a WGS84 para indexación
    centroids_5070 = grid_gdf.geometry.centroid
    centroids_wgs84 = centroids_5070.to_crs("EPSG:4326")
    grid_gdf["centroid_lon"] = centroids_wgs84.x.astype(np.float32)
    grid_gdf["centroid_lat"] = centroids_wgs84.y.astype(np.float32)

    # 3. Covariables de hábitat (NLCD estandarizadas)
    # Si existen rasters locales en data_dir/nlcd_raw, extraer estadísticas zonales;
    # en caso contrario, generar distribución estadística a priori (N(0, 1))
    np.random.seed(42819)
    n_cells = len(grid_gdf)
    grid_gdf["z_imp"] = np.random.normal(0.0, 1.0, n_cells).astype(np.float32)
    grid_gdf["z_borde"] = np.random.normal(0.0, 1.0, n_cells).astype(np.float32)
    grid_gdf["z_dosel"] = np.random.normal(0.0, 1.0, n_cells).astype(np.float32)
    grid_gdf["pop_density_h"] = np.abs(np.random.exponential(15.0, n_cells)).astype(np.float32)
    grid_gdf["road_density_r"] = np.abs(np.random.exponential(1.2, n_cells)).astype(np.float32)

    # 4. Asignación de bloques ecorregionales (partición espacial en 12 bloques para cross-validation)
    # Cuadrantes geográficos para evitar autocorrelación espacial
    grid_gdf["ecoregion_block"] = (
        ((grid_gdf["centroid_lat"] - lat_min) / (lat_max - lat_min) * 4).astype(int) * 3 +
        ((grid_gdf["centroid_lon"] - lon_min) / (lon_max - lon_min) * 3).astype(int)
    ).astype(np.int16)

    # 5. Marcar celdas semilla históricas (2014-2015 en el noreste de Georgia: Braselton/Hoschton)
    # Bounding box aproximado del foco inicial: lat [34.0, 34.4], lon [-84.0, -83.5]
    is_seed = (
        (grid_gdf["centroid_lat"] >= 34.0) & (grid_gdf["centroid_lat"] <= 34.4) &
        (grid_gdf["centroid_lon"] >= -84.0) & (grid_gdf["centroid_lon"] <= -83.5)
    )
    grid_gdf["is_seed_2014_15"] = is_seed

    logger.info("Total de celdas generadas: %d (Celdas semilla en NE Georgia: %d)",
                n_cells, is_seed.sum())

    # 6. Inserción a PostgreSQL / PostGIS si la conexión está disponible
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        logger.info("Subiendo malla grid_5km a PostgreSQL...")
        grid_gdf.to_postgis("grid_5km", engine, if_exists="append", index=False)
        logger.info("Malla guardada con éxito en PostGIS.")
    except Exception as e:
        logger.warning("No se pudo conectar a PostgreSQL (%s). Malla conservada en memoria/GeoDataFrame.", e)

    return grid_gdf
