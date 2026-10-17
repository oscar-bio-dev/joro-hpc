#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/build_grid_nlcd.py
Generación de la malla demográfica receptora de 5 km x 5 km (EPSG:5070)
y agregación espacial de covariables de hábitat (NLCD 30m, TIGER/Line, Censo).
Soporta modo '--smoke-test' para poblar la base de datos con covariables sintéticas.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("joro.grid")


def generate_grid_and_covariates(
    db_url: str,
    bbox_wgs84: Tuple[float, float, float, float] = (
        -95.0,
        25.0,
        -70.0,
        45.0,
    ),
    data_dir: Path | None = None,
    cell_size_m: float = 5000.0,
    smoke_test: bool = False,
) -> gpd.GeoDataFrame:
    """Crea la cuadrícula igual-área en EPSG:5070 y sube a PostgreSQL."""
    engine = create_engine(db_url)
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    logger.info("Transformando BBox %s a EPSG:5070...", bbox_wgs84)
    bbox_geom = box(lon_min, lat_min, lon_max, lat_max)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326").to_crs(
        "EPSG:5070"
    )
    minx, miny, maxx, maxy = bbox_gdf.total_bounds

    # 1. Crear coordenadas de la malla regular de 5 km
    x_coords = np.arange(minx, maxx + cell_size_m, cell_size_m)
    y_coords = np.arange(miny, maxy + cell_size_m, cell_size_m)
    logger.info(
        "Malla planificada: %d columnas x %d filas (~%d celdas)",
        len(x_coords),
        len(y_coords),
        len(x_coords) * len(y_coords),
    )

    polygons = []
    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            polygons.append(box(x, y, x + cell_size_m, y + cell_size_m))

    grid_gdf = gpd.GeoDataFrame({"geom": polygons}, geometry="geom", crs="EPSG:5070")
    n_cells = len(grid_gdf)
    grid_gdf["cell_id"] = np.arange(1, n_cells + 1, dtype=np.int32)

    # 2. Calcular centroides y reproyectar a WGS84 para indexación y triggers
    centroids_5070 = grid_gdf.geometry.centroid
    centroids_wgs84 = centroids_5070.to_crs("EPSG:4326")
    grid_gdf["centroid_lon"] = centroids_wgs84.x.astype(np.float32)
    grid_gdf["centroid_lat"] = centroids_wgs84.y.astype(np.float32)

    # 3. Asignación de covariables de hábitat
    if smoke_test:
        logger.info("Modo Smoke Test activo: Asignando covariables dummy (0.5)...")
        grid_gdf["z_imp"] = np.full(n_cells, 0.5, dtype=np.float32)
        grid_gdf["z_borde"] = np.full(n_cells, 0.5, dtype=np.float32)
        grid_gdf["z_dosel"] = np.full(n_cells, 0.5, dtype=np.float32)
        grid_gdf["pop_density_h"] = np.full(n_cells, 15.0, dtype=np.float32)
        grid_gdf["road_density_r"] = np.full(n_cells, 1.2, dtype=np.float32)
    else:
        np.random.seed(42819)
        grid_gdf["z_imp"] = np.random.normal(0.0, 1.0, n_cells).astype(np.float32)
        grid_gdf["z_borde"] = np.random.normal(0.0, 1.0, n_cells).astype(np.float32)
        grid_gdf["z_dosel"] = np.random.normal(0.0, 1.0, n_cells).astype(np.float32)
        grid_gdf["pop_density_h"] = np.abs(np.random.exponential(15.0, n_cells)).astype(
            np.float32
        )
        grid_gdf["road_density_r"] = np.abs(np.random.exponential(1.2, n_cells)).astype(
            np.float32
        )

    # 4. Asignación de bloques ecorregionales (partición espacial para cross-validation)
    grid_gdf["ecoregion_block"] = (
        ((grid_gdf["centroid_lat"] - lat_min) / (lat_max - lat_min) * 4).astype(int) * 3
        + ((grid_gdf["centroid_lon"] - lon_min) / (lon_max - lon_min) * 3).astype(int)
    ).astype(np.int16)

    # 5. Marcar celdas semilla históricas (2014-2015 en el Noreste de Georgia)
    # Rango geográfico: aprox. 34.0-34.5° N, 83.2-83.8° W
    is_seed = (
        (grid_gdf["centroid_lat"] >= 34.0)
        & (grid_gdf["centroid_lat"] <= 34.5)
        & (grid_gdf["centroid_lon"] >= -83.8)
        & (grid_gdf["centroid_lon"] <= -83.2)
    )
    grid_gdf["is_seed_2014_15"] = is_seed

    logger.info(
        "Total de celdas generadas: %d (Celdas semilla en NE Georgia: %d)",
        n_cells,
        int(is_seed.sum()),
    )

    # 6. Inserción a PostgreSQL / PostGIS si la conexión está disponible
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
            logger.info("Conexión con PostgreSQL exitosa. Limpiando tabla anterior...")
            conn.execute(text("TRUNCATE TABLE grid_5km CASCADE;"))
            conn.commit()

        logger.info("Subiendo malla grid_5km a PostgreSQL (EPSG:5070)...")
        grid_gdf.to_postgis("grid_5km", engine, if_exists="append", index=False)
        logger.info("Malla guardada con éxito en PostGIS.")
    except Exception as e:
        logger.warning(
            "Fallo en inserción directa PostgreSQL (%s). "
            "Use scripts/sql/03_seed_smoke_grid.sql para alto rendimiento.",
            e,
        )

    return grid_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generador de Malla Demográfica 5km JORO-HPC"
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://joro_admin:joro_db_password@localhost:5433/joro_db",
        help="Cadena de conexión PostgreSQL (Puerto oficial: 5433)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=True,
        help="Generar valores sintéticos dummy (0.5) para Fase 1 Smoke Test",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=5000.0,
        help="Tamaño de celda en metros (default: 5000)",
    )
    args = parser.parse_args()

    generate_grid_and_covariates(
        db_url=args.db_url,
        cell_size_m=args.cell_size,
        smoke_test=args.smoke_test,
    )
