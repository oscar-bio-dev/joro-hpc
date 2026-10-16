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
import geopandas as gpd
from sqlalchemy import create_engine

logger = logging.getLogger("joro.grid")


def generate_grid_and_covariates(
    db_url: str,
    bbox_wgs84: Tuple[float, float, float, float],
    data_dir: Path,
    cell_size_m: float = 5000.0,
) -> None:
    """Crea la cuadrícula igual-área en EPSG:5070, agrega capas NLCD y sube a PostgreSQL."""
    engine = create_engine(db_url)
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    logger.info("Generando malla regular EPSG:5070 de %.1f m...", cell_size_m)
    # TODO (Mes 1):
    # 1. Crear polígono envolvente en EPSG:4326 y proyectar a EPSG:5070.
    # 2. Generar malla regular de celdas cuadradas de 5 km x 5 km (~200,000 celdas).
    # 3. Calcular centroides (lon, lat) reproyectados a WGS84.
    # 4. Extraer zonificación NLCD:
    #      - Impervious surface (% urbano) -> estandarizar a z_imp
    #      - Forest edge density -> estandarizar a z_borde
    #      - Canopy cover -> estandarizar a z_dosel
    # 5. Agregar densidad vial TIGER/Line -> road_density_r
    # 6. Marcar celdas semilla 2014-2015 en NE Georgia (is_seed_2014_15 = True).
    # 7. Asignar bloques ecorregionales EPA Nivel III (ecoregion_block).
    # 8. Carga masiva con COPY a la tabla grid_5km.
    logger.info("Esqueleto de build_grid_nlcd listo para procesamiento raster/vector.")
