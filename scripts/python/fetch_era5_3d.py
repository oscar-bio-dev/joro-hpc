#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/fetch_era5_3d.py
Descarga de tensores 3D horarios de ERA5 vía Copernicus Climate Data Store (CDS API).

Variables 3D: u, v, w (omega), t en 12 niveles de presión (1000 a 700 hPa).
Variables superficiales: sp (surface pressure), t2m, u10, v10, blh (boundary layer height).
Meses objetivo: Mayo, Junio, Julio (temporada de dispersión juvenil aérea).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("joro.era5")

SELECTED_PRESSURE_LEVELS = [
    "1000", "975", "950", "925", "900", "875", "850", "825", "800", "775", "750", "700"
]

SURFACE_VARIABLES = [
    "surface_pressure", "2m_temperature", "10m_u_component_of_wind",
    "10m_v_component_of_wind", "boundary_layer_height"
]

PRESSURE_VARIABLES = [
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity", "temperature"
]


def download_era5_season(
    output_dir: Path,
    bbox: Tuple[float, float, float, float],
    years: List[int],
    months: List[str] = ["05", "06", "07"],
) -> None:
    """Descarga lotes anuales de ERA5 en formato NetCDF4 por año."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    # CDS API area format: [North, West, South, East]
    area = [lat_max, lon_min, lat_min, lon_max]

    logger.info("Directorio destino de NetCDF4: %s", output_dir)
    logger.info("Área CDS [N, W, S, E]: %s", area)
    logger.info("Años solicitados: %s", years)

    # TODO (Mes 1): Invocar cliente cdsapi.Client()
    # Para cada año y mes:
    # 1. Solicitar reanalysis-era5-pressure-levels en NetCDF
    # 2. Solicitar reanalysis-era5-single-levels en NetCDF
    # 3. Guardar como era5_uvwt_YYYY_MM.nc y era5_sfc_YYYY_MM.nc
    logger.info("Esqueleto de fetch_era5_3d listo para ejecución con credenciales CDS.")
