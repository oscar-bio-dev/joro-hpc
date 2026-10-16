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

HOURLY_STEPS = [f"{h:02d}:00" for h in range(24)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]


def get_cds_client():
    """Inicializa el cliente de CDS API verificando configuración."""
    try:
        import cdsapi
        return cdsapi.Client()
    except ImportError:
        logger.error("Librería 'cdsapi' no instalada. Ejecute: pip install cdsapi")
        return None
    except Exception as e:
        logger.error("Error al inicializar cdsapi.Client: %s", e)
        logger.info("Asegúrese de configurar el archivo ~/.cdsapirc con su URL y API Key de Copernicus CDS.")
        return None


def download_era5_season(
    output_dir: Path,
    bbox: Tuple[float, float, float, float],
    years: List[int],
    months: List[str] = ["05", "06", "07"],
) -> None:
    """Descarga lotes de ERA5 en formato NetCDF4 para los meses de baloneo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    # CDS API area format: [North, West, South, East]
    area = [lat_max, lon_min, lat_min, lon_max]

    logger.info("Iniciando descarga ERA5 hacia: %s", output_dir)
    logger.info("Área CDS [N, W, S, E]: %s", area)
    logger.info("Años solicitados: %s | Meses: %s", years, months)

    client = get_cds_client()
    if client is None:
        logger.warning("Operando en modo demostración/dry-run. Cree ~/.cdsapirc para habilitar descargas reales.")
        return

    for year in years:
        for month in months:
            # 1. Descarga de tensores 3D en niveles de presión
            target_3d = output_dir / f"era5_uvwt_{year}_{month}.nc"
            if target_3d.exists():
                logger.info("Archivo existente, omitiendo: %s", target_3d.name)
            else:
                logger.info("Solicitando campos 3D: %s...", target_3d.name)
                try:
                    client.retrieve(
                        "reanalysis-era5-pressure-levels",
                        {
                            "product_type": "reanalysis",
                            "format": "netcdf",
                            "variable": PRESSURE_VARIABLES,
                            "pressure_level": SELECTED_PRESSURE_LEVELS,
                            "year": str(year),
                            "month": month,
                            "day": ALL_DAYS,
                            "time": HOURLY_STEPS,
                            "area": area,
                        },
                        str(target_3d),
                    )
                    logger.info("Descargado con éxito: %s", target_3d.name)
                except Exception as e:
                    logger.error("Fallo al descargar campos 3D (%d-%s): %s", year, month, e)

            # 2. Descarga de variables superficiales
            target_sfc = output_dir / f"era5_sfc_{year}_{month}.nc"
            if target_sfc.exists():
                logger.info("Archivo existente, omitiendo: %s", target_sfc.name)
            else:
                logger.info("Solicitando variables superficiales: %s...", target_sfc.name)
                try:
                    client.retrieve(
                        "reanalysis-era5-single-levels",
                        {
                            "product_type": "reanalysis",
                            "format": "netcdf",
                            "variable": SURFACE_VARIABLES,
                            "year": str(year),
                            "month": month,
                            "day": ALL_DAYS,
                            "time": HOURLY_STEPS,
                            "area": area,
                        },
                        str(target_sfc),
                    )
                    logger.info("Descargado con éxito: %s", target_sfc.name)
                except Exception as e:
                    logger.error("Fallo al descargar variables superficiales (%d-%s): %s", year, month, e)

    logger.info("Proceso de descarga ERA5 completado.")
