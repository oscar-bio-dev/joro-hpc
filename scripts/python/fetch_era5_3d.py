#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/fetch_era5_3d.py
Descarga quirúrgica de tensores horarios de ERA5 vía CDS API.

Hito Fase 1 (Smoke Test):
  - Descarga exclusiva de 24h: 1 de mayo de 2014 (00:00 a 23:00 UTC).
  - Variables 3D (12 niveles 1000-700 hPa): u, v, w (omega), T.
  - Variables superficiales: sp, t2m, u10, v10, blh.
  - Destino: /mnt/hpc_scratch/joro-hpc/era5_nc_active/

Uso:
  python3 scripts/python/fetch_era5_3d.py
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("joro.era5")

# 12 niveles de presión estándar para la capa límite atmosférica y troposfera baja (hPa)
SELECTED_PRESSURE_LEVELS = [
    "1000",
    "975",
    "950",
    "925",
    "900",
    "875",
    "850",
    "825",
    "800",
    "775",
    "750",
    "700",
]

# Variables 3D de dinámica y termodinámica atmosférica
PRESSURE_VARIABLES = [
    "u_component_of_wind",  # u [m/s]
    "v_component_of_wind",  # v [m/s]
    "vertical_velocity",  # omega [Pa/s]
    "temperature",  # T [K]
]

# 5 variables superficiales requeridas para el modelo LPDM y acoplamiento biofísico
SURFACE_VARIABLES = [
    "surface_pressure",  # sp [Pa]
    "2m_temperature",  # t2m [K]
    "10m_u_component_of_wind",  # u10 [m/s]
    "10m_v_component_of_wind",  # v10 [m/s]
    "boundary_layer_height",  # blh [m]
]

HOURLY_STEPS = [f"{h:02d}:00" for h in range(24)]


def get_cds_client():
    """Inicializa el cliente de CDS API verificando credenciales."""
    try:
        import cdsapi

        return cdsapi.Client()
    except ImportError:
        logger.error(
            "Librería 'cdsapi' no disponible. Instale las dependencias con:\n"
            "  .venv/bin/pip install cdsapi"
        )
        return None
    except Exception as e:
        logger.error("Error al inicializar cdsapi.Client(): %s", e)
        logger.info(
            "Configure el archivo ~/.cdsapirc o las variables "
            "CDSAPI_URL y CDSAPI_KEY.\n"
            "Guía oficial: https://cds.climate.copernicus.eu/how-to-api"
        )
        return None


def _download_3d(
    client, output_dir: Path, year: str, month: str, day: str, area: list
) -> bool:
    """Descarga variables 3D en niveles isobáricos."""
    target_3d = output_dir / f"era5_uvwt_{year}_{month}.nc"
    target_3d_daily = output_dir / f"era5_uvwt_{year}_{month}_{day}.nc"

    if target_3d.exists():
        logger.info("Archivo 3D existente detectado: %s (Omitiendo)", target_3d.name)
        return True

    logger.info("Enviando petición CDS: Tensores 3D (u, v, omega, T)...")
    try:
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": PRESSURE_VARIABLES,
                "pressure_level": SELECTED_PRESSURE_LEVELS,
                "year": year,
                "month": month,
                "day": [day],
                "time": HOURLY_STEPS,
                "area": area,
            },
            str(target_3d),
        )
        logger.info("✔ Descarga 3D completada con éxito: %s", target_3d.name)
        if not target_3d_daily.exists():
            try:
                os.link(target_3d, target_3d_daily)
            except OSError:
                pass
        return True
    except Exception as e:
        logger.error("Error durante la descarga de tensores 3D: %s", e)
        return False


def _download_sfc(
    client, output_dir: Path, year: str, month: str, day: str, area: list
) -> bool:
    """Descarga variables superficiales en superficie simple."""
    target_sfc = output_dir / f"era5_sfc_{year}_{month}.nc"
    target_sfc_daily = output_dir / f"era5_sfc_{year}_{month}_{day}.nc"

    if target_sfc.exists():
        logger.info(
            "Archivo superficial existente detectado: %s (Omitiendo)", target_sfc.name
        )
        return True

    logger.info(
        "Enviando petición CDS: Variables superficiales (sp, t2m, u10, v10, blh)..."
    )
    try:
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": SURFACE_VARIABLES,
                "year": year,
                "month": month,
                "day": [day],
                "time": HOURLY_STEPS,
                "area": area,
            },
            str(target_sfc),
        )
        logger.info("✔ Descarga superficial completada con éxito: %s", target_sfc.name)
        if not target_sfc_daily.exists():
            try:
                os.link(target_sfc, target_sfc_daily)
            except OSError:
                pass
        return True
    except Exception as e:
        logger.error("Error durante la descarga de variables superficiales: %s", e)
        return False


def fetch_era5_smoke_test(
    output_dir: Path = Path("/mnt/hpc_scratch/joro-hpc/era5_nc_active"),
    bbox: Tuple[float, float, float, float] = (-95.0, 25.0, -70.0, 45.0),
    year: str = "2014",
    month: str = "05",
    day: str = "01",
) -> bool:
    """
    Descarga quirúrgica para el Smoke Test de la Fase 1:
    Exclusivamente 24 horas del 1 de mayo de 2014.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    area = [lat_max, lon_min, lat_min, lon_max]

    logger.info("════════════════════════════════════════════════════════════")
    logger.info("  JORO-HPC · Descarga Quirúrgica ERA5 (Fase 1 Smoke Test)  ")
    logger.info("════════════════════════════════════════════════════════════")
    logger.info("Directorio destino : %s", output_dir)
    logger.info("Fecha solicitada   : %s-%s-%s (24 horas UTC)", year, month, day)
    logger.info(
        "Dominio espacial   : Norte=%.1f, Oeste=%.1f, Sur=%.1f, Este=%.1f", *area
    )
    logger.info("Niveles de presión : %s", ", ".join(SELECTED_PRESSURE_LEVELS))

    client = get_cds_client()
    if client is None:
        logger.warning(
            "Cliente CDS no autenticado. "
            "Verifique ~/.cdsapirc para habilitar la descarga en vivo."
        )
        return False

    if not _download_3d(client, output_dir, year, month, day, area):
        return False

    if not _download_sfc(client, output_dir, year, month, day, area):
        return False

    logger.info("✔ Ingesta meteorológica para el Smoke Test finalizada exitosamente.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga Quirúrgica ERA5 (Smoke Test Fase 1 JORO-HPC)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/hpc_scratch/joro-hpc/era5_nc_active"),
        help="Ruta de almacenamiento NVMe Scratch para los archivos NetCDF4",
    )
    parser.add_argument(
        "--year",
        default="2014",
        help="Año objetivo (default: 2014)",
    )
    parser.add_argument(
        "--month",
        default="05",
        help="Mes objetivo (default: 05 - Mayo)",
    )
    parser.add_argument(
        "--day",
        default="01",
        help="Día objetivo (default: 01)",
    )
    args = parser.parse_args()

    fetch_era5_smoke_test(
        output_dir=args.output_dir,
        year=args.year,
        month=args.month,
        day=args.day,
    )
