#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/fetch_era5_3d.py
Descarga quirúrgica de tensores horarios de ERA5 vía CDS API.

Fase 2 (Piloto Aéreo de 7 Días):
  - Descarga de 168 horas: 1 al 7 de mayo de 2014 (00:00 a 23:00 UTC).
  - Variables 3D (12 niveles 1000-700 hPa): u, v, w (omega), T.
  - Variables superficiales: sp, t2m, u10, v10, blh.
  - Destino: /mnt/hpc_scratch/joro-hpc/era5_nc_active/

Uso:
  python3 scripts/python/fetch_era5_3d.py --week1 --force
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("joro.era5")

# 12 niveles de presión estándar para la capa límite y troposfera baja (hPa)
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

# Dinámica 3D
PRESSURE_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
]

# 5 Variables superficiales
SURFACE_VARIABLES = [
    "surface_pressure",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "boundary_layer_height",
]

HOURLY_STEPS = [f"{h:02d}:00" for h in range(24)]


def get_cds_client():
    """Inicializa cliente cdsapi verificando credenciales."""
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
    client,
    output_dir: Path,
    year: str,
    month: str,
    days: List[str],
    area: list,
    force: bool = False,
) -> bool:
    """Descarga variables 3D en niveles isobáricos."""
    target_3d = output_dir / f"era5_uvwt_{year}_{month}.nc"

    if target_3d.exists() and not force:
        logger.info(
            "Archivo 3D existente detectado: %s (Omitiendo. Use --force para re-descargar)",
            target_3d.name,
        )
        return True

    logger.info(
        "Enviando petición CDS: Tensores 3D (u, v, omega, T) para días %s...", days
    )
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
                "day": days,
                "time": HOURLY_STEPS,
                "area": area,
            },
            str(target_3d),
        )
        logger.info("✔ Descarga 3D completada con éxito: %s", target_3d.name)
        return True
    except Exception as e:
        logger.error("Error durante la descarga de tensores 3D: %s", e)
        return False


def _download_sfc(
    client,
    output_dir: Path,
    year: str,
    month: str,
    days: List[str],
    area: list,
    force: bool = False,
) -> bool:
    """Descarga variables superficiales en superficie simple."""
    target_sfc = output_dir / f"era5_sfc_{year}_{month}.nc"

    if target_sfc.exists() and not force:
        logger.info(
            "Archivo superficial existente detectado: %s (Omitiendo. Use --force)",
            target_sfc.name,
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
                "day": days,
                "time": HOURLY_STEPS,
                "area": area,
            },
            str(target_sfc),
        )
        logger.info("✔ Descarga superficial completada con éxito: %s", target_sfc.name)
        return True
    except Exception as e:
        logger.error("Error durante la descarga de variables superficiales: %s", e)
        return False


def fetch_era5(
    output_dir: Path = Path("/mnt/hpc_scratch/joro-hpc/era5_nc_active"),
    bbox: Tuple[float, float, float, float] = (-95.0, 25.0, -70.0, 45.0),
    year: str = "2014",
    month: str = "05",
    days: List[str] | None = None,
    force: bool = False,
) -> bool:
    """Descarga tensores horarios de ERA5 para el periodo especificado."""
    if days is None:
        days = ["01", "02", "03", "04", "05", "06", "07"]

    output_dir.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    area = [lat_max, lon_min, lat_min, lon_max]

    total_hours = len(days) * 24
    logger.info("════════════════════════════════════════════════════════════")
    logger.info(
        "  JORO-HPC · Descarga ERA5 (%d horas: %s días)  ", total_hours, len(days)
    )
    logger.info("════════════════════════════════════════════════════════════")
    logger.info("Directorio destino : %s", output_dir)
    logger.info("Periodo            : %s-%s, Días=%s", year, month, days)
    logger.info("Dominio espacial   : N=%.1f, W=%.1f, S=%.1f, E=%.1f", *area)
    logger.info("Niveles de presión : %s", ", ".join(SELECTED_PRESSURE_LEVELS))

    client = get_cds_client()
    if client is None:
        logger.warning("Cliente CDS no autenticado. Verifique ~/.cdsapirc.")
        return False

    if not _download_3d(client, output_dir, year, month, days, area, force=force):
        return False

    if not _download_sfc(client, output_dir, year, month, days, area, force=force):
        return False

    logger.info(
        "✔ Ingesta meteorológica ERA5 (%d horas) finalizada exitosamente.", total_hours
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga ERA5 (Fase 2 Piloto 7 Días JORO-HPC)"
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
        "--days",
        nargs="+",
        default=["01", "02", "03", "04", "05", "06", "07"],
        help="Días del mes a descargar (default: 01 a 07)",
    )
    parser.add_argument(
        "--week1",
        action="store_true",
        default=True,
        help="Descargar primera semana de mayo (días 01 a 07 = 168 horas)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Sobrescribir archivos existentes si ya fueron descargados",
    )
    args = parser.parse_args()

    fetch_era5(
        output_dir=args.output_dir,
        year=args.year,
        month=args.month,
        days=args.days,
        force=args.force,
    )
