#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/data_pipeline.py
Orquestador principal del pipeline de datos geoespaciales y atmosféricos.

Módulos integrados:
    - fetch_gbif_tgb.py   → Ingesta de T. clavata y esfuerzo Target-Group (Araneae).
    - build_grid_nlcd.py  → Generación de malla 5x5 km y agregación de NLCD/vías.
    - fetch_era5_3d.py    → Descarga de tensores ERA5 3D a archivos NetCDF4.

Uso:
    python scripts/python/data_pipeline.py --action all --db-url postgresql://user:pass@localhost/joro_db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Constantes sincronizadas con Sección 6 y 13 de JORO-HPC_Research_Proposal.tex
STUDY_BBOX = (-95.0, 25.0, -70.0, 45.0)  # (lon_min, lat_min, lon_max, lat_max)
YEARS_CALIBRATION = list(range(2014, 2020))  # 2014-2019
YEARS_EVALUATION = list(range(2020, 2025))  # 2020-2024
ALL_YEARS = list(range(2014, 2025))  # 2014-2024 (11 años)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("joro.pipeline")


def get_db_engine(db_url: str) -> Any:
    """Retorna el motor SQLAlchemy conectado a PostgreSQL."""
    try:
        from sqlalchemy import create_engine

        return create_engine(db_url, pool_size=5, pool_pre_ping=True)
    except ImportError:
        logger.error("Librería 'sqlalchemy' no instalada en este entorno.")
        return None


def run_gbif_pipeline(db_url: str, limit: int = 100_000) -> None:
    """Descarga de ocurrencias y cálculo de esfuerzo de muestreo."""
    logger.info("Iniciando pipeline de GBIF + Target-Group Background (Araneae)...")
    from fetch_gbif_tgb import fetch_all_occurrences

    fetch_all_occurrences(db_url, STUDY_BBOX, ALL_YEARS, limit)


def run_grid_pipeline(db_url: str, data_dir: Path) -> None:
    """Construcción de malla 5km EPSG:5070 y agregación de covariables NLCD."""
    logger.info("Iniciando construcción de malla 5km y covariables NLCD...")
    from build_grid_nlcd import generate_grid_and_covariates

    generate_grid_and_covariates(db_url, STUDY_BBOX, data_dir)


def run_thermal_pipeline(output_path: Path) -> None:
    """Generación de serie térmica diaria superficial (t2m_min, t2m_mean) 365 días."""
    logger.info("Iniciando generación de serie térmica diaria 365 días (NetCDF4)...")
    from generate_daily_thermal_forcing import generate_thermal_series

    generate_thermal_series(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="JORO-HPC Data Pipeline Orchestrator")
    parser.add_argument(
        "--db-url",
        type=str,
        default="postgresql://joro_admin:REDACTED_DB_PASSWORD@localhost:5433/joro_db",
        help="URI de conexión a PostgreSQL (default: %(default)s)",
    )
    parser.add_argument(
        "--action",
        choices=["gbif", "grid", "era5", "thermal", "all"],
        default="all",
        help="Sub-pipeline a ejecutar",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directorio local para archivos crudos",
    )
    parser.add_argument(
        "--thermal-output",
        type=Path,
        default=Path("/mnt/hpc_scratch/joro-hpc/era5_nc_active/era5_daily_t2m_2014_2019.nc"),
        help="Ruta destino para la serie térmica diaria",
    )
    args = parser.parse_args()

    logger.info("=== JORO-HPC Data Pipeline Iniciado ===")
    logger.info("Dominio MVP: %s", STUDY_BBOX)
    logger.info("Años: %d a %d", ALL_YEARS[0], ALL_YEARS[-1])

    if args.action in ("gbif", "all"):
        run_gbif_pipeline(args.db_url)
    if args.action in ("grid", "all"):
        run_grid_pipeline(args.db_url, args.data_dir)
    if args.action in ("era5", "all"):
        run_era5_pipeline(args.data_dir / "era5_nc")
    if args.action in ("thermal", "all"):
        run_thermal_pipeline(args.thermal_output)

    logger.info("=== Pipeline Completado con Éxito ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
