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
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Constantes sincronizadas con Sección 6 y 13 de JORO-HPC.tex
STUDY_BBOX = (-95.0, 25.0, -70.0, 45.0)  # (lon_min, lat_min, lon_max, lat_max)
YEARS_CALIBRATION = list(range(2014, 2020)) # 2014-2019
YEARS_EVALUATION = list(range(2020, 2025))  # 2020-2024
ALL_YEARS = list(range(2014, 2025))         # 2014-2024 (11 años)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("joro.pipeline")


def get_db_engine(db_url: str) -> Engine:
    """Retorna el motor SQLAlchemy conectado a PostgreSQL."""
    return create_engine(db_url, pool_size=5, pool_pre_ping=True)


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


def run_era5_pipeline(output_dir: Path) -> None:
    """Descarga de tensores 3D horarios de ERA5 vía Copernicus CDS."""
    logger.info("Iniciando descarga de campos 3D de ERA5 (NetCDF4)...")
    from fetch_era5_3d import download_era5_season
    download_era5_season(output_dir, STUDY_BBOX, ALL_YEARS)


def main() -> int:
    parser = argparse.ArgumentParser(description="JORO-HPC Data Pipeline Orchestrator")
    parser.add_argument(
        "--db-url",
        type=str,
        default="postgresql://joro_admin:secret@localhost:5432/joro_db",
        help="URI de conexión a PostgreSQL",
    )
    parser.add_argument(
        "--action",
        choices=["gbif", "grid", "era5", "all"],
        default="all",
        help="Sub-pipeline a ejecutar",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directorio local para archivos crudos",
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

    logger.info("=== Pipeline Completado con Éxito ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
