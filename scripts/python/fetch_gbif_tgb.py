#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/fetch_gbif_tgb.py
Descarga de ocurrencias de Trichonephila clavata y del grupo objetivo (Araneae)
para construir la matriz de detección y esfuerzo Target-Group Background (TGB).
"""

from __future__ import annotations

import logging
from typing import List, Tuple
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger("joro.gbif")

JORO_TAXON_KEY = 5167542      # Trichonephila clavata
ARANEAE_TAXON_KEY = 1496      # Orden Araneae (grupo objetivo para esfuerzo TGB)


def fetch_all_occurrences(
    db_url: str,
    bbox: Tuple[float, float, float, float],
    years: List[int],
    limit: int = 100_000,
) -> None:
    """Descarga ocurrencias de GBIF y puebla occurrences_raw y cell_annual_observations."""
    engine = create_engine(db_url)
    lon_min, lat_min, lon_max, lat_max = bbox

    logger.info("Consultando GBIF para T. clavata en bbox: %s", bbox)
    # TODO (Mes 1): Integrar pygbif.occurrences.search paginado
    # 1. Descargar registros de T. clavata con coordenadas válidas.
    # 2. Descargar conteos agregados por año/celda del orden Araneae (TGB effort).
    # 3. Intersectar espacialmente con grid_5km para asignar cell_id.
    # 4. Insertar o actualizar cell_annual_observations:
    #      INSERT INTO cell_annual_observations (cell_id, year, y_joro_detected, joro_count, effort_araneae)
    #      VALUES (...) ON CONFLICT (cell_id, year) DO UPDATE ...
    logger.info("Esqueleto de fetch_gbif_tgb listo para implementación completa.")
