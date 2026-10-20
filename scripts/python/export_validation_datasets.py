#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/export_validation_datasets.py
Exporta v_calibration_set (2014-2019) y v_evaluation_set (2020-2024)
hacia /mnt/hpc_scratch/joro-hpc/ para la validación fuera de muestra.
Incluye ecoregion_block y todas las covariables necesarias.
"""
import sys
import psycopg2
from joro_db import get_db_url

DB_URL = get_db_url()
CALIB_OUT = "/mnt/hpc_scratch/joro-hpc/calibration_set_2014_2019.csv"
EVAL_OUT = "/mnt/hpc_scratch/joro-hpc/evaluation_set_2020_2024.csv"

SQL_CALIB = """
COPY (
    SELECT cell_id, year, y_joro_detected, joro_count, effort_araneae,
           centroid_lon, centroid_lat, z_imp, z_borde, z_dosel, pop_density_h, road_density_r,
           ecoregion_block
    FROM v_calibration_set
    ORDER BY year, cell_id
) TO STDOUT WITH (FORMAT csv, HEADER true)
"""

SQL_EVAL = """
COPY (
    SELECT cell_id, year, y_joro_detected, joro_count, effort_araneae,
           centroid_lon, centroid_lat, z_imp, z_borde, z_dosel, pop_density_h, road_density_r,
           ecoregion_block
    FROM v_evaluation_set
    ORDER BY year, cell_id
) TO STDOUT WITH (FORMAT csv, HEADER true)
"""

def main():
    with psycopg2.connect(DB_URL) as con, con.cursor() as cur:
        print(f"Exportando v_calibration_set a {CALIB_OUT}...")
        with open(CALIB_OUT, "w") as fh:
            cur.copy_expert(SQL_CALIB, fh)
        print(f"✔ Exportado: {CALIB_OUT}")

        print(f"Exportando v_evaluation_set a {EVAL_OUT}...")
        with open(EVAL_OUT, "w") as fh:
            cur.copy_expert(SQL_EVAL, fh)
        print(f"✔ Exportado: {EVAL_OUT}")

if __name__ == "__main__":
    main()
