#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/export_calibration_set.py
Exporta `v_calibration_set` (2014-2019; la vista `v_evaluation_set` 2020-2024 NO se toca) al CSV
que lee el calibrador C++ (`load_calibration_dataset`). Debe regenerarse cada vez que cambian las
covariables de `grid_5km` o el esfuerzo TGB en `cell_annual_observations`.
"""
import sys

import psycopg2

DB_URL = "postgresql://joro_admin:REDACTED_DB_PASSWORD@localhost:5433/joro_db"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/hpc_scratch/joro-hpc/calibration_set_2014_2019.csv"

SQL = """
COPY (
    SELECT cell_id, year, y_joro_detected, joro_count, effort_araneae,
           centroid_lon, centroid_lat, z_imp, z_borde, z_dosel, pop_density_h, road_density_r
    FROM v_calibration_set
    ORDER BY year, cell_id
) TO STDOUT WITH (FORMAT csv, HEADER true)
"""

with psycopg2.connect(DB_URL) as con, con.cursor() as cur, open(OUT, "w") as fh:
    cur.copy_expert(SQL, fh)
print(f"Exportado: {OUT}")
