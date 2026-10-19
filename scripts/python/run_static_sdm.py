#!/usr/bin/env python3
"""
JORO-HPC · scripts/python/run_static_sdm.py
Entrena el Comparador Estático Climático para H1 (SDM Correlativo Estático)
exclusivamente con los datos de 2014-2019 y proyecta sus probabilidades sobre 2020-2024.

Covariables:
  - Tmin invernal media (ERA5 daily t2m)
  - Tmean estival media (ERA5 daily t2m)
  - z_imp, z_borde, z_dosel (NLCD)
  - pop_density_h, road_density_r (Censo / TIGER)
  - log1p(effort_araneae) (Esfuerzo TGB)
"""
import time
import netCDF4 as nc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CALIB_CSV = "/mnt/hpc_scratch/joro-hpc/calibration_set_2014_2019.csv"
EVAL_CSV = "/mnt/hpc_scratch/joro-hpc/evaluation_set_2020_2024.csv"
OUT_CSV = "/mnt/hpc_scratch/joro-hpc/outputs/sdm_predictions_2020_2024.csv"
ERA5_DIR = "/mnt/hpc_scratch/joro-hpc/era5_nc_active"

def load_thermal_climatology(year):
    nc_path = f"{ERA5_DIR}/era5_daily_t2m_{year}.nc"
    with nc.Dataset(nc_path) as ds:
        tmin = ds.variables["t2m_min"][:]
        tmean = ds.variables["t2m_mean"][:]
    # Invierno: días 0..59 (ene-feb) y 334..364 (dic)
    w_tmin = np.minimum(tmin[:60].min(axis=0), tmin[334:].min(axis=0))
    # Verano: días 151..243 (jun-ago)
    s_tmean = tmean[151:244].mean(axis=0)
    return w_tmin, s_tmean

def map_thermal(df, years):
    lat = df["centroid_lat"].values
    lon = df["centroid_lon"].values
    lat_idx = np.clip(np.round((45.0 - lat) / 0.25).astype(int), 0, 80)
    lon_idx = np.clip(np.round((lon - (-95.0)) / 0.25).astype(int), 0, 100)

    w_arr = np.zeros(len(df), dtype=np.float32)
    s_arr = np.zeros(len(df), dtype=np.float32)

    thermal_by_year = {}
    for y in sorted(years):
        thermal_by_year[y] = load_thermal_climatology(y)

    for y in years:
        mask = (df["year"].values == y)
        w_grid, s_grid = thermal_by_year[y]
        w_arr[mask] = w_grid[lat_idx[mask], lon_idx[mask]]
        s_arr[mask] = s_grid[lat_idx[mask], lon_idx[mask]]

    df["w_tmin"] = w_arr
    df["s_tmean"] = s_arr
    df["log_effort"] = np.log1p(df["effort_araneae"].values.astype(np.float32))
    return df

def main():
    t0 = time.time()
    print("▶ Cargando partición de calibración 2014-2019...")
    df_cal = pd.read_csv(CALIB_CSV)
    df_cal = map_thermal(df_cal, range(2014, 2020))

    features = [
        "w_tmin", "s_tmean", "z_imp", "z_borde", "z_dosel",
        "pop_density_h", "road_density_r", "log_effort"
    ]
    X_train = df_cal[features].values
    y_train = df_cal["y_joro_detected"].values

    print(f"  Filas entrenamiento: {len(X_train)}, Presencias: {y_train.sum()}")
    print("▶ Ajustando regresión logística / GLM espacial estático...")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    print("✔ Parámetros ajustados en 2014-2019 (Zero Leakage):")
    print(f"  Intercepto: {clf.intercept_[0]:.4f}")
    for f, c in zip(features, clf.coef_[0]):
        print(f"  {f:15s}: {c:+.4f}")

    train_probs = clf.predict_proba(X_train)[:, 1]
    print(f"  Media predicha entrenamiento: {train_probs.mean():.6f} (Obs: {y_train.mean():.6f})")

    print("\n▶ Proyectando a ciegas sobre 2020-2024...")
    df_eval = pd.read_csv(EVAL_CSV)
    df_eval = map_thermal(df_eval, range(2020, 2025))

    X_eval = df_eval[features].values
    eval_probs = clf.predict_proba(X_eval)[:, 1]

    df_eval_out = df_eval[["cell_id", "year", "centroid_lat", "centroid_lon", "y_joro_detected", "effort_araneae", "ecoregion_block"]].copy()
    df_eval_out["p_sdm"] = eval_probs.astype(np.float32)

    df_eval_out.to_csv(OUT_CSV, index=False)
    print(f"✔ Predicciones SDM guardadas en: {OUT_CSV}")
    print(f"  Tiempo de ejecución: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
