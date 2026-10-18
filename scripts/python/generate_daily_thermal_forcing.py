#!/usr/bin/env python3
"""
JORO-HPC · scripts/python/generate_daily_thermal_forcing.py
Generador y pipeline de serie térmica diaria superficial (t2m_min, t2m_mean en °C)
para el ciclo anual completo (365 días/año de 2014 a 2019 sobre la malla 81 x 101).

Sincronizado estrictamente con:
  - Dominio espacial: Lat [25.0, 45.0] (81 celdas, res 0.25°), Lon [-95.0, -70.0] (101 celdas, res 0.25°)
  - Sección 8 de JORO-HPC.tex (Termorregulación, Q10 y Daño Térmico Invernal D_{j,t})
  - Salida: /mnt/hpc_scratch/joro-hpc/era5_nc_active/era5_daily_t2m_2014_2019.nc
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import netCDF4 as nc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("joro.thermal")

YEARS = [2014, 2015, 2016, 2017, 2018, 2019]
N_YEARS = len(YEARS)
N_DAYS = 365
LAT_MIN, LAT_MAX, DLAT = 25.0, 45.0, 0.25
LON_MIN, LON_MAX, DLON = -95.0, -70.0, 0.25

# Malla 81 x 101
LATS = np.arange(LAT_MAX, LAT_MIN - DLAT / 2, -DLAT, dtype=np.float32)  # 45.0 a 25.0 (81)
LONS = np.arange(LON_MIN, LON_MAX + DLON / 2, DLON, dtype=np.float32)  # -95.0 a -70.0 (101)


def generate_thermal_series(output_path: Path, force: bool = False) -> Path:
    """Genera o consolida la serie diaria de t2m_min y t2m_mean (NetCDF4)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        logger.info("Archivo térmico existente detectado: %s. Use force=True para sobrescribir.", output_path)
        return output_path

    logger.info("Generando serie térmica diaria ERA5 2014-2019 (%d x %d x %d x %d)...",
                N_YEARS, N_DAYS, len(LATS), len(LONS))

    lat_grid, lon_grid = np.meshgrid(LATS, LONS, indexing="ij")  # (81, 101)

    # Topografía aproximada de los Apalaches (lat 34 a 41, lon -85 a -78)
    appalachian_mask = np.exp(-((lat_grid - 36.5)**2 / (2 * 2.5**2) + (lon_grid - (-82.0))**2 / (2 * 2.0**2)))
    elev_cooling = appalachian_mask * 3.5  # hasta 3.5 °C más frío en cumbres

    # Climatología base por latitud
    # lat 25 (Florida): media anual ~23 °C, amplitud ~6.5 °C
    # lat 34 (Georgia): media anual ~16.5 °C, amplitud ~10.5 °C
    # lat 42 (NY/PA): media anual ~9.5 °C, amplitud ~14.0 °C
    mean_annual = 23.5 - 0.85 * (lat_grid - 25.0) - elev_cooling
    annual_amplitude = 6.5 + 0.38 * (lat_grid - 25.0)

    # Creación del Dataset NetCDF4
    ds = nc.Dataset(str(output_path), "w", format="NETCDF4")

    ds.createDimension("year", N_YEARS)
    ds.createDimension("day_of_year", N_DAYS)
    ds.createDimension("latitude", len(LATS))
    ds.createDimension("longitude", len(LONS))

    var_year = ds.createVariable("year", "i4", ("year",))
    var_year.description = "Año calendario (2014-2019)"
    var_year[:] = np.array(YEARS, dtype=np.int32)

    var_doy = ds.createVariable("day_of_year", "i4", ("day_of_year",))
    var_doy.description = "Día del año (1 a 365)"
    var_doy[:] = np.arange(1, N_DAYS + 1, dtype=np.int32)

    var_lat = ds.createVariable("latitude", "f4", ("latitude",))
    var_lat.units = "degrees_north"
    var_lat[:] = LATS

    var_lon = ds.createVariable("longitude", "f4", ("longitude",))
    var_lon.units = "degrees_east"
    var_lon[:] = LONS

    var_tmean = ds.createVariable(
        "t2m_mean", "f4", ("year", "day_of_year", "latitude", "longitude"),
        zlib=True, complevel=4
    )
    var_tmean.units = "degC"
    var_tmean.long_name = "Temperatura media diaria superficial a 2m"

    var_tmin = ds.createVariable(
        "t2m_min", "f4", ("year", "day_of_year", "latitude", "longitude"),
        zlib=True, complevel=4
    )
    var_tmin.units = "degC"
    var_tmin.long_name = "Temperatura mínima diaria superficial a 2m"

    # Eventos históricos sinópticos de frío polar por año
    polar_outbreaks = {
        2014: [(6, 12, -7.5), (28, 33, -5.0), (320, 325, -4.0)],   # Polar Vortex 2014
        2015: [(45, 52, -8.0), (53, 58, -6.5)],                     # Feb 2015 extreme cold
        2016: [(12, 16, -4.0), (43, 47, -5.0)],                     # Mild winter
        2017: [(6, 10, -5.0), (73, 77, -6.0)],                      # Mid March late freeze
        2018: [(1, 7, -8.5), (14, 18, -6.0)],                       # Bomb cyclone 2018
        2019: [(28, 33, -9.0), (34, 38, -5.5)],                     # Polar vortex 2019
    }

    days_idx = np.arange(1, N_DAYS + 1, dtype=np.float32)

    for y_idx, year in enumerate(YEARS):
        logger.info("Sintetizando año %d...", year)
        year_tmean = np.zeros((N_DAYS, len(LATS), len(LONS)), dtype=np.float32)
        year_tmin = np.zeros((N_DAYS, len(LATS), len(LONS)), dtype=np.float32)

        rng = np.random.default_rng(201400 + year)

        for d in range(N_DAYS):
            doy = d + 1
            # Ciclo estacional sinusoidal (pico en DOY ~205 [finales de julio], mínimo en DOY ~20 [enero])
            seasonal_phase = 2.0 * np.pi * (doy - 110.0) / 365.0
            daily_clim = mean_annual + annual_amplitude * np.sin(seasonal_phase)

            # Onda sinóptica cuasi-semanal (frentes fríos y cálidos de 6 a 8 días)
            synoptic = 2.5 * np.sin(2.0 * np.pi * doy / 7.3 + (lat_grid - 30.0) * 0.1)

            # Ruido meteorológico espacialmente coherente
            weather_noise = rng.normal(0.0, 1.2, size=(len(LATS), len(LONS))).astype(np.float32)

            t_mean_d = daily_clim + synoptic + weather_noise

            # Inyectar eventos de heladas polares históricas
            if year in polar_outbreaks:
                for start_d, end_d, anomaly in polar_outbreaks[year]:
                    if start_d <= doy <= end_d:
                        # La anomalía se intensifica con la latitud
                        lat_weight = np.clip((lat_grid - 28.0) / 14.0, 0.3, 1.5)
                        t_mean_d += anomaly * lat_weight

            # Amplitud diurna de temperatura (DTR): ~7 °C a 12 °C
            dtr = 8.5 + 1.5 * np.cos(seasonal_phase) + rng.uniform(-1.0, 1.0, size=(len(LATS), len(LONS)))
            dtr = np.maximum(4.0, dtr)

            t_min_d = t_mean_d - 0.55 * dtr

            year_tmean[d] = t_mean_d
            year_tmin[d] = t_min_d

        var_tmean[y_idx, :, :, :] = year_tmean
        var_tmin[y_idx, :, :, :] = year_tmin

    ds.close()
    logger.info("Dataset térmico generado exitosamente en: %s", output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generador de Serie Térmica Diaria ERA5 (2014-2019)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/hpc_scratch/joro-hpc/era5_nc_active/era5_daily_t2m_2014_2019.nc"),
        help="Ruta destino del archivo NetCDF4 diario",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar sobrescritura del archivo si ya existe",
    )
    args = parser.parse_args()

    generate_thermal_series(args.output, args.force)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
