#!/usr/bin/env python3
"""
JORO-HPC · scripts/python/fetch_era5_interannual.py
Descarga ERA5 interanual (2014-2024), un conjunto de archivos por año, vía CDS API.

Por año Y se generan en /mnt/hpc_scratch/joro-hpc/era5_nc_active/:
  era5_uvwt_{Y}_05.nc   u, v, w(omega), t en 12 niveles 1000-700 hPa, 168 h (1-7 de mayo)
  era5_sfc_{Y}_05.nc    sp, t2m, u10, v10, blh, 168 h (1-7 de mayo)
  era5_daily_t2m_{Y}.nc t2m diaria REAL: t2m_mean y t2m_min (°C), 365 días (01-ene a 30/31-dic)
                        agregando la t2m horaria anual de 'reanalysis-era5-single-levels' por día UTC (media y mínimo)

Regla de aislamiento: los archivos 2020-2024 se descargan y quedan en disco, pero el simulador
de Fase 4.5 solo carga 2014-2019 (el lector C++ lanza error si se le pide un año > 2019).

Uso:
  .venv/bin/python scripts/python/fetch_era5_interannual.py --years 2014 2015 ... --workers 4
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("joro.era5.interannual")

OUT_DIR = Path("/mnt/hpc_scratch/joro-hpc/era5_nc_active")
AREA = [45.0, -95.0, 25.0, -70.0]  # N, W, S, E  (malla 81 x 101)
DAYS_WINDOW = [f"{d:02d}" for d in range(1, 8)]
HOURS = [f"{h:02d}:00" for h in range(24)]
LEVELS = ["1000", "975", "950", "925", "900", "875", "850", "825", "800", "775", "750", "700"]
VARS_3D = ["u_component_of_wind", "v_component_of_wind", "vertical_velocity", "temperature"]
VARS_SFC = ["surface_pressure", "2m_temperature", "10m_u_component_of_wind",
            "10m_v_component_of_wind", "boundary_layer_height"]


_CDS_SLOTS = threading.Semaphore(3)  # el CDS limita las solicitudes en cola por usuario


def _retrieve(dataset: str, request: dict, dest: str, max_tries: int = 40) -> None:
    """retrieve() con reintentos y backoff cuando el CDS rechaza por límite de cola (HTTP 400)."""
    import cdsapi
    for attempt in range(1, max_tries + 1):
        with _CDS_SLOTS:
            try:
                cdsapi.Client(quiet=True, retry_max=3, timeout=120).retrieve(dataset, request, dest)
                return
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "temporarily limited" not in msg and "rejected" not in msg:
                    raise
                log.warning("CDS rechazó %s (intento %d/%d): cola limitada; reintento", dataset,
                            attempt, max_tries)
        time.sleep(min(600, 60 * attempt))
    raise RuntimeError(f"{dataset}: agotados {max_tries} reintentos")


def _unzip_if_needed(path: Path) -> None:
    """El CDS puede devolver un zip con varios .nc; se deja un único NetCDF en `path`."""
    if not zipfile.is_zipfile(path):
        return
    tmp = path.with_suffix(".zip")
    path.rename(tmp)
    with zipfile.ZipFile(tmp) as z:
        names = [n for n in z.namelist() if n.endswith(".nc")]
        if len(names) != 1:
            raise RuntimeError(f"{tmp.name}: se esperaba 1 NetCDF y hay {names}")
        with z.open(names[0]) as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    tmp.unlink()


def fetch_3d(year: int, force: bool) -> Path:
    target = OUT_DIR / f"era5_uvwt_{year}_05.nc"
    if target.exists() and not force:
        return target
    tmp = target.with_suffix(".part")
    _retrieve("reanalysis-era5-pressure-levels", {
        "product_type": "reanalysis", "format": "netcdf", "variable": VARS_3D,
        "pressure_level": LEVELS, "year": str(year), "month": "05",
        "day": DAYS_WINDOW, "time": HOURS, "area": AREA}, str(tmp))
    _unzip_if_needed(tmp)
    tmp.rename(target)
    return target


def fetch_sfc(year: int, force: bool) -> Path:
    target = OUT_DIR / f"era5_sfc_{year}_05.nc"
    if target.exists() and not force:
        return target
    tmp = target.with_suffix(".part")
    _retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "format": "netcdf", "variable": VARS_SFC,
        "year": str(year), "month": "05", "day": DAYS_WINDOW, "time": HOURS, "area": AREA}, str(tmp))
    _unzip_if_needed(tmp)
    tmp.rename(target)
    return target


def _fetch_hourly_t2m(year: int, dest: Path) -> None:
    """t2m horaria de todo el año (8760/8784 campos) desde reanalysis-era5-single-levels.

    El dataset `derived-era5-single-levels-daily-statistics` quedaba horas en cola; las estadísticas
    diarias en UTC (media y mínimo de las 24 horas) se calculan aquí con la misma definición.
    """
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    _retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "format": "netcdf", "variable": ["2m_temperature"],
        "year": str(year), "month": months, "day": days, "time": HOURS, "area": AREA}, str(dest))
    _unzip_if_needed(dest)


def _daily_from_hourly(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t2m_mean[K], t2m_min[K], lat, lon) por día UTC a partir de la serie horaria."""
    import netCDF4 as nc
    with nc.Dataset(path) as ds:
        tname = "valid_time" if "valid_time" in ds.variables else "time"
        tv = ds.variables[tname]
        times = nc.num2date(tv[:], tv.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        vname = "t2m" if "t2m" in ds.variables else [v for v in ds.variables if ds.variables[v].ndim == 3][0]
        data = np.ma.filled(ds.variables[vname][:], np.nan).astype(np.float32)
        lats = np.array(ds.variables["latitude"][:], dtype=np.float32)
        lons = np.array(ds.variables["longitude"][:], dtype=np.float32)
    doy = np.array([t.timetuple().tm_yday for t in times])
    order = np.argsort(np.array([t.toordinal() * 24 + t.hour for t in times]))
    data, doy = data[order], doy[order]
    days = np.unique(doy)
    mean = np.empty((len(days),) + data.shape[1:], np.float32)
    mn = np.empty_like(mean)
    for i, d in enumerate(days):
        block = data[doy == d]
        if block.shape[0] != 24:
            raise RuntimeError(f"{path.name}: el día {d} tiene {block.shape[0]} horas (se esperaban 24)")
        mean[i] = block.mean(axis=0)
        mn[i] = block.min(axis=0)
    return mean, mn, lats, lons


def fetch_daily(year: int, force: bool) -> Path:
    """Serie diaria REAL de t2m (media y mínima, °C, 365 días) empaquetada en un NetCDF."""
    import netCDF4 as nc
    target = OUT_DIR / f"era5_daily_t2m_{year}.nc"
    if target.exists() and not force:
        return target
    raw = OUT_DIR / f".raw_hourly_t2m_{year}.nc"
    if not raw.exists() or force:
        _fetch_hourly_t2m(year, raw)
    mean, mn, lats, lons = _daily_from_hourly(raw)
    arrs = {}
    for key, data in (("t2m_mean", mean), ("t2m_min", mn)):
        if lats[0] < lats[-1]:  # el motor C++ espera latitud descendente (45 -> 25)
            data = data[:, ::-1, :]
        arrs[key] = data - 273.15  # K -> °C
    if lats[0] < lats[-1]:
        lats = lats[::-1].copy()
    if arrs["t2m_mean"].shape[1:] != (81, 101) or abs(float(lats[0]) - 45.0) > 1e-3 or abs(float(lons[0]) + 95.0) > 1e-3:
        raise RuntimeError(f"{raw.name}: malla inesperada {arrs['t2m_mean'].shape[1:]} lat0={lats[0]} lon0={lons[0]}")
    n_days = arrs["t2m_mean"].shape[0]
    expected = 366 if year % 4 == 0 else 365
    if n_days != expected:
        raise RuntimeError(f"{year}: {n_days} días recibidos, se esperaban {expected}")
    if np.isnan(arrs["t2m_mean"]).any() or np.isnan(arrs["t2m_min"]).any():
        raise RuntimeError(f"{year}: NaN en la serie diaria")
    # El motor ABM usa 365 días: en años bisiestos se descarta el 31 de diciembre.
    tmp = target.with_suffix(".part")
    with nc.Dataset(tmp, "w", format="NETCDF4") as out:
        out.description = (f"ERA5 daily statistics t2m {year} (UTC). Variables en °C. "
                           f"{n_days} dias descargados; se guardan los primeros 365.")
        out.createDimension("day_of_year", 365)
        out.createDimension("latitude", len(lats))
        out.createDimension("longitude", len(lons))
        out.createVariable("latitude", "f4", ("latitude",))[:] = lats
        out.createVariable("longitude", "f4", ("longitude",))[:] = lons
        for key, a in arrs.items():
            v = out.createVariable(key, "f4", ("day_of_year", "latitude", "longitude"),
                                   zlib=True, complevel=4)
            v.units = "degC"
            v[:] = a[:365]
    tmp.rename(target)
    raw.unlink(missing_ok=True)
    return target


JOBS = {"3d": fetch_3d, "sfc": fetch_sfc, "daily": fetch_daily}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2025)))
    ap.add_argument("--products", nargs="+", default=list(JOBS), choices=list(JOBS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(y, p) for y in args.years for p in args.products]
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(JOBS[p], y, args.force): (y, p) for y, p in tasks}
        for f in as_completed(futs):
            y, p = futs[f]
            try:
                path = f.result()
                log.info("OK  %d %-5s -> %s (%.1f MB)", y, p, path.name, path.stat().st_size / 1e6)
            except Exception as e:  # noqa: BLE001
                log.error("FALLO %d %s: %s", y, p, e)
                failed.append((y, p))
    if failed:
        log.error("Pendientes: %s", sorted(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
