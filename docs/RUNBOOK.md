# JORO-HPC — Runbook operativo

Checklist paso a paso para reproducir el flujo completo: compilación, datos, calibración inversa (2014–2019), validación ciega (2020–2024) y regeneración de informes. Los comandos se ejecutan desde la raíz del repositorio.

> [!IMPORTANT]
> La partición 2020–2024 permanece **bloqueada** hasta el paso 6. Ningún paso de calibración (pasos 3–5) puede leerla: el lector ERA5 rechaza años > 2019 salvo que `JORO_UNLOCK_EVALUATION=1` (la establece `joro_evaluate` internamente). No exporte esa variable en su shell durante la calibración.

## 0. Requisitos

- [ ] Intel oneAPI Base + HPC Toolkit 2026.1 (`icpx`, SYCL, Level-Zero) y GPU Intel con controlador Level-Zero (referencia: Arc A380).
- [ ] CPU con AVX-512 (referencia: i9-11900K; la compilación usa `-march=rocketlake`).
- [ ] CMake ≥ 3.22, `libnetcdf`, `libpqxx` (opcional; sin ella se usan *stubs*).
- [ ] PostgreSQL ≥ 14 + PostGIS ≥ 3.3 en el puerto 5433 (esquema en `scripts/sql/`).
- [ ] Credenciales de la base de datos **sólo por entorno**: `cp .env.example .env`, configure `JORO_DB_PASSWORD` y cargue con `set -a; source .env; set +a` (`.env` está ignorado por Git y con permisos `600`). Variables: `JORO_DB_HOST` (`localhost`), `JORO_DB_PORT` (`5433`), `JORO_DB_NAME` (`joro_db`), `JORO_DB_USER` (`joro_admin`), `JORO_DB_PASSWORD` (sin valor por defecto). Los scripts de calibración y Python cargan `.env` automáticamente si existe.
  ```bash
  # Verificación rápida de autenticación en PostGIS
  set -a; source .env; set +a
  PGPASSWORD="${JORO_DB_PASSWORD}" psql -h "${JORO_DB_HOST}" -p "${JORO_DB_PORT}" -U "${JORO_DB_USER}" -d "${JORO_DB_NAME}" \
    -c "SELECT COUNT(*) AS verified_grid_cells FROM grid_5km;"  # debe retornar 256,020 celdas
  ```
- [ ] Almacenamiento: `/mnt/hpc_scratch/joro-hpc` (el enlace `data/` del repositorio apunta ahí; está ignorado por Git) y repositorio atmosférico compartido en `/mnt/hpc_scratch/shared_env/era5_conus_east` (vinculado vía *hardlinks* XFS hacia `data/era5_nc_active/`).
- [ ] Entorno Python: `python3 -m venv .venv && .venv/bin/pip install -r scripts/python/requirements.txt`.

## 1. Compilar y probar

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure      # debe pasar 4/4
```

- [ ] Se generan `build/joro_hpc` (simulación y calibración) y `build/joro_evaluate` (validación fuera de muestra).
- [ ] `ctest`: `test_well_mixed`, `test_euler_maruyama_conv`, `test_cpu_gpu_equivalence`, `test_parameter_recovery` → 4/4.

## 2. Datos (Esquema SQL y Pipeline Local de Ingesta)

Los conjuntos de datos congelados (`calibration_set_2014_2019.csv` y `evaluation_set_2020_2024.csv`) y los tensores atmosféricos NetCDF4 se alojan en `data/` (`/mnt/hpc_scratch/joro-hpc`). Para reconstruir la base de datos PostGIS desde cero en la estación de trabajo local (`local_pipeline/`, ignorada por Git):

```bash
set -a; source .env; set +a

# 1. Esquema y particiones espaciales en PostGIS (públicos en scripts/sql/)
PGPASSWORD="${JORO_DB_PASSWORD}" psql -h "${JORO_DB_HOST:-localhost}" -p "${JORO_DB_PORT:-5433}" -U "${JORO_DB_USER:-joro_admin}" -d "${JORO_DB_NAME:-joro_db}" -f scripts/sql/01_schema_postgis.sql
PGPASSWORD="${JORO_DB_PASSWORD}" psql -h "${JORO_DB_HOST:-localhost}" -p "${JORO_DB_PORT:-5433}" -U "${JORO_DB_USER:-joro_admin}" -d "${JORO_DB_NAME:-joro_db}" -f scripts/sql/02_spatial_blocks.sql
# (03_seed_smoke_grid.sql sólo para pruebas de humo)

# 2. Ingesta local en estación de trabajo (local_pipeline/ingestion/)
.venv/bin/python local_pipeline/ingestion/fetch_era5_full_season.py --workers 1
.venv/bin/python local_pipeline/ingestion/fetch_era5_interannual.py
.venv/bin/python local_pipeline/ingestion/build_grid_nlcd.py
.venv/bin/python local_pipeline/ingestion/build_census_population.py
.venv/bin/python local_pipeline/ingestion/build_real_covariates.py
.venv/bin/python local_pipeline/ingestion/fetch_gbif_araneae_exact.py
.venv/bin/python local_pipeline/ingestion/fetch_gbif_tgb.py
.venv/bin/python local_pipeline/ingestion/export_validation_datasets.py
```

Verifique que `calibration_set_2014_2019.csv` y `evaluation_set_2020_2024.csv` no cambian de hash entre corridas.

## 3. Comparador estático (SDM — Hipótesis Nula H1)

- [ ] `.venv/bin/python scripts/python/run_static_sdm.py` — entrena con 2014–2019 y escribe `outputs/sdm_predictions_2020_2024.csv` y `outputs/sdm_coefficients.json`.

## 4. Calibración inversa 2014–2019

```bash
set -a; source .env; set +a
./build/joro_hpc --calibrate --engine=gpu --model M1
# o para correr M1, M0 y M3 con telemetría local: bash local_pipeline/telemetry/run_calibration.sh
```

- [ ] Salidas en `/mnt/hpc_scratch/joro-hpc/checkpoints/`: `calibration_summary_{M1,M0,M3}.json`, `theta_compatible_ensemble{,_M0,_M3}.csv`, `telemetry_{M1,M0,M3}.json`.
- [ ] Duración de referencia: ≈ 15 s por modelo en Arc A380.
- [ ] Para telemetría completa de hardware local: `local_pipeline/telemetry/run_calibration_telemetry.sh`.

## 5. Informe de calibración (generador local)

```bash
set -a; source .env; set +a
.venv/bin/python local_pipeline/reporting/generate_calibration_report.py
```
- [ ] Genera `docs/reports/03_Inverse_Calibration_2014_2019.md` y `docs/telemetry/calibration_hardware_profile.json`.

## 6. Validación ciega 2020–2024 y Análisis Estadístico (sin reajuste)

```bash
set -a; source .env; set +a
./build/joro_evaluate                                              # propaga los ensambles congelados 2014–2024
.venv/bin/python scripts/python/compute_validation_metrics.py      # métricas oficiales (AUC con corrección de empates, Brier, cinemática)
.venv/bin/python local_pipeline/reporting/generate_validation_report.py # docs/reports/04_Out_of_Sample_Validation_2020_2024.md
```

- [ ] Salidas en `outputs/`: `predictions_2020_2024_ensemble.csv`, `validation_sim_kinematics.json`, `validation_metrics.json`, `seed_geometry.json` (centroide y radio de las 124 celdas semilla, referencia del azimut) y `prevalence_scaling.json` (offset logit de prevalencia ajustado sólo con 2014–2019).
- [ ] Duración de referencia de `joro_evaluate`: ≈ 27 s.
- [ ] El informe 04 calcula los dictámenes de H1 y H2 con criterios fijos; revise la sección «Hallazgos que matizan los dictámenes».

## 7. Informes de verificación (01 y 02)

```bash
set -a; source .env; set +a
./build/test_well_mixed && ./build/test_euler_maruyama_conv && ./build/test_cpu_gpu_equivalence
./build/joro_hpc --engine=both --no-calibration          # informe 01 (CPU vs GPU)
./build/joro_hpc --engine=gpu --morris --no-calibration  # informe 02 (Morris)
./build/joro_hpc --diagnostics --engine=gpu              # transecto real (ABM + ERA5 diario) + Morris
```

## 8. Propuesta en PDF

```bash
docker run --rm --user $(id -u):$(id -g) -v "$PWD/docs/proposal:/work:z" -w /work \
  texlive/texlive:latest-medium sh -c 'for i in 1 2 3; do pdflatex -interaction=nonstopmode JORO-HPC_Research_Proposal.tex; done'
rm -f docs/proposal/*.aux docs/proposal/*.log docs/proposal/*.out docs/proposal/*.toc
```

## 9. Banderas, alias y variables de entorno

### Banderas de línea de comandos

| Bandera / variable | Efecto | Alias heredado (sigue funcionando) |
|---|---|---|
| `--calibrate` | sólo calibración inversa | `--phase4` |
| `--diagnostics` | transecto + Morris, sin calibración | `--phase3` |
| `--transect`, `--morris` | activan cada diagnóstico por separado | — |
| `--model M0\|M1\|M3\|all` | modelo de capacidad de carga | — |
| `--engine=cpu\|gpu\|both` | motor LPDM | — |
| `JORO_UNLOCK_EVALUATION=1` | desbloquea 2020–2024 (la fija `joro_evaluate`) | `JORO_UNLOCK_PHASE5` |

El ejecutable `phase5_runner` fue renombrado a `joro_evaluate`.

### Variables de entorno (`.env`)

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `JORO_DB_HOST` | Host de PostgreSQL / PostGIS | `localhost` |
| `JORO_DB_PORT` | Puerto de PostgreSQL / PostGIS | `5433` |
| `JORO_DB_NAME` | Nombre de la base de datos | `joro_db` |
| `JORO_DB_USER` | Usuario de PostgreSQL | `joro_admin` |
| `JORO_DB_PASSWORD` | Contraseña obligatoria (leída del entorno) | *sin valor por defecto* |
| `JORO_UNLOCK_EVALUATION` | Desbloquea ERA5 2020–2024 (fijada por `joro_evaluate`) | `0` |
| `OMP_NUM_THREADS` | Hilos OpenMP para el ABM en CPU | `16` (o número de núcleos de la CPU) |
| `OMP_PROC_BIND` | Afinidad de hilos OpenMP | `close` |
| `OMP_PLACES` | Ubicación de hilos OpenMP | `cores` |

## 10. Problemas conocidos

1. **Identificabilidad en $\mathcal{M}_1$.** En $\Theta_{\text{comp}}$ hay pares con $|\rho| \ge 0.9$ (`gamma_hab`–`a0`, `k0`–`a0`, `k0`–`gamma_hab`); estos parámetros no se interpretan por separado (informe 03).
2. **Calibración absoluta.** El ensamble discrimina bien (AUC ≈ 0.97) pero subestima el total anual de detecciones 4.0–7.7 veces (crudo). El offset logit de prevalencia, ajustado sólo con 2014–2019, lo reduce a 2.4–4.9 veces, pero no cubre lo observado en el intervalo P5–P95 en ninguno de los 5 años (informe 04, tabla 5).
3. **Dirección del frente.** Con el azimut referido al centroide real de las 124 celdas semilla, el error angular medio del LPDM (≈ 160°) sigue siendo peor que el del kernel isotrópico (≈ 125°): el LPDM no mejora la dirección de expansión sobre las celdas colonizadas fuera del radio semilla (informe 04, tabla 3). La discrepancia direccional medida con ψ (≈ 25°) sugiere que el desacople proviene de dónde se detecta, no de dónde llegan los propágulos.
4. **CPU frente a GPU.** Las corridas acopladas divergen estocásticamente (hasta 9.5 % en propágulos en 2019); calibrar y evaluar siempre con el mismo motor (informe 01).
5. **`-march=rocketlake`.** Las banderas de producción son específicas de la CPU de referencia; sin ellas, CMake cae a `-march=x86-64-v3` (portable).
6. **Hardware en la propuesta.** El texto de la propuesta menciona una Arc A770; las corridas y *benchmarks* documentados se hicieron en la GPU de desarrollo Arc A380 y están listos para la A770.
7. **Transecto con parámetros nominales.** `--transect` usa los parámetros demográficos nominales (no $\Theta_{\text{comp}}$ calibrado): la persistencia falla desde ≈ 36.5 °N, lo que contradice la expansión observada (informe 02, sección 4).
