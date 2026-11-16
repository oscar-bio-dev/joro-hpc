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
JORO_ERA5_TIME_STRIDE=3 ./build/joro_hpc --calibrate --engine=gpu --model all
# o para correr M1, M0 y M3 con telemetría local: bash local_pipeline/telemetry/run_calibration.sh
```

- [ ] Salidas en `/mnt/hpc_scratch/joro-hpc/checkpoints/`: `calibration_summary_{M1,M0,M3}.json`, `theta_compatible_ensemble{,_M0,_M3}.csv`, `telemetry_{M1,M0,M3}.json`.
- [ ] Duración de referencia (temporada completa mayo–julio, `JORO_ERA5_TIME_STRIDE=3`, `3.64 GiB` en VRAM sobre Arc A380): ≈ `119.9 s` (`M1`), `83.8 s` (`M0`), `127.6 s` (`M3`).
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
JORO_ERA5_TIME_STRIDE=3 ./build/joro_evaluate                      # propaga los ensambles congelados 2014–2024
.venv/bin/python scripts/python/compute_validation_metrics.py      # métricas oficiales (AUC con corrección de empates, Brier, cinemática, Cox)
.venv/bin/python local_pipeline/reporting/generate_validation_report.py # docs/reports/04_Out_of_Sample_Validation_2020_2024.md
```

- [ ] Salidas en `outputs/`: `predictions_2020_2024_ensemble.csv`, `validation_sim_kinematics.json`, `validation_metrics.json`, `hypothesis_diagnostics.json`, `seed_geometry.json` (centroide y radio de las 125 celdas semilla de Braselton/Hoschton, `37.6 km`, referencia del azimut) y `prevalence_scaling.json` (offset logit de prevalencia ajustado sólo con 2014–2019).
- [ ] Duración de referencia de `joro_evaluate` (339 miembros de $\mathcal{M}_1$ + comparadores, 11 años): ≈ `211.9 s` (`0.215 s` por configuración de 11 años).
- [ ] El informe 04 calcula los dictámenes de H1, H2, H3a y H3b con criterios fijos, así como la tabla de sensibilidad temporal `1 h` vs `3 h`.

## 7. Informes de verificación (01 y 02)

```bash
set -a; source .env; set +a
./build/test_well_mixed && ./build/test_euler_maruyama_conv && ./build/test_cpu_gpu_equivalence && ./build/test_parameter_recovery
./build/joro_hpc --engine=both --no-calibration          # informe 01 (CPU vs GPU)
./build/joro_hpc --engine=gpu --morris --no-calibration  # informe 02 (Morris)
JORO_ERA5_TIME_STRIDE=3 ./build/joro_hpc --diagnostics --engine=gpu # transecto real con MAP de M1 + Morris
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
| --- | --- | --- |
| `--calibrate` | sólo calibración inversa | `--phase4` |
| `--diagnostics` | transecto + Morris, sin calibración | `--phase3` |
| `--transect`, `--morris` | activan cada diagnóstico por separado | — |
| `--model M0, M1, M3, all` | modelo de capacidad de carga | — |
| `--engine=cpu, gpu, both` | motor LPDM | — |
| `--benchmark-single-eval` | perfila UNA evaluación completa de J(θ) (6 años): ABM / costo / H2D / kernel / D2H | — |
| `JORO_UNLOCK_EVALUATION=1` | desbloquea 2020–2024 (la fija `joro_evaluate`) | `JORO_UNLOCK_PHASE5` |

El ejecutable `phase5_runner` fue renombrado a `joro_evaluate`.

### Variables de entorno (`.env`)

| Variable | Descripción | Valor por defecto |
| --- | --- | --- |
| `JORO_DB_HOST` | Host de PostgreSQL / PostGIS | `localhost` |
| `JORO_DB_PORT` | Puerto de PostgreSQL / PostGIS | `5433` |
| `JORO_DB_NAME` | Nombre de la base de datos | `joro_db` |
| `JORO_DB_USER` | Usuario de PostgreSQL | `joro_admin` |
| `JORO_DB_PASSWORD` | Contraseña obligatoria (leída del entorno) | *sin valor por defecto* |
| `JORO_UNLOCK_EVALUATION` | Desbloquea ERA5 2020–2024 (fijada por `joro_evaluate`) | `0` |
| `JORO_ERA5_TIME_STRIDE` | Paso temporal (h) de almacenamiento ERA5 en FP16. `1` = serie horaria; `3` deja los 6 años de calibración (3.64 GiB) 100 % residentes en la VRAM (0 H2D por evaluación, $\lvert\Delta\text{AUC}\rvert = 0.0003$ vs `1 h`). **Usar el mismo valor en calibración y evaluación** | `1` |
| `JORO_VRAM_BUDGET_GIB` | Tope de VRAM para campos ERA5 (≤ 2 años simultáneos si se excede) | `3.8` |
| `JORO_MAX_SUPER` | Tope de súper-partículas por despacho anual (v2.0: hasta 5000000) | `120000` |
| `JORO_P_HUMAN`, `JORO_LAMBDA_HUMAN_KM` | Kernel mixto humano en calibración: p_h (≡ w_h) y escala λ_h; `0` = solo LPDM | `0`, `150` |
| `JORO_LAMBDA_M`, `JORO_KAPPA_TAU` | Coeficientes fijos de f_est(i,t) (notoriedad y habituación, calibrados externamente) | `1.15`, `0.25` |
| `JORO_ATTENTION_INDEX` | CSV `year,M_t,M_ratio_2019` (proxy exógeno de actividad de plataforma) | `/mnt/hpc_scratch/joro-hpc/attention_index_Mt.csv` |
| `JORO_CHECKPOINT_DIR` | Directorio de checkpoints de `--calibrate` (usar uno temporal en pruebas de humo) | `/mnt/hpc_scratch/joro-hpc/checkpoints/` |
| `JORO_EVAL_MAX_MEMBERS` | Limita el ensamble M1 propagado por `joro_evaluate` (verificación rápida) | todos |
| `OMP_NUM_THREADS` | Hilos OpenMP para el ABM en CPU | `16` (o número de núcleos de la CPU) |
| `OMP_PROC_BIND` | Afinidad de hilos OpenMP | `close` |
| `OMP_PLACES` | Ubicación de hilos OpenMP | `cores` |

## 10. Problemas resueltos en `v2.0.0` (Auditoría `M1`–`M16`) y notas operativas

1. **Trazabilidad de las 16 brechas de auditoría (`07_Pre_Run_Scientific_Code_Audit.md`):**
   - `M1`–`M8` (Bloque 1): Lector NetCDF4 multi-mes estacional (mayo–julio = 92 días, `2,208 h/año`) con soporte CF `scale_factor`/`add_offset`; cuantización `FP16` (`Wind4Half`, 8 bytes/vóxel) en USM con interpolación tetralineal en `FP32`; horizonte de vuelo acoplado a la ventana de 92 días; integración exacta de Ornstein-Uhlenbeck con corrección de Thomson (1987) acotada; y compuerta de despegue fenológico-convectiva ($u_{10} < 3.0\text{ m/s}$, ventana diurna 14–22 UTC, $S_{\text{conv}}$). **✔ Resuelto.**
   - `M9`–`M16` (Bloque 2): Fusión tripartita deduplicada GBIF + iNaturalist + EDDMapS (`1,359` registros únicos nuevos) y esfuerzo exógeno $E_{i,t}^{-J} = \max(0, C_{i,t}^{\text{Araneae}} - C_{i,t}^J)$; polígono semilla limpio de Braselton/Hoschton (`125` celdas contiguas, radio `37.6 km`); amortiguación térmica microclimática $\delta_{\text{dosel}} = 0.35\text{ °C/SD}$ y $\delta_{\text{imp}} = 0.50\text{ °C/SD}$; emergencia logística primaveral ($DD_{\text{emerg}} = 0.12, \kappa_{DD} = 8.0$); colonización mecanística $\operatorname{cloglog}$ sin generación espontánea ($\gamma_{i,t} = 0$ si $A_{i,t}^{\text{total}} = 0$) y kernel gravitacional humano ponderado por red vial ($K_{\text{mixto}}$); proceso de observación no estacionario $f_{\text{est}}(i,t)$ ($\lambda_M = 1.15, \kappa_\tau = 0.25$); centrado de $H_i^{\text{hab}}$ sobre el soporte informativo $S$ ($\rho(K_0, \gamma_{\text{hab}}) = -0.38$, 0 parámetros truncados en bordes); y calibración logística de Cox ($c_1 = 0.641$). **✔ Resuelto.**
2. **Cobertura empírica y prevalencia (H3b):** Con $f_{\text{est}}(i,t)$ activa ($\delta = 0$), el modelo predice `2,849.5` detecciones totales en 2020–2024 frente a `2,787` observadas (`+2.2 %`), y el ensamble $\Theta_{\text{comp}}$ (`339` miembros) cubre el total anual observado en **3 de 5 años** (`2020, 2023, 2024`), con una cobertura empírica por celda de **91.92 %** (nominal `90.0 %`).
3. **Geometría del frente por cuadrante vs azimut medio (H2b):** Aunque el error angular circular medio sigue siendo alto (`140.1°`) por la forma multilobulada de la expansión y el peso demográfico de Atlanta (SW), el desglose por cuadrantes (Tabla 6 del informe 04) demuestra que $\mathcal{M}_1$ (`256.8 km`) y $\mathcal{M}_1 + K_{\text{mixto}}$ (`263.0 km`) capturan la rápida expansión hacia el **noreste (`NE = 270.6 km` observado)**, donde el kernel isotrópico colapsa (`104.0 km`).
4. **Transecto con parámetros calibrados:** `--transect` carga ahora automáticamente $\hat{\theta}_{\text{MAP}}$ de `calibration_summary_M1.json`.
5. **CPU frente a GPU y banderas de compilación:** Calibrar y evaluar siempre con el mismo motor (`gpu`); `-march=rocketlake` se aplica cuando `JORO_ENABLE_AVX512=ON` y cae a `-march=x86-64-v3` en contenedores/CI portables.
