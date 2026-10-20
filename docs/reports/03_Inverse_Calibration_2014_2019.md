# JORO-HPC — Calibración inversa 2014–2019: datos, identificabilidad y ensamble de parámetros compatibles (Θ_comp)

_Generado automáticamente el 2026-10-19 21:31 por `scripts/python/generate_calibration_report.py` a partir de los artefactos de la corrida. Los veredictos se calculan con reglas fijas._

- Dispositivo LPDM: **Intel(R) Arc(TM) A380 Graphics | Intel(R) oneAPI Unified Runtime over Level-Zero | GPU**.
- Hilos OpenMP: **16**. Evaluaciones por modelo: **2000** (5 cadenas × 400).

## 1. Ingesta de datos

### 1.1 ERA5 interanual (2014–2024) en disco; solo 2014–2019 leído

| Año | uvwt 3D | superficial | t2m diaria (365 d) | Rol |
|---|---|---|---|---|
| 2014 | ✔ 155.8 MB | ✔ 13.1 MB | ✔ 12.4 MB | calibración (leído) |
| 2015 | ✔ 155.9 MB | ✔ 13.2 MB | ✔ 12.2 MB | calibración (leído) |
| 2016 | ✔ 158.0 MB | ✔ 13.2 MB | ✔ 12.2 MB | calibración (leído) |
| 2017 | ✔ 157.3 MB | ✔ 13.2 MB | ✔ 12.2 MB | calibración (leído) |
| 2018 | ✔ 159.5 MB | ✔ 13.2 MB | ✔ 12.3 MB | calibración (leído) |
| 2019 | ✔ 160.9 MB | ✔ 13.2 MB | ✔ 12.3 MB | calibración (leído) |
| 2020 | ✔ 158.5 MB | ✔ 13.2 MB | ✔ 12.3 MB | reservado para validación (NO leído) |
| 2021 | ✔ 161.3 MB | ✔ 13.1 MB | ✔ 12.2 MB | reservado para validación (NO leído) |
| 2022 | ✔ 154.3 MB | ✔ 13.2 MB | ✔ 12.4 MB | reservado para validación (NO leído) |
| 2023 | ✔ 152.6 MB | ✔ 13.1 MB | ✔ 12.3 MB | reservado para validación (NO leído) |
| 2024 | ✔ 158.0 MB | ✔ 13.1 MB | ✔ 12.2 MB | reservado para validación (NO leído) |

- Archivos ERA5 efectivamente abiertos por el simulador: **21**; años distintos leídos: **[2014, 2015, 2016, 2017, 2018, 2019]**.
- Solicitudes bloqueadas por la regla de aislamiento (año > 2019): **[]**.
- Campos ERA5 3D+superficiales residentes en la GPU (uno por año): **6** (cada año simulado usa SU meteorología; ya no se recicla el 1–7 de mayo de 2014).
- Modo estricto del lector (sin datos sintéticos de respaldo): **True**; directorio: `/mnt/hpc_scratch/joro-hpc/era5_nc_active`.
- **Veredicto:** ✔ CUMPLE.

> **Alcance meteorológico (limitación):** los tensores 3D/superficiales cubren la ventana 1–7 de mayo de cada año (la fase de baloneo), y la serie `t2m` diaria cubre los 365 días. Es lo que consume el simulador.

### 1.2 Población censal real (`pop_density_h`)

- Fuente: Censo 2020 (PL 94-171, encabezado geográfico por bloque: POP100 + punto interno), asignado a la celda de 5 km (EPSG:5070) que contiene el punto interno; `pop_density_h = estandarizar(log1p(hab/km²))`.
- `grid_5km`: **256,020** celdas, media **-0.0000**, sd **1.0000**, **11,803** valores distintos.
- Población total asignada: **215,846,400** hab. en **114,070** celdas pobladas.
- Correlación con el proxy NLCD anterior (respaldado en `grid_5km_pop_proxy_backup`): **0.920**.
- Limitación: censo 2020 aplicado a toda la ventana 2014–2019 (sin serie anual); asignación por punto interno.

### 1.3 Esfuerzo exacto GBIF *Araneae* (`taxonKey = 1496`)

| Año | Celdas | Σ effort_araneae | máx | celdas con effort>0 | valores distintos | presencias *T. clavata* |
|---|---|---|---|---|---|---|
| 2014 | 39,917 | 493 | 25 | 280 | 13 | 1 |
| 2015 | 39,917 | 928 | 75 | 418 | 19 | 0 |
| 2016 | 39,917 | 1,501 | 32 | 737 | 22 | 2 |
| 2017 | 39,917 | 2,666 | 39 | 1,219 | 25 | 5 |
| 2018 | 39,917 | 4,904 | 52 | 1,917 | 35 | 26 |
| 2019 | 39,917 | 7,379 | 118 | 3,001 | 41 | 72 |
| 2020–2024 | — | reservado para la validación fuera de muestra: este informe no consulta estos años | | | | |

- Método: conteo de registros georreferenciados (`hasCoordinate`, sin problemas geoespaciales) asignados a la celda por punto en la celda EPSG:5070; `effort = GREATEST(conteo, joro_count)`. Respaldo previo: `cell_annual_observations_effort_density_backup`.
- `v_calibration_set`: años **2014–2019**, 239,502 filas, 106 presencias.

## 2. Telemetría de tiempos y utilización GPU/CPU (M1, modelo principal)

| Métrica | Calibración actual (M1) | Línea base (commit `3fa0d37`) |
|---|---|---|
| Tiempo total (2,000 evaluaciones) | **13.2 s** | 25.9 s |
| H2D + D2H | **1.06 s** (8.0 % de pared) | 2.08 s |
| Kernel SYCL en GPU | 8.48 s (64.3 % de pared) | — |
| ABM (OpenMP + AVX-512) | 1.63 s | — |
| Ocupación + costo (OpenMP) | 0.75 s | — |
| LPDM host (init + reducción) | 0.56 s | — |
| Super-partículas despachadas | 203.4 M | — |
| Despachos con fallback a CPU | **0** | 0 |

Muestreo externo durante la corrida (`sample_telemetry.py`; `intel_gpu_top` requiere CAP_PERFMON, no disponible, por lo que la ocupación GPU se **estima** como 100 % − residencia RC6, que mide actividad del GT, no del motor Compute):

| Recurso | media | P95 | máx |
|---|---|---|---|
| CPU, promedio de 16 hilos (%) | 66.7 | 100.0 | 100.0 |
| GPU activa (100 − RC6, %) | 91.9 | 100.0 | 100.0 |
| Frecuencia GPU (MHz) | 1773 | 2450 | 2450 |
| Potencia GPU (W) | 30.5 | 39.3 | 39.9 |

Hilos con carga media > 80 %: **1 de 16**.

## 3. Convergencia de las 5 cadenas (M1)

Componentes normalizados por la escala del punto nominal (1.000 = costo en θ nominal). Se reporta el estado inicial (J0) y el mejor de cada cadena (J*).

| Cadena | semilla | J0 | J* | J̃_llegada (J0 → J*) | J̃_frente (J0 → J*) | J̃_persist. (J0 → J*) | Aceptadas | t (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1001 | 1.0000 | **0.8089** | 1.0000 → **1.0485** | 1.0000 → **0.3477** | 1.0000 → **1.0306** | 70 % | 2.4 |
| 2 | 2002 | 0.8862 | **0.8100** | 1.0631 → **0.9901** | 0.5521 → **0.4840** | 1.0434 → **0.9558** | 72 % | 2.7 |
| 3 | 3003 | 0.8820 | **0.8224** | 1.0579 → **1.0517** | 0.5521 → **0.3781** | 1.0362 → **1.0375** | 88 % | 1.8 |
| 4 | 4004 | 0.9176 | **0.8228** | 0.9733 → **0.9847** | 0.8313 → **0.5521** | 0.9482 → **0.9317** | 60 % | 3.6 |
| 5 | 5005 | 1.0605 | **0.8024** | 0.8697 → **0.9673** | 1.5150 → **0.5445** | 0.7967 → **0.8953** | 67 % | 2.6 |

**MAP (mejor global):** J = 0.8024; J̃_llegada = **0.9673**, J̃_frente = **0.5445**, J̃_persistencia = **0.8953**.

- Criterio de convergencia (los tres componentes por debajo de 1.00 simultáneamente en el MAP): **✔ CUMPLE**.
- Línea base previa: J̃_llegada 1.01–1.05, J̃_persistencia ≈1.00.

Conjunto de calibración: 239,502 filas; llegada 89 pos / 7,326 neg (231,929 ausencias lejanas excluidas); persistencia 17 pos / 141 neg. Esfuerzo real (conteos GBIF, sin imputación).

## 4. Parámetros estimados (M1) y correlaciones en Θ_comp

| Parámetro | Prior [min, max] | θ̂_MAP | Mediana Θ_comp | P5–P95 Θ_comp |
|---|---|---|---|---|
| `t_c` | [-5, 0] | -1.297 | -0.7638 | [-3.986, -0.03961] |
| `alpha_damage` | [0.01, 0.1] | 0.09337 | 0.09388 | [0.05939, 0.1] |
| `beta_damage` | [1, 3] | 1 | 1 | [1, 3] |
| `k0` | [1000, 15000] | 1.463e+04 | 9626 | [1064, 1.493e+04] |
| `gamma_hab` | [0, 2] | 0.2776 | 0.797 | [0.1267, 1.709] |
| `q_balloon` | [0.1, 0.6] | 0.1034 | 0.1141 | [0.1, 0.3895] |
| `q_viab` | [0.2, 0.9] | 0.7346 | 0.7191 | [0.6474, 0.8028] |
| `a0` | [-6, -1] | -1.095 | -1.772 | [-3.712, -1.016] |
| `b0` | [-2, 2.5] | 0.3453 | 0.7278 | [0.2014, 2.328] |

Miembros de Θ_comp (ΔJ ≤ 0.05): **439**.

**Matriz de correlación de Pearson en Θ_comp (M1):**

| |t_c|alpha_damage|beta_damage|k0|gamma_hab|q_balloon|q_viab|a0|b0|
|---|---|---|---|---|---|---|---|---|---|
| **t_c** |+1.00|-0.30|-0.77|-0.20|+0.48|-0.65|+0.08|-0.42|+0.17|
| **alpha_damage** |-0.30|+1.00|+0.15|-0.06|+0.02|+0.13|+0.01|-0.15|+0.39|
| **beta_damage** |-0.77|+0.15|+1.00|+0.17|-0.41|+0.46|+0.01|+0.39|+0.06|
| **k0** |-0.20|-0.06|+0.17|+1.00|-0.92|-0.34|+0.56|+0.92|-0.57|
| **gamma_hab** |+0.48|+0.02|-0.41|-0.92|+1.00|+0.12|-0.43|-0.96|+0.52|
| **q_balloon** |-0.65|+0.13|+0.46|-0.34|+0.12|+1.00|-0.23|-0.12|-0.01|
| **q_viab** |+0.08|+0.01|+0.01|+0.56|-0.43|-0.23|+1.00|+0.42|-0.34|
| **a0** |-0.42|-0.15|+0.39|+0.92|-0.96|-0.12|+0.42|+1.00|-0.55|
| **b0** |+0.17|+0.39|+0.06|-0.57|+0.52|-0.01|-0.34|-0.55|+1.00|

Mayores |ρ|: `gamma_hab`–`a0` = -0.96; `k0`–`a0` = +0.92; `k0`–`gamma_hab` = -0.92.

- Criterio |ρ| < 0.5 para todos los pares: **✘ NO CUMPLE** (máx |ρ| = 0.96). Línea base: ρ(γ1,γ2) = -0.82, ρ(K0,γ3) = -0.75.
- ρ(K0, γ_hab) = **-0.92**.
- Truncamiento de γ_hab en el borde (dentro del 2 % del rango): superior **0.0 %**, inferior **3.0 %** de Θ_comp; P95 = 1.709 (techo 2; el techo anterior de γ₂ era 1.2).
- K0 en el borde: superior 9.3 %, inferior 10.5 %.
- a0 en el borde: superior 9.8 %, inferior 0.0 %.
- b0 en el borde: superior 1.8 %, inferior 0.0 %.
- Pares con |ρ| ≥ 0.5: **9 de 36**.
- Parámetros con ≥ 10 % de Θ_comp pegado a un borde del prior: `alpha_damage` (superior, 16 %), `k0` (inferior, 10 %), `q_balloon` (inferior, 44 %). Es truncamiento por el prior y limita la interpretación de esos parámetros.

## 5. Comparación de costo M0 vs M1 vs M3 (mismo presupuesto de evaluaciones por modelo)

| Modelo | K_i | Parámetros libres | J_min | J̃_llegada | J̃_frente | J̃_persist. | Tiempo (s) | Θ_comp |
|---|---|---|---|---|---|---|---|---|
| **M1** | K0·exp(γ_hab·H_hab) | 9 | 0.8024 | 0.9673 | 0.5445 | 0.8953 | 13.2 | 439 |
| **M0** | K0 | 8 | 0.7958 | 0.9885 | 0.4565 | 0.9424 | 12.0 | 74 |
| **M3** | K0·exp(γ1 Z_imp + γ2 Z_borde + γ3 Z_dosel) | 11 | 0.8129 | 1.0019 | 0.4471 | 0.9898 | 16.1 | 38 |

- ΔJ(M1 − M0) = **+0.0066**; ΔJ(M3 − M1) = **+0.0106**. El costo es un Brier/RMSE normalizado, no una verosimilitud, así que no se calcula AIC; con 2,000 evaluaciones por modelo y una cadena estocástica, diferencias del orden de la variabilidad entre cadenas (ver Sección 3) no son concluyentes.
- Parsimonia: M1 solo añade γ_hab a M0; en M1 la mediana de γ_hab en Θ_comp es 0.797.
- Correlaciones en Θ_comp de M3 (sobreparametrizado): `t_c`–`q_balloon` = +0.84; `alpha_damage`–`beta_damage` = -0.83; `beta_damage`–`b0` = +0.77.

## 6. Aislamiento de `v_evaluation_set` (2020–2024)

- Años de ERA5 abiertos por el simulador: **[2014, 2015, 2016, 2017, 2018, 2019]** → ninguno > 2019.
- Solicitudes de lectura rechazadas por el guardia del lector: **0**.
- `v_calibration_set` exporta solo 2014–2019; el calibrador lee únicamente `calibration_set_2014_2019.csv` y el binario aborta si encuentra una fila fuera de 2014–2019.
- Búsqueda estática de lecturas SQL de `v_evaluation_set` en `scripts/`, `src/` y `tests/`: **1** (scripts/python/export_validation_datasets.py:31).
- El motor de PostgreSQL no registra lecturas de vistas sin `pg_stat_statements`; la afirmación «0 lecturas» se apoya en las tres comprobaciones anteriores (código, CSV exportado y guardia del lector), no en un contador del servidor.
- **Veredicto:** ✘ revisar.

## 7. Resumen de criterios de aceptación

| Criterio | Estado |
|---|---|
| ERA5 2014–2024 en disco, solo 2014–2019 leído | ✔ |
| Censo real y conteo exacto GBIF en `grid_5km`/`cell_annual_observations` | ✔ |
| Exactamente 9 parámetros libres en M1 | ✔ |
| `ctest` 4/4 | ver nota |
| J̃_llegada, J̃_frente y J̃_persistencia < 1.00 simultáneamente (MAP) | ✔ |
| |ρ| < 0.5 en Θ_comp (M1) | ✘ |
| `v_evaluation_set` con 0 lecturas verificables | ✘ |

_`ctest` (4/4) se verifica por separado; ver `docs/RUNBOOK.md`._

