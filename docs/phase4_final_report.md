# JORO-HPC — Informe final Fase 4.5: cierre de datos interanuales y reparametrización de identificabilidad

_Generado automáticamente el 2026-10-19 02:34 por `scripts/python/build_phase4_report.py` a partir de los artefactos de la corrida. Los veredictos se calculan con reglas fijas._

- Dispositivo LPDM: **Intel(R) Arc(TM) A380 Graphics | Intel(R) oneAPI Unified Runtime over Level-Zero | GPU** (pruebas); la corrida de producción final será en la Arc A770.
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
| 2020 | ✔ 158.5 MB | ✔ 13.2 MB | ✘ falta | reservado Fase 5 (NO leído) |
| 2021 | ✘ falta | ✘ falta | ✘ falta | reservado Fase 5 (NO leído) |
| 2022 | ✘ falta | ✘ falta | ✘ falta | reservado Fase 5 (NO leído) |
| 2023 | ✘ falta | ✘ falta | ✘ falta | reservado Fase 5 (NO leído) |
| 2024 | ✘ falta | ✘ falta | ✘ falta | reservado Fase 5 (NO leído) |

- Archivos ERA5 efectivamente abiertos por el simulador: **21**; años distintos leídos: **[2014, 2015, 2016, 2017, 2018, 2019]**.
- Solicitudes bloqueadas por la regla de aislamiento (año > 2019): **[]**.
- Campos ERA5 3D+superficiales residentes en la GPU (uno por año): **6** (cada año simulado usa SU meteorología; ya no se recicla el 1–7 de mayo de 2014).
- Modo estricto del lector (sin datos sintéticos de respaldo): **True**; directorio: `/mnt/hpc_scratch/joro-hpc/era5_nc_active`.
- **Veredicto:** ✘ NO CUMPLE.

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
| 2020–2024 | — | reservado para la Fase 5: el informe no consulta estos años | | | | |

- Método: conteo de registros georreferenciados (`hasCoordinate`, sin problemas geoespaciales) asignados a la celda por punto en la celda EPSG:5070; `effort = GREATEST(conteo, joro_count)`. Respaldo previo: `cell_annual_observations_effort_density_backup`.
- `v_calibration_set`: años **2014–2019**, 239,502 filas, 106 presencias.

## 2. Telemetría de tiempos y utilización GPU/CPU (M1, modelo principal)

| Métrica | Fase 4.5 (M1) | Línea base (commit `3fa0d37`) |
|---|---|---|
| Tiempo total (2,000 evaluaciones) | **12.4 s** | 25.9 s |
| H2D + D2H | **0.94 s** (7.6 % de pared) | 2.08 s |
| Kernel SYCL en GPU | 7.97 s (64.1 % de pared) | — |
| ABM (OpenMP + AVX-512) | 1.45 s | — |
| Ocupación + costo (OpenMP) | 0.87 s | — |
| LPDM host (init + reducción) | 0.50 s | — |
| Super-partículas despachadas | 237.5 M | — |
| Despachos con fallback a CPU | **0** | 0 |

Muestreo externo durante la corrida (`sample_telemetry.py`; `intel_gpu_top` requiere CAP_PERFMON, no disponible, por lo que la ocupación GPU se **estima** como 100 % − residencia RC6, que mide actividad del GT, no del motor Compute):

| Recurso | media | P95 | máx |
|---|---|---|---|
| CPU, promedio de 16 hilos (%) | 63.7 | 100.0 | 100.0 |
| GPU activa (100 − RC6, %) | 91.0 | 100.0 | 100.0 |
| Frecuencia GPU (MHz) | 1838 | 2450 | 2450 |
| Potencia GPU (W) | 30.8 | 41.0 | 41.7 |

Hilos con carga media > 80 %: **0 de 16**.

## 3. Convergencia de las 5 cadenas (M1)

Componentes normalizados por la escala del punto nominal (1.000 = costo en θ nominal). Se reporta el estado inicial (J0) y el mejor de cada cadena (J*).

| Cadena | semilla | J0 | J* | J̃_llegada (J0 → J*) | J̃_frente (J0 → J*) | J̃_persist. (J0 → J*) | Aceptadas | t (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1001 | 1.0000 | **0.8683** | 1.0000 → **0.9984** | 1.0000 → **0.6189** | 1.0000 → **0.9875** | 75 % | 2.9 |
| 2 | 2002 | 0.9260 | **0.8403** | 1.0655 → **1.0193** | 0.6602 → **0.4886** | 1.0524 → **1.0131** | 88 % | 2.8 |
| 3 | 3003 | 0.9119 | **0.7954** | 1.0474 → **0.9308** | 0.6602 → **0.5719** | 1.0281 → **0.8836** | 79 % | 2.3 |
| 4 | 4004 | 0.9179 | **0.8073** | 0.9308 → **0.9139** | 0.9400 → **0.6602** | 0.8829 → **0.8480** | 87 % | 2.3 |
| 5 | 5005 | 1.0305 | **0.8067** | 0.8802 → **1.0201** | 1.4060 → **0.3978** | 0.8052 → **1.0021** | 61 % | 2.1 |

**MAP (mejor global):** J = 0.7954; J̃_llegada = **0.9308**, J̃_frente = **0.5719**, J̃_persistencia = **0.8836**.

- Criterio de la tarea 3 (los tres componentes por debajo de 1.00 simultáneamente en el MAP): **✔ CUMPLE**.
- Línea base previa: J̃_llegada 1.01–1.05, J̃_persistencia ≈1.00.

Conjunto de calibración: 239,502 filas; llegada 89 pos / 7,326 neg (231,929 ausencias lejanas excluidas); persistencia 17 pos / 141 neg. Esfuerzo real (conteos GBIF, sin imputación).

## 4. Parámetros estimados (M1) y correlaciones en Θ_comp

| Parámetro | Prior [min, max] | θ̂_MAP | Mediana Θ_comp | P5–P95 Θ_comp |
|---|---|---|---|---|
| `t_c` | [-5, 0] | -2.175 | -2.249 | [-3.031, -1.132] |
| `alpha_damage` | [0.01, 0.1] | 0.02896 | 0.07645 | [0.02896, 0.09546] |
| `beta_damage` | [1, 3] | 2 | 2 | [1, 3] |
| `k0` | [1000, 15000] | 9204 | 5424 | [1480, 1.438e+04] |
| `gamma_hab` | [0, 2] | 0.7963 | 0.6789 | [0.05347, 1.206] |
| `q_balloon` | [0.1, 0.6] | 0.4722 | 0.3019 | [0.1663, 0.4473] |
| `q_viab` | [0.2, 0.9] | 0.3518 | 0.7914 | [0.6503, 0.8653] |
| `a0` | [-6, -1] | -1.329 | -1.424 | [-1.866, -1] |
| `b0` | [-2, 2.5] | 1.733 | 2.278 | [-0.3852, 2.5] |

Miembros de Θ_comp (ΔJ ≤ 0.05): **277**.

**Matriz de correlación de Pearson en Θ_comp (M1):**

| |t_c|alpha_damage|beta_damage|k0|gamma_hab|q_balloon|q_viab|a0|b0|
|---|---|---|---|---|---|---|---|---|---|
| **t_c** |+1.00|+0.43|-0.57|+0.68|-0.61|-0.34|+0.41|+0.43|-0.63|
| **alpha_damage** |+0.43|+1.00|-0.67|+0.69|-0.24|-0.78|+0.22|+0.20|-0.60|
| **beta_damage** |-0.57|-0.67|+1.00|-0.77|+0.53|+0.68|-0.37|-0.55|+0.81|
| **k0** |+0.68|+0.69|-0.77|+1.00|-0.69|-0.82|+0.33|+0.67|-0.93|
| **gamma_hab** |-0.61|-0.24|+0.53|-0.69|+1.00|+0.54|-0.59|-0.77|+0.72|
| **q_balloon** |-0.34|-0.78|+0.68|-0.82|+0.54|+1.00|-0.27|-0.54|+0.78|
| **q_viab** |+0.41|+0.22|-0.37|+0.33|-0.59|-0.27|+1.00|+0.54|-0.39|
| **a0** |+0.43|+0.20|-0.55|+0.67|-0.77|-0.54|+0.54|+1.00|-0.73|
| **b0** |-0.63|-0.60|+0.81|-0.93|+0.72|+0.78|-0.39|-0.73|+1.00|

Mayores |ρ|: `k0`–`b0` = -0.93; `k0`–`q_balloon` = -0.82; `beta_damage`–`b0` = +0.81.

- Criterio |ρ| < 0.5 para todos los pares: **✘ NO CUMPLE** (máx |ρ| = 0.93). Línea base: ρ(γ1,γ2) = -0.82, ρ(K0,γ3) = -0.75.
- ρ(K0, γ_hab) = **-0.69**.
- Truncamiento de γ_hab en el borde (dentro del 2 % del rango): superior **0.0 %**, inferior **1.1 %** de Θ_comp; P95 = 1.206 (techo 2; el techo anterior de γ₂ era 1.2).
- K0 en el borde: superior 2.9 %, inferior 3.6 %.
- a0 en el borde: superior 36.1 %, inferior 0.0 %.
- b0 en el borde: superior 23.5 %, inferior 0.0 %.
- Pares con |ρ| ≥ 0.5: **25 de 36**.
- Parámetros con ≥ 10 % de Θ_comp pegado a un borde del prior: `a0` (superior, 36 %), `b0` (superior, 23 %). Es truncamiento por el prior y limita la interpretación de esos parámetros.

## 5. Comparación de costo M0 vs M1 vs M3 (mismo presupuesto de evaluaciones por modelo)

| Modelo | K_i | Parámetros libres | J_min | J̃_llegada | J̃_frente | J̃_persist. | Tiempo (s) | Θ_comp |
|---|---|---|---|---|---|---|---|---|
| **M1** | K0·exp(γ_hab·H_hab) | 9 | 0.7954 | 0.9308 | 0.5719 | 0.8836 | 12.4 | 277 |
| **M0** | K0 | 8 | 0.7388 | 0.9138 | 0.4532 | 0.8494 | 8.0 | 134 |
| **M3** | K0·exp(γ1 Z_imp + γ2 Z_borde + γ3 Z_dosel) | 11 | 0.7869 | 0.9119 | 0.6058 | 0.8431 | 14.6 | 191 |

- ΔJ(M1 − M0) = **+0.0566**; ΔJ(M3 − M1) = **-0.0085**. El costo es un Brier/RMSE normalizado, no una verosimilitud, así que no se calcula AIC; con 2,000 evaluaciones por modelo y una cadena estocástica, diferencias del orden de la variabilidad entre cadenas (ver Sección 3) no son concluyentes.
- Parsimonia: M1 solo añade γ_hab a M0; en M1 la mediana de γ_hab en Θ_comp es 0.679.
- Correlaciones en Θ_comp de M3 (sobreparametrizado): `alpha_damage`–`q_balloon` = -0.97; `a0`–`gamma2_borde` = -0.90; `alpha_damage`–`gamma2_borde` = -0.89.

## 6. Aislamiento de `v_evaluation_set` (2020–2024)

- Años de ERA5 abiertos por el simulador: **[2014, 2015, 2016, 2017, 2018, 2019]** → ninguno > 2019.
- Solicitudes de lectura rechazadas por el guardia del lector: **0**.
- `v_calibration_set` exporta solo 2014–2019; el calibrador lee únicamente `calibration_set_2014_2019.csv` y el binario aborta si encuentra una fila fuera de 2014–2019.
- Búsqueda estática de lecturas SQL de `v_evaluation_set` en `scripts/`, `src/` y `tests/`: **0** .
- El motor de PostgreSQL no registra lecturas de vistas sin `pg_stat_statements`; la afirmación «0 lecturas» se apoya en las tres comprobaciones anteriores (código, CSV exportado y guardia del lector), no en un contador del servidor.
- **Veredicto:** ✔ 0 lecturas verificables; lista para desbloquear la Fase 5.

## 7. Resumen de criterios de la Fase 4.5

| Criterio | Estado |
|---|---|
| ERA5 2014–2024 en disco, solo 2014–2019 leído | ✘ |
| Censo real y conteo exacto GBIF en `grid_5km`/`cell_annual_observations` | ✔ |
| Exactamente 9 parámetros libres en M1 | ✔ |
| `ctest` 4/4 | ver nota |
| J̃_llegada, J̃_frente y J̃_persistencia < 1.00 simultáneamente (MAP) | ✔ |
| |ρ| < 0.5 en Θ_comp (M1) | ✘ |
| `v_evaluation_set` con 0 lecturas verificables | ✔ |

_`ctest` se verifica por separado (ver mensaje de commit)._

