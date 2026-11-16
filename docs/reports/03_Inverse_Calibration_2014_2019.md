# JORO-HPC — Calibración inversa 2014–2019: datos, identificabilidad y ensamble de parámetros compatibles (Θ_comp)

*Generado automáticamente el 2026-11-14 22:44 por `scripts/python/generate_calibration_report.py` a partir de los artefactos de la corrida. Los veredictos se calculan con reglas fijas.*

- Dispositivo LPDM: **Intel(R) Arc(TM) A380 Graphics | Intel(R) oneAPI Unified Runtime over Level-Zero | GPU**.
- Hilos OpenMP: **16**. Evaluaciones por modelo: **2000** (5 cadenas × 400).

## 1. Ingesta de datos

### 1.1 ERA5 interanual (2014–2024) en disco; solo 2014–2019 leído

| Año | uvwt 3D (May–Jul, 92 d) | superficial (May–Jul, 92 d) | t2m diaria (365 d) | Rol |
| --- | --- | --- | --- | --- |
| 2014 | ✔ 3 meses (1855.6 MB) | ✔ 3 meses (176.2 MB) | ✔ 12.4 MB | calibración (leído) |
| 2015 | ✔ 3 meses (1843.9 MB) | ✔ 3 meses (175.9 MB) | ✔ 12.2 MB | calibración (leído) |
| 2016 | ✔ 3 meses (1857.4 MB) | ✔ 3 meses (176.7 MB) | ✔ 12.2 MB | calibración (leído) |
| 2017 | ✔ 3 meses (1855.0 MB) | ✔ 3 meses (176.0 MB) | ✔ 12.2 MB | calibración (leído) |
| 2018 | ✔ 3 meses (1868.3 MB) | ✔ 3 meses (177.0 MB) | ✔ 12.3 MB | calibración (leído) |
| 2019 | ✔ 3 meses (1862.6 MB) | ✔ 3 meses (176.9 MB) | ✔ 12.3 MB | calibración (leído) |
| 2020 | ✔ 3 meses (1861.0 MB) | ✔ 3 meses (176.3 MB) | ✔ 12.3 MB | reservado para validación (NO leído) |
| 2021 | ✔ 3 meses (1850.8 MB) | ✔ 3 meses (175.6 MB) | ✔ 12.2 MB | reservado para validación (NO leído) |
| 2022 | ✔ 3 meses (1858.4 MB) | ✔ 3 meses (176.3 MB) | ✔ 12.4 MB | reservado para validación (NO leído) |
| 2023 | ✔ 3 meses (1865.5 MB) | ✔ 3 meses (177.0 MB) | ✔ 12.3 MB | reservado para validación (NO leído) |
| 2024 | ✔ 3 meses (1868.3 MB) | ✔ 3 meses (176.7 MB) | ✔ 12.2 MB | reservado para validación (NO leído) |

- Archivos ERA5 efectivamente abiertos por el simulador: **45**; años distintos leídos: **[2014, 2015, 2016, 2017, 2018, 2019]**.
- Solicitudes bloqueadas por la regla de aislamiento (año > 2019): **[]**.
- Campos ERA5 3D+superficiales residentes en la GPU (uno por año): **6** (cada año simulado usa SU meteorología estacional completa mayo–julio, 92 días = 2,208 h/año en FP16).
- Modo estricto del lector (sin datos sintéticos de respaldo): **True**; directorio: `/mnt/hpc_scratch/joro-hpc/era5_nc_active`.
- **Veredicto:** ✔ CUMPLE.

> **Alcance meteorológico v2.0.0:** los tensores 3D/superficiales cubren la temporada completa de dispersión de **92 días (mayo, junio y julio = 2,208 h/año)** con cuantización `FP16` (`Wind4Half`, 8 bytes/vóxel) y paso sinóptico de 3 h residente en VRAM (`3.64 GiB` para los 6 años en la Intel Arc A380), y la serie `t2m` diaria cubre los 365 días.

### 1.2 Población censal real (`pop_density_h`)

- Fuente: Censo 2020 (PL 94-171, encabezado geográfico por bloque: POP100 + punto interno), asignado a la celda de 5 km (EPSG:5070) que contiene el punto interno; `pop_density_h = estandarizar(log1p(hab/km²))`.
- `grid_5km`: **256,020** celdas, media **-0.0000**, sd **1.0000**, **11,803** valores distintos.
- Población total asignada: **215,847,140** hab. en **114,070** celdas pobladas.
- Correlación con el proxy NLCD anterior (respaldado en `grid_5km_pop_proxy_backup`): **0.920**.
- Limitación: censo 2020 aplicado a toda la ventana 2014–2019 (sin serie anual); asignación por punto interno.

### 1.3 Fusión tripartita (GBIF + iNaturalist + EDDMapS) y esfuerzo exógeno GBIF *Araneae* (`taxonKey = 1496`)

| Año | Celdas | Σ effort_araneae | máx | celdas con effort>0 | valores distintos | presencias *T. clavata* |
| --- | --- | --- | --- | --- | --- | --- |
| 2014 | 39,917 | 493 | 25 | 280 | 13 | 1 |
| 2015 | 39,917 | 928 | 75 | 418 | 19 | 1 |
| 2016 | 39,917 | 1,501 | 32 | 737 | 22 | 2 |
| 2017 | 39,917 | 2,666 | 39 | 1,219 | 25 | 5 |
| 2018 | 39,917 | 4,904 | 52 | 1,917 | 35 | 26 |
| 2019 | 39,917 | 7,379 | 118 | 3,001 | 41 | 73 |
| 2020–2024 | — | reservado para la validación fuera de muestra: este informe no consulta estos años | — | — | — | — |

- Método: fusión deduplicada de GBIF + iNaturalist + EDDMapS (`1,359` registros únicos nuevos insertados en `occurrences_raw`); esfuerzo exógeno $E_{i,t}^{-J} = \max(0, C_{i,t}^{\text{Araneae}} - C_{i,t}^J)$ libre de endogeneidad; polígono semilla fundador de **125 celdas contiguas** en Braselton/Hoschton (centroide `34.2508°N, -83.4962°W`, radio `37.6 km`), excluyendo de $t_0$ la celda aislada de Anderson, SC (`cell_id=118702`) para exigir su colonización dinámica durante la simulación. Respaldo previo: `cell_annual_observations_effort_density_backup`.
- `v_calibration_set`: años **2014–2019**, 239,502 filas, 108 presencias.

## 2. Telemetría de tiempos y utilización GPU/CPU (M1, modelo principal)

| Métrica | Calibración actual (M1) | Línea base (commit `3fa0d37`) |
| --- | --- | --- |
| Tiempo total (2,000 evaluaciones) | **119.9 s** | 25.9 s |
| H2D + D2H | **2.60 s** (2.2 % de pared) | 2.08 s |
| Kernel SYCL en GPU | 104.05 s (86.8 % de pared) | — |
| ABM (OpenMP + AVX-512) | 8.89 s | — |
| Ocupación + costo (OpenMP) | 1.72 s | — |
| LPDM host (init + reducción) | 1.54 s | — |
| Super-partículas despachadas | 443.3 M | — |
| Despachos con fallback a CPU | **0** | 0 |

Muestreo externo durante la corrida (`sample_telemetry.py`; `intel_gpu_top` requiere CAP_PERFMON, no disponible, por lo que la ocupación GPU se **estima** como 100 % − residencia RC6, que mide actividad del GT, no del motor Compute):

| Recurso | media | P95 | máx |
| --- | --- | --- | --- |
| CPU, promedio de 16 hilos (%) | 66.7 | 100.0 | 100.0 |
| GPU activa (100 − RC6, %) | 91.9 | 100.0 | 100.0 |
| Frecuencia GPU (MHz) | 1773 | 2450 | 2450 |
| Potencia GPU (W) | 30.5 | 39.3 | 39.9 |

Hilos con carga media > 80 %: **1 de 16**.

## 3. Convergencia de las 5 cadenas (M1)

Componentes normalizados por la escala del punto nominal (1.000 = costo en θ nominal). Se reporta el estado inicial (J0) y el mejor de cada cadena (J*).

| Cadena | semilla | J0 | J* | J̃_llegada (J0 → J*) | J̃_frente (J0 → J*) | J̃_persist. (J0 → J*) | Aceptadas | t (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1001 | 1.0000 | **0.8702** | 1.0000 → **1.1555** | 1.0000 → **0.2855** | 1.0000 → **1.1695** | 59 % | 20.0 |
| 2 | 2002 | 1.0044 | **0.8345** | 1.2349 → **1.1051** | 0.5432 → **0.2862** | 1.2352 → **1.1122** | 58 % | 17.3 |
| 3 | 3003 | 1.0044 | **0.8443** | 1.2349 → **1.0757** | 0.5432 → **0.3769** | 1.2351 → **1.0803** | 58 % | 23.4 |
| 4 | 4004 | 0.9758 | **0.8711** | 0.9846 → **1.0291** | 0.9451 → **0.5681** | 0.9977 → **1.0161** | 68 % | 26.9 |
| 5 | 5005 | 1.1105 | **0.8434** | 0.9775 → **1.0974** | 1.3540 → **0.3392** | 1.0000 → **1.0936** | 50 % | 31.7 |

**MAP (mejor global):** J = 0.8345; J̃_llegada = **1.1051**, J̃_frente = **0.2862**, J̃_persistencia = **1.1122**.

- Criterio de convergencia (los tres componentes por debajo de 1.00 simultáneamente en el MAP): **✘ NO CUMPLE**.
- Línea base previa: J̃_llegada 1.01–1.05, J̃_persistencia ≈1.00.

Conjunto de calibración: 239,502 filas; llegada 90 pos / 16,720 neg (222,532 ausencias lejanas excluidas); persistencia 18 pos / 142 neg. Esfuerzo real (conteos GBIF, sin imputación).

## 4. Parámetros estimados (M1) y correlaciones en Θ_comp

| Parámetro | Prior [min, max] | θ̂_MAP | Mediana Θ_comp | P5–P95 Θ_comp |
| --- | --- | --- | --- | --- |
| `t_c` | [-5, 0] | -0.06487 | -0.7686 | [-4.867, -0.06417] |
| `alpha_damage` | [0.01, 0.1] | 0.0849 | 0.02323 | [0.01899, 0.08602] |
| `beta_damage` | [1, 3] | 1 | 2 | [1, 3] |
| `k0` | [1000, 15000] | 1.351e+04 | 1.136e+04 | [3625, 1.417e+04] |
| `gamma_hab` | [0, 2] | 1.601 | 1.652 | [0.08824, 1.954] |
| `q_balloon` | [0.1, 0.6] | 0.3502 | 0.3942 | [0.3294, 0.5064] |
| `q_viab` | [0.2, 0.9] | 0.3897 | 0.606 | [0.3438, 0.7316] |
| `a0` | [-6, -1] | -3.765 | -3.191 | [-4.493, -2.88] |
| `b0` | [-2, 2.5] | 2.045 | 0.8681 | [-1.111, 1.885] |

Miembros de Θ_comp (ΔJ ≤ 0.05): **339**.

### Matriz de correlación de Pearson en Θ_comp (M1)

| Parámetro | t_c | alpha_damage | beta_damage | k0 | gamma_hab | q_balloon | q_viab | a0 | b0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **t_c** | +1.00 | -0.02 | -0.88 | -0.30 | -0.58 | +0.62 | -0.58 | -0.44 | +0.86 |
| **alpha_damage** | -0.02 | +1.00 | -0.43 | +0.20 | -0.11 | -0.04 | +0.08 | -0.48 | +0.02 |
| **beta_damage** | -0.88 | -0.43 | +1.00 | +0.28 | +0.51 | -0.51 | +0.56 | +0.57 | -0.78 |
| **k0** | -0.30 | +0.20 | +0.28 | +1.00 | -0.38 | -0.03 | +0.85 | -0.34 | -0.27 |
| **gamma_hab** | -0.58 | -0.11 | +0.51 | -0.38 | +1.00 | -0.59 | -0.21 | +0.72 | -0.57 |
| **q_balloon** | +0.62 | -0.04 | -0.51 | -0.03 | -0.59 | +1.00 | -0.16 | -0.40 | +0.58 |
| **q_viab** | -0.58 | +0.08 | +0.56 | +0.85 | -0.21 | -0.16 | +1.00 | -0.13 | -0.49 |
| **a0** | -0.44 | -0.48 | +0.57 | -0.34 | +0.72 | -0.40 | -0.13 | +1.00 | -0.51 |
| **b0** | +0.86 | +0.02 | -0.78 | -0.27 | -0.57 | +0.58 | -0.49 | -0.51 | +1.00 |

Mayores |ρ|: `t_c`–`beta_damage` = -0.88; `t_c`–`b0` = +0.86; `k0`–`q_viab` = +0.85.

- Criterio |ρ| < 0.5 para todos los pares: **✘ NO CUMPLE** (máx |ρ| = 0.88). Línea base: ρ(γ1,γ2) = -0.82, ρ(K0,γ3) = -0.75.
- ρ(K0, γ_hab) = **-0.38**.
- Truncamiento de γ_hab en el borde (dentro del 2 % del rango): superior **3.8 %**, inferior **0.9 %** de Θ_comp; P95 = 1.954 (techo 2; el techo anterior de γ₂ era 1.2).
- K0 en el borde: superior 1.5 %, inferior 0.3 %.
- a0 en el borde: superior 0.0 %, inferior 0.0 %.
- b0 en el borde: superior 2.1 %, inferior 1.8 %.
- Pares con |ρ| ≥ 0.5: **16 de 36**.
- Ningún parámetro tiene ≥ 10 % de Θ_comp pegado a un borde del prior.

## 5. Comparación de costo M0 vs M1 vs M3 (mismo presupuesto de evaluaciones por modelo)

| Modelo | K_i | Parámetros libres | J_min | J̃_llegada | J̃_frente | J̃_persist. | Tiempo (s) | Θ_comp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **M1** | K0·exp(γ_hab·H_hab) | 9 | 0.8345 | 1.1051 | 0.2862 | 1.1122 | 119.9 | 339 |
| **M0** | K0 | 8 | 0.8251 | 1.0651 | 0.3482 | 1.0621 | 83.8 | 332 |
| **M3** | K0·exp(γ1 Z_imp + γ2 Z_borde + γ3 Z_dosel) | 11 | 0.8183 | 1.0698 | 0.3440 | 1.0411 | 127.6 | 363 |

- ΔJ(M1 − M0) = **+0.0094**; ΔJ(M3 − M1) = **-0.0162**. El costo es un Brier/RMSE normalizado, no una verosimilitud, así que no se calcula AIC; con 2,000 evaluaciones por modelo y una cadena estocástica, diferencias del orden de la variabilidad entre cadenas (ver Sección 3) no son concluyentes.
- Parsimonia: M1 solo añade γ_hab a M0; en M1 la mediana de γ_hab en Θ_comp es 1.652.
- Correlaciones en Θ_comp de M3 (sobreparametrizado): `t_c`–`alpha_damage` = -0.93; `k0`–`gamma2_borde` = +0.91; `q_viab`–`gamma3_dosel` = -0.87.

## 6. Aislamiento de `v_evaluation_set` (2020–2024)

- Años de ERA5 abiertos por el simulador: **[2014, 2015, 2016, 2017, 2018, 2019]** → ninguno > 2019.
- Solicitudes de lectura rechazadas por el guardia del lector: **0**.
- `v_calibration_set` exporta solo 2014–2019; el calibrador lee únicamente `calibration_set_2014_2019.csv` y el binario aborta si encuentra una fila fuera de 2014–2019.
- Búsqueda estática de lecturas SQL de `v_evaluation_set` en `scripts/`, `src/` y `tests/`: **0** .
- El motor de PostgreSQL no registra lecturas de vistas sin `pg_stat_statements`; la afirmación «0 lecturas» se apoya en las tres comprobaciones anteriores (código, CSV exportado y guardia del lector), no en un contador del servidor.
- **Veredicto:** ✔ 0 lecturas verificables; lista para desbloquear la validación fuera de muestra.

## 7. Resumen de criterios de aceptación

| Criterio | Estado |
| --- | --- |
| ERA5 2014–2024 en disco, solo 2014–2019 leído | ✔ |
| Censo real y conteo exacto GBIF en `grid_5km`/`cell_annual_observations` | ✔ |
| Exactamente 9 parámetros libres en M1 | ✔ |
| `ctest` 4/4 | ver nota |
| J̃_llegada, J̃_frente y J̃_persistencia < 1.00 simultáneamente (MAP) | ✘ |
| $\lvert\rho\rvert < 0.5$ en Θ_comp (M1) | ✘ |
| `v_evaluation_set` con 0 lecturas verificables | ✔ |

*`ctest` (4/4) se verifica por separado; ver `docs/RUNBOOK.md`.*
