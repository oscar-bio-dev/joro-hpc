# INFORME — FASE 4 (REPETIDA): CALIBRACIÓN INVERSA MULTI-INICIO 2014–2019 CON PARALELISMO HÍBRIDO CPU+GPU

**Proyecto:** JORO-HPC (*Trichonephila clavata*) · **Plataforma de pruebas:** i9-11900K (8 núcleos × 2 SMT = 16 hilos, AVX-512) + Intel Arc A380 (SYCL / Level-Zero). La corrida de producción usará una Arc A770.
**Ventana de calibración:** 2014–2019 · **Ventana bloqueada:** 2020–2024 (`v_evaluation_set`)
**Compilación:** `icpx -O3 -fsycl -qopenmp -march=rocketlake` (Release) · **ctest:** 4/4 ✔
**Lanzamiento:** `OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores` vía `scripts/telemetry/run_phase4_telemetry.sh`
**Artefactos crudos:** `/mnt/hpc_scratch/joro-hpc/telemetry/phase4_final/` (copia del log y del resumen en `docs/telemetry/`)

> [!WARNING]
> **Leer antes de interpretar las tablas.** La repetición corrige la paralelización y las tres fallas matemáticas, pero la auditoría de datos reveló que `grid_5km` y el esfuerzo TGB siguen siendo marcadores de posición. Varios objetivos numéricos pedidos (J̃_llegada < 0.65; K₀ "interior" con significado científico) **no se cumplen o no son atribuibles** (ver §6). No se ajustó nada para forzarlos.

---

## 1. Qué se corrigió

| # | Problema de la corrida anterior | Corrección |
|---|---|---|
| 1 | Las 5 cadenas corrían en paralelo sobre CPU; la GPU nunca se usaba (el SA instanciaba `create_cpu_baseline_engine()`) | Cadenas **secuenciales**; cada evaluación usa los 16 hilos y el motor SYCL persistente (Device USM `d_particles_`). El SA aborta si el motor no es GPU (`JORO_ALLOW_CPU_FALLBACK=1` para anularlo). |
| 2 | β_s continuo (2.166 / 2.643) | β_s ∈ {1,2,3}: valores iniciales aleatorios discretos y saltos discretos (p = 0.30) a otro valor. Potencias enteras por multiplicación directa (`ipow123`, [fast_math.hpp](file:///home/ecotech/Projects/joro-hpc/src/common/fast_math.hpp)). |
| 3 | J̃_llegada congelado en 0.9974 | Brier **balanceado**: 0.5·Brier(positivos) + 0.5·Brier(negativos en la envolvente activa). Envolvente del año *t* = 50 km + distancia máxima acumulada de presencias de años < *t*. Se excluyen 231,929 ausencias lejanas. |
| 4 | Otros defectos hallados | Se usaba la grilla térmica de 2014 para los 6 años (ahora una por año). D̄ estaba fijo en 0.05 (ahora real, desde las cohortes del ABM). Θ_comp usaba solo estados aceptados (ahora todas las evaluaciones). Etiqueta "ROBUSTO" fija (ahora calculada). |
| 5 | No determinismo del ABM (J₀ nominal = 1.0039) | Depósito de llegadas y creación de cohortes en orden de `cell_id`. J₀ = 1.0000, reproducible. |

---

## 2. Telemetría híbrida CPU+GPU (corrida completa: 5 × 400 = 2,000 evaluaciones + 6 de referencia)

Fuente: contadores DRM fdinfo del proceso + `/proc/stat`, muestreo 0.25 s, 222 muestras en los 56.2 s de calibración. `intel_gpu_top` **no se pudo usar** (requiere CAP_PERFMON, `perf_event_paranoid=2`, sin sudo); fdinfo mide el tiempo ocupado por motor del propio proceso, que es equivalente para este fin.

### 2.1 Tiempos por cadena (Tabla 7 del programa)

| Cadena | Total (s) | ABM CPU (s) | Kernel GPU (s) | H2D+D2H (s) | LPDM host (s) | Costo CPU (s) | Kernel % wall | Super-part. (M) | Mpart/s kernel |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.71 | 0.33 | 0.35 | 1.43 | 0.82 | 0.20 | 9.5% | 85.7 | 244.0 |
| 2 | 9.99 | 0.37 | 0.91 | 4.42 | 2.34 | 0.17 | 9.2% | 309.1 | 337.8 |
| 3 | 4.50 | 0.29 | 0.46 | 1.86 | 0.97 | 0.15 | 10.2% | 119.6 | 261.6 |
| 4 | 5.89 | 0.37 | 0.59 | 2.55 | 1.23 | 0.16 | 10.0% | 161.4 | 273.3 |
| 5 | 31.99 | 0.45 | 2.62 | 15.15 | 7.71 | 0.17 | 8.2% | 1114.6 | 425.6 |
| **Total** | **56.17** | 1.80 | 4.94 | 25.45 | 13.10 | 0.85 | **8.8%** | 1792.2 | **362.7** |

- **Cuello de botella de 281 s eliminado:** la cadena más lenta ahora tarda 31.99 s (8.8× menos). Calibración completa: 56.2 s.
- **0 fallbacks a CPU** en 2,006 despachos GPU.
- **Balance entre cadenas: NO es similar** (min 3.71 s, máx 31.99 s, máx/mín = 8.62). El requisito de "tiempos similares" no se cumple. La causa no es de paralelización: el trabajo por evaluación es proporcional al número de súper-partículas, que depende de θ (cadena 5 explora K₀ y q_balloon altos → 1,115 M súper-partículas frente a 86 M en la cadena 1). Es desbalance de carga dependiente de los datos y del parámetro, no de hilos.

### 2.2 CPU (16 hilos)

| Métrica | Valor |
|---|---|
| Utilización media global | **100.0 %** (mín 99.9 %, máx 100.0 %) |
| Dispersión entre hilos (máx–mín) | 0.1 pp |
| Muestras con los 16 hilos > 50 % | 222 / 222 |

Los 16 hilos se mantienen simétricos y saturados (antes: 1–5 hilos al 100 % y el resto al 0 %). Nota: 16 hilos = 8 núcleos × SMT2, por lo que `OMP_PLACES=cores` coloca 2 hilos por núcleo físico.

### 2.3 GPU Intel Arc A380 (motores del proceso)

| Motor | Media | P95 | Máx | Muestras activas (> 1 %) |
|---|---|---|---|---|
| compute | **8.0 %** | 9.3 % | 11.9 % | 222 / 222 |
| copy (DMA H2D/D2H) | **38.1 %** | 46.9 % | 51.6 % | 222 / 222 |
| render / video / video-enhance | 0.0 % | 0.0 % | 0.0 % | 0 / 222 |

Frecuencia GPU: media 2,214 MHz, máx 2,450 MHz.

> [!IMPORTANT]
> **La GPU está activa en todo momento, pero NO saturada al 100 %.** El motor de cómputo solo está ocupado ~8–9 % del tiempo. El kernel SYCL (60 pasos × 60 s) es muy rápido (21,762 M partícula-pasos/s); el tiempo de pared lo dominan la transferencia PCIe (45.3 %), la preparación/reducción en el host (23.3 %) y trabajo no instrumentado (~18 %, propuesta del SA, ψ recursivo, trazas). No se alcanzará carga sostenida alta en cómputo sin rediseñar el flujo (ver §7).

### 2.4 Reparto del tiempo de pared (56.17 s)

| Fase | Segundos | % |
|---|---|---|
| Kernel GPU SYCL | 4.94 | 8.8 % |
| Transferencias H2D+D2H | 25.45 | 45.3 % |
| LPDM en host (init + reducción) | 13.10 | 23.3 % |
| ABM CPU (OpenMP, 16 hilos) | 1.80 | 3.2 % |
| Reducción de costo CPU | 0.85 | 1.5 % |
| No instrumentado | 10.03 | 17.9 % |

Throughput medio: **362.7 M súper-partículas/s** (solo kernel) y **31.9 M súper-partículas/s** de extremo a extremo (1,792.2 M / 56.17 s).

---

## 3. Tabla 4 — Convergencia de las 5 cadenas

Componentes normalizados por la escala nominal (1.0000 = valor en θ_nominal). Escalas de referencia: S_llegada = 0.36067, S_frente = 54.002 km, S_persistencia = 0.61031.

| Cadena | Semilla | J₀ | J* | J̃_llegada | J̃_frente | J̃_persistencia | Aceptadas | Tiempo |
|---|---|---|---|---|---|---|---|---|
| 1 | 1001 | 1.0000 | 0.6968 | 1.0568 | 0.2999 | 0.7336 | 62 % | 3.71 s |
| 2 | 2002 | 0.8235 | 0.6801 | 1.0718 | 0.2888 | 0.6798 | 53 % | 9.99 s |
| 3 | 3003 | 0.7068 | 0.6850 | 1.0696 | 0.2963 | 0.6891 | 60 % | 4.50 s |
| 4 | 4004 | 0.9834 | **0.6740** | 1.0402 | 0.2876 | 0.6941 | 60 % | 5.89 s |
| 5 | 5005 | 1.0472 | 0.6855 | 1.0806 | 0.3220 | 0.6538 | 56 % | 31.99 s |

Las 5 cadenas convergen a J* ∈ [0.674, 0.697] (dispersión 0.023). **Solo J̃_frente baja de forma real (≈0.29).** J̃_persistencia baja a 0.65–0.73 (> 0.65 en 4 de 5 cadenas). **J̃_llegada no baja: queda en 1.04–1.08 (> nominal)**; el optimizador lo sacrifica a cambio de ganar en frente y persistencia. El objetivo "< 0.65 en llegada y persistencia" **no se alcanzó**.

El J_min previo (0.7348) **no es comparable** con el actual (0.6740): cambiaron la definición de J̃_llegada, la envolvente, D̄ y las grillas térmicas.

Conjuntos de evaluación en la envolvente: llegada 89 positivos / 7,326 negativos; persistencia 17 positivos / 141 negativos. Con 106 presencias en total, la muestra es pequeña.

---

## 4. Tabla 5 — Parámetros calibrados (θ̂_MAP y Θ_comp, 778 miembros, Δ ≤ 0.05)

| Parámetro | Prior | θ̂_MAP | Mediana Θ_comp | P5–P95 Θ_comp |
|---|---|---|---|---|
| T_c,s | [−5.00, 0.00] | −1.968 | −1.942 | [−2.835, −1.397] |
| α_s | [0.01, 0.10] | 0.083 | 0.068 | [0.034, 0.088] |
| **β_s** | **{1, 2, 3}** | **2** | 2 | **[2, 3]** (480 con β=2, 298 con β=3) |
| K₀ | [1,000, 20,000] | 3,908 | 15,557 | [2,585, 19,945] |
| q_balloon | [0.10, 0.60] | 0.432 | 0.445 | [0.304, 0.566] |
| q_viab | [0.20, 0.90] | 0.807 | 0.766 | [0.428, 0.852] |
| γ₁ | [0.10, 1.00] | 0.376 | 0.807 | [0.387, 0.998] |
| γ₂ | [0.20, 1.20] | 0.530 | 0.539 | [0.219, 1.197] |
| γ₃ | [0.05, 0.80] | 0.220 | 0.335 | [0.112, 0.576] |

Fijos: F = 450.0, Q₁₀ = 2.2. β_s es siempre entero (violación corregida). θ̂_MAP viene de la cadena 4, iteración 242 (J = 0.6740).

> [!CAUTION]
> **K₀ y γ₁ ya no están pegados al borde, pero esto NO es una convergencia identificada.** El MAP de K₀ (3,908) cae justo por debajo del rango "plausible" pedido [4,000, 14,000], y el intervalo P5–P95 de Θ_comp abarca casi todo el prior (K₀: 2,585–19,945; γ₁: 0.387–0.998). Como las covariables de `grid_5km` son constantes (z_imp = z_borde = z_dosel = 0.5 en las 256,020 celdas), K_i es idéntico en todas las celdas y solo **K_eff = K₀·exp(½Σγ)** es identificable. K_eff en Θ_comp: mediana 34,607, P5–P95 [4,536, 50,000] (con tope en 50,000), sin acotar tampoco. K_eff del MAP: 6,862. La columna `k_eff_seed` del CSV permite verificarlo.

---

## 5. Tabla 6 — Sensibilidad a los pesos multicriterio

| Configuración | (w₁, w₂, w₃) | J_min | Retención Θ_comp | Diagnóstico |
|---|---|---|---|---|
| Base (iguales) | (0.333, 0.333, 0.334) | 0.6740 | 100.0 % | Referencia |
| A (énfasis en frente) | (0.2, 0.6, 0.2) | 0.5194 | 89.3 % | ROBUSTO (≥ 70 %) |
| B (énfasis en persistencia) | (0.2, 0.2, 0.6) | 0.6668 | 75.8 % | ROBUSTO (≥ 70 %) |

Θ_comp^A y Θ_comp^B se reconstruyen sobre todas las evaluaciones, cada uno con su propio J_min; la retención es la fracción del Θ_comp base que permanece en ellos. La robustez es parcialmente esperable: J̃_frente es pequeño y casi plano en Θ_comp (0.288–0.426), y Θ_comp es muy amplio por no identificabilidad.

**`v_evaluation_set` (2020–2024): intacta y bloqueada para la Fase 5.**
- El código de calibración no consulta la vista; carga solo 239,502 filas 2014–2019 (log: "v_evaluation_set: 0 filas cargadas") y lanza excepción si aparece una fila fuera de 2014–2019.
- La vista sigue con 199,602 filas (verificado en PostGIS tras la corrida).

---

## 6. Limitaciones de datos detectadas (importan más que cualquier mejora de J)

1. **Covariables de `grid_5km` son marcadores de posición**: z_imp = z_borde = z_dosel = 0.5, densidad de población = 15.0 y de carreteras = 1.2 en todas las celdas. K₀, γ₁, γ₂, γ₃ no son identificables por separado.
2. **El esfuerzo TGB (`E_Araneae`) es sintético**: celdas de fondo = 238 constante; celdas con presencia = 3. Esto invierte el modelo de detección. No se descargaron conteos reales de Araneae de GBIF, y el informe anterior sugería lo contrario. Corrección aplicada **solo en memoria**: el esfuerzo en filas con Y = 1 se imputa con la mediana anual del esfuerzo de fondo (106 filas). La base de datos no se modificó.
3. **Coeficientes del modelo de ocupación**: el informe anterior citaba valores incorrectos. Los reales (`occupancy_model.hpp`) son a₀ = −4.0, a₁ = 1.2, a₂ = 0.35, b₀ = 1.5, b₁ = 0.8, b₂ = 0.45, β₀ = −3.2, β₁ = 0.75, β₂ = 0.25, β₃ = 0.30.
4. **LPDM de calibración**: 60 pasos × 60 s (1 h de vuelo) y viento fijo (u = 4, v = 1.5 m/s) dentro del kernel SYCL, que ignora la atmósfera ERA5.
5. **Muestra pequeña**: 106 presencias celda-año (2014: 1, 2015: 0, 2016: 2, 2017: 5, 2018: 26, 2019: 72).

---

## 7. Pendientes recomendados antes de la Fase 5

1. Descargar TGB real (GBIF Araneae, EE. UU. este, 2014–2019) y covariables reales (NLCD impermeabilidad/borde/dosel, población, carreteras); sin ello, Tabla 5 no se puede interpretar biológicamente.
2. Subir la ocupación del cómputo GPU: partículas en SoA, histograma/reducción de aterrizajes en GPU (elimina D2H y la reducción en host), memoria anclada. Es previsible que baje el 45 % de transferencia y el 23 % de host; vale la pena antes de pasar a la A770.
3. Balancear carga entre cadenas si se quiere tiempos similares (p. ej. limitar súper-partículas por evaluación o remuestrear con tope).
4. Usar la atmósfera ERA5 real en el kernel SYCL.
