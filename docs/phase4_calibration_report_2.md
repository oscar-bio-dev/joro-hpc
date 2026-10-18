# INFORME — FASE 4 (CIERRE DE LA SECCIÓN 7): DATOS REALES, ERA5 EN EL KERNEL SYCL Y LPDM ZERO-COPY

**Proyecto:** JORO-HPC (*Trichonephila clavata*) · **Plataforma de pruebas:** i9-11900K (8 núcleos × 2 SMT = 16 hilos, AVX-512) + Intel Arc A380 (SYCL / Level-Zero). La corrida de producción usará una Arc A770.
**Ventana de calibración:** 2014–2019 · **Ventana bloqueada:** 2020–2024 (`v_evaluation_set`)
**Compilación:** `icpx -O3 -fsycl -qopenmp -march=rocketlake` (Release) · **ctest:** 4/4 ✔
**Lanzamiento:** `OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores` vía `scripts/telemetry/run_phase4_telemetry.sh`
**Artefactos crudos:** `docs/telemetry/phase4_closure_run.log`, `phase4_closure_telemetry_summary.txt` (corrida final), `phase4_closure_intermedia_stdout.txt` (corrida previa a la consolidación de transferencias), `covariates_real.log`.
**Informe anterior:** [phase4_calibration_report.md](phase4_calibration_report.md) (Fase 4 repetida, con marcadores de posición). Este documento lo reemplaza como referencia vigente; el anterior se conserva como histórico.

> [!WARNING]
> **Resumen honesto.** Las 4 correcciones estructurales están implementadas y verificadas, y el flujo de cómputo mejoró de forma medible (pared 56.2 s → 25.9 s, GPU compute 8 % → 68 %, desbalance 8.6× → 3.7×). Pero **tres objetivos numéricos NO se cumplieron**: (i) `H2D+D2H` bajó a 2.08 s, no a < 1.5 s; (ii) las cadenas siguen sin tener tiempos "similares" (máx/mín = 3.67); (iii) con covariables y esfuerzo reales, **K₀, γ₁, γ₂, γ₃ siguen sin ser identificables** y J̃_llegada **no cae por debajo de 1.0**. No se ajustó nada para forzarlos (§6).

---

## 1. Tarea 1 — Datos reales en PostGIS

### 1.1 Covariables espaciales de `grid_5km` (256,020 celdas)

Script: [build_real_covariates.py](../scripts/python/build_real_covariates.py). Copias de seguridad de los valores constantes: tablas `grid_5km_covariates_placeholder_backup` y `cell_annual_observations_effort_placeholder_backup`. Capas crudas: tabla `grid_5km_covariates_raw`.

| Capa | Fuente | Definición |
|---|---|---|
| `z_imp` | NLCD 2019 Impervious (MRLC WCS, 250 m, EPSG:5070) | log1p(% impermeable), estandarizado |
| `z_dosel` | NLCD 2019 Land Cover | fracción de bosque (clases 41, 42, 43, 90), estandarizada |
| `z_borde` | NLCD 2019 Land Cover | densidad de transiciones entre clases amplias de cobertura, estandarizada |
| `road_density_r` | TIGER 2019 Primary Roads (I-85, I-95, etc.) | log1p(10 × km/km²), estandarizada |
| `pop_density_h` | **PROXY**, no censo | densidad típica por clase NLCD *Developed* (30 / 700 / 2,500 / 7,000 hab/km²), log1p, estandarizada |

**Verificación SQL** (`SELECT stddev(z_imp), stddev(z_borde), stddev(z_dosel) FROM grid_5km;`):

| Capa | desv. estándar | media |
|---|---|---|
| z_imp | **1.0000001** | −1.4e-7 |
| z_borde | **0.9999999** | −4.7e-8 |
| z_dosel | **0.9999998** | 1.5e-7 |
| pop_density_h | 1.0000001 | −1.9e-7 |
| road_density_r | 1.0000001 | −3.9e-7 |

La varianza espacial es ≈ 1.0 (antes 0.0). Correlaciones entre capas estandarizadas: z_imp–z_dosel 0.29 · z_imp–z_borde 0.65 · z_dosel–z_borde 0.68 · z_imp–pop 0.93 · z_imp–road 0.52.

> [!NOTE]
> **Limitaciones de la capa.** (1) Solo ≈ 49.5 % de los píxeles del dominio tienen dato NLCD (el resto es océano o fuera de CONUS) y se trataron como 0 antes de estandarizar; esto distorsiona la media y la escala de las capas (las celdas sin dato quedan como un grupo con valor común). (2) `pop_density_h` es un proxy derivado de NLCD, **no** datos censales; su correlación de 0.93 con `z_imp` lo hace casi redundante. (3) `z_borde` y `z_dosel` correlacionan 0.68. Estas colinealidades limitan la identificabilidad de γ₁, γ₂, γ₃ (§4).

### 1.2 Esfuerzo TGB real (`effort_araneae`)

Script: [fetch_effort_tgb_real.py](../scripts/python/fetch_effort_tgb_real.py). Fuente: API de mapas de densidad de GBIF (teselas vectoriales, `taxonKey = 1496`, *Araneae*, 2014–2024), zoom 8, `squareSize = 8`, agregadas a las celdas por `ST_Contains` en PostGIS. Se reemplazaron los valores constantes `238` (fondo) y `3` (presencia). `effort_araneae = GREATEST(conteo_GBIF, joro_count)`.

> [!IMPORTANT]
> Es una **aproximación** al conteo por celda (rejilla de densidad de GBIF, no un volcado de registros individuales). Para conteos exactos habría que usar una descarga DwC-A con DOI.

| Año | Celdas con esfuerzo > 0 | Máx. | Media |
|---|---|---|---|
| 2014 | 279 | 25 | 0.012 |
| 2015 | 417 | 75 | 0.023 |
| 2016 | 734 | 32 | 0.038 |
| 2017 | 1,222 | 39 | 0.067 |
| 2018 | 1,916 | 52 | 0.123 |
| 2019 | 3,007 | 118 | 0.185 |

El esfuerzo real es **muy disperso** (≈ 99.3 % de las celdas-año en 2014 tienen esfuerzo 0) y crece entre 2014 y 2019: el esfuerzo medio por celda se multiplica ×15 (0.012 → 0.185) y el total de registros de *Araneae* en el dominio pasa de 15,044 a 36,889 (×2.5), reflejando el crecimiento de ciencia ciudadana. Hay 46 valores distintos en 2014–2019, de modo que la imputación en memoria del informe anterior **queda desactivada** (se activaba solo con < 8 valores distintos): el log del SA indica "conteos reales de Araneae (sin imputación)".

El CSV de calibración se regeneró con [export_calibration_set.py](../scripts/python/export_calibration_set.py): 239,502 filas, 106 presencias (igual que antes).

> [!NOTE]
> **`v_evaluation_set`:** solo cambiaron los valores de `effort_araneae` y de las covariables de 2020–2024 (que ahora también son reales). Las etiquetas `y_joro_detected` no se tocaron (199,602 filas, 2,442 presencias, verificado tras la corrida).

---

## 2. Tarea 2 — Atmósfera ERA5 dentro del kernel SYCL

Código: [era5_view.hpp](../src/lpdm/era5_view.hpp), [lpdm_trace.hpp](../src/lpdm/lpdm_trace.hpp), [lpdm_sycl_kernel.cpp](../src/lpdm/lpdm_sycl_kernel.cpp).

- Los tensores `u, v, w, blh` (+ presión superficial para la altura sobre el terreno) se suben **una sola vez** a Device USM al iniciar la calibración: 168 h × 12 niveles × 81 × 101, empaquetado `float4` (u, v, w) = **262.1 MB**, carga única 0.09 s. Caben holgadamente en los 6 GB de la A380.
- Dentro del `parallel_for`, cada partícula y paso interpola **bilinealmente en horizontal, linealmente en el tiempo y linealmente en altura sobre el terreno** (U, V, W) y en el tiempo la BLH (acotada a [250, 3,000] m; w acotada a ±1 m/s). Las constantes `u = 4, v = 1.5` solo siguen existiendo en el modo heredado (Fases 1–3, Morris, test 1), activo únicamente si no se llama a `set_atmosphere`.
- Hora de liberación uniforme en [0, n_t − 1 − horas de vuelo].
- Test nuevo (`test_cpu_gpu_equivalence`, caso 2: ERA5 sintético no uniforme en ambos motores): razón de aterrizaje CPU 0.98875 vs GPU 0.98925; salto medio 3.31 vs 3.30 km.

> [!WARNING]
> **Limitación importante:** en disco solo hay **7 días de mayo de 2014** (168 h). El mismo campo atmosférico se reutiliza para los 6 años 2014–2019. El viento es real, pero no es el viento de cada año ni de toda la estación de dispersión. Descargar la ventana completa de ERA5 por año (≈ 150 MB × 6 siguiendo el mismo formato) es un pendiente para la A770.

---

## 3. Tarea 3 — Eliminación del cuello de botella PCIe

### 3.1 Diseño (LPDM zero-copy)

- **H2D:** solo la tabla compacta de celdas emisoras (`EmissionJob`: `cell_id, lon, lat, total, n_sim, offset`; < 1 MB) vía memoria anclada. Ya no existe `d_particles_`: no se crea ni se copia ningún vector de partículas.
- **Kernel:** work-group de 128; cada *work-item* localiza su lote por búsqueda binaria del `offset`, integra la trayectoria completa en registros con **Philox-4x32-10** (contador = índice de super-partícula, sin estado compartido; reproducible), busca la celda de aterrizaje en el `SpatialGridLookup` (Device USM) y acumula con `sycl::atomic_ref<uint32_t, relaxed, device>` en `d_arrivals[cell_id]`. Histograma de distancias en memoria local por work-group + `reduce_over_group` para parciales.
- **D2H:** un solo bloque contiguo `[llegadas | histograma | parciales]` (≈ 1.03 MB: 256,020 enteros + 2,048 bins + parciales) en **una** copia hacia memoria anclada.
- **Tope adaptativo de súper-partículas:** tope por celda 25,000 y presupuesto por despacho 120,000 (variable `JORO_MAX_SUPER`). Si lo excede, se reduce `n_sim` por celda (`ceil`) y el peso de cada súper-partícula crece a `total/n_sim`. **La suma de pesos por celda sigue siendo exactamente el total de propágulos** (test 3: conservación exacta con tope).
- Estadísticos de salto ponderados por propágulos; P95 desde el histograma (resolución 0.25 km).

### 3.2 Resultados de telemetría (5 × 400 = 2,000 evaluaciones + 6 de referencia)

Se muestran tres estados: **antes** (informe anterior), **intermedio** (zero-copy, pero 3 copias H2D + 5 D2H paginadas por despacho) y **final** (bloque único + memoria anclada). Los tres resultados científicos de las corridas intermedia y final son **idénticos** (J_min = 0.7525, mismo Θ_comp de 546 miembros): el refactor de transferencias no cambia el resultado, solo el tiempo.

| Métrica | Antes | Intermedio | **Final** | Objetivo |
|---|---|---|---|---|
| Tiempo de pared de la calibración | 56.17 s | 31.73 s | **25.90 s** | — |
| `H2D + D2H` | 25.45 s (45.3 %) | 5.25 s (16.5 %) | **2.08 s (8.0 %)** | **< 1.5 s ✘ no alcanzado** |
| Kernel GPU | 4.94 s (8.8 %) | 19.89 s (62.7 %) | **17.98 s (69.4 %)** | — |
| LPDM host (init + reducción) | 13.10 s (23.3 %) | 1.48 s | **1.13 s** | — |
| ABM CPU | 1.80 s | 3.10 s | 2.92 s | — |
| Costo CPU | 0.85 s | 1.17 s | 1.02 s | — |
| GPU motor *compute* (media / P95) | 8.0 % / 9.3 % | 59.9 % / 74.6 % | **68.3 % / 81.3 %** | ↑ ✔ |
| GPU motor *copy* (media) | 38.1 % | 6.1 % | **4.0 %** | ↓ ✔ |
| Super-partículas totales | 1,792 M | 555 M | 555 M | — |
| Despachos con tope adaptativo | — | 3,933 / 12,036 | 3,933 / 12,036 | — |
| Fallbacks a CPU | 0 | 0 | **0** | 0 ✔ |
| CPU, 16 hilos (media global) | 100.0 % | 99.4 % | **99.2 %** (102/103 muestras con los 16 hilos > 50 %) | ✔ |

- **Transferencias:** 25.45 s → 2.08 s (−92 %). El objetivo de < 1.5 s **no se alcanzó**. Con 12,036 despachos anuales el costo es ≈ 173 µs por despacho, dominado por el D2H del bloque de ≈ 1 MB (≈ 12 GB movidos en total sobre PCIe de la A380, que va a x8). Para bajarlo más haría falta compactar en el dispositivo (devolver solo las celdas con llegada ≠ 0, que son unos pocos miles) o fusionar despachos de varios años; no se hizo. En la A770 (PCIe 4.0 x16) se espera un valor menor, pero no está medido.
- **Rendimiento del kernel no comparable:** el kernel previo recibía partículas ya inicializadas, con viento fijo y sin atomics; el actual hace la trayectoria completa con ERA5, Philox y `atomic_ref`. Por eso el throughput de kernel por súper-partícula cae de 362.7 a 30.8 M/s (1,850 M partícula-pasos/s) aunque el tiempo total baja 2.2×. El total de súper-partículas cae 3.2× (1,792 M → 555 M) por el tope, pero **el número de propágulos simulados es el mismo** (se conserva por peso).
- **Cuello de botella actual:** el propio kernel (69.4 % de la pared); el motor *compute* alcanza 82.7 % de máximo.

### 3.3 Balance entre cadenas

| Cadena | Antes (s) | Final (s) | Super-part. antes (M) | Super-part. final (M) |
|---|---|---|---|---|
| 1 | 3.71 | 4.83 | 85.7 | 99.6 |
| 2 | 9.99 | 3.44 | 309.1 | 64.2 |
| 3 | 4.50 | 3.56 | 119.6 | 54.7 |
| 4 | 5.89 | **11.02** | 161.4 | 282.1 |
| 5 | **31.99** | 3.00 | 1,114.6 | 53.2 |
| **máx / mín** | **8.62** | **3.67** | | |

Mejora de 8.62× a 3.67×, pero **no es "similar"**. La cadena 4 (J* = 0.875, la peor) explora una región de K₀ y q_balloon altos donde casi todos los despachos alcanzan el tope: 120,000 × 6 años × 401 evaluaciones ≈ 289 M súper-partículas, coherente con los 282 M medidos. El tope acota el peor caso por evaluación (≈ 27 ms) pero no lo iguala con las cadenas que viven en regiones de baja emisión. Con estas semillas, igualar tiempos exigiría un tope mucho menor, a costa de ruido adicional en el recuento de llegadas.

---

## 4. Tarea 4 — Recalibración (5 cadenas × 400 evaluaciones, 2014–2019)

Escalas de referencia: S_llegada = 0.45432, S_frente = 75.151 km, S_persistencia = 0.45355. Llegada: 89 positivos / 7,326 negativos en la envolvente activa; persistencia: 17 / 141; 231,929 ausencias lejanas excluidas. Con 106 presencias la muestra sigue siendo pequeña.

### 4.1 Tabla 4 — Convergencia de las 5 cadenas

| Cadena | Semilla | J₀ | J* | J̃_llegada | J̃_frente | J̃_persist. | Aceptadas | Tiempo |
|---|---|---|---|---|---|---|---|---|
| 1 | 1001 | 1.0000 | 0.7538 | 1.0529 | 0.2001 | 1.0083 | 59 % | 4.83 s |
| 2 | 2002 | 0.8533 | **0.7525** | 1.0414 | 0.2146 | 1.0016 | 77 % | 3.44 s |
| 3 | 3003 | 0.8711 | 0.7546 | 1.0538 | 0.1989 | 1.0112 | 66 % | 3.56 s |
| 4 | 4004 | 0.9650 | 0.8751 | 1.0126 | 0.6132 | 0.9995 | 79 % | 11.02 s |
| 5 | 5005 | 1.0184 | 0.7578 | 1.0316 | 0.2418 | 1.0000 | 66 % | 3.00 s |

**Lectura honesta:**
- **J̃_frente** baja con claridad (0.20–0.24 en 4 de 5 cadenas; 0.29–0.32 en el informe anterior).
- **J̃_llegada** queda en **1.013–1.054** (antes 1.040–1.081): ligeramente menor, pero **no baja de 1.0** y está lejos del 0.65 buscado. El optimizador no mejora el acierto de llegada respecto a θ_nominal.
- **J̃_persistencia ≈ 1.00**: no mejora. En el informe anterior bajaba a 0.65–0.73; con esfuerzo real y covariables heterogéneas esa mejora **desapareció**, lo que sugiere que era un artefacto del esfuerzo imputado y de las covariables constantes.
- **J_min = 0.7525 no es comparable con 0.6740 del informe anterior** (cambian datos, escalas de normalización y covariables). No debe leerse como un empeoramiento ni una mejora del modelo.
- Los resultados de las cadenas 1–5 son deterministas y reproducibles bit a bit entre la corrida intermedia y la final.

### 4.2 Tabla 5 — Parámetros calibrados (θ̂_MAP y Θ_comp, 546 miembros, Δ ≤ 0.05)

| Parámetro | Prior | θ̂_MAP | Mediana Θ_comp | P5–P95 Θ_comp | Ancho P5–P95 / prior (nuevo) | (anterior) |
|---|---|---|---|---|---|---|
| T_c,s | [−5.00, 0.00] | −1.329 | −0.720 | [−2.038, −0.060] | 0.40 | 0.29 |
| α_s | [0.01, 0.10] | 0.050 | 0.065 | [0.031, 0.087] | 0.63 | 0.60 |
| β_s | {1, 2, 3} | 2 | 2 | [1, 3] (358 con 2, 146 con 1, 42 con 3) | — | 2–3 |
| **K₀** | [1,000, 20,000] | 11,839 | 6,483 | [2,507, 18,637] | **0.85** | 0.91 |
| q_balloon | [0.10, 0.60] | 0.128 | 0.560 | [0.118, 0.599] | 0.96 | 0.52 |
| q_viab | [0.20, 0.90] | 0.268 | 0.442 | [0.255, 0.875] | 0.89 | 0.61 |
| **γ₁** | [0.10, 1.00] | 0.106 | 0.358 | [0.107, 0.845] | **0.82** | 0.68 |
| **γ₂** | [0.20, 1.20] | **1.200** | 0.779 | [0.312, 1.194] | **0.88** | 0.98 |
| **γ₃** | [0.05, 0.80] | 0.588 | 0.627 | [0.332, 0.799] | **0.62** | 0.62 |

Fijos: F = 450.0, Q₁₀ = 2.2. β_s es siempre entero.

> [!CAUTION]
> **La identificabilidad de K₀, γ₁, γ₂ y γ₃ NO mejoró de forma sustancial.** Los intervalos P5–P95 siguen cubriendo entre 62 % y 88 % del prior (K₀ 85 %, γ₁ 82 %, γ₂ 88 %, γ₃ 62 %); γ₂ queda **en el borde superior** (MAP = 1.200, el límite del prior) y los intervalos de q_balloon y q_viab son ahora *más* anchos que antes. Que el MAP de K₀ (11,839) caiga en el rango "plausible" [4,000, 14,000] no es evidencia de identificación: la mediana de Θ_comp es 6,483 y el intervalo cubre 2,507–18,637.

**Lo que sí cambió:** con covariables heterogéneas dejó de haber una degeneración exacta (K_i idéntico en todas las celdas). Θ_comp muestra ahora correlaciones estructurales, que son la huella de la identificabilidad parcial:

| Par | Correlación en Θ_comp |
|---|---|
| γ₁ – γ₂ | **−0.82** |
| K₀ – γ₃ | **−0.75** |
| K₀ – q_viab | +0.69 |
| γ₃ – q_viab | −0.62 |
| γ₂ – q_balloon | −0.49 |

Es decir, los datos restringen **combinaciones** de parámetros (p. ej. γ₁ + γ₂ trade-off a través de la colinealidad z_imp–z_borde de 0.65; K₀ frente a γ₃), pero no cada uno por separado. Una cantidad que sí se podría reportar con más confianza es K_i por celda, no los parámetros individuales.

> [!NOTE]
> La columna `k_eff_seed` del CSV (`/mnt/hpc_scratch/joro-hpc/checkpoints/theta_compatible_ensemble.csv`) es K_i **en la celda semilla**; vale 50,000 (el tope de `compute_carrying_capacity`) en ≥ 50 % de los miembros, es decir, en la celda semilla el factor exp(γᵀZ) satura el tope. No es un indicador de identificabilidad.

### 4.3 Tabla 6 — Sensibilidad a los pesos multicriterio

| Configuración | (w₁, w₂, w₃) | J_min | Retención Θ_comp | Diagnóstico |
|---|---|---|---|---|
| Base (iguales) | (0.333, 0.333, 0.334) | 0.7525 | 100.0 % | Referencia |
| A (énfasis en frente) | (0.2, 0.6, 0.2) | 0.5299 | 84.8 % | ROBUSTO (≥ 70 %) |
| B (énfasis en persistencia) | (0.2, 0.2, 0.6) | 0.8521 | 96.0 % | ROBUSTO (≥ 70 %) |

La robustez es parcialmente esperable: J̃_frente es pequeño y Θ_comp es muy amplio por falta de identificabilidad, de modo que cualquier reponderación conserva casi todos los miembros.

### 4.4 `v_evaluation_set` (2020–2024): intacta y bloqueada para la Fase 5

- El código de calibración no consulta la vista: log de la corrida, "v_evaluation_set: 0 filas cargadas (bloqueada)"; lanza excepción si aparece una fila fuera de 2014–2019.
- Verificado en PostGIS tras la corrida: **199,602 filas, 2,442 presencias (`y_joro_detected`), años 2020–2024**.
- Lo único modificado en esa ventana: `effort_araneae` y las covariables `z_*`, `pop_density_h`, `road_density_r` (ahora reales). Las etiquetas de presencia no cambiaron. Ninguna de estas columnas se usó para ajustar parámetros.
- Calibración: `v_calibration_set` con 239,502 filas, 106 presencias, años 2014–2019.

---

## 5. Cambios de código (resumen)

| Archivo | Cambio |
|---|---|
| [era5_view.hpp](../src/lpdm/era5_view.hpp) (nuevo) | Muestreador ERA5 (bilineal/lineal) compartido por kernel y host |
| [lpdm_trace.hpp](../src/lpdm/lpdm_trace.hpp) (nuevo) | `EmissionJob`, Philox-4x32-10, `trace_super_particle()` compartido |
| [lpdm_sycl_kernel.cpp](../src/lpdm/lpdm_sycl_kernel.cpp) | Reescrito: zero-copy, `atomic_ref`, tope adaptativo, bloque de resultados único + memoria anclada |
| [lpdm_cpu_baseline.cpp](../src/lpdm/lpdm_cpu_baseline.cpp), [lpdm_engine.hpp](../src/lpdm/lpdm_engine.hpp) | `set_atmosphere`, modo ERA5, estadísticos ponderados |
| [spatial_grid_lookup.hpp](../src/common/spatial_grid_lookup.hpp) | `SpatialGridView` (POD, utilizable en el dispositivo) |
| [era5_netcdf_reader.cpp](../src/io/era5_netcdf_reader.cpp) | Lee latitud/longitud del NetCDF |
| [simulated_annealing.cpp](../src/inference/simulated_annealing.cpp) | H y R normalizados min-max; imputación de esfuerzo solo si es sintético |
| [main.cpp](../src/main.cpp) | `set_atmosphere` antes del SA; Tabla 7 con despachos con tope y % de pared H2D+D2H |
| [test_cpu_gpu_equivalence.cpp](../tests/test_cpu_gpu_equivalence.cpp) | Casos 2 (ERA5 no uniforme) y 3 (conservación con tope) |
| `scripts/python/*.py` (nuevos) | `build_real_covariates.py`, `fetch_effort_tgb_real.py`, `export_calibration_set.py` |

---

## 6. Limitaciones y objetivos no cumplidos

1. **`H2D+D2H` = 2.08 s, no < 1.5 s.** Reducción de 92 % lograda; el resto es el D2H de ≈ 1 MB × 12,036 despachos.
2. **Balance máx/mín = 3.67, no "similar".** Acotado por el tope, pero dependiente del θ que explora cada cadena.
3. **J̃_llegada ≥ 1.0 y J̃_persistencia ≈ 1.0.** El modelo, con los datos reales, no mejora el acierto de llegada ni de persistencia respecto a θ_nominal; solo mejora el frente.
4. **K₀, γ₁, γ₂, γ₃ no identificables por separado**; γ₂ pegado al borde del prior. Hay trade-offs fuertes (γ₁–γ₂ −0.82; K₀–γ₃ −0.75).
5. **Atmósfera:** solo 7 días de mayo de 2014 para todos los años.
6. **Covariables:** población es un proxy de NLCD, no censo; ~50 % de los píxeles sin dato NLCD (océano/fuera de CONUS).
7. **Esfuerzo:** aproximación por rejilla de densidad de GBIF, muy disperso (≈ 99 % de ceros en 2014).
8. **Muestra:** 106 presencias celda-año en 2014–2019 (89 positivos de llegada, 17 de persistencia).

## 7. Pendientes recomendados antes de la Fase 5

1. Descargar ERA5 de la ventana completa de dispersión por año (mismo formato; ≈ 150 MB/año) y quitar la repetición del campo de mayo de 2014.
2. Sustituir `pop_density_h` por densidad censal real (p. ej. Census/SEDAC GPW) y revisar la colinealidad con `z_imp`; considerar reparametrizar (γ₁ + γ₂, K₀·exp(γ₃·z̄)) o fijar con a priori informativos los parámetros que no se identifican.
3. Usar conteos exactos de *Araneae* (descarga GBIF DwC-A con DOI) en vez de la rejilla de densidad.
4. Compactar el D2H en el dispositivo (solo celdas con llegada ≠ 0) para llegar a < 1.5 s; medir en la A770.
5. Revisar los límites del prior de γ₂ (MAP en 1.2) antes de interpretar el parámetro.
