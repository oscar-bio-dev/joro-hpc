# JORO-HPC — Validación ciega fuera de muestra (2020–2024) y contraste de las hipótesis H1 y H2

*Generado automáticamente el 2026-11-14 23:01:31 por `scripts/python/generate_validation_report.py` a partir de los artefactos de la corrida; ninguna cifra de resultados está escrita a mano.*

- **Plataforma de cómputo:** Intel(R) Arc(TM) A380 Graphics | Intel(R) oneAPI Unified Runtime over Level-Zero | GPU + 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz (16 hilos OpenMP, AVX-512).
- **Compilador:** icpx (versión no disponible en este entorno); banderas `-O3 -fsycl -qopenmp -march=rocketlake -std=c++20`.
- **Ventana de evaluación:** 2020–2024 (**199,602 filas celda-año**, **2,787 presencias observadas**); ventana de calibración: 2014–2019 (239,502 filas).
- **Protocolo de integridad:** los parámetros de JORO-HPC ($\mathcal{M}_1$ y $\mathcal{M}_0$) y de los comparadores se congelaron con datos 2014–2019; ninguna fila de 2020–2024 participó en la calibración.
- **Pruebas unitarias:** `ctest` 4/4 (ver `docs/RUNBOOK.md`).

---

## 1. Protocolo de aislamiento y desbloqueo controlado

1. **Desbloqueo controlado:** el ejecutable `joro_evaluate` habilita la lectura de los tensores ERA5 2020–2024 (`era5_uvwt_202X_{05,06,07}.nc`, `era5_sfc_202X_{05,06,07}.nc`, `era5_daily_t2m_202X.nc`) y de `evaluation_set_2020_2024.csv` mediante la variable de entorno `JORO_UNLOCK_EVALUATION=1` (alias heredado: `JORO_UNLOCK_PHASE5`). Sin ella, el lector rechaza cualquier año posterior a 2019.
2. **Cero reajuste paramétrico:** se propagan los estimadores MAP y el ensamble completo $\Theta_{\text{comp}}$ congelados: **339** miembros para $\mathcal{M}_1$ y **332** para $\mathcal{M}_0$.
3. **Simulación continua de 11 años (2014–2024):** el acoplamiento LPDM–ABM se inicializa en 2014 con la cohorte semilla de Georgia (125 celdas contiguas en Braselton/Hoschton, radio 37.6 km) y se propaga sin interrupción por los 6 años de calibración y los 5 de evaluación, con los tensores ERA5 estacionales completos (mayo–julio, 2,208 h/año en `FP16`) reales de cada año en la GPU y las series térmicas diarias de 365 días en la CPU.
4. **Tiempo de ejecución:** la corrida completa de `joro_evaluate` (ensambles de $\mathcal{M}_1$ y $\mathcal{M}_0$, comparadores y exportación de métricas) tomó **211.86 s** en pared.

---

## 2. Comparadores nulos bajo particiones idénticas

1. **Comparador espacial para H2 — kernel exponencial isotrópico $K_0(d \mid \lambda)$:**
   $$K_0(d \mid \lambda) = \frac{1}{\lambda} \exp\left(-\frac{d}{\lambda}\right), \qquad d \ge 0$$
   - Sustituye el transporte 3D de ERA5 y la turbulencia de Langevin por dispersión radial simétrica $O(1)$ (`SpatialGridLookup`).
   - Conserva el reloj de grados-día $Q_{10}$, el daño térmico $D_{j,t}$, la capacidad de carga $K_i$ y el submodelo de detección.
   - **Calibración estricta en 2014–2019:** $\lambda^* = 8.0\text{ km}$ (congelado para 2020–2024).
2. **Comparador climático para H1 — SDM correlativo estático (regresión logística):**
   $$\operatorname{logit}(P(Y_{i,t}=1)) = \beta_0 + \beta_1 T_{\min,\text{inv}} + \beta_2 T_{\text{med},\text{ver}} + \beta_3 Z_{\text{imp}} + \beta_4 Z_{\text{borde}} + \beta_5 Z_{\text{dosel}} + \beta_6 H_i + \beta_7 R_i + \beta_8 \log(1 + E_{i,t}^{\text{Araneae}})$$
   - Entrenado por máxima verosimilitud con las filas 2014–2019; carece de memoria multianual de daño ($D_{j,t}$) y de dispersión de propágulos.
   - Coeficientes (2014–2019): intercepto +0.0502; `w_tmin` +0.2560; `s_tmean` -0.4441; `z_imp` +0.6122; `z_borde` +1.5846; `z_dosel` +0.7290; `pop_density_h` +0.9251; `road_density_r` +0.0888; `log_effort` +1.6655.

---

## 3. Desempeño probabilístico global y por año

### Tabla 1: Desempeño global fuera de muestra (2020–2024, 199,602 filas celda-año)

| Modelo | Brier global | Brier balanc. (350 km) | Log-Loss | AUC-ROC | PR-AUC | Pendiente calib. |
| --- | --- | --- | --- | --- | --- | --- |
| JORO-HPC $\mathcal{M}_1$ (MAP) | 0.01094 | 0.31759 | 0.0545 | 0.9217 | 0.3532 | 0.641 |
| JORO-HPC $\mathcal{M}_1$ (mediana del ensamble) | 0.01311 | 0.34528 | 0.0698 | 0.9037 | 0.2087 | 0.488 |
| JORO-HPC $\mathcal{M}_1$ + $K_{\text{mixto}}$ humano (MAP) | 0.01094 | 0.31812 | 0.0537 | 0.9245 | 0.3497 | 0.645 |
| JORO-HPC $\mathcal{M}_0$ ($K_0$ constante, MAP) | 0.01227 | 0.31628 | 0.0613 | 0.9113 | 0.2786 | 0.575 |
| Kernel exponencial isotrópico (H2) | 0.01045 | 0.34324 | 0.0629 | 0.8921 | 0.4182 | 0.642 |
| SDM correlativo estático (H1) | 0.01329 | 0.44285 | 0.0743 | 0.8952 | 0.1592 | 0.586 |

*El AUC-ROC se calcula con la corrección de empates de Mann–Whitney (rangos promedio); el Log-Loss recorta las probabilidades a $[10^{-6}, 1-10^{-6}]$.*

### Tabla 2: Desglose anual (2020–2024)

| Año | Presencias | Brier $\mathcal{M}_1$ MAP | Brier kernel isot. | Brier SDM | AUC $\mathcal{M}_1$ MAP | AUC kernel isot. | AUC SDM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **2020** | 167 | 0.00343 | 0.00322 | 0.00399 | 0.9731 | 0.9681 | 0.9653 |
| **2021** | 316 | 0.00595 | 0.00541 | 0.00715 | 0.9680 | 0.9675 | 0.9554 |
| **2022** | 536 | 0.01005 | 0.00936 | 0.01312 | 0.9376 | 0.9224 | 0.9163 |
| **2023** | 758 | 0.01474 | 0.01432 | 0.01771 | 0.9218 | 0.8942 | 0.9053 |
| **2024** | 1010 | 0.02053 | 0.01993 | 0.02446 | 0.8737 | 0.8313 | 0.8850 |

---

## 4. Cinemática y geometría del frente de invasión (prueba de H2)

### Tabla 3: Radio P95 y azimut del frente (azimut respecto al centroide del polígono semilla)

| Año | P95 obs (km) | P95 $\mathcal{M}_1$ (km) | P95 kernel isot. (km) | Azimut obs (°) | Azimut $\mathcal{M}_1$ (°) | Azimut kernel (°) | Error ang. $\mathcal{M}_1$ (°) | Error ang. kernel (°) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **2020** | 89.6 | 194.6 | 101.5 | 248.9 | 113.5 | 176.3 | 135.4 | 72.5 |
| **2021** | 117.8 | 218.6 | 125.9 | 246.0 | 118.0 | 175.4 | 128.0 | 70.6 |
| **2022** | 179.0 | 244.2 | 151.1 | 255.1 | 127.1 | 170.9 | 128.0 | 84.2 |
| **2023** | 179.0 | 297.8 | 165.0 | 277.9 | 130.8 | 166.1 | 147.1 | 111.7 |
| **2024** | 215.2 | 335.7 | 180.6 | 291.0 | 128.8 | 161.5 | 162.2 | 129.5 |

- **Error absoluto medio en la fecha de primera llegada por celda:** $\mathcal{M}_1$: **1.67 años**; kernel isotrópico: **1.39 años**; SDM estático: **1.86 años**.
- **Dirección según la ocupación latente (celdas con $\psi \cdot p_{\text{det}} > 0.001$):** error angular medio de $\mathcal{M}_1$ 142.7° y del kernel 93.7°. Es del mismo orden que el error del frente biológico de la Tabla 3.
- El frente se mide estrictamente sobre las celdas con colonización activa del ABM (`adult_density_per_cell > 2.0`); el transporte basal por carreteras actúa sobre la ocupación $\psi$, no sobre el radio del frente biológico.

---

## 5. Persistencia térmica y transferencia por ecorregiones (prueba de H1)

### Tabla 4: Restricción térmica invernal y transferencia ecorregional

| Métrica | JORO-HPC $\mathcal{M}_1$ | JORO-HPC $\mathcal{M}_0$ | Kernel isotrópico | SDM estático |
| --- | --- | --- | --- | --- |
| **Tasa de falsos positivos al norte (>38°–40°N, $\tau^*=0.0012$)** | 0.00 % | 0.00 % | 0.00 % | 9.79 % |
| **Brier de persistencia interanual ($\ge 2$ años)** | 0.3370 | 0.3381 | 0.3517 | 0.4870 |
| **AUC ecorregión vista (bloque 5, Piedmont)** | 0.9247 | 0.8986 | 0.9167 | 0.9069 |
| **Brier ecorregión vista (bloque 5)** | 0.02195 | 0.02478 | 0.02079 | 0.02594 |
| **AUC ecorregiones nuevas (bloques 2, 7–12)** | 0.6927 | 0.7045 | 0.5529 | 0.8184 |
| **Brier ecorregiones nuevas** | 0.00246 | 0.00263 | 0.00248 | 0.00355 |

---

## 6. Propagación de incertidumbre y cobertura empírica

### Tabla 5: Cobertura agregada por año del ensamble $\Theta_{\text{comp}}$ (339 miembros), conteo esperado crudo y ajustado por prevalencia

Para cada miembro se suma, sobre todas las celdas, la probabilidad predicha de detección de cada año; la tabla muestra los percentiles 5, 50 y 95 de esos totales anuales entre miembros frente a las presencias observadas. El **conteo ajustado** aplica a cada miembro un offset logit de solo intercepto $\delta_m$ calculado **exclusivamente con 2014–2019** (hace que el total esperado de 2014–2019 iguale las presencias observadas en esa ventana) y se aplica sin cambios a 2020–2024; las celdas con probabilidad cero permanecen en cero.

| Año | Obs. | Crudo P5 | Crudo mediana | Crudo P95 | ¿Cubre? | Ajustado P5 | Ajustado mediana | Ajustado P95 | ¿Cubre? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **2020** | 167 | 163.2 | 356.2 | 568.0 | Sí | 272.7 | 392.8 | 718.7 | No |
| **2021** | 316 | 424.0 | 693.4 | 979.3 | No | 476.7 | 750.1 | 1585.3 | No |
| **2022** | 536 | 633.2 | 1001.8 | 1421.1 | No | 653.3 | 1119.4 | 2451.7 | No |
| **2023** | 758 | 502.0 | 679.9 | 1164.8 | Sí | 384.9 | 900.6 | 2176.5 | Sí |
| **2024** | 1010 | 834.7 | 1367.8 | 1778.6 | Sí | 926.3 | 1479.3 | 3174.2 | Sí |

- **Offset de prevalencia (solo 2014–2019, 108 presencias observadas):** $\mathcal{M}_1$ MAP: $\delta = +0.745$, esperado 55.1 → 108.0; $\mathcal{M}_0$ MAP: $\delta = +0.005$, esperado 107.5 → 108.0; ensamble: $\delta$ mediana +0.108 (P5–P95: -0.815 a +1.826).
- **Cobertura agregada:** crudo 3 de 5 años; ajustado 2 de 5 años con el total observado dentro del intervalo P5–P95.
- **Subpredicción del total anual:** observado / mediana = 0.5–1.1 (crudo) y 0.4–0.8 (ajustado). El ajuste corrige la prevalencia de la ventana de calibración, pero la distribución anual de las detecciones sigue creciendo más rápido que el simulador (el intervalo solo propaga incertidumbre paramétrica).
- **Ancho medio de banda $P_{95} - P_{5}$ por celda:** 0.03053.

### Tabla 6: Contraste de H2b — Radio P95 por cuadrante (media 2020–2024, km desde centroide semilla) y cola > 300 km

| Cuadrante | Observado (km) | $\mathcal{M}_1$ solo LPDM (km) | $\mathcal{M}_1$ + $K_{\text{mixto}}$ (km) | Kernel isotrópico (km) |
| --- | --- | --- | --- | --- |
| **NE** | 270.6 | 256.8 | 263.0 | 104.0 |
| **SE** | 145.4 | 272.1 | 273.8 | 155.4 |
| **SW** | 122.0 | 181.8 | 169.4 | 140.0 |
| **NW** | 159.7 | 144.1 | 149.3 | 103.2 |

- **Cola de larga distancia (> 300 km, 34 presencias observadas):** recall ($\psi \ge 0.10$) de $\mathcal{M}_1$ solo LPDM = **0.029** vs $\mathcal{M}_1$ + $K_{\text{mixto}}$ ($p_h = 1e-06$, $\lambda_h = 150\text{ km}$) = **0.029**.

### Tabla 7: Contraste de H1b y H3a — Correlación de rango de Spearman de la CPUE observada

| Hipótesis | Par evaluado (celdas-año con presencia, $n = 2787$) | $\rho$ de Spearman | Signo esperado |
| --- | --- | --- | --- |
| **H1b** | $\text{CPUE}_{\text{obs}}$ vs. daño invernal medio $\overline{D}_{i,t}$ | **0.0593** | Negativo (mayor daño reduce abundancia) |
| **H3a** | $\text{CPUE}_{\text{obs}}$ vs. capacidad de carga $K_i$ ($\mathcal{M}_1$) | **0.1000** | Positivo (hábitat sinantrópico sostiene mayor densidad) |

### Tabla 8: Contraste de H3b — Proceso de observación no estacionario $f_{\text{est}}(i,t)$ y calibración de Cox

| Variante del modelo de observación | Pendiente Cox ($c_1$, ideal 1) | Intercepto Cox ($c_0$, ideal 0) | Total esperado 2020–2024 | Total observado 2020–2024 |
| --- | --- | --- | --- | --- |
| 1. Crudo ($\delta = 0, f_{\text{est}} = 0$) | 0.794 | 0.672 | 859.2 | 2787 |
| 2. Offset estacionario de Platt ($\hat{\delta}_{\text{2014–19}}, f_{\text{est}} = 0$) | 0.662 | -0.629 | 2715.0 | 2787 |
| 3a. Con $f_{\text{est}}(i,t)$ activa ($\delta = 0$, Ec. 25) | 0.641 | -0.713 | 2849.5 | 2787 |
| 3b. Con $f_{\text{est}}(i,t)$ + offset base 2014–2019 ($\hat{\delta}_{\text{base}} + f_{\text{est}}$) | 0.588 | -1.237 | 4843.5 | 2787 |

### Tabla 8b: Sensibilidad a la resolución temporal de ERA5 en VRAM (`1 h`, 2,208 pasos vs `3 h`, 737 pasos en $\hat{\theta}_{\text{MAP}}$)

| Métrica fuera de muestra (2020–2024) | Paso `3 h` (737 pasos/año, 3.64 GiB VRAM) | Paso `1 h` (2,208 pasos/año, horario completo) | Diferencia absoluta $\lvert\Delta\rvert$ |
| --- | --- | --- | --- |
| **AUC-ROC global ($\mathcal{M}_1$ MAP)** | 0.9217 | 0.9214 | **0.0003** (0.03 %) |
| **Brier global ($\mathcal{M}_1$ MAP)** | 0.01094 | 0.01091 | **0.00003** (0.27 %) |
| **Log-Loss global ($\mathcal{M}_1$ MAP)** | 0.0545 | 0.0545 | **0.0000** |
| **Pendiente de calibración de Cox ($c_1$)** | 0.641 | 0.640 | **0.001** |
| **Falsos positivos al norte (>38°–40°N)** | 0.00 % | 0.00 % | **0.00 %** |
| **Radio $P_{95}$ medio 2020–2024 (km)** | 258.2 km | 254.0 km | **4.2 km** (1.65 %) |
| **Azimut medio del frente 2020–2024 (°)** | 123.6° | 122.6° | **1.0°** |

---

## 7. Criterios de decisión y dictamen sobre H1 y H2

Los dictámenes se calculan con criterios fijos, declarados antes de leer los resultados: **CONFIRMADA** si se cumplen todos; **SOPORTADA CON SALVEDADES** si se cumple ≥ 75 %; **PARCIALMENTE SOPORTADA** si ≥ 50 %; **NO SOPORTADA** en otro caso.

### Dictamen sobre H1: restricción térmica y memoria fisiológica (M1 frente al SDM estático)

| # | Criterio | Resultado | Evidencia |
| --- | --- | --- | --- |
| 1 | AUC-ROC global de M1 > SDM | ✔ | 0.9217 vs 0.8952 |
| 2 | Log-Loss global de M1 < SDM | ✔ | 0.0545 vs 0.0743 |
| 3 | Brier global de M1 < SDM | ✔ | 0.01094 vs 0.01329 |
| 4 | AUC anual de M1 > SDM en los 5 años | ✘ | 2020: 0.973/0.965, 2021: 0.968/0.955, 2022: 0.938/0.916, 2023: 0.922/0.905, 2024: 0.874/0.885 |
| 5 | Falsos positivos al norte (>38°N) de M1 < SDM | ✔ | 0.00 % vs 9.79 % |
| 6 | AUC en ecorregiones nuevas de M1 > SDM | ✘ | 0.6927 vs 0.8184 |
| 7 | Brier en ecorregiones nuevas de M1 < SDM | ✔ | 0.00246 vs 0.00355 |
| 8 | Brier de persistencia interanual de M1 < SDM | ✔ | 0.3370 vs 0.4870 |

> **DICTAMEN H1: SOPORTADA CON SALVEDADES** (6 de 8 criterios).

### Dictamen sobre H2: dispersión atmosférica LPDM 3D (M1 frente al kernel isotrópico)

| # | Criterio | Resultado | Evidencia |
| --- | --- | --- | --- |
| 1 | Error absoluto del P95 del frente en 2024: M1 < kernel | ✘ | 120.5 km vs 34.7 km |
| 2 | Error absoluto medio del P95 (2020–2024): M1 < kernel | ✘ | 102.0 km vs 19.3 km |
| 3 | AUC-ROC global de M1 > kernel | ✔ | 0.9217 vs 0.8921 |
| 4 | Log-Loss global de M1 < kernel | ✔ | 0.0545 vs 0.0629 |
| 5 | Brier global de M1 < kernel | ✘ | 0.01094 vs 0.01045 |
| 6 | PR-AUC global de M1 > kernel | ✘ | 0.3532 vs 0.4182 |
| 7 | Error angular medio de M1 menor que el del kernel en ≥ 10° | ✘ | 140.1° vs 93.7° |
| 8 | MAE de la fecha de primera llegada: M1 < kernel | ✘ | 1.67 vs 1.39 años |

> **DICTAMEN H2: NO SOPORTADA** (2 de 8 criterios).

### Hallazgos que matizan los dictámenes

- **Dirección del frente (referencia: centroide del polígono semilla):** error angular medio de $\mathcal{M}_1$ 140° y del kernel 94° (diferencia a favor de $\mathcal{M}_1$: -46°). El LPDM no mejora la dirección de expansión con los datos y la calibración actuales.
- **Calibración absoluta:** la pendiente de calibración de $\mathcal{M}_1$ es 0.641 (ideal 1) y el ensamble cubre el total observado en 3 de 5 años (crudo) y 2 de 5 (con offset de prevalencia): el modelo ordena bien las celdas pero subestima la magnitud.
- **$\mathcal{M}_1$ frente a $\mathcal{M}_0$:** AUC-ROC 0.9217 vs 0.9113; Brier global 0.01094 vs 0.01227; PR-AUC 0.3532 vs 0.2786. La capacidad de carga dependiente del hábitat ($\mathcal{M}_1$) supera de forma consistente a la constante ($\mathcal{M}_0$) en las tres métricas probabilísticas fuera de muestra.
- **Persistencia:** el SDM estático logra un Brier de persistencia interanual mayor que $\mathcal{M}_1$ (0.4870 vs 0.3370).
- **Muestras pequeñas:** varias ecorregiones tienen < 10 presencias, por lo que su AUC (0.5 por construcción cuando no hay presencias) no es informativo.

---

## 8. Manifiesto de reproducibilidad

### 8.1 Hashes SHA-256

| Archivo / artefacto | Descripción | SHA-256 |
| --- | --- | --- |
| `calibration_set_2014_2019.csv` | Dataset de calibración (2014–2019) | `63b47915032fe88874c8b0fa77a4ce204c0ff12f04ed67613a6c276e48e8a427` |
| `evaluation_set_2020_2024.csv` | Dataset de evaluación ciega (2020–2024) | `5bcd7a82b1ed35c9cd91b8d939fe86a2e56cff03a577515ed951cae816cc1b61` |
| `theta_compatible_ensemble.csv` | Ensamble $\mathcal{M}_1$ (Θ_comp) | `1e7721266ea89fd77234ddc897261e33627b66874a80d164c43e4f55cbe6abc7` |
| `theta_compatible_ensemble_M0.csv` | Ensamble $\mathcal{M}_0$ (Θ_comp) | `4f006525fef5a663c7083193ef084ca2d2f55ffd9f3a6cbcc34296a25da56c37` |
| `predictions_2020_2024_ensemble.csv` | Predicciones espaciales de ensamble 2020–2024 | `908ff4f6ac164d5aa5757405c5f927ab9d19c2771b165ae2932dfe5229680e95` |
| `sdm_predictions_2020_2024.csv` | Predicciones del SDM estático | `3fe6c7ca5b68d07560500a26e084deb17d46a3629691860aba66e9d67541979c` |
| `joro_hpc` | Binario de simulación y calibración (`joro_hpc`) | `467bc7b8a9bfef396e1036c3d08953649ecf5276cf57d4f12949e149c70b2f69` |
| `joro_evaluate` | Binario de validación fuera de muestra (`joro_evaluate`) | `aba238265d53c25ed1ac22d411b4d52452415f2cdeb5a658977a3ad1360e45a1` |
| `joro_hpc.def` | Receta del contenedor Apptainer | `ddc1e272b749049f06f3cf65e5fa5b5abe8e1742142c74fa29bbcf55d6b55198` |

### 8.2 Entorno de ejecución

- **Kernel Linux:** `7.3.3-1-default x86_64`.
- **Acelerador:** Intel(R) Arc(TM) A380 Graphics | Intel(R) oneAPI Unified Runtime over Level-Zero | GPU.
- **Procesador:** 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz.
- **Semillas:** 1001, 2002, 3003, 4004, 5005 (recocido simulado) y Philox 4x32-10 (LPDM SYCL).
- **Contenedor:** `containers/joro_hpc.def` (Apptainer/Singularity).
- **Procedimiento paso a paso:** `docs/RUNBOOK.md`.
