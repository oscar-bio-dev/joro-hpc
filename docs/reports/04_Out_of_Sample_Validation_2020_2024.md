# JORO-HPC — Validación ciega fuera de muestra (2020–2024) y contraste de las hipótesis H1 y H2

_Generado automáticamente el 2026-10-19 20:07:11 por `scripts/python/generate_validation_report.py` a partir de los artefactos de la corrida; ninguna cifra de resultados está escrita a mano._

- **Plataforma de cómputo:** Intel(R) Arc(TM) A380 Graphics | Intel(R) oneAPI Unified Runtime over Level-Zero | GPU + 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz (16 hilos OpenMP, AVX-512).
- **Compilador:** Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724); banderas `-O3 -fsycl -qopenmp -march=rocketlake -std=c++20`.
- **Ventana de evaluación:** 2020–2024 (**199,602 filas celda-año**, **2,442 presencias observadas**); ventana de calibración: 2014–2019 (239,502 filas).
- **Protocolo de integridad:** los parámetros de JORO-HPC ($\mathcal{M}_1$ y $\mathcal{M}_0$) y de los comparadores se congelaron con datos 2014–2019; ninguna fila de 2020–2024 participó en la calibración.
- **Pruebas unitarias:** `ctest` 4/4 (ver `docs/RUNBOOK.md`).

---

## 1. Protocolo de aislamiento y desbloqueo controlado

1. **Desbloqueo controlado:** el ejecutable `joro_evaluate` habilita la lectura de los tensores ERA5 2020–2024 (`era5_uvwt_202X_05.nc`, `era5_sfc_202X_05.nc`, `era5_daily_t2m_202X.nc`) y de `evaluation_set_2020_2024.csv` mediante la variable de entorno `JORO_UNLOCK_EVALUATION=1` (alias heredado: `JORO_UNLOCK_PHASE5`). Sin ella, el lector rechaza cualquier año posterior a 2019.
2. **Cero reajuste paramétrico:** se propagan los estimadores MAP y el ensamble completo $\Theta_{\text{comp}}$ congelados: **439** miembros para $\mathcal{M}_1$ y **74** para $\mathcal{M}_0$.
3. **Simulación continua de 11 años (2014–2024):** el acoplamiento LPDM–ABM se inicializa en 2014 con la cohorte semilla de Georgia y se propaga sin interrupción por los 6 años de calibración y los 5 de evaluación, con los tensores ERA5 reales de cada año en la GPU y las series térmicas diarias de 365 días en la CPU.
4. **Tiempo de ejecución:** la corrida completa de `joro_evaluate` (ensambles de $\mathcal{M}_1$ y $\mathcal{M}_0$, comparadores y exportación de métricas) tomó **26.56 s** en pared; propagar un miembro de $\mathcal{M}_1$ cuesta ≈ 0.03 s.

---

## 2. Comparadores nulos bajo particiones idénticas

1. **Comparador espacial para H2 — kernel exponencial isotrópico $K_0(d \mid \lambda)$:**
   $$K_0(d \mid \lambda) = \frac{1}{\lambda} \exp\left(-\frac{d}{\lambda}\right), \qquad d \ge 0$$
   - Sustituye el transporte 3D de ERA5 y la turbulencia de Langevin por dispersión radial simétrica $O(1)$ (`SpatialGridLookup`).
   - Conserva el reloj de grados-día $Q_{10}$, el daño térmico $D_{j,t}$, la capacidad de carga $K_i$ y el submodelo de detección.
   - **Calibración estricta en 2014–2019:** $\lambda^* = 1.5\text{ km}$ (congelado para 2020–2024).
2. **Comparador climático para H1 — SDM correlativo estático (regresión logística):**
   $$\operatorname{logit}(P(Y_{i,t}=1)) = \beta_0 + \beta_1 T_{\min,\text{inv}} + \beta_2 T_{\text{med},\text{ver}} + \beta_3 Z_{\text{imp}} + \beta_4 Z_{\text{borde}} + \beta_5 Z_{\text{dosel}} + \beta_6 H_i + \beta_7 R_i + \beta_8 \log(1 + E_{i,t}^{\text{Araneae}})$$
   - Entrenado por máxima verosimilitud con las filas 2014–2019; carece de memoria multianual de daño ($D_{j,t}$) y de dispersión de propágulos.
   - Coeficientes (2014–2019): intercepto +0.0495; `w_tmin` +0.2658; `s_tmean` -0.4786; `z_imp` +0.6759; `z_borde` +1.8578; `z_dosel` +0.9285; `pop_density_h` +1.0575; `road_density_r` +0.0646; `log_effort` +1.6441.

---

## 3. Desempeño probabilístico global y por año

### Tabla 1: Desempeño global fuera de muestra (2020–2024, 199,602 filas celda-año)

| Modelo | Brier global | Brier balanc. (350 km) | Log-Loss | AUC-ROC | PR-AUC | Pendiente calib. |
|---|---|---|---|---|---|---|
| JORO-HPC $\mathcal{M}_1$ (MAP) | 0.01079 | 0.43724 | 0.0425 | 0.9712 | 0.4475 | 0.009 |
| JORO-HPC $\mathcal{M}_1$ (mediana del ensamble) | 0.01107 | 0.45027 | 0.0489 | 0.9675 | 0.4168 | 0.012 |
| JORO-HPC $\mathcal{M}_0$ ($K_0$ constante, MAP) | 0.01069 | 0.43500 | 0.0432 | 0.9678 | 0.6046 | 0.014 |
| Kernel exponencial isotrópico (H2) | 0.01113 | 0.45382 | 0.0661 | 0.8884 | 0.5057 | 0.023 |
| SDM correlativo estático (H1) | 0.01159 | 0.43631 | 0.0634 | 0.9043 | 0.1619 | 0.008 |

_El AUC-ROC se calcula con la corrección de empates de Mann–Whitney (rangos promedio); el Log-Loss recorta las probabilidades a $[10^{-6}, 1-10^{-6}]$._

### Tabla 2: Desglose anual (2020–2024)

| Año | Presencias | Brier $\mathcal{M}_1$ MAP | Brier kernel isot. | Brier SDM | AUC $\mathcal{M}_1$ MAP | AUC kernel isot. | AUC SDM |
|---|---|---|---|---|---|---|---|
| **2020** | 166 | 0.00369 | 0.00367 | 0.00393 | 0.9834 | 0.9815 | 0.9657 |
| **2021** | 303 | 0.00654 | 0.00663 | 0.00681 | 0.9866 | 0.9727 | 0.9567 |
| **2022** | 463 | 0.01008 | 0.01037 | 0.01132 | 0.9733 | 0.9259 | 0.9263 |
| **2023** | 606 | 0.01354 | 0.01398 | 0.01404 | 0.9663 | 0.8788 | 0.9183 |
| **2024** | 904 | 0.02009 | 0.02098 | 0.02185 | 0.9554 | 0.8293 | 0.8921 |

---

## 4. Cinemática y geometría del frente de invasión (prueba de H2)

### Tabla 3: Radio P95 y azimut del frente respecto al epicentro (Braselton, GA)

| Año | P95 obs (km) | P95 $\mathcal{M}_1$ (km) | P95 kernel isot. (km) | Azimut obs (°) | Azimut $\mathcal{M}_1$ (°) | Azimut kernel (°) | Error ang. $\mathcal{M}_1$ (°) | Error ang. kernel (°) |
|---|---|---|---|---|---|---|---|---|
| **2020** | 69.5 | 122.3 | 65.4 | 258.7 | 86.3 | 89.5 | 172.4 | 169.2 |
| **2021** | 110.5 | 155.2 | 70.8 | 255.1 | 86.7 | 93.8 | 168.4 | 161.4 |
| **2022** | 178.1 | 181.6 | 76.6 | 273.1 | 86.9 | 93.4 | 173.8 | 179.8 |
| **2023** | 182.9 | 202.5 | 77.5 | 306.7 | 95.3 | 99.0 | 148.6 | 152.4 |
| **2024** | 234.7 | 228.6 | 81.6 | 322.6 | 95.1 | 98.5 | 132.5 | 135.9 |

- **Error absoluto medio en la fecha de primera llegada por celda:** $\mathcal{M}_1$: **1.43 años**; kernel isotrópico: **1.64 años**; SDM estático: **1.83 años**.
- El frente se mide estrictamente sobre las celdas con colonización activa del ABM (`adult_density_per_cell > 2.0`); el transporte basal por carreteras actúa sobre la ocupación $\psi$, no sobre el radio del frente biológico.

---

## 5. Persistencia térmica y transferencia por ecorregiones (prueba de H1)

### Tabla 4: Restricción térmica invernal y transferencia ecorregional

| Métrica | JORO-HPC $\mathcal{M}_1$ | JORO-HPC $\mathcal{M}_0$ | Kernel isotrópico | SDM estático |
|---|---|---|---|---|
| **Tasa de falsos positivos al norte (>38°–40°N, $\tau^*=0.0012$)** | 0.00 % | 0.00 % | 0.00 % | 9.33 % |
| **Brier de persistencia interanual ($\ge 2$ años)** | 0.4683 | 0.4590 | 0.4830 | 0.4554 |
| **AUC ecorregión vista (bloque 5, Piedmont)** | 0.9651 | 0.9750 | 0.9229 | 0.9164 |
| **Brier ecorregión vista (bloque 5)** | 0.02201 | 0.02178 | 0.02267 | 0.02237 |
| **AUC ecorregiones nuevas (bloques 2, 7–12)** | 0.9055 | 0.8541 | 0.5350 | 0.8239 |
| **Brier ecorregiones nuevas** | 0.00214 | 0.00215 | 0.00224 | 0.00329 |

---

## 6. Propagación de incertidumbre y cobertura empírica

### Tabla 5: Cobertura agregada por año del ensamble $\Theta_{\text{comp}}$ (439 miembros)

Suma anual, sobre todas las celdas, de la probabilidad predicha de detección (percentiles 5, 50 y 95 del ensamble) frente al número de presencias observadas.

| Año | Presencias observadas | Suma P5 | Suma mediana | Suma P95 | ¿Observado dentro del intervalo 90 %? |
|---|---|---|---|---|---|
| **2020** | 166 | 2.6 | 40.8 | 70.7 | No |
| **2021** | 303 | 3.7 | 61.9 | 108.2 | No |
| **2022** | 463 | 3.8 | 79.7 | 136.0 | No |
| **2023** | 606 | 3.7 | 91.9 | 152.3 | No |
| **2024** | 904 | 4.4 | 111.7 | 182.4 | No |

- **Cobertura agregada:** 0 de 5 años con el total observado dentro del intervalo P5–P95.
- **Subpredicción del total anual:** las presencias observadas superan a la mediana del ensamble en un factor 4.1–8.1; el ensamble discrimina bien (AUC) pero no reproduce la magnitud absoluta de las detecciones (ver pendiente de calibración en la Tabla 1).
- **Ancho medio de banda $P_{95} - P_{5}$ por celda:** 0.00316.

---

## 7. Criterios de decisión y dictamen sobre H1 y H2

Los dictámenes se calculan con criterios fijos, declarados antes de leer los resultados: **CONFIRMADA** si se cumplen todos; **SOPORTADA CON SALVEDADES** si se cumple ≥ 75 %; **PARCIALMENTE SOPORTADA** si ≥ 50 %; **NO SOPORTADA** en otro caso.

### Dictamen sobre H1: restricción térmica y memoria fisiológica (M1 frente al SDM estático)

| # | Criterio | Resultado | Evidencia |
|---|---|---|---|
| 1 | AUC-ROC global de M1 > SDM | ✔ | 0.9712 vs 0.9043 |
| 2 | Log-Loss global de M1 < SDM | ✔ | 0.0425 vs 0.0634 |
| 3 | Brier global de M1 < SDM | ✔ | 0.01079 vs 0.01159 |
| 4 | AUC anual de M1 > SDM en los 5 años | ✔ | 2020: 0.983/0.966, 2021: 0.987/0.957, 2022: 0.973/0.926, 2023: 0.966/0.918, 2024: 0.955/0.892 |
| 5 | Falsos positivos al norte (>38°N) de M1 < SDM | ✔ | 0.00 % vs 9.33 % |
| 6 | AUC en ecorregiones nuevas de M1 > SDM | ✔ | 0.9055 vs 0.8239 |
| 7 | Brier en ecorregiones nuevas de M1 < SDM | ✔ | 0.00214 vs 0.00329 |
| 8 | Brier de persistencia interanual de M1 < SDM | ✘ | 0.4683 vs 0.4554 |

> **DICTAMEN H1: SOPORTADA CON SALVEDADES** (7 de 8 criterios).

### Dictamen sobre H2: dispersión atmosférica LPDM 3D (M1 frente al kernel isotrópico)

| # | Criterio | Resultado | Evidencia |
|---|---|---|---|
| 1 | Error absoluto del P95 del frente en 2024: M1 < kernel | ✔ | 6.1 km vs 153.2 km |
| 2 | Error absoluto medio del P95 (2020–2024): M1 < kernel | ✔ | 25.3 km vs 80.8 km |
| 3 | AUC-ROC global de M1 > kernel | ✔ | 0.9712 vs 0.8884 |
| 4 | Log-Loss global de M1 < kernel | ✔ | 0.0425 vs 0.0661 |
| 5 | Brier global de M1 < kernel | ✔ | 0.01079 vs 0.01113 |
| 6 | PR-AUC global de M1 > kernel | ✘ | 0.4475 vs 0.5057 |
| 7 | Error angular medio de M1 menor que el del kernel en ≥ 10° | ✘ | 159.2° vs 159.7° |
| 8 | MAE de la fecha de primera llegada: M1 < kernel | ✔ | 1.43 vs 1.64 años |

> **DICTAMEN H2: SOPORTADA CON SALVEDADES** (6 de 8 criterios).

### Hallazgos que matizan los dictámenes

- **Dirección del frente:** tanto $\mathcal{M}_1$ (159° de error medio) como el kernel (160°) fallan por más de 130° el azimut observado; el LPDM no mejora la dirección de expansión con los datos y la calibración actuales.
- **Calibración absoluta:** la pendiente de calibración de $\mathcal{M}_1$ es 0.009 (ideal 1) y el ensamble cubre el total observado en 0 de 5 años: el modelo ordena bien las celdas pero subestima la magnitud.
- **$\mathcal{M}_1$ frente a $\mathcal{M}_0$:** AUC-ROC 0.9712 vs 0.9678; Brier global 0.01079 vs 0.01069; PR-AUC 0.4475 vs 0.6046. La capacidad de carga dependiente del hábitat ($\mathcal{M}_1$) no supera de forma consistente a la constante ($\mathcal{M}_0$).
- **Persistencia:** el SDM estático logra un Brier de persistencia interanual menor que $\mathcal{M}_1$ (0.4554 vs 0.4683).
- **Muestras pequeñas:** varias ecorregiones tienen < 10 presencias, por lo que su AUC (0.5 por construcción cuando no hay presencias) no es informativo.

---

## 8. Manifiesto de reproducibilidad

### 8.1 Hashes SHA-256

| Archivo / artefacto | Descripción | SHA-256 |
|---|---|---|
| `calibration_set_2014_2019.csv` | Dataset de calibración (2014–2019) | `7acd442f1d08c22c8f41f8d3dff406bf75abc7afb6912c8d26a0a93e376e704b` |
| `evaluation_set_2020_2024.csv` | Dataset de evaluación ciega (2020–2024) | `c1fa45ee94a3965b9c45a3cbe55f7296e5753eae5e9352d29f2d1626205d603e` |
| `theta_compatible_ensemble.csv` | Ensamble $\mathcal{M}_1$ (Θ_comp) | `12061119974bdd46443c76c5a51bb3533fa1ec1493e89662342e1d96a128a99e` |
| `theta_compatible_ensemble_M0.csv` | Ensamble $\mathcal{M}_0$ (Θ_comp) | `cc26bcc89301c24ef5b35cb96ccae8c38432a1ccacbfb9cdcab5bba34397dbdb` |
| `predictions_2020_2024_ensemble.csv` | Predicciones espaciales de ensamble 2020–2024 | `d5010d3e65a8ede26abe5b1cc2548cd3f518d3f2a01e11c98080c6d028e0e7d2` |
| `sdm_predictions_2020_2024.csv` | Predicciones del SDM estático | `30c6d3dbf9105da8c78ede8c4c46e807a5892899787be539a970e095cdca1d71` |
| `joro_hpc` | Binario de simulación y calibración (`joro_hpc`) | `ebc75853047ffe3a36f30347a02dd0fe3ba75e26ec3ea2523f2e30517ac88ba6` |
| `joro_evaluate` | Binario de validación fuera de muestra (`joro_evaluate`) | `adca67fa19febeb6fea44c809512ea3efd3e80466b5265f36c5f09eba414fbb2` |
| `joro_hpc.def` | Receta del contenedor Apptainer | `ddc1e272b749049f06f3cf65e5fa5b5abe8e1742142c74fa29bbcf55d6b55198` |

### 8.2 Entorno de ejecución

- **Kernel Linux:** `7.2.9-1-default x86_64`.
- **Acelerador:** Intel(R) Arc(TM) A380 Graphics | Intel(R) oneAPI Unified Runtime over Level-Zero | GPU.
- **Procesador:** 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz.
- **Semillas:** 1001, 2002, 3003, 4004, 5005 (recocido simulado) y Philox 4x32-10 (LPDM SYCL).
- **Contenedor:** `containers/joro_hpc.def` (Apptainer/Singularity).
- **Procedimiento paso a paso:** `docs/RUNBOOK.md`.

