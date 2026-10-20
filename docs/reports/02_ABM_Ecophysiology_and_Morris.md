# JORO-HPC — ABM ecofisiológico: estructura del modelo y cribado de sensibilidad de Morris

_Informe del módulo demográfico (ABM) y del análisis de sensibilidad global previo a la calibración. Las cifras provienen de `joro_hpc --engine=gpu --diagnostics` (120 evaluaciones acopladas) y de `test_parameter_recovery`, sobre la versión del repositorio que acompaña este documento. Comandos de reproducción en la sección 7._

- **Plataforma:** Intel Arc A380 (SYCL) + CPU Intel Core i9-11900K (OpenMP, AVX-512).
- **Ventana térmica y de simulación:** 2014–2019, forzamiento térmico diario ERA5 (365 días × 81 × 101 celdas).

---

## 1. Estructura del ABM

Las cohortes de *T. clavata* (huevo/ooteca → juvenil → adulto) se integran en pasos diarios sobre una malla de 256 020 celdas de 5 km (EPSG:5070).

| Proceso | Formulación | Parámetros (cota de búsqueda en `configs/priors_bounds.toml`) |
|---|---|---|
| Desarrollo térmico | Grados-día con acoplamiento $Q_{10}$ entre $T_{\min,\text{dev}}$ = 13 °C y $T_{\max,\text{dev}}$ = 35 °C | $Q_{10}$ ∈ [2.0, 3.0] |
| Daño invernal por frío | Daño acumulado $D_{j,t}$ cuando la temperatura de microhábitat cae bajo $T_c$; tasa $\alpha$ y exponente $\beta$ | $T_c$ ∈ [−5, 0] °C; $\alpha$ ∈ [0.01, 0.10]; $\beta$ ∈ {1, 2, 3} |
| Reproducción | Fecundidad por saco $F$, dispersión binomial negativa ($\kappa_B$), viabilidad de ootecas | $F$ ∈ [400, 500]; $q_{\text{viab}}$ ∈ [0.20, 0.90] |
| Dispersión aérea | Fracción de juveniles que inician baloneo; sus propágulos pasan al LPDM y regresan como llegadas | $q_{\text{balloon}}$ ∈ [0.10, 0.60] |
| Capacidad de carga $K_i$ | ver abajo | $K_0$, $\gamma$ |

**Capacidad de carga** (acotada a [100, 50 000]):

| Modelo | $K_i$ | Parámetros libres |
|---|---|---|
| $\mathcal{M}_0$ | $K_0$ | 8 |
| $\mathcal{M}_1$ (principal) | $K_0 \exp(\gamma_{\text{hab}} H_i^{\text{hab}})$ | 9 |
| $\mathcal{M}_3$ | $K_0 \exp(\gamma_1 Z_{\text{imp}} + \gamma_2 Z_{\text{borde}} + \gamma_3 Z_{\text{dosel}})$ | 11 |

con el índice de hábitat ortogonalizado $H_i^{\text{hab}} = \operatorname{std}(0.50\,Z_{\text{borde}} + 0.30\,Z_{\text{imp}} + 0.20\,Z_{\text{dosel}})$, centrado y escalado a varianza unitaria sobre todas las celdas.

---

## 2. Verificación de recuperación de parámetros

`test_parameter_recovery` ajusta un parámetro sintético con la función de costo del calibrador y verifica además el enlace de reparametrización de $K_i$ ($\mathcal{M}_0$/$\mathcal{M}_1$) y de los parámetros de ocupación $a_0$, $b_0$:

| Magnitud | Valor |
|---|---|
| Valor inicial (perturbado) | 0.15 |
| Valor verdadero | 0.85 |
| Valor recuperado | 1.081 |
| Costo inicial → final | 0.2291 → 0.0202 |
| Error de recuperación | 0.231 |

> [!NOTE]
> El costo se reduce un 91 %, pero el parámetro no se recupera con exactitud (error 0.23 sobre un valor verdadero de 0.85). La prueba confirma que el calibrador mejora el ajuste y que las reparametrizaciones están bien enlazadas; no demuestra identificabilidad plena. La colinealidad entre parámetros se cuantifica con datos reales en `03_Inverse_Calibration_2014_2019.md`.

---

## 3. Cribado global de sensibilidad de Morris (1991)

Diseño: $k$ = 11 parámetros, $r$ = 10 trayectorias aleatorias, $p$ = 4 niveles, 120 evaluaciones del modelo acoplado (4.35 s en total). Salidas: población adulta final y radio del frente de invasión.

### Tabla 1: Índices $\mu^*$ y $\sigma$

| Parámetro | $\mu^*$ población | $\sigma$ población | $\mu^*$ frente (km) | $\sigma$ frente (km) | Clasificación del código |
|---|---|---|---|---|---|
| `t_c` | 1 394 502 | 3 242 546 | 49.9 | 51.4 | Dominante |
| `alpha_damage` | 1 759 067 | 3 319 375 | **73.0** | 78.4 | Dominante |
| `beta_damage` | 2 714 737 | 5 020 091 | 65.6 | 53.9 | Dominante |
| `k0` | 2 795 363 | 4 648 844 | 3.1 | 7.1 | Dominante |
| `q_viab` | 552 073 | 801 227 | 13.4 | 14.5 | Dominante |
| `q_balloon` | 417 975 | 931 838 | 11.5 | 12.1 | Dominante |
| `gamma3_dosel` | 665 133 | 1 242 904 | 1.1 | 2.5 | Dominante |
| `gamma2_borde` | 472 144 | 793 009 | 2.3 | 3.8 | Dominante |
| `gamma1_imp` | 229 678 | 679 119 | 1.6 | 3.1 | Dominante |
| `q10` | 41 907 | 58 453 | 0.6 | 2.0 | Interacción fuerte |
| `fecundity_F` | 16 863 | 43 107 | 1.1 | 3.3 | Interacción fuerte |

### Lectura

- **Frente de invasión:** domina el bloque de daño por frío ($\alpha$, $\beta$, $T_c$: 50–73 km de efecto elemental medio); $q_{\text{balloon}}$ y $q_{\text{viab}}$ aportan 11–13 km; los $\gamma$ y $K_0$ casi no mueven el frente (< 4 km).
- **Población final:** $\beta$, $K_0$, $\alpha$ y $T_c$ son los más influyentes; $Q_{10}$ y $F$ tienen un efecto 30–160 veces menor, lo que justifica fijarlos en la calibración (los 9 parámetros libres de $\mathcal{M}_1$ no los incluyen).
- **No linealidad:** $\sigma > \mu^*$ en todos los parámetros, es decir, los efectos dependen fuertemente del resto de parámetros (interacciones o no linealidad). Con $r$ = 10 trayectorias los índices tienen incertidumbre apreciable; sirven para **jerarquizar**, no para cuantificar efectos con precisión.
- **Clasificación:** la etiqueta «Dominante» del código se asigna con un umbral relativo y es poco discriminante (9 de 11 parámetros); la jerarquía útil es la de los valores de $\mu^*$.

---

## 4. Diagnósticos de saturación y transecto latitudinal (`--transect`)

Ambos experimentos se simulan con el ABM real (`step_spring_phenology` y `step_summer_autumn_phenology`) y con la serie térmica diaria ERA5 **de cada año** 2014–2019 (modo estricto: sin climatología de respaldo). Todas las cifras se imprimen desde `joro_hpc --diagnostics --engine=gpu` (registro en `logs/diagnostics.log`); ninguna está escrita en el código. Usan el conjunto **nominal** de parámetros demográficos del ABM (`DemographicParameters` por defecto: $T_c$ = −4 °C, $\alpha$ = 0.04, $\beta$ = 2, $F$ = 450, $q_{\text{viab}}$ = 0.65, $q_{\text{balloon}}$ = 0.35, $K_0$ = 5000 con $\gamma_1$–$\gamma_3$ nominales), **no** los estimadores calibrados $\Theta_{\text{comp}}$.

### Tabla 2: Saturación y crecimiento en el epicentro (celda semilla activa, ABM acoplado al LPDM)

La celda testigo es la celda **semilla** más cercana a (34.25 °N, −83.50 °E) (`is_seed == true`). Celda 106645 (34.248°N, -83.519°E) | K_i efectiva = 45965.

| Año | $N_{i,t}$ adultas | $K_i$ | $N/K$ | Supervivencia invernal | Propágulos aéreos | Estado |
|---|---|---|---|---|---|---|
| 2014 | 0 | 45964 | 0.0 % | 0.0 % | 0 | SIN ADULTAS |
| 2015 | 0 | 45964 | 0.0 % | 1.3 % | 0 | SIN ADULTAS |
| 2016 | 50 | 45964 | 0.1 % | 83.1 % | 13 | Crecimiento |
| 2017 | 7436 | 45964 | 16.2 % | 85.7 % | 2907 | Crecimiento |
| 2018 | 3124 | 45964 | 6.8 % | 7.9 % | 1195 | Crecimiento |
| 2019 | 37781 | 45964 | 82.2 % | 100.0 % | 15250 | Crecimiento |

La supervivencia invernal es la de las ootecas de la propia celda (no un agregado continental). La celda semilla pierde casi toda su cohorte inicial en los inviernos de 2014 y 2015 y se repuebla con llegadas del LPDM desde las demás celdas semilla; la saturación $N/K$ alcanza 82.2% en 2019 sin llegar al umbral de 0.85, por lo que la curva de saturación Beverton–Holt todavía está en la fase de crecimiento al terminar la ventana.

### Tabla 3: Celda testigo B (NC/VA, ≈ 37 °N, cohorte de 1,000 ootecas inyectada en 2014)

Celda 142360 (36.980°N, -78.984°E) | K_i efectiva = 21182.

| Año | $N_{i,t}$ adultas | $K_i$ | $N/K$ | Supervivencia invernal | Propágulos aéreos | Estado |
|---|---|---|---|---|---|---|
| 2014 | 0 | 21181 | 0.0 % | 0.0 % | — | SIN ADULTAS |
| 2015 | 0 | 21181 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |
| 2016 | 0 | 21181 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |
| 2017 | 0 | 21181 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |
| 2018 | 0 | 21181 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |
| 2019 | 0 | 21181 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |

La cohorte inyectada en 2014 no sobrevive a ese invierno y la celda no es recolonizada en 2015–2019: el ABM con parámetros nominales y ERA5 no sostiene población a esta latitud en la ventana de calibración.

### Tabla 4: Gradiente latitudinal sur–norte (30–44 °N), celdas reales y cohorte estandarizada de 1,000 ootecas

Cada fila es una celda real de `grid_5km` (la más cercana a la latitud y longitud de referencia); las celdas se simulan **aisladas** (sin LPDM: los propágulos aéreos emigran sin reposición), de modo que el gradiente refleja la barrera térmica local y no la dispersión.

| Latitud | Celda | Zona | $T_{\min}$ media inv. (°C) | $T_{\min}$ extrema (°C) | $\overline{D}_{j,t}$ | Supervivencia de ootecas | Adultas 2019 | Persistencia ($\ge 2$ años) |
|---|---|---|---|---|---|---|---|---|
| 30.0°N | 117596 | FL norte / GA sur | 9.6 | -5.1 | 0.01 | 99.1 % | 14829 | PERSISTENTE |
| 32.0°N | 110109 | Georgia central | 7.2 | -9.3 | 0.38 | 75.2 % | 10484 | PERSISTENTE |
| 34.0°N | 107142 | NE Georgia (epicentro) | 4.7 | -13.9 | 2.53 | 49.9 % | 19745 | PERSISTENTE |
| 36.5°N | 129795 | Carolina del Norte | 0.9 | -17.5 | 26.92 | 0.0 % | 0 | FALLIDA |
| 38.0°N | 140374 | Virginia (Apalaches) | -1.2 | -25.8 | 63.25 | 0.0 % | 0 | FALLIDA |
| 40.0°N | 148957 | Pensilvania central | -3.0 | -23.1 | 107.40 | 0.0 % | 0 | FALLIDA |
| 42.0°N | 152518 | Nueva York (S. Tier) | -6.3 | -26.7 | 295.61 | 0.0 % | 0 | FALLIDA |
| 44.0°N | 156080 | N. York (St. Lawrence) | -9.5 | -35.1 | 418.37 | 0.0 % | 0 | FALLIDA |

- $T_{\min}$: `t2m_min` ERA5 en enero–marzo (media de las medias anuales 2014–2019; la extrema es el mínimo diario del periodo), sin amortiguación de dosel.
- $\overline{D}_{j,t}$: daño acumulado de la cohorte invernante (otoño previo + invierno), media de los años con cohorte viva.
- Supervivencia de ootecas: media de $\sum w_0 e^{-\Delta D}/\sum w_0$ en los años con cohorte viva. Persistencia: ≥ 2 años consecutivos con ≥ 1 adulta.

Serie anual de adultas por celda (otoño de 2014 a 2019):

```text
     30.0°N:       652     12777     14815     14831     14229     14829
     32.0°N:       191      4152     10367     10125      6942     10484
     34.0°N:         0         0        54      5546      3144     19745
     36.5°N:         0         0         0         0         0         0
     38.0°N:         0         0         0         0         0         0
     40.0°N:         0         0         0         0         0         0
     42.0°N:         0         0         0         0         0         0
     44.0°N:         0         0         0         0         0         0
```

> [!IMPORTANT]
> Con los parámetros nominales, la persistencia solo se sostiene hasta ≈ 34 °N; desde 36.5 °N todas las celdas fracasan. Esto contrasta con la expansión observada más al norte (informe 04), de modo que el conjunto nominal es **demasiado severo** (o el $T_{\min}$ ERA5 sin dosel sobreestima el daño). Es un resultado del experimento, no un valor ajustado; la comparación con las observaciones debe hacerse con $\Theta_{\text{comp}}$ y se deja como trabajo futuro.

---

## 5. Implicaciones para la calibración

1. Calibrar $T_c$, $\alpha$ y $\beta$ con prioridad: gobiernan el frente y la supervivencia invernal.
2. Fijar $F$ y $Q_{10}$ en sus valores biológicos de referencia (450 huevos/saco; 2.2).
3. En $\mathcal{M}_3$, $\gamma_1$–$\gamma_3$ afectan poco al frente y mucho a la abundancia local: su identificabilidad depende del esfuerzo observacional (véase informe 03, correlaciones en $\Theta_{\text{comp}}$).

## 6. Limitaciones

- $r$ = 10 trayectorias de Morris: jerarquización cualitativa.
- Los índices dependen del rango de cada prior; los rangos son cotas físicas, no distribuciones.
- El transecto (sección 4) usa parámetros demográficos nominales, cohortes aisladas y una sola semilla de RNG; no está contrastado con observaciones.

## 7. Reproducción

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build && cmake --build build -j
./build/test_parameter_recovery
./build/joro_hpc --diagnostics --engine=gpu     # transecto (--transect) + cribado de Morris (--morris)
```
