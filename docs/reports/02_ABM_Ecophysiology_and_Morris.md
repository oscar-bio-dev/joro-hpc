# JORO-HPC — ABM ecofisiológico: estructura del modelo y cribado de sensibilidad de Morris

*Informe del módulo demográfico (ABM) y del análisis de sensibilidad global previo a la calibración. Las cifras provienen de `joro_hpc --engine=gpu --diagnostics` (120 evaluaciones acopladas) y de `test_parameter_recovery`, sobre la versión del repositorio que acompaña este documento. Comandos de reproducción en la sección 7.*

- **Plataforma:** Intel Arc A380 (SYCL) + CPU Intel Core i9-11900K (OpenMP, AVX-512).
- **Ventana térmica y de simulación:** 2014–2019, forzamiento térmico diario ERA5 (365 días × 81 × 101 celdas).

---

## 1. Estructura del ABM

Las cohortes de *T. clavata* (huevo/ooteca → juvenil → adulto) se integran en pasos diarios sobre una malla de 256 020 celdas de 5 km (EPSG:5070).

| Proceso | Formulación | Parámetros (cota de búsqueda en `configs/priors_bounds.toml`) |
| --- | --- | --- |
| Desarrollo térmico | Grados-día con acoplamiento $Q_{10}$ entre $T_{\min,\text{dev}}$ = 13 °C y $T_{\max,\text{dev}}$ = 35 °C; emergencia logística con umbral $DD_{\text{emerg}} = 0.12$ y pendiente $\kappa_{DD} = 8.0$ | $Q_{10}$ ∈ [2.0, 3.0]; $DD_{\text{emerg}} = 0.12$, $\kappa_{DD} = 8.0$ (fijos) |
| Daño invernal por frío | Daño acumulado $D_{j,t}$ cuando la temperatura de microhábitat (con amortiguación térmica de dosel $\delta_{\text{dosel}} = 0.35\text{ °C/SD}$ e impermeabilidad urbana $\delta_{\text{imp}} = 0.50\text{ °C/SD}$) cae bajo $T_c$; tasa $\alpha$ y exponente $\beta$ | $T_c$ ∈ [−5, 0] °C; $\alpha$ ∈ [0.01, 0.10]; $\beta$ ∈ {1, 2, 3} |
| Reproducción | Fecundidad por saco $F$, dispersión binomial negativa ($\kappa_B$), viabilidad de ootecas | $F$ ∈ [400, 500]; $q_{\text{viab}}$ ∈ [0.20, 0.90] |
| Dispersión aérea | Fracción de juveniles que inician baloneo; sus propágulos pasan al LPDM estacional (mayo–julio, 92 días en `FP16`) y regresan como llegadas | $q_{\text{balloon}}$ ∈ [0.10, 0.60] |
| Capacidad de carga $K_i$ | ver abajo | $K_0$, $\gamma$ |

**Capacidad de carga** (acotada a [100, 50 000]):

| Modelo | $K_i$ | Parámetros libres |
| --- | --- | --- |
| $\mathcal{M}_0$ | $K_0$ | 8 |
| $\mathcal{M}_1$ (principal) | $K_0 \exp(\gamma_{\text{hab}} H_i^{\text{hab}})$ | 9 |
| $\mathcal{M}_3$ | $K_0 \exp(\gamma_1 Z_{\text{imp}} + \gamma_2 Z_{\text{borde}} + \gamma_3 Z_{\text{dosel}})$ | 11 |

con el índice de hábitat ortogonalizado $H_i^{\text{hab}} = \operatorname{std}(0.50\,Z_{\text{borde}} + 0.30\,Z_{\text{imp}} + 0.20\,Z_{\text{dosel}})$, centrado y escalado a varianza unitaria sobre el soporte informativo $S$ ($4,967$ celdas con esfuerzo u observación en 2014–2019; $m_S = 1.1729, s_S = 0.5205$).

---

## 2. Verificación de recuperación de parámetros

`test_parameter_recovery` ajusta un parámetro sintético con la función de costo del calibrador y verifica además el enlace de reparametrización de $K_i$ ($\mathcal{M}_0$/$\mathcal{M}_1$), de los parámetros de ocupación $a_0$, $b_0$ y del kernel gravitacional humano ponderado por red vial ($K_{\text{mixto}}$):

| Magnitud | Valor |
| --- | --- |
| Valor inicial (perturbado) | 0.15 |
| Valor verdadero | 0.85 |
| Valor recuperado | 1.081 |
| Costo inicial → final | 0.2291 → 0.0202 |
| Error de recuperación | 0.231 |

> [!NOTE]
> El costo se reduce un 91 %, pero el parámetro no se recupera con exactitud (error 0.23 sobre un valor verdadero de 0.85). La prueba confirma que el calibrador mejora el ajuste y que las reparametrizaciones están bien enlazadas; no demuestra identificabilidad plena. La colinealidad entre parámetros se cuantifica con datos reales en `03_Inverse_Calibration_2014_2019.md`.

---

## 3. Cribado global de sensibilidad de Morris (1991)

Diseño: $k$ = 11 parámetros, $r$ = 10 trayectorias aleatorias, $p$ = 4 niveles, 120 evaluaciones del modelo acoplado sobre la temporada estacional completa mayo–julio (5.18 s en total sobre la Intel Arc A380 con paso sinóptico de 3 h en `FP16`). Salidas: población adulta final y radio del frente de invasión desde el polígono fundador limpio de Braselton/Hoschton (125 celdas contiguas, radio 37.6 km).

### Tabla 1: Índices $\mu^*$ y $\sigma$

| Parámetro | $\mu^*$ población | $\sigma$ población | $\mu^*$ frente (km) | $\sigma$ frente (km) | Clasificación del código |
| --- | --- | --- | --- | --- | --- |
| `t_c` | 1 356 057 | 2 761 233 | 57.4 | 52.7 | Dominante |
| `alpha_damage` | 1 983 336 | 3 475 478 | **75.4** | 71.9 | Dominante |
| `beta_damage` | 2 806 936 | 4 861 385 | 63.7 | 53.5 | Dominante |
| `k0` | **2 889 106** | 4 573 488 | 2.3 | 6.6 | Dominante |
| `q_viab` | 653 138 | 866 228 | 7.8 | 11.4 | Dominante |
| `q_balloon` | 205 281 | 416 236 | 10.9 | 12.7 | Dominante |
| `gamma3_dosel` | 638 354 | 1 188 512 | 0.4 | 1.3 | Dominante |
| `gamma2_borde` | 586 401 | 947 855 | 5.8 | 16.6 | Dominante |
| `gamma1_imp` | 301 685 | 861 958 | 6.0 | 16.7 | Dominante |
| `fecundity_F` | 18 595 | 46 900 | 0.6 | 1.8 | Interacción fuerte |
| `q10` | 8 084 | 17 291 | 0.0 | 0.0 | Despreciable |

### Lectura

- **Frente de invasión:** domina el bloque de daño por frío ($\alpha$, $\beta$, $T_c$: 57.4–75.4 km de efecto elemental medio); $q_{\text{balloon}}$ y $q_{\text{viab}}$ aportan 7.8–10.9 km; $\gamma_1$ y $\gamma_2$ aportan ~6 km y $K_0$/$\gamma_3$ < 2.5 km.
- **Población final:** $K_0$, $\beta$, $\alpha$ y $T_c$ son los más influyentes; $Q_{10}$ y $F$ tienen un efecto 70–350 veces menor, lo que justifica fijarlos en la calibración (los 9 parámetros libres de $\mathcal{M}_1$ no los incluyen).
- **No linealidad:** $\sigma > \mu^*$ en la mayoría de los parámetros, es decir, los efectos dependen fuertemente del resto de parámetros (interacciones o no linealidad). Con $r$ = 10 trayectorias los índices tienen incertidumbre apreciable; sirven para **jerarquizar**, no para cuantificar efectos con precisión.

---

## 4. Diagnósticos de saturación y transecto latitudinal (`--transect`)

Ambos experimentos se simulan con el ABM real (`step_spring_phenology` y `step_summer_autumn_phenology`), los tensores ERA5 estacionales completos (mayo–julio, 2,208 h/año) y la serie térmica diaria ERA5 **de cada año** 2014–2019 (modo estricto: sin climatología de respaldo). En `v2.0.0`, el experimento carga automáticamente el estimador calibrado $\hat{\theta}_{\text{MAP}}$ de $\mathcal{M}_1$ desde `calibration_summary_M1.json` ($T_c = -0.06\text{ °C}$, $\alpha_s = 0.0849$, $\beta_s = 1$, $K_0 = 13,505$, $\gamma_{\text{hab}} = 1.601$, $q_{\text{balloon}} = 0.350$, $q_{\text{viab}} = 0.390$, $\delta_{\text{dosel}} = 0.35$, $\delta_{\text{imp}} = 0.50$).

### Tabla 2: Saturación y crecimiento en el epicentro (celda semilla activa, ABM acoplado al LPDM)

La celda testigo es la celda **semilla** más cercana a (34.25 °N, −83.50 °E) (`is_seed == true`). Celda 106645 (34.248°N, -83.519°E) | $K_i$ efectiva = 49,781.

| Año | $N_{i,t}$ adultas | $K_i$ | $N/K$ | Supervivencia invernal | Propágulos aéreos | Estado |
| --- | --- | --- | --- | --- | --- | --- |
| 2014 | 0 | 49781 | 0.0 % | 0.0 % | 2 | SIN ADULTAS |
| 2015 | 0 | 49781 | 0.0 % | 0.5 % | 0 | SIN ADULTAS |
| 2016 | 0 | 49781 | 0.0 % | 11.5 % | 1 | SIN ADULTAS |
| 2017 | 52 | 49781 | 0.1 % | 36.8 % | 6 | Crecimiento |
| 2018 | 39 | 49781 | 0.1 % | 0.9 % | 9 | Crecimiento |
| 2019 | 847 | 49781 | 1.7 % | 35.8 % | 262 | Crecimiento |

La supervivencia invernal es la de las ootecas de la propia celda (no un agregado continental). La celda testigo 106645 recibe recolonización aérea desde el resto del clúster semilla de 125 celdas y consolida crecimiento sostenido a partir de 2017–2019.

### Tabla 3: Celda testigo B (NC/VA, ≈ 37 °N, cohorte de 1,000 ootecas inyectada en 2014)

Celda 142360 (36.980°N, -78.984°E) | $K_i$ efectiva = 9,390.

| Año | $N_{i,t}$ adultas | $K_i$ | $N/K$ | Supervivencia invernal | Propágulos aéreos | Estado |
| --- | --- | --- | --- | --- | --- | --- |
| 2014 | 0 | 9389 | 0.0 % | 0.0 % | — | SIN ADULTAS |
| 2015 | 0 | 9389 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |
| 2016 | 0 | 9389 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |
| 2017 | 0 | 9389 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |
| 2018 | 0 | 9389 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |
| 2019 | 0 | 9389 | 0.0 % | — (sin ootecas) | — | SIN ADULTAS |

La cohorte inyectada en 2014 no sobrevive a los inviernos de 2014–2015 (vórtice polar norteamericano) y la celda no es alcanzada por dispersión natural en 2015–2019.

### Tabla 4: Gradiente latitudinal sur–norte (30–44 °N), celdas reales y cohorte estandarizada de 1,000 ootecas

Cada fila es una celda real de `grid_5km` (la más cercana a la latitud y longitud de referencia); las celdas se simulan **aisladas** (sin LPDM: los propágulos aéreos emigran sin reposición), de modo que el gradiente refleja la barrera térmica local y no la dispersión.

| Latitud | Celda | Zona | $T_{\min}$ media inv. (°C) | $T_{\min}$ extrema (°C) | $\overline{D}_{j,t}$ | Supervivencia de ootecas | Adultas 2019 | Persistencia ($\ge 2$ años) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30.0°N | 117596 | FL norte / GA sur | 9.6 | -5.1 | 0.75 | 58.1 % | 6824 | PERSISTENTE |
| 32.0°N | 110109 | Georgia central | 7.2 | -9.3 | 2.73 | 18.2 % | 489 | PERSISTENTE |
| 34.0°N | 107142 | NE Georgia (epicentro) | 4.7 | -13.9 | 4.45 | 11.7 % | 5 | MARGINAL (1 año) |
| 36.5°N | 129795 | Carolina del Norte | 0.9 | -17.5 | 18.60 | 0.0 % | 0 | FALLIDA |
| 38.0°N | 140374 | Virginia (Apalaches) | -1.2 | -25.8 | 28.65 | 0.0 % | 0 | FALLIDA |
| 40.0°N | 148957 | Pensilvania central | -3.0 | -23.1 | 44.29 | 0.0 % | 0 | FALLIDA |
| 42.0°N | 152518 | Nueva York (S. Tier) | -6.3 | -26.7 | 62.77 | 0.0 % | 0 | FALLIDA |
| 44.0°N | 156080 | N. York (St. Lawrence) | -9.5 | -35.1 | 86.74 | 0.0 % | 0 | FALLIDA |

- $T_{\min}$: `t2m_min` ERA5 en enero–marzo (media de las medias anuales 2014–2019; la extrema es el mínimo diario del periodo), sin amortiguación de dosel/impermeabilidad.
- $\overline{D}_{j,t}$: daño acumulado de la cohorte invernante (otoño previo + invierno), media de los años con cohorte viva (incluyendo amortiguación microclimática $\delta_{\text{dosel}} Z_{\text{dosel}} + \delta_{\text{imp}} Z_{\text{imp}}$).
- Supervivencia de ootecas: media de $\sum w_0 e^{-\Delta D}/\sum w_0$ en los años con cohorte viva. Persistencia: ≥ 2 años consecutivos con ≥ 1 adulta.

Serie anual de adultas por celda (otoño de 2014 a 2019):

```text
     30.0°N:       323      1980      4079      5306      1198      6824
     32.0°N:         7        11        77       653        47       489
     34.0°N:         0         0         0         1         0         5
     36.5°N:         0         0         0         0         0         0
     38.0°N:         0         0         0         0         0         0
     40.0°N:         0         0         0         0         0         0
     42.0°N:         0         0         0         0         0         0
     44.0°N:         0         0         0         0         0         0
```

> [!IMPORTANT]
> Con una única celda aislada sin rescate metapoblacional por dispersión aérea LPDM, la persistencia en 2014–2019 (que incluye las olas de frío de 2014 y 2015) se sostiene de forma autónoma en 30–32 °N y marginalmente en 34 °N (`11.7 %` de supervivencia invernal media de ootecas), mientras que en el sistema acoplado ABM+LPDM (Tabla 2) el rescate espacial entre celdas del clúster fundador y los refugios microclimáticos urbanos ($\delta_{\text{imp}} = 0.50\text{ °C/SD}$) sostienen el crecimiento regional.

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
