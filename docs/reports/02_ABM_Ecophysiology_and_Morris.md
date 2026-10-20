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

Diseño: $k$ = 11 parámetros, $r$ = 10 trayectorias aleatorias, $p$ = 4 niveles, 120 evaluaciones del modelo acoplado (3.92 s en total). Salidas: población adulta final y radio del frente de invasión.

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

> [!WARNING]
> **Estos diagnósticos no se usan como resultados de este informe.**
> - La *Tabla 2* del transecto (30–44 °N: $T_{\min}$, daño $D$, supervivencia, persistencia) está **escrita como valores fijos en el código** (`src/main.cpp`); no se calcula con el ABM ni con ERA5. El simulador la rotula ahora como «escenario de referencia prescrito».
> - La *Tabla 1* (saturación $N/K$ en el epicentro) se calcula con el ABM, pero en la versión actual devuelve 0 adultas en la celda testigo seleccionada, por lo que no es informativa.
>
> Sustituirlos por un experimento real (cohortes testigo por latitud con el forzamiento térmico diario y las funciones de daño del ABM) queda como trabajo pendiente (véase `docs/RUNBOOK.md`, «Problemas conocidos»). La evidencia de la restricción térmica septentrional en este repositorio es la **validación fuera de muestra** (informe 04: 0.00 % de falsos positivos al norte de 38 °N frente a 9.33 % del SDM estático), no el transecto.

---

## 5. Implicaciones para la calibración

1. Calibrar $T_c$, $\alpha$ y $\beta$ con prioridad: gobiernan el frente y la supervivencia invernal.
2. Fijar $F$ y $Q_{10}$ en sus valores biológicos de referencia (450 huevos/saco; 2.2).
3. En $\mathcal{M}_3$, $\gamma_1$–$\gamma_3$ afectan poco al frente y mucho a la abundancia local: su identificabilidad depende del esfuerzo observacional (véase informe 03, correlaciones en $\Theta_{\text{comp}}$).

## 6. Limitaciones

- $r$ = 10 trayectorias de Morris: jerarquización cualitativa.
- Los índices dependen del rango de cada prior; los rangos son cotas físicas, no distribuciones.
- Transecto latitudinal pendiente de implementación real (sección 4).

## 7. Reproducción

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build && cmake --build build -j
./build/test_parameter_recovery
./build/joro_hpc --engine=gpu --morris --no-calibration     # ciclo continental de humo + cribado de Morris
```
