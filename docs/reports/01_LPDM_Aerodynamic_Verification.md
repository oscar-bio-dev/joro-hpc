# JORO-HPC — Verificación del LPDM 3D y del acoplamiento aerodinámico

*Informe de verificación numérica del módulo de dispersión lagrangiana (LPDM). Todas las cifras provienen de la ejecución de `ctest` y de `joro_hpc --engine=both --no-calibration` sobre la versión del repositorio que acompaña este documento (comandos de reproducción en la sección 8).*

- **Plataforma:** Intel Arc A380 (SYCL / Level-Zero, controlador 1.15.39122) y CPU Intel Core i9-11900K (OpenMP, AVX-512).
- **Alcance:** verificación (¿se resuelven correctamente las ecuaciones?). La validación contra observaciones se trata en `04_Out_of_Sample_Validation_2020_2024.md`.

---

## 1. Modelo verificado

El LPDM integra la posición 3D de super-partículas (propágulos de *T. clavata* en baloneo) sobre la temporada fenológica de mayo–julio (92 días = 2,208 h/año):

- **Advección horizontal y cuantización `FP16`:** viento ERA5 interpolado tetralinealmente en 12 niveles isobáricos (1000–700 hPa), almacenado en `FP16` (`Wind4Half`, 8 bytes/vóxel) en memoria unificada del dispositivo (`Device USM`) y promovido al vuelo a `FP32`.
- **Velocidad vertical:** conversión de $\omega$ (Pa/s) a $w$ (m/s) con la relación hidrostática y la temperatura virtual del nivel.
- **Turbulencia vertical:** integración exacta de Ornstein–Uhlenbeck con deriva *well-mixed* de Thomson (1987) acotada dentro de la capa límite planetaria (altura $h$ = BLH).
- **Sedimentación araña–seda y compuerta de despegue:** velocidad terminal $v_s(M, L_s, \chi)$ dependiente de la masa, la longitud de seda y la postura (`calculate_settling_velocity_fast`), y compuerta fenológico-convectiva de despegue ($u_{10} < 3.0\text{ m/s}$, ventana diurna 14–22 UTC y término convectivo $S_{\text{conv}}$).
- **Fronteras:** suelo absorbente (aterrizaje). En la sub-prueba A ($v_s = 0$) el techo de la capa límite es reflectante; en las corridas de producción el techo es abierto (`configs/default_mvp.toml`).

---

## 2. Condición *well-mixed* de Thomson (sub-prueba A, $v_s = 0$)

Partículas: 10 000; duración: 168 h (30 240 pasos, $\Delta t = 20$ s); BLH = 1 200 m; fronteras reflectantes. Bajo la condición *well-mixed*, la distribución vertical estacionaria debe ser uniforme (1 000 partículas por bin).

| Bin (m) | 0–120 | 120–240 | 240–360 | 360–480 | 480–600 | 600–720 | 720–840 | 840–960 | 960–1080 | 1080–1200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Partículas | 985 | 1006 | 1033 | 982 | 1007 | 976 | 1030 | 981 | 1003 | 997 |

- **Estadístico $\chi^2$ = 3.578 con 9 g.l.** El valor crítico al 5 % es 16.92: no se rechaza la uniformidad.
- El umbral de aceptación codificado en la prueba es $\chi^2 < 60$ (más laxo que el crítico estadístico); el resultado observado lo cumple con holgura.

---

## 3. Dispersión con sedimentación (sub-prueba B, $v_s > 0$)

Juveniles en baloneo con $M = 0.6$ mg, $L_s = 1.2$ m, $\chi = 0.85$; 10 000 partículas; frontera inferior absorbente.

| Magnitud | Valor |
| --- | --- |
| Velocidad de sedimentación $v_s$ | 0.2947 m/s |
| Tasa de asentamiento | 100 % (10 000 / 10 000) |
| Distancia media de salto | 3.082 km |
| Mediana (P50) | 2.051 km |
| P90 | 6.750 km |
| P99 | 16.746 km |

La distribución de saltos es de cola larga (media > mediana), coherente con un régimen de dispersión dominado por unos pocos eventos de viento fuerte en la capa límite.

---

## 4. Convergencia débil de Euler–Maruyama

Segundo momento de la velocidad turbulenta, $E[w'^2(T)]$ (valor teórico 0.25), frente a una referencia con $\Delta t_{\text{ref}} = 0.05$ s ($E = 0.24976$):

| $\Delta t$ (s) | Pasos | $E[w'^2(T)]$ | Error débil absoluto |
| --- | --- | --- | --- |
| 2.00 | 2 | 0.250202 | 4.42 × 10⁻⁴ |
| 1.00 | 4 | 0.249998 | 2.38 × 10⁻⁴ |
| 0.50 | 8 | 0.249817 | 5.72 × 10⁻⁵ |
| 0.25 | 16 | 0.248843 | 9.17 × 10⁻⁴ |

> [!NOTE]
> Todos los errores son < 10⁻³, es decir, del orden del ruido de muestreo Monte Carlo. El error **no** decrece monótonamente (el de $\Delta t = 0.25$ s es el mayor), por lo que esta prueba **no permite estimar un orden de convergencia**; sólo acota el sesgo del integrador. El criterio codificado (`error(Δt fino) < 0.25`) es muy permisivo. Una prueba de orden requeriría más réplicas Monte Carlo o una solución analítica de referencia.

---

## 5. Equivalencia CPU ↔ GPU y Precisión `FP16` ↔ `FP32`

### 5.1 Prueba unitaria (`test_cpu_gpu_equivalence`)

| Caso | CPU | GPU (`FP16` USM) |
| --- | --- | --- |
| Viento sintético, 200 partículas: aterrizadas | 99 | 95 |
| Diferencia en la tasa de aterrizaje | 0.02 | — |
| Campo ERA5 real (6 h): tasa de aterrizaje | 0.98475 | 0.98925 |
| Distancia media (km) | 3.313 | 3.300 |
| **Sub-prueba 3 (`FP16` vs `FP32`, misma semilla Philox):** error relativo en cuantiles $P_{50}/P_{90}/P_{99}$ | `< 0.5 %` (cota `< 2.0 %`) | **✔ APROBADA** |

Los motores no son idénticos bit a bit (generadores y orden de operaciones distintos), pero coinciden estadísticamente en tasa de aterrizaje y distancia media, y la cuantización `FP16` preserva los cuantiles de salto dentro del ruido Monte Carlo (`< 0.5 %`).

### 5.2 Ciclo continental acoplado ABM–LPDM 2014–2019

| Año | Propágulos CPU | Propágulos GPU | Diferencia | Aterrizaje CPU / GPU | Distancia media CPU / GPU (km) |
| --- | --- | --- | --- | --- | --- |
| 2014 | 1 602 399 | 1 602 399 | 0.0 % | 99.96 % / 99.97 % | 3.38 / 3.39 |
| 2015 | 2 206 999 | 2 207 959 | +0.04 % | 99.96 % / 99.97 % | 3.39 / 3.39 |
| 2016 | 3 246 150 | 3 135 239 | −3.4 % | 99.96 % / 99.97 % | 3.39 / 3.39 |
| 2017 | 4 408 529 | 4 144 959 | −6.0 % | 99.96 % / 99.97 % | 3.38 / 3.39 |
| 2018 | 5 775 547 | 5 314 263 | −8.0 % | 99.96 % / 99.96 % | 3.38 / 3.39 |
| 2019 | 7 280 919 | 6 590 530 | −9.5 % | 99.96 % / 99.96 % | 3.38 / 3.38 |

- **Conservación 1:1:** en los 12 ciclos, los propágulos liberados al LPDM igualan exactamente a los generados por el ABM (0 pérdidas).
- **Divergencia demográfica acumulada:** las trayectorias de CPU y GPU son estadísticamente equivalentes por año (aterrizaje y distancia), pero las llegadas estocásticas difieren y el ABM las amplifica año tras año; la población propagada difiere hasta un 9.5 % en 2019. Es una propiedad esperable de sistemas estocásticos acoplados, y debe tenerse en cuenta al comparar corridas entre dispositivos: **las calibraciones deben ejecutarse siempre con el mismo motor** (GPU, por defecto).

---

## 6. Rendimiento del kernel

| Motor | Tiempo de kernel 2019 | Observación |
| --- | --- | --- |
| CPU (OpenMP + AVX-512) | 0.179 s | todas las partículas integradas |
| GPU (SYCL, Arc A380) | 0.0016 s | despacho con tope adaptativo de super-partículas |

> [!WARNING]
> El *throughput* en p/s que imprime el simulador para la GPU (hasta ~3 × 10⁹ p/s) cuenta propágulos **representados**, no super-partículas integradas realmente (el despacho aplica un tope adaptativo). No debe compararse con el *throughput* de la CPU como si fuera trabajo equivalente.

---

## 7. Conclusiones y limitaciones

1. La condición *well-mixed* y la dispersión con sedimentación se verifican con las pruebas del repositorio ($\chi^2 = 3.58$; $v_s = 0.295$ m/s; salto medio 3.08 km).
2. CPU y GPU son estadísticamente equivalentes en las magnitudes del LPDM por año; la divergencia demográfica acumulada (hasta 9.5 % en 2019) es inherente al acoplamiento estocástico.
3. La verificación de convergencia de Euler–Maruyama es sólo una cota de sesgo; no demuestra un orden de convergencia.
4. Las pruebas usan escenarios sintéticos y ventanas de 168 h; la validación frente a datos de campo está en el informe 04.

## 8. Reproducción

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/test_well_mixed && ./build/test_euler_maruyama_conv && ./build/test_cpu_gpu_equivalence
./build/joro_hpc --engine=both --no-calibration
```
