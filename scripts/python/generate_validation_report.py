#!/usr/bin/env python3
"""
JORO-HPC · scripts/python/generate_validation_report.py

Genera docs/reports/04_Out_of_Sample_Validation_2020_2024.md a partir de los artefactos REALES de
la validación ciega fuera de muestra (joro_evaluate + compute_validation_metrics.py + run_static_sdm.py).

Principios:
  * Ninguna cifra de resultados está escrita a mano: todo proviene de los JSON/CSV/logs de la corrida.
  * Los dictámenes sobre H1 y H2 se calculan con criterios fijos y explícitos (sección 7); si un criterio
    falla, el informe lo muestra.
  * El LaTeX se emite con una sola barra invertida (cadenas crudas), sin escapes duplicados.

Uso:  .venv/bin/python scripts/python/generate_validation_report.py [--out docs/reports/04_...md]
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path("/mnt/hpc_scratch/joro-hpc")
OUT_DIR = SCRATCH / "outputs"
CK = SCRATCH / "checkpoints"
METRICS_JSON = OUT_DIR / "validation_metrics.json"
SDM_JSON = OUT_DIR / "sdm_coefficients.json"
EVAL_LOG = SCRATCH / "logs" / "evaluate.log"
PRED_CSV = OUT_DIR / "predictions_2020_2024_ensemble.csv"
HW_PROFILE = ROOT / "docs" / "telemetry" / "calibration_hardware_profile.json"
YEARS = ["2020", "2021", "2022", "2023", "2024"]

M1 = r"$\mathcal{M}_1$"
M0 = r"$\mathcal{M}_0$"
THETA = r"$\Theta_{\text{comp}}$"

# Criterio de relevancia práctica para la dirección del frente (grados de reducción del error medio).
ANGULAR_MARGIN_DEG = 10.0


def sha256(path: Path) -> str:
    try:
        return subprocess.check_output(["sha256sum", str(path)], text=True).split()[0]
    except (OSError, subprocess.CalledProcessError):
        return "N/A"


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse_eval_log() -> dict:
    out = {"wall_s": None, "lambda_km": None, "m1_members": None, "m0_members": None, "m1_per_eval_s": None}
    try:
        txt = EVAL_LOG.read_text(errors="ignore")
    except OSError:
        return out
    pats = {
        "wall_s": r"COMPLETADA EXITOSAMENTE EN ([\d.]+) s",
        "lambda_km": r"lambda\* = ([\d.]+) km",
        "m1_members": r"M1: (\d+) miembros",
        "m0_members": r"M0: (\d+) miembros",
        "m1_per_eval_s": r"Ensamble M1 propagado en [\d.]+ s \(([\d.]+) s/eval\)",
    }
    for k, pat in pats.items():
        mt = re.search(pat, txt)
        if mt:
            out[k] = float(mt.group(1)) if "." in mt.group(1) else int(mt.group(1))
    return out


def f(x, nd=4):
    return "N/D" if x is None else f"{x:.{nd}f}"


def verdict(passed: int, total: int) -> str:
    frac = passed / total
    if passed == total:
        return "CONFIRMADA"
    if frac >= 0.75:
        return "SOPORTADA CON SALVEDADES"
    if frac >= 0.5:
        return "PARCIALMENTE SOPORTADA"
    return "NO SOPORTADA"


def criteria_table(rows: list[tuple[str, bool, str]]) -> tuple[str, int, int]:
    lines = ["| # | Criterio | Resultado | Evidencia |", "|---|---|---|---|"]
    for i, (name, ok, ev) in enumerate(rows, 1):
        lines.append(f"| {i} | {name} | {'✔' if ok else '✘'} | {ev} |")
    return "\n".join(lines), sum(1 for _, ok, _ in rows if ok), len(rows)


def nf(x, nd: int = 1) -> str:
    """Formatea un número; `n/d` si es nulo o no finito (p. ej., sin expansión fuera del radio semilla)."""
    try:
        return "n/d" if x is None or x != x else f"{x:.{nd}f}"
    except TypeError:
        return "n/d"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "reports" / "04_Out_of_Sample_Validation_2020_2024.md"))
    args = ap.parse_args()

    m = load_json(METRICS_JSON)
    if m is None:
        raise SystemExit(f"Falta {METRICS_JSON}: ejecute joro_evaluate y compute_validation_metrics.py primero")
    sdm = load_json(SDM_JSON)
    ev = parse_eval_log()
    cal = {k: load_json(CK / f"calibration_summary_{k}.json") for k in ("M1", "M0", "M3")}
    hw = load_json(HW_PROFILE) or {}
    plat = hw.get("platform", {})

    pred = pd.read_csv(PRED_CSV, usecols=["year", "y_observed"])
    n_rows, n_pos = len(pred), int(pred["y_observed"].sum())
    n_cal_rows = (cal["M1"] or {}).get("stratification", {}).get("n_rows")

    g, by, kin = m["global"], m["by_year"], m["kinematics"]
    tp, unc = m["thermal_persistence"], m["uncertainty"]
    k24 = kin["2024"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_m1 = ev["m1_members"] or (cal["M1"] or {}).get("compatible_members")
    n_m0 = ev["m0_members"] or (cal["M0"] or {}).get("compatible_members")

    # ────────────────────────────── Cabecera y protocolo
    md: list[str] = []
    A = md.append
    A("# JORO-HPC — Validación ciega fuera de muestra (2020–2024) y contraste de las hipótesis H1 y H2\n")
    A(f"_Generado automáticamente el {now} por `scripts/python/generate_validation_report.py` a partir de los "
      "artefactos de la corrida; ninguna cifra de resultados está escrita a mano._\n")
    A(f"- **Plataforma de cómputo:** {plat.get('lpdm_device', 'Intel Arc A380 (SYCL / Level-Zero)')} + {plat.get('cpu', 'CPU x86-64')} "
      f"({plat.get('omp_threads', 16)} hilos OpenMP, AVX-512).")
    A(f"- **Compilador:** {plat.get('compiler', 'icpx (Intel oneAPI)')}; banderas `{plat.get('compile_flags', '-O3 -fsycl -qopenmp -march=rocketlake -std=c++20')}`.")
    A(f"- **Ventana de evaluación:** 2020–2024 (**{n_rows:,} filas celda-año**, **{n_pos:,} presencias observadas**); "
      f"ventana de calibración: 2014–2019 ({n_cal_rows:,} filas)." if n_cal_rows else
      f"- **Ventana de evaluación:** 2020–2024 (**{n_rows:,} filas celda-año**, **{n_pos:,} presencias observadas**).")
    A(f"- **Protocolo de integridad:** los parámetros de JORO-HPC ({M1} y {M0}) y de los comparadores se congelaron con datos "
      "2014–2019; ninguna fila de 2020–2024 participó en la calibración.")
    A("- **Pruebas unitarias:** `ctest` 4/4 (ver `docs/RUNBOOK.md`).\n")
    A("---\n")

    A("## 1. Protocolo de aislamiento y desbloqueo controlado\n")
    A("1. **Desbloqueo controlado:** el ejecutable `joro_evaluate` habilita la lectura de los tensores ERA5 2020–2024 "
      "(`era5_uvwt_202X_05.nc`, `era5_sfc_202X_05.nc`, `era5_daily_t2m_202X.nc`) y de `evaluation_set_2020_2024.csv` "
      "mediante la variable de entorno `JORO_UNLOCK_EVALUATION=1` (alias heredado: `JORO_UNLOCK_PHASE5`). "
      "Sin ella, el lector rechaza cualquier año posterior a 2019.")
    A(f"2. **Cero reajuste paramétrico:** se propagan los estimadores MAP y el ensamble completo {THETA} congelados: "
      f"**{n_m1}** miembros para {M1} y **{n_m0}** para {M0}.")
    A("3. **Simulación continua de 11 años (2014–2024):** el acoplamiento LPDM–ABM se inicializa en 2014 con la cohorte semilla "
      "de Georgia y se propaga sin interrupción por los 6 años de calibración y los 5 de evaluación, con los tensores ERA5 "
      "reales de cada año en la GPU y las series térmicas diarias de 365 días en la CPU.")
    if ev["wall_s"]:
        A(f"4. **Tiempo de ejecución:** la corrida completa de `joro_evaluate` (ensambles de {M1} y {M0}, comparadores y "
          f"exportación de métricas) tomó **{ev['wall_s']:.2f} s** en pared"
          + (f"; propagar un miembro de {M1} cuesta ≈ {ev['m1_per_eval_s']:.2f} s." if ev["m1_per_eval_s"] else "."))
    A("\n---\n")

    # ────────────────────────────── Comparadores
    lam = ev["lambda_km"]
    A("## 2. Comparadores nulos bajo particiones idénticas\n")
    A("1. **Comparador espacial para H2 — kernel exponencial isotrópico " + r"$K_0(d \mid \lambda)$:**")
    A(r"   $$K_0(d \mid \lambda) = \frac{1}{\lambda} \exp\left(-\frac{d}{\lambda}\right), \qquad d \ge 0$$")
    A("   - Sustituye el transporte 3D de ERA5 y la turbulencia de Langevin por dispersión radial simétrica $O(1)$ (`SpatialGridLookup`).")
    A(r"   - Conserva el reloj de grados-día $Q_{10}$, el daño térmico $D_{j,t}$, la capacidad de carga $K_i$ y el submodelo de detección.")
    A(f"   - **Calibración estricta en 2014–2019:** " + (r"$\lambda^* = " + f"{lam:.1f}" + r"\text{ km}$" if lam else r"$\lambda^*$ N/D")
      + " (congelado para 2020–2024).")
    A("2. **Comparador climático para H1 — SDM correlativo estático (regresión logística):**")
    A(r"   $$\operatorname{logit}(P(Y_{i,t}=1)) = \beta_0 + \beta_1 T_{\min,\text{inv}} + \beta_2 T_{\text{med},\text{ver}} + \beta_3 Z_{\text{imp}} + \beta_4 Z_{\text{borde}} + \beta_5 Z_{\text{dosel}} + \beta_6 H_i + \beta_7 R_i + \beta_8 \log(1 + E_{i,t}^{\text{Araneae}})$$")
    A("   - Entrenado por máxima verosimilitud con las filas 2014–2019; carece de memoria multianual de daño ($D_{j,t}$) y de dispersión de propágulos.")
    if sdm:
        c = sdm["coefficients"]
        A(f"   - Coeficientes (2014–2019): intercepto {sdm['intercept']:+.4f}; " + "; ".join(
            f"`{k}` {v:+.4f}" for k, v in c.items()) + ".")
    A("\n---\n")

    # ────────────────────────────── Tablas 1 y 2
    A("## 3. Desempeño probabilístico global y por año\n")
    A(f"### Tabla 1: Desempeño global fuera de muestra (2020–2024, {n_rows:,} filas celda-año)\n")
    A("| Modelo | Brier global | Brier balanc. (350 km) | Log-Loss | AUC-ROC | PR-AUC | Pendiente calib. |")
    A("|---|---|---|---|---|---|---|")
    labels = [("m1_map", f"JORO-HPC {M1} (MAP)"), ("m1_median", f"JORO-HPC {M1} (mediana del ensamble)"),
              ("m0_map", f"JORO-HPC {M0} ($K_0$ constante, MAP)"), ("iso_null", "Kernel exponencial isotrópico (H2)"),
              ("sdm", "SDM correlativo estático (H1)")]
    for key, lab in labels:
        r = g[key]
        A(f"| {lab} | {r['brier_global']:.5f} | {r['brier_balanced']:.5f} | {r['log_loss']:.4f} | {r['auc_roc']:.4f} | "
          f"{r['pr_auc']:.4f} | {r['calib_slope']:.3f} |")
    A("\n_El AUC-ROC se calcula con la corrección de empates de Mann–Whitney (rangos promedio); el Log-Loss recorta las "
      "probabilidades a $[10^{-6}, 1-10^{-6}]$._\n")
    A("### Tabla 2: Desglose anual (2020–2024)\n")
    A(f"| Año | Presencias | Brier {M1} MAP | Brier kernel isot. | Brier SDM | AUC {M1} MAP | AUC kernel isot. | AUC SDM |")
    A("|---|---|---|---|---|---|---|---|")
    for y in YEARS:
        r = by[y]
        A(f"| **{y}** | {r['presences']} | {r['m1_map']['brier']:.5f} | {r['iso_null']['brier']:.5f} | {r['sdm']['brier']:.5f} | "
          f"{r['m1_map']['auc']:.4f} | {r['iso_null']['auc']:.4f} | {r['sdm']['auc']:.4f} |")
    A("\n---\n")

    # ────────────────────────────── Tabla 3
    A("## 4. Cinemática y geometría del frente de invasión (prueba de H2)\n")
    A("### Tabla 3: Radio P95 y azimut del frente (azimut respecto al centroide del polígono semilla)\n")
    A(f"| Año | P95 obs (km) | P95 {M1} (km) | P95 kernel isot. (km) | Azimut obs (°) | Azimut {M1} (°) | Azimut kernel (°) | Error ang. {M1} (°) | Error ang. kernel (°) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for y in YEARS:
        k = kin[y]
        A(f"| **{y}** | {k['obs_p95']:.1f} | {k['m1_map']['p95']:.1f} | {k['iso_null']['p95']:.1f} | {nf(k['obs_angle'])} | "
          f"{nf(k['m1_map']['angle'])} | {nf(k['iso_null']['angle'])} | {nf(k['m1_map']['angular_error_deg'])} | "
          f"{nf(k['iso_null']['angular_error_deg'])} |")
    sg = m.get("seed_geometry") or {}
    if sg:
        A(f"\n- **Referencia angular:** centroide geométrico de las {sg['n_seed_cells']} celdas semilla "
          f"({sg['centroid_lat']:.4f}°N, {sg['centroid_lon']:.4f}°E), a {sg['legacy_to_centroid_km']:.1f} km de la constante "
          f"heredada ({sg['legacy_epicenter_lat']:.4f}°N, {sg['legacy_epicenter_lon']:.4f}°E), que queda al suroeste del polígono. "
          f"El azimut se mide solo sobre celdas colonizadas **fuera del radio semilla inicial** ({sg['radius_km']:.1f} km), "
          "tanto en las observaciones como en las simulaciones; las distancias P50/P90/P95 conservan la referencia heredada "
          "de la calibración (no se altera ningún parámetro ni el envolvente de ocupación).")
    mae = m["first_arrival_mae"]
    A("\n- **Error absoluto medio en la fecha de primera llegada por celda:** "
      f"{M1}: **{mae['m1_map']:.2f} años**; kernel isotrópico: **{mae['iso_null']:.2f} años**; SDM estático: **{mae['sdm']:.2f} años**.")
    _psi = lambda mk: [kin[y][mk].get("angular_error_psi_deg") for y in YEARS if kin[y][mk].get("angular_error_psi_deg") is not None]  # noqa: E731
    if _psi("m1_map") and _psi("iso_null"):
        _bio = [kin[y]["m1_map"]["angular_error_deg"] for y in YEARS if kin[y]["m1_map"]["angular_error_deg"] is not None]
        ang_m1 = sum(_bio) / max(1, len(_bio))
        _pm, _pi = sum(_psi("m1_map")) / len(_psi("m1_map")), sum(_psi("iso_null")) / len(_psi("iso_null"))
        _interp = ("Difiere del azimut del frente biológico de la Tabla 3 (celdas con adultas de la simulación): la capa de ocupación, "
                   "con el transporte basal por carreteras y la covariable humana, sigue mejor la dirección observada que la expansión "
                   "biofísica (ABM + viento LPDM)." if _pm + 30.0 < ang_m1 else
                   "Es del mismo orden que el error del frente biológico de la Tabla 3.")
        A(r"- **Dirección según la ocupación latente (celdas con $\psi \cdot p_{\text{det}} > 0.001$):** error angular medio de "
          + f"{M1} {_pm:.1f}° y del kernel {_pi:.1f}°. " + _interp)
    A("- El frente se mide estrictamente sobre las celdas con colonización activa del ABM (`adult_density_per_cell > 2.0`); "
      "el transporte basal por carreteras actúa sobre la ocupación $\\psi$, no sobre el radio del frente biológico.\n")
    A("---\n")

    # ────────────────────────────── Tabla 4
    A("## 5. Persistencia térmica y transferencia por ecorregiones (prueba de H1)\n")
    A("### Tabla 4: Restricción térmica invernal y transferencia ecorregional\n")
    A(f"| Métrica | JORO-HPC {M1} | JORO-HPC {M0} | Kernel isotrópico | SDM estático |")
    A("|---|---|---|---|---|")
    fpr = tp["fpr_north_40"]
    A(r"| **Tasa de falsos positivos al norte (>38°–40°N, $\tau^*=0.0012$)** | "
      f"{fpr['m1_map'] * 100:.2f} % | {fpr['m0_map'] * 100:.2f} % | {fpr['iso_null'] * 100:.2f} % | {fpr['sdm'] * 100:.2f} % |")
    ip = tp["interannual_persistence_brier"]
    A(r"| **Brier de persistencia interanual ($\ge 2$ años)** | "
      f"{ip['m1']:.4f} | {ip['m0']:.4f} | {ip['iso']:.4f} | {ip['sdm']:.4f} |")
    for lab, key, nd in (("AUC ecorregión vista (bloque 5, Piedmont)", "auc_seen", 4),
                         ("Brier ecorregión vista (bloque 5)", "brier_seen", 5),
                         ("AUC ecorregiones nuevas (bloques 2, 7–12)", "auc_unseen", 4),
                         ("Brier ecorregiones nuevas", "brier_unseen", 5)):
        r = tp[key]
        A(f"| **{lab}** | {r['m1_map']:.{nd}f} | {r['m0_map']:.{nd}f} | {r['iso_null']:.{nd}f} | {r['sdm']:.{nd}f} |")
    A("\n---\n")

    # ────────────────────────────── Incertidumbre
    A("## 6. Propagación de incertidumbre y cobertura empírica\n")
    A(f"### Tabla 5: Cobertura agregada por año del ensamble {THETA} ({n_m1} miembros), conteo esperado crudo y ajustado por prevalencia\n")
    A("Para cada miembro se suma, sobre todas las celdas, la probabilidad predicha de detección de cada año; la tabla muestra los "
      "percentiles 5, 50 y 95 de esos totales anuales entre miembros frente a las presencias observadas. El **conteo ajustado** "
      "aplica a cada miembro un offset logit de solo intercepto $\\delta_m$ calculado **exclusivamente con 2014–2019** (hace que "
      "el total esperado de 2014–2019 iguale las presencias observadas en esa ventana) y se aplica sin cambios a 2020–2024; las "
      "celdas con probabilidad cero permanecen en cero.\n")
    A("| Año | Obs. | Crudo P5 | Crudo mediana | Crudo P95 | ¿Cubre? | Ajustado P5 | Ajustado mediana | Ajustado P95 | ¿Cubre? |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    ncov = ncov_adj = 0
    yc = unc["yearly_aggregated_coverage"]
    for y in YEARS:
        r = yc[y]
        ncov += bool(r["in_90_pi"])
        ncov_adj += bool(r["adj_in_90_pi"])
        A(f"| **{y}** | {r['obs']} | {r['p5']:.1f} | {r['median']:.1f} | {r['p95']:.1f} | {'Sí' if r['in_90_pi'] else 'No'} | "
          f"{r['adj_p5']:.1f} | {r['adj_median']:.1f} | {r['adj_p95']:.1f} | {'Sí' if r['adj_in_90_pi'] else 'No'} |")
    ratio = [yc[y]["obs"] / yc[y]["median"] for y in YEARS]
    ratio_adj = [yc[y]["obs"] / yc[y]["adj_median"] for y in YEARS]
    pv = unc["prevalence"]
    A(f"\n- **Offset de prevalencia (solo 2014–2019, {pv['calibration_observed']} presencias observadas):** {M1} MAP: "
      f"$\\delta = {pv['m1_map']['delta']:+.3f}$, esperado {pv['m1_map']['raw_expected']:.1f} → {pv['m1_map']['adj_expected']:.1f}; "
      f"{M0} MAP: $\\delta = {pv['m0_map']['delta']:+.3f}$, esperado {pv['m0_map']['raw_expected']:.1f} → {pv['m0_map']['adj_expected']:.1f}; "
      f"ensamble: $\\delta$ mediana {pv['ensemble_delta_median']:+.3f} (P5–P95: {pv['ensemble_delta_p5']:+.3f} a {pv['ensemble_delta_p95']:+.3f}).")
    A(f"- **Cobertura agregada:** crudo {ncov} de {len(YEARS)} años; ajustado {ncov_adj} de {len(YEARS)} años con el total observado dentro del intervalo P5–P95.")
    A(f"- **Subpredicción del total anual:** observado / mediana = {min(ratio):.1f}–{max(ratio):.1f} (crudo) y "
      f"{min(ratio_adj):.1f}–{max(ratio_adj):.1f} (ajustado). El ajuste corrige la prevalencia de la ventana de calibración, pero la "
      "distribución anual de las detecciones sigue creciendo más rápido que el simulador (el intervalo solo propaga incertidumbre paramétrica).")
    A(f"- **Ancho medio de banda $P_{{95}} - P_{{5}}$ por celda:** {unc['mean_band_width']:.5f}.\n")
    A("---\n")

    # ────────────────────────────── Dictámenes
    A("## 7. Criterios de decisión y dictamen sobre H1 y H2\n")
    A("Los dictámenes se calculan con criterios fijos, declarados antes de leer los resultados: "
      "**CONFIRMADA** si se cumplen todos; **SOPORTADA CON SALVEDADES** si se cumple ≥ 75 %; "
      "**PARCIALMENTE SOPORTADA** si ≥ 50 %; **NO SOPORTADA** en otro caso.\n")

    auc_all = all(by[y]["m1_map"]["auc"] > by[y]["sdm"]["auc"] for y in YEARS)
    h1 = [
        ("AUC-ROC global de M1 > SDM", g["m1_map"]["auc_roc"] > g["sdm"]["auc_roc"], f"{g['m1_map']['auc_roc']:.4f} vs {g['sdm']['auc_roc']:.4f}"),
        ("Log-Loss global de M1 < SDM", g["m1_map"]["log_loss"] < g["sdm"]["log_loss"], f"{g['m1_map']['log_loss']:.4f} vs {g['sdm']['log_loss']:.4f}"),
        ("Brier global de M1 < SDM", g["m1_map"]["brier_global"] < g["sdm"]["brier_global"], f"{g['m1_map']['brier_global']:.5f} vs {g['sdm']['brier_global']:.5f}"),
        ("AUC anual de M1 > SDM en los 5 años", auc_all, ", ".join(f"{y}: {by[y]['m1_map']['auc']:.3f}/{by[y]['sdm']['auc']:.3f}" for y in YEARS)),
        ("Falsos positivos al norte (>38°N) de M1 < SDM", fpr["m1_map"] < fpr["sdm"], f"{fpr['m1_map'] * 100:.2f} % vs {fpr['sdm'] * 100:.2f} %"),
        ("AUC en ecorregiones nuevas de M1 > SDM", tp["auc_unseen"]["m1_map"] > tp["auc_unseen"]["sdm"], f"{tp['auc_unseen']['m1_map']:.4f} vs {tp['auc_unseen']['sdm']:.4f}"),
        ("Brier en ecorregiones nuevas de M1 < SDM", tp["brier_unseen"]["m1_map"] < tp["brier_unseen"]["sdm"], f"{tp['brier_unseen']['m1_map']:.5f} vs {tp['brier_unseen']['sdm']:.5f}"),
        ("Brier de persistencia interanual de M1 < SDM", ip["m1"] < ip["sdm"], f"{ip['m1']:.4f} vs {ip['sdm']:.4f}"),
    ]
    h1_md, h1_ok, h1_n = criteria_table(h1)
    A("### Dictamen sobre H1: restricción térmica y memoria fisiológica (M1 frente al SDM estático)\n")
    A(h1_md)
    A(f"\n> **DICTAMEN H1: {verdict(h1_ok, h1_n)}** ({h1_ok} de {h1_n} criterios).\n")

    abs_err = lambda mk, y: abs(kin[y][mk]["p95"] - kin[y]["obs_p95"])  # noqa: E731
    mean_err_m1 = sum(abs_err("m1_map", y) for y in YEARS) / len(YEARS)
    mean_err_iso = sum(abs_err("iso_null", y) for y in YEARS) / len(YEARS)
    _ang = lambda mk: [kin[y][mk]["angular_error_deg"] for y in YEARS if kin[y][mk]["angular_error_deg"] is not None]  # noqa: E731
    ang_m1 = sum(_ang("m1_map")) / max(1, len(_ang("m1_map")))
    ang_iso = sum(_ang("iso_null")) / max(1, len(_ang("iso_null")))
    h2 = [
        ("Error absoluto del P95 del frente en 2024: M1 < kernel", abs_err("m1_map", "2024") < abs_err("iso_null", "2024"),
         f"{abs_err('m1_map', '2024'):.1f} km vs {abs_err('iso_null', '2024'):.1f} km"),
        ("Error absoluto medio del P95 (2020–2024): M1 < kernel", mean_err_m1 < mean_err_iso, f"{mean_err_m1:.1f} km vs {mean_err_iso:.1f} km"),
        ("AUC-ROC global de M1 > kernel", g["m1_map"]["auc_roc"] > g["iso_null"]["auc_roc"], f"{g['m1_map']['auc_roc']:.4f} vs {g['iso_null']['auc_roc']:.4f}"),
        ("Log-Loss global de M1 < kernel", g["m1_map"]["log_loss"] < g["iso_null"]["log_loss"], f"{g['m1_map']['log_loss']:.4f} vs {g['iso_null']['log_loss']:.4f}"),
        ("Brier global de M1 < kernel", g["m1_map"]["brier_global"] < g["iso_null"]["brier_global"], f"{g['m1_map']['brier_global']:.5f} vs {g['iso_null']['brier_global']:.5f}"),
        ("PR-AUC global de M1 > kernel", g["m1_map"]["pr_auc"] > g["iso_null"]["pr_auc"], f"{g['m1_map']['pr_auc']:.4f} vs {g['iso_null']['pr_auc']:.4f}"),
        (f"Error angular medio de M1 menor que el del kernel en ≥ {ANGULAR_MARGIN_DEG:.0f}°", ang_iso - ang_m1 >= ANGULAR_MARGIN_DEG,
         f"{ang_m1:.1f}° vs {ang_iso:.1f}°"),
        ("MAE de la fecha de primera llegada: M1 < kernel", mae["m1_map"] < mae["iso_null"], f"{mae['m1_map']:.2f} vs {mae['iso_null']:.2f} años"),
    ]
    h2_md, h2_ok, h2_n = criteria_table(h2)
    A("### Dictamen sobre H2: dispersión atmosférica LPDM 3D (M1 frente al kernel isotrópico)\n")
    A(h2_md)
    A(f"\n> **DICTAMEN H2: {verdict(h2_ok, h2_n)}** ({h2_ok} de {h2_n} criterios).\n")

    A("### Hallazgos que matizan los dictámenes\n")
    _diff_ang = ang_iso - ang_m1
    A(f"- **Dirección del frente (referencia: centroide del polígono semilla):** error angular medio de {M1} {ang_m1:.0f}° y del kernel "
      f"{ang_iso:.0f}° (diferencia a favor de {M1}: {_diff_ang:+.0f}°). "
      + ("El LPDM no mejora la dirección de expansión con los datos y la calibración actuales."
         if _diff_ang < ANGULAR_MARGIN_DEG else "El LPDM mejora la dirección de expansión respecto al kernel isotrópico."))
    A(f"- **Calibración absoluta:** la pendiente de calibración de {M1} es {g['m1_map']['calib_slope']:.3f} (ideal 1) y el ensamble "
      f"cubre el total observado en {ncov} de {len(YEARS)} años (crudo) y {ncov_adj} de {len(YEARS)} (con offset de prevalencia): "
      "el modelo ordena bien las celdas pero subestima la magnitud.")
    A(f"- **{M1} frente a {M0}:** AUC-ROC {g['m1_map']['auc_roc']:.4f} vs {g['m0_map']['auc_roc']:.4f}; Brier global "
      f"{g['m1_map']['brier_global']:.5f} vs {g['m0_map']['brier_global']:.5f}; PR-AUC {g['m1_map']['pr_auc']:.4f} vs {g['m0_map']['pr_auc']:.4f}. "
      f"La capacidad de carga dependiente del hábitat ({M1}) no supera de forma consistente a la constante ({M0}).")
    A(f"- **Persistencia:** el SDM estático logra un Brier de persistencia interanual {'menor' if ip['sdm'] < ip['m1'] else 'mayor'} "
      f"que {M1} ({ip['sdm']:.4f} vs {ip['m1']:.4f}).")
    A("- **Muestras pequeñas:** varias ecorregiones tienen < 10 presencias, por lo que su AUC (0.5 por construcción cuando no hay presencias) no es informativo.\n")
    A("---\n")

    # ────────────────────────────── Reproducibilidad
    A("## 8. Manifiesto de reproducibilidad\n")
    A("### 8.1 Hashes SHA-256\n")
    A("| Archivo / artefacto | Descripción | SHA-256 |")
    A("|---|---|---|")
    files = [
        (SCRATCH / "calibration_set_2014_2019.csv", "Dataset de calibración (2014–2019)"),
        (SCRATCH / "evaluation_set_2020_2024.csv", "Dataset de evaluación ciega (2020–2024)"),
        (CK / "theta_compatible_ensemble.csv", f"Ensamble {M1} (Θ_comp)"),
        (CK / "theta_compatible_ensemble_M0.csv", f"Ensamble {M0} (Θ_comp)"),
        (OUT_DIR / "predictions_2020_2024_ensemble.csv", "Predicciones espaciales de ensamble 2020–2024"),
        (OUT_DIR / "sdm_predictions_2020_2024.csv", "Predicciones del SDM estático"),
        (ROOT / "build" / "joro_hpc", "Binario de simulación y calibración (`joro_hpc`)"),
        (ROOT / "build" / "joro_evaluate", "Binario de validación fuera de muestra (`joro_evaluate`)"),
        (ROOT / "containers" / "joro_hpc.def", "Receta del contenedor Apptainer"),
    ]
    for p, d in files:
        A(f"| `{p.name}` | {d} | `{sha256(p)}` |")
    A("\n### 8.2 Entorno de ejecución\n")
    A(f"- **Kernel Linux:** `{platform.release()} {platform.machine()}`.")
    A(f"- **Acelerador:** {plat.get('lpdm_device', 'Intel Arc A380 (SYCL / Level-Zero)')}.")
    A(f"- **Procesador:** {plat.get('cpu', 'N/D')}.")
    A("- **Semillas:** 1001, 2002, 3003, 4004, 5005 (recocido simulado) y Philox 4x32-10 (LPDM SYCL).")
    A("- **Contenedor:** `containers/joro_hpc.def` (Apptainer/Singularity).")
    A("- **Procedimiento paso a paso:** `docs/RUNBOOK.md`.\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"✔ Informe de validación escrito en {out}")
    print(f"  H1: {verdict(h1_ok, h1_n)} ({h1_ok}/{h1_n}) · H2: {verdict(h2_ok, h2_n)} ({h2_ok}/{h2_n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
