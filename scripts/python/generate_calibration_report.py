#!/usr/bin/env python3
"""
JORO-HPC · scripts/python/generate_calibration_report.py
Genera docs/reports/03_Inverse_Calibration_2014_2019.md y docs/telemetry/calibration_hardware_profile.json
a partir de los artefactos REALES de la calibración inversa 2014–2019:
  checkpoints/calibration_summary_{M1,M0,M3}.json     (resumen de cada calibración, escrito por el binario)
  checkpoints/theta_compatible_ensemble{,_M0,_M3}.csv
  checkpoints/telemetry_{M1,M0,M3}.json          (sample_telemetry.py)
  PostGIS (grid_5km, cell_annual_observations)   y el inventario de /era5_nc_active

Todos los veredictos (cumple / NO cumple) se calculan aquí con reglas fijas; nada se redacta a mano
sobre resultados numéricos, para que el informe no pueda sobrestimar lo logrado.

Uso:  .venv/bin/python scripts/python/generate_calibration_report.py [--out docs/reports/03_Inverse_Calibration_2014_2019.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg2

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path("/mnt/hpc_scratch/joro-hpc")
CK = SCRATCH / "checkpoints"
ERA5 = SCRATCH / "era5_nc_active"
DB_URL = "postgresql://joro_admin:REDACTED_DB_PASSWORD@localhost:5433/joro_db"
MODELS = ["M1", "M0", "M3"]
ENSEMBLE = {"M1": CK / "theta_compatible_ensemble.csv", "M0": CK / "theta_compatible_ensemble_M0.csv",
            "M3": CK / "theta_compatible_ensemble_M3.csv"}
BASE = {"h2d_d2h_s": 2.08, "runtime_s": 25.9, "j_arrival": "1.01–1.05", "j_persist": "≈1.00",
        "rho_g1g2": -0.82, "rho_k0g3": -0.75}


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def load_ensemble(p: Path):
    if not p.exists():
        return None, None
    with open(p) as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    return rd.fieldnames, rows


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def era5_inventory():
    lines = ["| Año | uvwt 3D | superficial | t2m diaria (365 d) | Rol |", "|---|---|---|---|---|"]
    ok_all = True
    for y in range(2014, 2025):
        cells = []
        for pat in (f"era5_uvwt_{y}_05.nc", f"era5_sfc_{y}_05.nc", f"era5_daily_t2m_{y}.nc"):
            p = ERA5 / pat
            if p.exists():
                cells.append(f"✔ {p.stat().st_size / 1e6:.1f} MB")
            else:
                cells.append("✘ falta")
                ok_all = False
        role = "calibración (leído)" if y <= 2019 else "reservado para validación (NO leído)"
        lines.append(f"| {y} | {cells[0]} | {cells[1]} | {cells[2]} | {role} |")
    return "\n".join(lines), ok_all


def db_stats():
    out = {}
    with psycopg2.connect(DB_URL) as con, con.cursor() as cur:
        cur.execute("SELECT count(*), avg(pop_density_h), stddev_samp(pop_density_h), "
                    "count(DISTINCT pop_density_h) FROM grid_5km")
        out["pop"] = cur.fetchone()
        cur.execute("SELECT to_regclass('public.grid_5km_pop_proxy_backup')")
        if cur.fetchone()[0]:
            cur.execute("SELECT corr(g.pop_density_h, b.pop_density_h) FROM grid_5km g "
                        "JOIN grid_5km_pop_proxy_backup b USING (cell_id)")
            out["pop_corr_old"] = cur.fetchone()[0]
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='grid_5km_covariates_raw' "
                    "AND column_name IN ('pop_count','pop_census_km2')")
        out["pop_cols"] = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT sum(pop_count), count(*) FILTER (WHERE pop_count > 0) FROM grid_5km_covariates_raw")
        out["pop_tot"] = cur.fetchone()
        cur.execute("SELECT year, count(*), sum(effort_araneae), max(effort_araneae), "
                    "sum((effort_araneae>0)::int), count(DISTINCT effort_araneae), sum(y_joro_detected) "
                    "FROM cell_annual_observations WHERE year BETWEEN 2014 AND 2019 GROUP BY year ORDER BY year")
        out["effort"] = cur.fetchall()
        cur.execute("SELECT to_regclass('public.cell_annual_observations_effort_density_backup')")
        out["effort_backup"] = cur.fetchone()[0] is not None
        cur.execute("SELECT min(year), max(year), count(*), sum(y_joro_detected) FROM v_calibration_set")
        out["calib_view"] = cur.fetchone()
    return out


def eval_set_scan():
    """Búsqueda estática de lecturas SQL de v_evaluation_set en el código del proyecto."""
    hits = []
    for base in ("scripts", "src", "tests"):
        for p in (ROOT / base).rglob("*"):
            if p.suffix not in (".py", ".cpp", ".hpp", ".sql", ".sh"):
                continue
            if p.name == "generate_calibration_report.py":
                continue
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                if re.search(r"(FROM|JOIN)\s+v_evaluation_set", line, re.I):
                    hits.append(f"{p.relative_to(ROOT)}:{i}")
    return hits


def corr_matrix(fields, rows, names):
    X = np.array([[float(r[n]) for n in names] for r in rows])
    if len(X) < 3:
        return None, X
    sd = X.std(axis=0)
    ok = sd > 0
    C = np.full((len(names), len(names)), np.nan)
    Xo = X[:, ok]
    Cc = np.corrcoef(Xo, rowvar=False)
    idx = np.where(ok)[0]
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            C[ia, ib] = Cc[a, b]
    return C, X


def md_corr(C, names):
    h = "| |" + "|".join(names) + "|\n|---|" + "|".join("---" for _ in names) + "|\n"
    for i, n in enumerate(names):
        h += f"| **{n}** |" + "|".join("—" if np.isnan(C[i, j]) else f"{C[i, j]:+.2f}" for j in range(len(names))) + "|\n"
    return h


def write_hardware_profile(path: Path, S: dict, T: dict) -> None:
    """Consolida en un único JSON la telemetría interna (temporizadores del binario) y externa (sysfs/fdinfo)."""
    import platform
    import subprocess
    cpu = "desconocido"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        compiler = subprocess.check_output(["icpx", "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        compiler = "icpx (versión no disponible en este entorno)"
    main = S["M1"]
    prof = {
        "schema": "joro-hpc/calibration_hardware_profile/v1",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {
            "lpdm_device": main["device"],
            "cpu": cpu,
            "omp_threads": main["omp_threads"],
            "kernel": platform.release(),
            "compiler": compiler,
            "compile_flags": "-O3 -fsycl -qopenmp -march=rocketlake -std=c++20",
        },
        "evaluations_per_model": main["total_evaluations"],
        "note_gpu_metric": "gpu_busy_pct_rc6_based = 100 - residencia RC6 (actividad del GT, no del motor Compute).",
        "models": {
            m: {
                "wall_s": S[m]["runtime_s"],
                "compatible_members": S[m]["compatible_members"],
                "internal_timers_s": S[m]["telemetry"],
                "external_sampling": T.get(m),
            }
            for m in MODELS if S.get(m)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prof, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "reports" / "03_Inverse_Calibration_2014_2019.md"))
    ap.add_argument("--telemetry-out", default=str(ROOT / "docs" / "telemetry" / "calibration_hardware_profile.json"))
    args = ap.parse_args()

    S = {m: load_json(CK / f"calibration_summary_{m}.json") for m in MODELS}
    T = {m: load_json(CK / f"telemetry_{m}.json") for m in MODELS}
    main_s = S["M1"]
    if main_s is None:
        raise SystemExit("Falta calibration_summary_M1.json: ejecute la calibración primero")
    db = db_stats()
    inv_md, inv_ok = era5_inventory()
    eval_hits = eval_set_scan()
    L: list[str] = []
    A = L.append

    A("# JORO-HPC — Calibración inversa 2014–2019: datos, identificabilidad y ensamble de parámetros compatibles (Θ_comp)\n")
    A(f"_Generado automáticamente el {datetime.now():%Y-%m-%d %H:%M} por `scripts/python/generate_calibration_report.py` "
      "a partir de los artefactos de la corrida. Los veredictos se calculan con reglas fijas._\n")
    A(f"- Dispositivo LPDM: **{main_s['device']}**.")
    A(f"- Hilos OpenMP: **{main_s['omp_threads']}**. Evaluaciones por modelo: **{main_s['total_evaluations']}** (5 cadenas × 400).\n")

    # ───────────── 1. Datos
    A("## 1. Ingesta de datos\n")
    A("### 1.1 ERA5 interanual (2014–2024) en disco; solo 2014–2019 leído\n")
    A(inv_md + "\n")
    files = main_s["era5_files_read"]
    yrs = sorted({int(m.group(1)) for f in files if (m := re.search(r"_(20\d\d)", f))})
    A(f"- Archivos ERA5 efectivamente abiertos por el simulador: **{len(files)}**; años distintos leídos: **{yrs}**.")
    A(f"- Solicitudes bloqueadas por la regla de aislamiento (año > 2019): **{main_s['era5_blocked_year_requests']}**.")
    A(f"- Campos ERA5 3D+superficiales residentes en la GPU (uno por año): **{main_s['atmosphere_slots']}** "
      "(cada año simulado usa SU meteorología; ya no se recicla el 1–7 de mayo de 2014).")
    A(f"- Modo estricto del lector (sin datos sintéticos de respaldo): **{main_s.get('era5_strict')}**; directorio: `{main_s.get('era5_dir')}`.")
    verd_era5 = bool(main_s.get("era5_strict")) and main_s.get("era5_dir") == str(ERA5) and inv_ok and max(yrs) <= 2019 and not main_s["era5_blocked_year_requests"] and main_s["atmosphere_slots"] == 6
    A(f"- **Veredicto:** {'✔ CUMPLE' if verd_era5 else '✘ NO CUMPLE'}.\n")
    A("> **Alcance meteorológico (limitación):** los tensores 3D/superficiales cubren la ventana 1–7 de mayo de cada año "
      "(la fase de baloneo), y la serie `t2m` diaria cubre los 365 días. Es lo que consume el simulador.\n")

    A("### 1.2 Población censal real (`pop_density_h`)\n")
    n, mu, sd, nd = db["pop"]
    A("- Fuente: Censo 2020 (PL 94-171, encabezado geográfico por bloque: POP100 + punto interno), asignado a la celda de 5 km "
      "(EPSG:5070) que contiene el punto interno; `pop_density_h = estandarizar(log1p(hab/km²))`.")
    A(f"- `grid_5km`: **{n:,}** celdas, media **{mu:+.4f}**, sd **{sd:.4f}**, **{nd:,}** valores distintos.")
    if db.get("pop_tot") and db["pop_tot"][0]:
        A(f"- Población total asignada: **{db['pop_tot'][0]:,.0f}** hab. en **{db['pop_tot'][1]:,}** celdas pobladas.")
    if db.get("pop_corr_old") is not None:
        A(f"- Correlación con el proxy NLCD anterior (respaldado en `grid_5km_pop_proxy_backup`): **{db['pop_corr_old']:.3f}**.")
    A("- Limitación: censo 2020 aplicado a toda la ventana 2014–2019 (sin serie anual); asignación por punto interno.\n")

    A("### 1.3 Esfuerzo exacto GBIF *Araneae* (`taxonKey = 1496`)\n")
    A("| Año | Celdas | Σ effort_araneae | máx | celdas con effort>0 | valores distintos | presencias *T. clavata* |\n|---|---|---|---|---|---|---|")
    for y, c, s, mx, pos, dist, pres in db["effort"]:
        A(f"| {y} | {c:,} | {int(s or 0):,} | {mx} | {pos:,} | {dist:,} | {pres or 0} |")
    A("| 2020–2024 | — | reservado para la validación fuera de muestra: este informe no consulta estos años | | | | |")
    A("")
    A("- Método: conteo de registros georreferenciados (`hasCoordinate`, sin problemas geoespaciales) asignados a la celda "
      "por punto en la celda EPSG:5070; `effort = GREATEST(conteo, joro_count)`. Respaldo previo: "
      f"{'`cell_annual_observations_effort_density_backup`' if db['effort_backup'] else '(no disponible)'}.")
    A(f"- `v_calibration_set`: años **{db['calib_view'][0]}–{db['calib_view'][1]}**, {db['calib_view'][2]:,} filas, "
      f"{db['calib_view'][3]} presencias.\n")

    # ───────────── 2. Telemetría
    A("## 2. Telemetría de tiempos y utilización GPU/CPU (M1, modelo principal)\n")
    t = main_s["telemetry"]
    wall = main_s["runtime_s"]
    h2d = t["h2d_s"] + t["d2h_s"]
    A("| Métrica | Calibración actual (M1) | Línea base (commit `3fa0d37`) |\n|---|---|---|")
    A(f"| Tiempo total (2,000 evaluaciones) | **{wall:.1f} s** | {BASE['runtime_s']} s |")
    A(f"| H2D + D2H | **{h2d:.2f} s** ({100 * h2d / wall:.1f} % de pared) | {BASE['h2d_d2h_s']} s |")
    A(f"| Kernel SYCL en GPU | {t['kernel_s']:.2f} s ({100 * t['kernel_s'] / wall:.1f} % de pared) | — |")
    A(f"| ABM (OpenMP + AVX-512) | {t['abm_s']:.2f} s | — |")
    A(f"| Ocupación + costo (OpenMP) | {t['cost_s']:.2f} s | — |")
    A(f"| LPDM host (init + reducción) | {t['init_s'] + t['reduce_s']:.2f} s | — |")
    A(f"| Super-partículas despachadas | {t['particles'] / 1e6:.1f} M | — |")
    A(f"| Despachos con fallback a CPU | **{t['cpu_fallbacks']}** | 0 |")
    A("")
    tm = T["M1"]
    if tm:
        g = tm["gpu_busy_pct_rc6_based"] or {}
        c = tm["cpu_util_pct_all_threads"] or {}
        A("Muestreo externo durante la corrida (`sample_telemetry.py`; `intel_gpu_top` requiere CAP_PERFMON, no disponible, "
          "por lo que la ocupación GPU se **estima** como 100 % − residencia RC6, que mide actividad del GT, no del motor Compute):\n")
        A("| Recurso | media | P95 | máx |\n|---|---|---|---|")
        A(f"| CPU, promedio de {tm['cpu_threads']} hilos (%) | {fmt(c.get('mean'), 1)} | {fmt(c.get('p95'), 1)} | {fmt(c.get('max'), 1)} |")
        A(f"| GPU activa (100 − RC6, %) | {fmt(g.get('mean'), 1)} | {fmt(g.get('p95'), 1)} | {fmt(g.get('max'), 1)} |")
        fq = tm.get("gpu_act_freq_mhz") or {}
        A(f"| Frecuencia GPU (MHz) | {fmt(fq.get('mean'), 0)} | {fmt(fq.get('p95'), 0)} | {fmt(fq.get('max'), 0)} |")
        pw = tm.get("gpu_power_w") or {}
        A(f"| Potencia GPU (W) | {fmt(pw.get('mean'), 1)} | {fmt(pw.get('p95'), 1)} | {fmt(pw.get('max'), 1)} |")
        A(f"\nHilos con carga media > 80 %: **{tm['cpu_threads_above_80pct_mean']} de {tm['cpu_threads']}**.\n")
    else:
        A("_No se capturó telemetría externa (falta `telemetry_M1.json`)._\n")

    # ───────────── 3. Convergencia
    A("## 3. Convergencia de las 5 cadenas (M1)\n")
    A("Componentes normalizados por la escala del punto nominal (1.000 = costo en θ nominal). "
      "Se reporta el estado inicial (J0) y el mejor de cada cadena (J*).\n")
    A("| Cadena | semilla | J0 | J* | J̃_llegada (J0 → J*) | J̃_frente (J0 → J*) | J̃_persist. (J0 → J*) | Aceptadas | t (s) |\n|---|---|---|---|---|---|---|---|---|")
    for c in main_s["chains"]:
        A(f"| {c['id']} | {c['seed']} | {c['j0']:.4f} | **{c['jmin']:.4f}** | "
          f"{c['j0_arrival']:.4f} → **{c['best_arrival']:.4f}** | {c['j0_front']:.4f} → **{c['best_front']:.4f}** | "
          f"{c['j0_persistence']:.4f} → **{c['best_persistence']:.4f}** | "
          f"{100 * c['accepted'] // c['iters']} % | {c['time_s']:.1f} |")
    mc = main_s["map_components"]
    A(f"\n**MAP (mejor global):** J = {mc['j_total']:.4f}; J̃_llegada = **{mc['j_arrival_norm']:.4f}**, "
      f"J̃_frente = **{mc['j_front_norm']:.4f}**, J̃_persistencia = **{mc['j_persistence_norm']:.4f}**.\n")
    all_lt1 = all(mc[k] < 1.0 for k in ("j_arrival_norm", "j_front_norm", "j_persistence_norm"))
    A(f"- Criterio de convergencia (los tres componentes por debajo de 1.00 simultáneamente en el MAP): "
      f"**{'✔ CUMPLE' if all_lt1 else '✘ NO CUMPLE'}**.")
    A(f"- Línea base previa: J̃_llegada {BASE['j_arrival']}, J̃_persistencia {BASE['j_persist']}.\n")
    st = main_s["stratification"]
    A(f"Conjunto de calibración: {st['n_rows']:,} filas; llegada {st['arrival_pos']} pos / {st['arrival_neg']:,} neg "
      f"({st['outside_envelope']:,} ausencias lejanas excluidas); persistencia {st['persistence_pos']} pos / {st['persistence_neg']} neg. "
      f"Esfuerzo {'SINTÉTICO (imputado)' if st['effort_synthetic'] else 'real (conteos GBIF, sin imputación)'}.\n")

    # ───────────── 4. Parámetros
    A("## 4. Parámetros estimados (M1) y correlaciones en Θ_comp\n")
    names = [p["name"] for p in main_s["params"]]
    A("| Parámetro | Prior [min, max] | θ̂_MAP | Mediana Θ_comp | P5–P95 Θ_comp |\n|---|---|---|---|---|")
    for i, p in enumerate(main_s["params"]):
        A(f"| `{p['name']}` | [{p['min']:g}, {p['max']:g}] | {main_s['map_theta'][i]:.4g} | "
          f"{main_s['median_theta'][i]:.4g} | [{main_s['p5_theta'][i]:.4g}, {main_s['p95_theta'][i]:.4g}] |")
    A(f"\nMiembros de Θ_comp (ΔJ ≤ 0.05): **{main_s['compatible_members']}**.\n")
    fields, rows = load_ensemble(ENSEMBLE["M1"])
    max_rho = None
    if rows:
        C, X = corr_matrix(fields, rows, names)
        if C is not None:
            A("**Matriz de correlación de Pearson en Θ_comp (M1):**\n")
            A(md_corr(C, names))
            iu = np.triu_indices(len(names), 1)
            vals = np.abs(C[iu])
            vals = vals[~np.isnan(vals)]
            max_rho = float(vals.max()) if len(vals) else None
            pairs = sorted(((abs(C[i, j]), names[i], names[j], C[i, j]) for i, j in zip(*iu) if not np.isnan(C[i, j])),
                           reverse=True)[:3]
            A("Mayores |ρ|: " + "; ".join(f"`{a}`–`{b}` = {r:+.2f}" for _, a, b, r in pairs) + ".\n")
            A(f"- Criterio |ρ| < 0.5 para todos los pares: **{'✔ CUMPLE' if (max_rho is not None and max_rho < 0.5) else '✘ NO CUMPLE'}** "
              f"(máx |ρ| = {fmt(max_rho, 2)}). Línea base: ρ(γ1,γ2) = {BASE['rho_g1g2']}, ρ(K0,γ3) = {BASE['rho_k0g3']}.")
            if "k0" in names and "gamma_hab" in names:
                A(f"- ρ(K0, γ_hab) = **{C[names.index('k0'), names.index('gamma_hab')]:+.2f}**.")
        if "gamma_hab" in names:
            j = names.index("gamma_hab")
            g = X[:, j]
            lo, hi = main_s["params"][j]["min"], main_s["params"][j]["max"]
            span = hi - lo
            at_hi = float((g >= hi - 0.02 * span).mean()) * 100
            at_lo = float((g <= lo + 0.02 * span).mean()) * 100
            A(f"- Truncamiento de γ_hab en el borde (dentro del 2 % del rango): superior **{at_hi:.1f} %**, inferior **{at_lo:.1f} %** "
              f"de Θ_comp; P95 = {main_s['p95_theta'][j]:.3f} (techo {hi:g}; el techo anterior de γ₂ era 1.2).")
        if "k0" in names:
            j = names.index("k0")
            lo, hi = main_s["params"][j]["min"], main_s["params"][j]["max"]
            span = hi - lo
            k = X[:, j]
            A(f"- K0 en el borde: superior {100 * float((k >= hi - 0.02 * span).mean()):.1f} %, inferior {100 * float((k <= lo + 0.02 * span).mean()):.1f} %.")
        for nm in ("a0", "b0"):
            if nm in names:
                j = names.index(nm)
                lo, hi = main_s["params"][j]["min"], main_s["params"][j]["max"]
                span = hi - lo
                v = X[:, j]
                A(f"- {nm} en el borde: superior {100 * float((v >= hi - 0.02 * span).mean()):.1f} %, inferior {100 * float((v <= lo + 0.02 * span).mean()):.1f} %.")
        # Interpretación automática (reglas fijas, sin ajuste posterior)
        if C is not None:
            iu = np.triu_indices(len(names), 1)
            tot = int((~np.isnan(C[iu])).sum())
            high = int((np.abs(C[iu][~np.isnan(C[iu])]) >= 0.5).sum())
            A(f"- Pares con |ρ| ≥ 0.5: **{high} de {tot}**.")
            edge = []
            for j, p_ in enumerate(main_s["params"]):
                if p_["name"] == "beta_damage":
                    continue
                span = p_["max"] - p_["min"]
                v = X[:, j]
                hi_f = float((v >= p_["max"] - 0.02 * span).mean())
                lo_f = float((v <= p_["min"] + 0.02 * span).mean())
                if max(hi_f, lo_f) >= 0.10:
                    edge.append(f"`{p_['name']}` ({'superior' if hi_f >= lo_f else 'inferior'}, {100 * max(hi_f, lo_f):.0f} %)")
            if edge:
                A("- Parámetros con ≥ 10 % de Θ_comp pegado a un borde del prior: " + ", ".join(edge) +
                  ". Es truncamiento por el prior y limita la interpretación de esos parámetros.")
            else:
                A("- Ningún parámetro tiene ≥ 10 % de Θ_comp pegado a un borde del prior.")
        A("")

    # ───────────── 5. Modelos
    A("## 5. Comparación de costo M0 vs M1 vs M3 (mismo presupuesto de evaluaciones por modelo)\n")
    A("| Modelo | K_i | Parámetros libres | J_min | J̃_llegada | J̃_frente | J̃_persist. | Tiempo (s) | Θ_comp |\n|---|---|---|---|---|---|---|---|---|")
    desc = {"M0": "K0", "M1": "K0·exp(γ_hab·H_hab)", "M3": "K0·exp(γ1 Z_imp + γ2 Z_borde + γ3 Z_dosel)"}
    for m in MODELS:
        s = S[m]
        if s is None:
            A(f"| {m} | {desc[m]} | — | no ejecutado | | | | | |")
            continue
        c = s["map_components"]
        A(f"| **{m}** | {desc[m]} | {s['n_params']} | {s['global_min_cost']:.4f} | {c['j_arrival_norm']:.4f} | "
          f"{c['j_front_norm']:.4f} | {c['j_persistence_norm']:.4f} | {s['runtime_s']:.1f} | {s['compatible_members']} |")
    A("")
    if all(S[m] for m in MODELS):
        j = {m: S[m]["global_min_cost"] for m in MODELS}
        A(f"- ΔJ(M1 − M0) = **{j['M1'] - j['M0']:+.4f}**; ΔJ(M3 − M1) = **{j['M3'] - j['M1']:+.4f}**. "
          "El costo es un Brier/RMSE normalizado, no una verosimilitud, así que no se calcula AIC; con 2,000 evaluaciones "
          "por modelo y una cadena estocástica, diferencias del orden de la variabilidad entre cadenas "
          "(ver Sección 3) no son concluyentes.")
        if "gamma_hab" in names:
            j_ = names.index("gamma_hab")
            A(f"- Parsimonia: M1 solo añade γ_hab a M0; en M1 la mediana de γ_hab en Θ_comp es {main_s['median_theta'][j_]:.3f}.")
        # correlaciones máximas M3 para contraste
        f3, r3 = load_ensemble(ENSEMBLE["M3"])
        if r3:
            n3 = [p["name"] for p in S["M3"]["params"]]
            C3, _ = corr_matrix(f3, r3, n3)
            if C3 is not None:
                iu = np.triu_indices(len(n3), 1)
                pr = sorted(((abs(C3[i, k]), n3[i], n3[k], C3[i, k]) for i, k in zip(*iu) if not np.isnan(C3[i, k])), reverse=True)[:3]
                A("- Correlaciones en Θ_comp de M3 (sobreparametrizado): " + "; ".join(f"`{a}`–`{b}` = {r:+.2f}" for _, a, b, r in pr) + ".")
        A("")

    # ───────────── 6. Aislamiento
    A("## 6. Aislamiento de `v_evaluation_set` (2020–2024)\n")
    A(f"- Años de ERA5 abiertos por el simulador: **{yrs}** → ninguno > 2019.")
    A(f"- Solicitudes de lectura rechazadas por el guardia del lector: **{len(main_s['era5_blocked_year_requests'])}**.")
    A(f"- `v_calibration_set` exporta solo {db['calib_view'][0]}–{db['calib_view'][1]}; el calibrador lee únicamente "
      "`calibration_set_2014_2019.csv` y el binario aborta si encuentra una fila fuera de 2014–2019.")
    A(f"- Búsqueda estática de lecturas SQL de `v_evaluation_set` en `scripts/`, `src/` y `tests/`: **{len(eval_hits)}** "
      f"{'(' + ', '.join(eval_hits) + ')' if eval_hits else ''}.")
    A("- El motor de PostgreSQL no registra lecturas de vistas sin `pg_stat_statements`; la afirmación «0 lecturas» se apoya en las "
      "tres comprobaciones anteriores (código, CSV exportado y guardia del lector), no en un contador del servidor.")
    zero = (max(yrs) <= 2019) and (not main_s["era5_blocked_year_requests"]) and len(eval_hits) == 0
    A(f"- **Veredicto:** {'✔ 0 lecturas verificables; lista para desbloquear la validación fuera de muestra' if zero else '✘ revisar'}.\n")

    # ───────────── 7. Resumen
    A("## 7. Resumen de criterios de aceptación\n")
    crit = [
        ("ERA5 2014–2024 en disco, solo 2014–2019 leído", verd_era5),
        ("Censo real y conteo exacto GBIF en `grid_5km`/`cell_annual_observations`", bool(db["pop_cols"]) and not db["calib_view"][2] == 0),
        ("Exactamente 9 parámetros libres en M1", main_s["n_params"] == 9),
        ("`ctest` 4/4", None),
        ("J̃_llegada, J̃_frente y J̃_persistencia < 1.00 simultáneamente (MAP)", all_lt1),
        ("|ρ| < 0.5 en Θ_comp (M1)", max_rho is not None and max_rho < 0.5),
        ("`v_evaluation_set` con 0 lecturas verificables", zero),
    ]
    A("| Criterio | Estado |\n|---|---|")
    for name, ok in crit:
        A(f"| {name} | {'ver nota' if ok is None else ('✔' if ok else '✘')} |")
    A("\n_`ctest` (4/4) se verifica por separado; ver `docs/RUNBOOK.md`._\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"Informe escrito en {args.out}")
    write_hardware_profile(Path(args.telemetry_out), S, T)
    print(f"Perfil de hardware escrito en {args.telemetry_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
