#!/usr/bin/env python3
"""Resume la telemetría de hardware (CSV de sample_hw.py) dentro de la ventana de calibración.

La ventana se delimita con las marcas de tiempo del log de la corrida:
  inicio : línea 'Evaluando escala de referencia inicial'
  fin    : línea 'TABLA 4'
"""
import argparse
import csv
import statistics as st
import sys


def parse_log_window(log_path):
    t0 = t1 = None
    with open(log_path, errors="replace") as fh:
        for line in fh:
            try:
                ts, rest = line.split(" ", 1)
                ts = float(ts)
            except ValueError:
                continue
            if t0 is None and "Evaluando escala de referencia inicial" in rest:
                t0 = ts
            if "TABLA 4" in rest:
                t1 = ts
    return t0, t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    t0, t1 = parse_log_window(args.log)
    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        print("Sin muestras.")
        return 1
    ts = [float(r["t"]) for r in rows]
    if t0 is None or t1 is None:
        t0, t1 = ts[0], ts[-1]
    win = [r for r in rows if t0 <= float(r["t"]) <= t1]
    print(f"Ventana de calibración: {t1 - t0:.1f} s, {len(win)} muestras (de {len(rows)})")

    cpu_cols = [c for c in rows[0] if c.startswith("cpu")]
    per_cpu = {c: st.mean(float(r[c]) for r in win) for c in cpu_cols}
    means = list(per_cpu.values())
    print(f"\nCPU (hilos lógicos: {len(cpu_cols)})")
    print("  utilización media por hilo [%]: " + " ".join(f"{m:5.1f}" for m in means))
    print(f"  media global {st.mean(means):.1f} % | min {min(means):.1f} % | max {max(means):.1f} % | "
          f"dispersión (max-min) {max(means) - min(means):.1f} pp")
    frames_all16 = sum(1 for r in win if all(float(r[c]) > 50.0 for c in cpu_cols))
    print(f"  muestras con los {len(cpu_cols)} hilos > 50 %: {frames_all16}/{len(win)}")
    per_sample_mean = [st.mean(float(r[c]) for c in cpu_cols) for r in win]
    print(f"  utilización media de CPU por muestra: P50 {st.median(per_sample_mean):.1f} % | max {max(per_sample_mean):.1f} %")

    gpu_cols = [c for c in rows[0] if c.startswith("gpu_") and c != "gpu_freq_mhz"]
    print("\nGPU Intel Arc A380 (motores del proceso, fdinfo DRM)")
    if not gpu_cols:
        print("  (sin contadores fdinfo disponibles)")
    for c in gpu_cols:
        vals = [float(r[c]) for r in win]
        busy = sum(1 for v in vals if v > 1.0)
        print(f"  {c[4:]:<10} media {st.mean(vals):5.1f} % | P95 {sorted(vals)[int(0.95 * (len(vals) - 1))]:5.1f} % | "
              f"max {max(vals):5.1f} % | muestras activas (>1 %) {busy}/{len(vals)}")
    freqs = [float(r["gpu_freq_mhz"]) for r in win if float(r["gpu_freq_mhz"]) >= 0]
    if freqs:
        print(f"  frecuencia actual: media {st.mean(freqs):.0f} MHz | max {max(freqs):.0f} MHz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
