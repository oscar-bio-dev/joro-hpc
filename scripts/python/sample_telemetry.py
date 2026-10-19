#!/usr/bin/env python3
"""
JORO-HPC · scripts/python/sample_telemetry.py
Muestrea utilización de CPU (/proc/stat) y de la GPU Intel Arc (sysfs, sin root) mientras corre la calibración.

intel_gpu_top exige CAP_PERFMON; aquí se estima la ocupación de la GPU como 100 % - residencia RC6
(fracción del tiempo que el GT NO está en reposo) y se registra la frecuencia activa y la potencia.

Uso:
  .venv/bin/python scripts/python/sample_telemetry.py --pid <PID> --out telemetry.json [--card card2]
Termina cuando el proceso `--pid` finaliza y escribe un resumen JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def read_cpu() -> list[tuple[int, int]]:
    out = []
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            f = [int(x) for x in line.split()[1:]]
            idle = f[3] + f[4]
            out.append((sum(f), idle))
    return out


def read_int(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", default="card2")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    gt = Path(f"/sys/class/drm/{args.card}/gt/gt0")
    freq = Path(f"/sys/class/drm/{args.card}/gt_act_freq_mhz")
    energy = next(iter(Path(f"/sys/class/drm/{args.card}/device/hwmon").glob("hwmon*/energy1_input")), None)

    cpu_prev = read_cpu()
    rc6_prev = read_int(gt / "rc6_residency_ms")
    e_prev = read_int(energy) if energy else None
    t_prev = time.monotonic()
    t_start = t_prev

    cpu_samples, gpu_busy, gpu_freq, gpu_power, per_core_sum = [], [], [], [], None
    n_samples = 0
    while os.path.exists(f"/proc/{args.pid}"):
        time.sleep(args.interval)
        t = time.monotonic()
        dt_ms = (t - t_prev) * 1000.0
        cpu_now = read_cpu()
        per_core = [100.0 * (1 - (i1 - i0) / max(1, (t1 - t0)))
                    for (t0, i0), (t1, i1) in zip(cpu_prev, cpu_now)]
        cpu_samples.append(sum(per_core) / len(per_core))
        per_core_sum = per_core if per_core_sum is None else [a + b for a, b in zip(per_core_sum, per_core)]
        rc6 = read_int(gt / "rc6_residency_ms")
        if rc6 is not None and rc6_prev is not None:
            gpu_busy.append(max(0.0, min(100.0, 100.0 * (1 - (rc6 - rc6_prev) / dt_ms))))
        f = read_int(freq)
        if f is not None:
            gpu_freq.append(f)
        if energy is not None:
            e = read_int(energy)
            if e is not None and e_prev is not None:
                gpu_power.append((e - e_prev) / 1e6 / ((t - t_prev)))  # µJ -> W
                e_prev = e
        cpu_prev, rc6_prev, t_prev = cpu_now, rc6, t
        n_samples += 1

    def stats(v):
        if not v:
            return None
        s = sorted(v)
        return {"mean": sum(v) / len(v), "p95": s[int(0.95 * (len(s) - 1))], "max": s[-1]}

    summary = {
        "wall_s": time.monotonic() - t_start,
        "samples": n_samples,
        "cpu_util_pct_all_threads": stats(cpu_samples),
        "cpu_threads_above_80pct_mean": None if per_core_sum is None else
            sum(1 for a in per_core_sum if a / n_samples > 80.0),
        "cpu_threads": 0 if per_core_sum is None else len(per_core_sum),
        "cpu_per_thread_mean_pct": None if per_core_sum is None else [a / n_samples for a in per_core_sum],
        "gpu_busy_pct_rc6_based": stats(gpu_busy),
        "gpu_act_freq_mhz": stats(gpu_freq),
        "gpu_power_w": stats(gpu_power),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
