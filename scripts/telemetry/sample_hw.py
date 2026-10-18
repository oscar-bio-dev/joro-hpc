#!/usr/bin/env python3
"""Muestreador de telemetría de hardware para JORO-HPC (solo biblioteca estándar, sin privilegios).

Muestrea cada `--interval` segundos:
  * Utilización por hilo lógico de CPU desde /proc/stat (16 hilos en el i9-11900K).
  * Tiempo ocupado por motor de la GPU Intel Arc A380 del PROCESO objetivo, leído de los contadores
    DRM fdinfo (drm-engine-*), que no requieren CAP_PERFMON (a diferencia de intel_gpu_top).
  * Frecuencia actual de la GPU (gt_act_freq_mhz).

Escribe un CSV con la serie temporal. El resumen se calcula con summarize_telemetry.py.
"""
import argparse
import os
import sys
import time


def read_cpu():
    stats = {}
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("cpu") and line[3].isdigit():
                parts = line.split()
                vals = list(map(int, parts[1:9]))
                idle = vals[3] + vals[4]
                stats[parts[0]] = (sum(vals) - idle, sum(vals))
    return stats


def find_pid(name):
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/comm") as fh:
                    if fh.read().strip() == name:
                        return int(entry)
            except OSError:
                continue
    return None


def gpu_pdev(card):
    try:
        return os.path.basename(os.path.realpath(f"/sys/class/drm/{card}/device"))
    except OSError:
        return None


def read_gpu_engines(pid, pdev):
    """Devuelve {motor: ns acumulados} sumando clientes DRM distintos del proceso en la GPU `pdev`."""
    engines = {}
    seen_clients = set()
    fd_dir = f"/proc/{pid}/fdinfo"
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        return engines
    for fd in fds:
        try:
            with open(f"{fd_dir}/{fd}") as fh:
                info = dict(l.rstrip("\n").split(":", 1) for l in fh if ":" in l)
        except OSError:
            continue
        info = {k.strip(): v.strip() for k, v in info.items()}
        if "drm-driver" not in info or (pdev and info.get("drm-pdev") != pdev):
            continue
        cid = info.get("drm-client-id")
        if cid in seen_clients:
            continue
        seen_clients.add(cid)
        for key, val in info.items():
            if key.startswith("drm-engine-") and not key.startswith("drm-engine-capacity"):
                try:
                    engines[key[len("drm-engine-"):]] = engines.get(key[len("drm-engine-"):], 0) + int(val.split()[0])
                except ValueError:
                    pass
    return engines


def read_freq(card):
    try:
        with open(f"/sys/class/drm/{card}/gt_act_freq_mhz") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="joro_hpc")
    ap.add_argument("--card", default="card2")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait", type=float, default=120.0, help="segundos máximos esperando al proceso")
    args = ap.parse_args()

    pdev = gpu_pdev(args.card)
    pid = None
    t_wait = time.time()
    while pid is None and time.time() - t_wait < args.wait:
        pid = find_pid(args.process)
        time.sleep(0.05)
    if pid is None:
        print("sample_hw: proceso no encontrado", file=sys.stderr)
        return 1

    prev_cpu = read_cpu()
    prev_gpu = read_gpu_engines(pid, pdev)
    prev_t = time.time()
    cpu_ids = sorted(prev_cpu, key=lambda c: int(c[3:]))

    # Las filas se acumulan en memoria: los motores de la GPU solo aparecen en fdinfo cuando el proceso
    # abre el dispositivo (después de iniciar), así que el encabezado se escribe al final con la unión.
    rows = []
    all_engines = set()
    while os.path.exists(f"/proc/{pid}"):
        time.sleep(args.interval)
        now = time.time()
        cur_cpu = read_cpu()
        cur_gpu = read_gpu_engines(pid, pdev)
        dt = now - prev_t
        cpu_util = []
        for c in cpu_ids:
            db = cur_cpu[c][0] - prev_cpu[c][0]
            dtot = cur_cpu[c][1] - prev_cpu[c][1]
            cpu_util.append(100.0 * db / dtot if dtot > 0 else 0.0)
        all_engines.update(cur_gpu)
        gpu_ns = {e: (cur_gpu.get(e, 0) - prev_gpu.get(e, cur_gpu.get(e, 0))) for e in cur_gpu}
        rows.append((now, cpu_util, gpu_ns, dt, read_freq(args.card)))
        prev_cpu, prev_gpu, prev_t = cur_cpu, cur_gpu, now

    engine_names = sorted(all_engines)
    with open(args.out, "w") as out:
        out.write("t," + ",".join(f"cpu{i}" for i in range(len(cpu_ids))) + "," +
                  ",".join(f"gpu_{e}" for e in engine_names) + ",gpu_freq_mhz\n")
        for now, cpu_util, gpu_ns, dt, freq in rows:
            gpu_util = [100.0 * gpu_ns.get(e, 0) / (dt * 1e9) for e in engine_names]
            out.write(f"{now:.3f}," + ",".join(f"{u:.1f}" for u in cpu_util) + "," +
                      ",".join(f"{u:.1f}" for u in gpu_util) + f",{freq}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
