#!/usr/bin/env bash
# JORO-HPC · scripts/run_phase4_final.sh
# Calibración final Fase 4.5 (2014-2019): M1 (principal), M0 y M3 con el mismo presupuesto (5 x 400),
# GPU SYCL + 16 hilos OpenMP, con telemetría externa (CPU/GPU vía sysfs) por modelo.
# Uso: bash scripts/run_phase4_final.sh
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CK=/mnt/hpc_scratch/joro-hpc/checkpoints
LOGS=/mnt/hpc_scratch/joro-hpc/logs
mkdir -p "$CK" "$LOGS"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
export OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
unset JORO_ERA5_DIR JORO_ERA5_NONSTRICT JORO_UNLOCK_PHASE5 JORO_ALLOW_CPU_FALLBACK JORO_SA_CHAINS JORO_SA_ITERS

for M in M1 M0 M3; do
  echo "=== Calibrando modelo $M ==="
  "$ROOT/build/joro_hpc" --phase4 --engine=gpu --model "$M" > "$LOGS/phase4_final_$M.log" 2>&1 &
  PID=$!
  "$ROOT/.venv/bin/python" "$ROOT/scripts/python/sample_telemetry.py" --pid "$PID" --out "$CK/telemetry_$M.json" &
  SPID=$!
  wait "$PID" && echo "$M: OK" || { echo "$M: FALLÓ (ver $LOGS/phase4_final_$M.log)"; wait "$SPID" || true; exit 1; }
  wait "$SPID" || true
done
echo "Corridas completas. Logs en $LOGS"
