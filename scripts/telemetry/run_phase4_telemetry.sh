#!/usr/bin/env bash
# Lanza la calibración de Fase 4 con afinidad OpenMP fija y captura telemetría real de CPU y GPU.
# Uso: scripts/telemetry/run_phase4_telemetry.sh [directorio_de_salida]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-/mnt/hpc_scratch/joro-hpc/telemetry/phase4_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

# shellcheck disable=SC1091
set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1 || true
set -eu

# Afinidad: 16 hilos (8 núcleos x 2 SMT) fijados a núcleos, sin migración entre ellos
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores

PY="${ROOT}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "[telemetría] salida en $OUT"
"$PY" "$ROOT/scripts/telemetry/sample_hw.py" --process joro_hpc --card card2 --interval 0.25 --out "$OUT/hw_samples.csv" &
SAMPLER=$!

cd "$ROOT"
stdbuf -oL ./build/joro_hpc --phase4 2>&1 | while IFS= read -r line; do
    printf '%s %s\n' "$(date +%s.%N)" "$line"
done | tee "$OUT/run.log" | sed -E 's/^[0-9.]+ //'

wait "$SAMPLER" || true
echo
echo "================ RESUMEN DE TELEMETRÍA ================"
"$PY" "$ROOT/scripts/telemetry/summarize_telemetry.py" --csv "$OUT/hw_samples.csv" --log "$OUT/run.log" | tee "$OUT/telemetry_summary.txt"
