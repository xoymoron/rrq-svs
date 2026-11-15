#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
WORKSPACE="${SVS_WORKSPACE:-$(cd .. && pwd)}"
PYTHON="${UNIAUDIO_PYTHON:-$WORKSPACE/.venv-uniaudio/bin/python}"
CONFIG="${UNIAUDIO_RECIPE:-config/uniaudio-gtsinger.json}"
OUT="${1:-artifacts/uniaudio-preparation}"
mkdir -p "$OUT"
"$PYTHON" -m rrq.uniaudio_data prepare --config "$CONFIG" --workspace "$WORKSPACE"
pids=()
for rank in 0 1; do
  "$PYTHON" -u -m rrq.uniaudio_data cache --config "$CONFIG" --workspace "$WORKSPACE" \
    --rank "$rank" --world-size 2 > >(tee "$OUT/cache-$rank.log") 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
((status == 0)) || exit "$status"
bash scripts/run_uniaudio_stage.sh preflight "$OUT/preflight.json" 2>&1 | tee "$OUT/preflight.log"
"$PYTHON" -m rrq.uniaudio_ready --config "$CONFIG" --workspace "$WORKSPACE" \
  --preflight "$OUT/preflight.json" --output "$OUT/readiness.json"
echo 'Preparation only finished. Zero optimizer steps. Synthesis quality is not established.'
