#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${UNIAUDIO_PYTHON:-../.venv-uniaudio/bin/python}"
OUTPUT="${1:-artifacts/uniaudio-native-20261116}"
MODEL_DIR="${UNIAUDIO_MODEL_DIR:-../data/uniaudio/checkpoints/uniaudio_v1}"
MODEL_FILE="${UNIAUDIO_MODEL_FILE:-ep5-iter340000.checkpoint}"
mkdir -p "$OUTPUT"
"$PYTHON" -m rrq.uniaudio_pilot prepare --manifest ../data/toksing/gtsinger-full.jsonl --output "$OUTPUT"
args=(--upstream ../references/UniAudio
      --config "$MODEL_DIR/config.yaml"
      --checkpoint "$MODEL_DIR/$MODEL_FILE"
      --codec ../data/uniaudio/checkpoints/universal_model --output "$OUTPUT" --world-size 2)
"$PYTHON" -u -m rrq.uniaudio_pilot run "${args[@]}" --rank 0 > >(tee "$OUTPUT/gpu0.log") 2>&1 &
pid0=$!
"$PYTHON" -u -m rrq.uniaudio_pilot run "${args[@]}" --rank 1 > >(tee "$OUTPUT/gpu1.log") 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if ((status)); then
    echo 'Worker failed. Full training is not authorized by this diagnostic.'
    exit "$status"
fi
WANDB_ENTITY=oxymoron_music "$PYTHON" -m rrq.uniaudio_pilot aggregate --output "$OUTPUT" --wandb
echo 'Frozen baseline/oracle diagnostic finished. No optimizer steps were run.'
