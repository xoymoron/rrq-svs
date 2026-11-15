#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SVS_ROOT="$(cd .. && pwd)"
PYTHON="$SVS_ROOT/.venv-toksing/bin/python"
export OMP_NUM_THREADS=2
export WANDB_PROJECT=rrq-svs
export WANDB_ENTITY=oxymoron_music
export CUDA_VISIBLE_DEVICES=0,1
MANIFEST="${1:?Pass a reviewed training manifest; full acquisition inventories are not finalized training splits}"
OUTPUT="${2:?Pass a new experiment output directory}"
mkdir -p "$OUTPUT" "$SVS_ROOT/data/toksing" "$SVS_ROOT/logs"
CODEC="$SVS_ROOT/data/toksing/dac44k.pth"
if [[ ! -f "$CODEC" ]]; then
  "$PYTHON" -c 'import dac, shutil, sys; shutil.copy2(dac.utils.download(model_type="44khz"), sys.argv[1])' "$CODEC"
fi
ARGS=(--manifest "$MANIFEST" --output "$OUTPUT" --codec "$CODEC")
for stage in cache reconstruction train oracle; do
  printf '%s stage=%s\n' "$(date -Is)" "$stage"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 -m rrq.toksing_run "$stage" "${ARGS[@]}" --epochs 20
done
printf '%s All diagnostic stages completed. Inspect reconstruction.json and oracle.json.\n' "$(date -Is)"
