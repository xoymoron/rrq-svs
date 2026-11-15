#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SVS_ROOT="$(cd .. && pwd)"
PYTHON="$SVS_ROOT/.venv-toksing/bin/python"
export OMP_NUM_THREADS=2
export WANDB_PROJECT=rrq-svs
export WANDB_ENTITY=oxymoron_music
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p artifacts/toksing-dac-pilot "$SVS_ROOT/data/toksing" "$SVS_ROOT/logs"
CODEC="$SVS_ROOT/data/toksing/dac44k.pth"
if [[ ! -f "$CODEC" ]]; then
  "$PYTHON" -c 'import dac, shutil, sys; shutil.copy2(dac.utils.download(model_type="44khz"), sys.argv[1])' "$CODEC"
fi
MANIFEST="$SVS_ROOT/data/toksing/gtsinger-pilot.jsonl"
"$PYTHON" -m rrq.toksing_acquire --root "$SVS_ROOT/data/gtsinger-pilot" --output "$MANIFEST" --existing-only
ARGS=(--manifest "$MANIFEST" --output artifacts/toksing-dac-pilot --codec "$CODEC")
for stage in cache reconstruction train oracle; do
  printf '%s stage=%s\n' "$(date -Is)" "$stage"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 -m rrq.toksing_run "$stage" "${ARGS[@]}" --epochs 20
done
printf '%s All pilot stages completed. Inspect reconstruction.json and oracle.json.\n' "$(date -Is)"
