#!/usr/bin/env bash
# Preparation only: no optimizer/research training or test evaluation is launched.
set -euo pipefail
cd "$(dirname "$0")"
SVS_ROOT="$(cd .. && pwd)"
PYTHON="$SVS_ROOT/.venv-toksing/bin/python"
unset WANDB_PROJECT WANDB_ENTITY
"$PYTHON" -m rrq.gtsinger_git --root "$SVS_ROOT/data/gtsinger" --repo "$SVS_ROOT/data/gtsinger/source-git"
"$PYTHON" -m rrq.gtsinger_prepare --source "$SVS_ROOT/data/toksing/gtsinger-full.jsonl" \
  --root "$SVS_ROOT/data/gtsinger" --output "$SVS_ROOT/data/toksing/gtsinger-prepared" \
  --aliases config/gtsinger-song-aliases.json
bash run_matched.sh cache
bash run_matched.sh readiness
printf 'Preparation verified. Research training has NOT been started.\n'
