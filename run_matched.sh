#!/usr/bin/env bash
# Explicit stages only. Calling this script without a stage starts no experiment.
set -euo pipefail
cd "$(dirname "$0")"
SVS_ROOT="$(cd .. && pwd)"
PYTHON="$SVS_ROOT/.venv-toksing/bin/python"
STAGE="${1:?Choose cache, readiness, common, targets, branches, valid, or final-test}"
shift
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
PREPARED="${PREPARED:-$SVS_ROOT/data/toksing/gtsinger-prepared}"
OUTPUT="${OUTPUT:-artifacts/gtsinger-rrq-v2}"
ACOUSTIC_CACHE="${ACOUSTIC_CACHE:-$SVS_ROOT/data/toksing/dac-cache}"
ARGS=(--prepared "$PREPARED" --output "$OUTPUT" --acoustic-cache "$ACOUSTIC_CACHE" --codec "$SVS_ROOT/data/toksing/dac44k.pth" --config config/gtsinger-rrq-v1.json)
run() { "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 -m rrq.matched_run "$@" "${ARGS[@]}"; }
case "$STAGE" in
  cache|readiness) unset WANDB_PROJECT WANDB_ENTITY; run "$STAGE" ;;
  common)
    export WANDB_PROJECT=rrq-svs WANDB_ENTITY=oxymoron_music
    run common ;;
  targets) unset WANDB_PROJECT WANDB_ENTITY; run targets ;;
  branches)
    export WANDB_PROJECT=rrq-svs WANDB_ENTITY=oxymoron_music
    for arm in baseline exposure rrq; do run branch --arm "$arm"; done ;;
  valid)
    for arm in baseline exposure rrq; do run evaluate --arm "$arm" --split valid; done
    "$PYTHON" -m rrq.compare_matched --root "$OUTPUT" --split valid ;;
  final-test)
    [[ "${1:-}" == "--unlock-test" ]] || { printf 'Test is locked. Pass --unlock-test only after freezing the experiment.\n' >&2; exit 2; }
    for arm in baseline exposure rrq; do run evaluate --arm "$arm" --split test --unlock-test; done
    "$PYTHON" -m rrq.compare_matched --root "$OUTPUT" --split test ;;
  *) printf 'Unknown stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
