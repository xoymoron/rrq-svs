#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
WORKSPACE="${SVS_WORKSPACE:-$(cd .. && pwd)}"
PYTHON="${UNIAUDIO_PYTHON:-$WORKSPACE/.venv-uniaudio/bin/python}"
CONFIG="${UNIAUDIO_RECIPE:-config/uniaudio-gtsinger.json}"
STAGE="${1:?Pass preflight, evaluate, bank or train explicitly}"
OUTPUT="${2:?Pass an output path}"
shift 2
export OMP_NUM_THREADS=2
CUDA_LIBS="$($PYTHON -c 'import site,pathlib; print(":".join(str(p) for root in site.getsitepackages() for p in pathlib.Path(root).glob("nvidia/*/lib")))')"
export LD_LIBRARY_PATH="${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"
case "$STAGE" in
  evaluate) module=(rrq.uniaudio_evaluate);;
  preflight|bank|train) module=(rrq.uniaudio_train "$STAGE");;
  *) echo "Unknown stage: $STAGE" >&2; exit 2;;
esac
exec "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m "${module[@]}" --config "$CONFIG" --workspace "$WORKSPACE" --output "$OUTPUT" "$@"
