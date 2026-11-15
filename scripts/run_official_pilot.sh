#!/usr/bin/env bash
# Bounded baseline diagnostics only. No full corpus or RRQ continuation launch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ESPNET_ROOT=${ESPNET_ROOT:-../references/espnet-toksing}
export TORCH_HOME=${TORCH_HOME:-../data/toksing/encodec}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
venv_path=${SVS_VENV:-../.venv-toksing}
mkdir -p ../logs
for mode in parallel prefix; do
    "$venv_path/bin/torchrun" --standalone --nproc_per_node=2 \
        -m rrq.official_pilot train --output artifacts/official-pilot \
        --mode "$mode" --arm baseline 2>&1 | tee "../logs/official-pilot-${mode}.log"
done
