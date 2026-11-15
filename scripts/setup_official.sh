#!/usr/bin/env bash
set -euo pipefail
revision=5f142ef90bf87ee0f4c2d6907f7a9462b3854316
target=${ESPNET_ROOT:-artifacts/upstream-espnet}
if [[ ! -d "$target/.git" ]]; then
    git clone --depth 1 --single-branch --no-tags https://github.com/espnet/espnet.git "$target"
fi
if [[ $(git -C "$target" rev-parse HEAD) != "$revision" ]]; then
    if [[ -n $(git -C "$target" status --porcelain) ]]; then
        echo 'Refusing to change a modified official checkout' >&2; exit 1
    fi
    git -C "$target" fetch --depth 1 origin "$revision"
    git -C "$target" checkout --detach "$revision"
fi
printf 'Official ESPnet: %s at %s\n' "$revision" "$target"
