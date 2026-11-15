# RRQ with official UniAudio: pretrained feasibility audit

The active branch is `official-uniaudio-rrq`. Historical custom DAC and
TokSing/EnCodec models are not the current baseline. Their results do not
establish UniAudio performance. The consolidated history is in the workspace's
`SVS_RESEARCH_RECORD.md`; old implementations remain in Git history.

The adapter directly imports the original MegaByteModel and SoundStream from
https://github.com/yangdongchao/UniAudio at
`c1146a8cecf84f1c2e877170d84998393b7e1e78`. No generator, prefix head or decoder
is substituted. Existing `rrq.core.requantize` uses the native first three
quantizers. The entire codec is frozen.

## Pin the checkpoint revision

The same public ZIP filename has two different releases:

|HF revision|Date|Global/local layers|Width|Parameters|
|---|---|---|---|---|
|`ba6d20045d7cdfacedf6c2707d0641361f4e2c3b`|2024-07-22|16 / 8|768|184,581,120|
|`835f07dcf1bdf586b0f427be59e4d48d0320f00e`|2025-01-07|24 / 8|1536|935,933,952|

Download from
`https://huggingface.co/Dongchao/UniAudio/resolve/REVISION/uniaudio_v1.zip`.
The older member is `uniaudio_v1/v1.checkpoint`; the newer member is
`uniaudio_v1/ep5-iter340000.checkpoint`. Use each archive's own config and
vocabulary, never the small README settings with large weights.

Both use upstream's `universal_model.zip` codec. Although it has eight books,
the generator uses three: 1.5 kbps, 16 kHz, 50 frames/s. No codec pretraining
is required.

## Current experiment

The full GTSinger singing corpus is preserved: 9 languages, 28,628 phrases,
81.02 annotated hours. Frozen diagnosis uses four Mandarin validation phrases,
two distinct songs per singer, selected by ID and duration before generation.
Three-second prompts come from different training songs of the same singer.
The native singing dictionary is Mandarin: this is not multilingual validation.

Input is phonemes, note MIDI, provided performed phoneme duration and an audio
prompt. This is annotation-conditioned acoustic SVS. Generation accepts no
target waveform, latent, F0 or tokens. Score duration constrains output length;
EOS is suppressed during these frames. The original SVS preprocessing recipe
is not released, which limits interpretation of out-of-domain performance.

Compare native codec reconstruction, free-running model output, predicted q0
with original suffix (exposure), and the identical q0 with re-quantized suffix
(RRQ oracle). Oracle requires target latent and is not an inference method.
Lower latent error does not prove better voicing or learnable improvement.

## Run

Create `../.venv-uniaudio`, install official CUDA PyTorch 2.6 and
`requirements-uniaudio.txt`. Clone the pinned upstream into
`../references/UniAudio`. Put the codec at
`../data/uniaudio/checkpoints/universal_model`.

```bash
../.venv-uniaudio/bin/python -m pytest rrq_tests/test_uniaudio.py rrq_tests/test_metrics.py -q

UNIAUDIO_MODEL_DIR=../data/uniaudio/checkpoints-20240722/uniaudio_v1 \
UNIAUDIO_MODEL_FILE=v1.checkpoint \
bash scripts/run_uniaudio_pilot.sh artifacts/uniaudio-small-native
```

The launcher processes disjoint examples on two GPUs and provides phrase/frame
tqdm, per-GPU logs, JSON/WAV/token outputs and a W&B audition table containing
every selected example. It reports failures and exits unsuccessfully if any
item fails. No training is started, and the agent does not claim listening-based
quality judgments.

Compatibility handling: direct original-class import (upstream infer.py
imports a missing class), allowlisted numeric/config checkpoint metadata,
FP32 LayerNorm in half inference, and target-free cached generation. Tests
compare cached decisions to original full-forward logits and input ordering
to the original collator. The optional depth-forward helper preserves global
history and clean-forward equivalence; learned RRQ effectiveness is untested.

Old runners and environments are inactive. Broad deletion waits until the
replacement baseline passes synthesis validation; raw corpus and historical
evidence are preserved.
