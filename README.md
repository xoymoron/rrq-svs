# RRQ with official UniAudio: conventional GTSinger recipe

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

The full GTSinger singing inventory is preserved: 9 languages, 28,628 phrases,
81.02 annotated hours. The new PCM/score audit retains 26,287 unique phrases;
2,115 conflicting/invalid records and 226 duplicates are listed in audit.json.
Song families, including explicit translated-title aliases and matching PCM,
are kept within one split. Three-second prompts come from different training
songs of the same singer. The native Chinese IDs are preserved. Additional
language-qualified phoneme rows are added from train data only. New rows use
the pretrained row mean, preserving its logit calibration; zero-mean random
head rows otherwise stole probability mass from the released model. This is
preparation for multilingual adaptation, not evidence that adaptation works.

Input is phonemes, note MIDI, provided performed phoneme duration and an audio
prompt. This is annotation-conditioned acoustic SVS. Generation accepts no
target waveform, latent, F0 or tokens. Score duration constrains output length;
EOS is suppressed during these frames. Phoneme and note labels are selected at
50-Hz codec frame centers. This is our explicit conventional conversion, not a
claim to reproduce the authors' unpublished training-data transformation.
Training uses nonoverlapping chunks of at most 400 frames, retaining all frames.
The full-waveform FP32 latent and native codes are cached separately from score
alignment and split: changing annotations or splits does not re-encode audio.

Issues #25 and #26 have no author replies at the time of inspection. A bounded
audit of 47 public repositories / 48 branches found no runnable SVS conversion
recipe. The missing example does not prevent constructing our own recipe.
The frozen 185M model's earlier Chinese pilot did not establish usable singing;
there is no assumption that fine-tuning necessarily repairs it.

Compare native codec reconstruction, free-running model output, predicted q0
with original suffix (exposure), and the identical q0 with re-quantized suffix
(RRQ oracle). Oracle requires target latent and is not an inference method.
Lower latent error does not prove better voicing or learnable improvement.

## Preparation and explicit training commands

Create `../.venv-uniaudio`, install official CUDA PyTorch 2.6 and
`requirements-uniaudio.txt`. Clone the pinned upstream into
`../references/UniAudio`. Put the codec at
`../data/uniaudio/checkpoints/universal_model`.

```bash
UNIAUDIO_ROOT=../references/UniAudio ../.venv-uniaudio/bin/python -m pytest rrq_tests/test_uniaudio*.py -q

# Audit, frozen native cache, real two-GPU forward/backward, 0 optimizer steps.
tmux new -s svs-uniaudio-prepare 'bash scripts/prepare_uniaudio.sh'

# Frozen release evaluation; ground truth is opened only after generation.
bash scripts/run_uniaudio_stage.sh evaluate artifacts/uniaudio-release --languages Chinese

# These are separate, explicit commands. Preparation never executes them.
# Choose the common training budget after the readiness report; 1 is an example.
bash scripts/run_uniaudio_stage.sh train artifacts/common --arm common --epochs 1 --languages Chinese --wandb

# Only a common checkpoint passing the matching validation gate unlocks branches.
bash scripts/run_uniaudio_stage.sh bank artifacts/prefix-bank --common artifacts/common/last.pt --languages Chinese
for arm in baseline exposure rrq; do
  extra=()
  if [[ "$arm" != baseline ]]; then extra=(--bank artifacts/prefix-bank); fi
  bash scripts/run_uniaudio_stage.sh train "artifacts/$arm" --arm "$arm" \
    --epochs 1 --common artifacts/common/last.pt --languages Chinese --wandb "${extra[@]}"
done
```

Omit `--languages Chinese` consistently to use all nine prepared languages.
Do not mix common/branch language scopes. `--resume path/to/last.pt` resumes an
epoch-complete checkpoint with matching recipe, seed, optimizer and prefix-bank
identity. Epochs are the total requested budget, not additional epochs.

The common model initializes from the public 185M release. All three branches
start from its identical frozen checkpoint with fresh identical optimizers.
The same deterministic batches and global token-mean loss cover every training
segment once per epoch. Codec weights remain frozen. The original all-token CE
is retained. RRQ changes q1/q2 targets and within-frame local history only;
q0 targets and temporal/global history stay clean. Exposure uses the identical
frozen common model's sampled q0 and original suffix. This isolates residual
depth exposure; it does not claim to solve autoregressive temporal exposure.

Every epoch records clean validation CE by codebook, reversed-phone and shifted-
MIDI CE gaps, free-running F0/voicing/spectral diagnostics and token diversity.
W&B receives per-language metrics and reference/codec/generated audio tables.
The fixed panel contains two validation utterances per language. Its gate is
an operational pilot filter, not intelligibility, identity or MOS validation;
the untouched test set is not used for model selection. Oracle suffix audio
and latent-error diagnostics are kept separate from generated audio.

The earlier frozen-only `run_uniaudio_pilot.sh` is retained for reproducibility.
The new entry points provide epoch/batch tqdm, JSON/WAV/token outputs and exact
recipe/checkpoint/prefix-bank provenance. Corpus download/cache progress is
not uploaded to W&B. The agent makes no listening-based quality claims.

Compatibility handling: direct original-class import (upstream infer.py
imports a missing class), allowlisted numeric/config checkpoint metadata,
FP32 LayerNorm in half inference, and target-free cached generation. Tests
compare cached decisions to original full-forward logits and input ordering
to the original collator. The optional depth-forward helper preserves global
history and clean-forward equivalence; learned RRQ effectiveness is untested.

Old runners and environments are inactive. Broad deletion waits until the
replacement baseline passes synthesis validation; raw corpus and historical
evidence are preserved.
