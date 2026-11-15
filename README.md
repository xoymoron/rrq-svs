# RRQ: GTSinger score-to-DAC experiment

Current scope: GTSinger singing only; no AIHub audio. This is a TokSing-derived
PyTorch score/length-regulation model with frozen DAC44k (9 RVQ codebooks), not
an official TokSing reproduction or a pretrained zero-shot model.

Inputs are phonemes, MIDI note pitch, supplied phoneme durations, singer,
language, and the six supplied per-phoneme technique flags. Duration and
technique are explicit user controls/annotations, not predictions. No target
waveform, latent, extracted F0 or target codec suffix enters primary generation.
Original TokSing's SSL+LF0 vocoder and melody enhancement are not claimed here.

## Data and split

The pinned GTSinger source revision is 4426c862beed558b7e1cb8a4dce7e8c0c83bb208.
Full acquisition targets all 28,628 singing WAVs and original JSON/TextGrid/MusicXML;
Speech takes are excluded. No singer/language/duration cap is used at acquisition.
Git/LFS batch transfer avoids per-file resolver throttling; rrq.gtsinger_git
verifies original Git/LFS hashes before materializing the complete inventory.
Acquisition is background-only, without W&B or a download tmux dashboard.

rrq.gtsinger_prepare checks finite mono PCM, nonzero energy, sample clock,
MIDI/phoneme correspondence, and binary technique flags. Original recordings
are not stretched, normalized or truncated. Invalid records are explicitly
reported; raw data remain intact. Exact PCM duplicates are collapsed after
family union; identical PCM with conflicting conditioning is rejected and recorded. Normalized titles, explicit reviewed aliases (including translated
versions), and exact PCM matches share one song family. Deterministic approximately
80/10/10 family split retains each singer in train/valid/test. Unidentified
translations/arrangements remain a limitation; this is not a universal composition
identity annotation. Vocabulary is train-only, with an explicit held-out UNK.

Test model evaluation is locked. Deterministic source validation/codec caching
may read test audio; no test metrics select a model or hyperparameters.

## Matched experiment

Common warmup: 20 epochs. Each continuation: 10 epochs, same common best weights,
reset AdamW, seed 17, sample schedule, gradient accumulation and optimizer budget.
Two RTX 3090s, batch 1/GPU, accumulation 4, effective batch 8. Language sampling is
uniform except Chinese weight 0.25 (about 3.03% of draws); singer/phrase sampling
is uniform within the selected language, with replacement. Raw Chinese data
are retained. An epoch is a fixed number of weighted draws, not an exhaustive pass.

|Arm|Selected-prefix input|Suffix supervision and teacher-forced suffix context|
|---|---|---|
|baseline|Native target prefix|Native codec tokens|
|exposure|Frozen common-model predicted q0 on 50% of draws|Native codec tokens|
|RRQ|Same frozen predicted q0 and same 50% mask|Native residual re-quantization against that prefix|

The first-codebook supervision is always the original native target. Predicted
q0 is never used as its own label. On frames where predicted q0 equals native
q0, native suffix labels remain unchanged, isolating DAC STE rounding artifacts.
The shared frozen bank is built from train data and the fixed validation panel,
never test data. Train prefixes are in-sample, not out-of-fold predictions.

Exposure vs RRQ tests the complete re-quantized training map: both suffix labels
and their subsequent teacher-forced context change. It does not isolate label
changes while holding every later context token fixed.

Primary inference self-generates all 9 codebooks. A frozen-common-prefix decode
is diagnostic only. Prefix-bank agreement measures drift after continuation;
fixed-prefix-only improvement cannot establish better ordinary generation.

All checkpoints use the same fixed 72-phrase validation panel (8/language), chosen
before model outputs and balanced across singers. Selection is language-macro
free-running MRSTFT spectral convergence, never arm-specific target CE. Per-language
F0/voicing/support, latent MSE, token diversity, phone/note condition ablations,
and fixed audio examples are logged. These are diagnostics, not MOS or proof
of lyric intelligibility. Final evaluation uses the full held-out split and
paired song-family bootstrap; one-seed CI does not represent seed variability.

Before branch training, operational gates require common-model generation and
condition-use improvement, then matched oracle room on the validation panel.
Thresholds are predeclared heuristics, not statistical/perceptual certificates.
If either fails, stop and review rather than automatically lengthening training.
Seeds 29/43 are planned replications after seed 17 feasibility, not completed runs.

## Server commands (stages do not run automatically)

Repository:~/SVS/rrq-svs. Environment:~/SVS/.venv-toksing (Python 3.10,
torch/torchaudio 2.6.0 CUDA 12.4, DAC 1.0.0). All current source/package versions,
manifest, aliases, codec, panels and training configuration are bound to protocol.

```bash
cd ~/SVS/rrq-svs
# After verified full acquisition:
../.venv-toksing/bin/python -m rrq.gtsinger_prepare \
  --source ../data/toksing/gtsinger-full.jsonl --root ../data/gtsinger \
  --output ../data/toksing/gtsinger-prepared --aliases config/gtsinger-song-aliases.json
bash run_matched.sh cache
bash run_matched.sh readiness
# Start actual training ONLY when requested, inside tmux:
tmux new-session -s svs-train -c ~/SVS/rrq-svs
bash run_matched.sh common 2>&1 | tee ../logs/gtsinger-common.log
# Review common quality gate before these stages:
bash run_matched.sh targets
bash run_matched.sh branches 2>&1 | tee ../logs/gtsinger-branches.log
bash run_matched.sh valid
# After freezing the protocol/model choices:
bash run_matched.sh final-test --unlock-test
```

Training has epoch/batch tqdm and W&B group gtsinger-rrq-v1 with separate arm
runs. Resume restores model+optimizer and deterministic epoch RNG; incomplete
epochs are replayed. The committed last checkpoint contains a recoverable best
snapshot; best.pt/report writes precede the epoch commit. No optimizer tensors
are loaded onto a GPU for inference.

Isolated rrq.verify_runtime checks maximum source length, real score shapes,
synthetic audio, three objectives, accumulation and two-GPU synchronization.
It saves no research checkpoint and does not establish synthesis quality.

## Historical evidence and boundaries

The retired 77-minute training pilot had only 3.3% coarse-code accuracy.
Its 50-phrase oracle MRSTFT-SC 0.9772→0.3376 supports testing the mechanism, not
learnability or a strong multilingual baseline. Old diagnostics are consolidated
in server logs/toksing-initial-diagnostic-history.json. Prompt-Singer source is
recoverable at commit 00e364e; its source/env/checkpoints are retired.

AIHub 473 audio acquisition was stopped by user request; completed label archives
are not a complete training corpus. The optional adapter remains unused.

## Primary sources

- TokSing: https://arxiv.org/abs/2406.08416
- Original implementation: https://github.com/espnet/espnet/blob/master/espnet2/svs/discrete/toksing.py
- DAC: https://github.com/descriptinc/descript-audio-codec
- GTSinger: https://arxiv.org/abs/2409.13832
- Related correction precedent (not proof for SVS): https://arxiv.org/abs/2412.04431

Original repository MIT license retained. No ESPnet source is redistributed.
Dataset/pretrained-model licenses are independent of this repository license.
