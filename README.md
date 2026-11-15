# RRQ with official TokSing: feasibility audit

The previous custom DAC experiment is stopped. It is not an official TokSing
baseline and its checkpoints cannot establish TokSing or RRQ performance.

The `official-toksing-rrq` branch directly imports the clean ESPnet TokSing at
`5f142ef90bf87ee0f4c2d6907f7a9462b3854316`. Its encoder, decoder, duration/F0
predictors, continuous-LF0 enhancement, and supervised losses remain original.
`rrq.official_toksing` adapts only input/output contracts and optionally the final
projection's prefix conditioning. Disabled-adapter logits/loss match upstream;
zero-initialized prefix embeddings preserve the original parallel projection.

**This is an EnCodec adaptation, not an exact reproduction of the TokSing paper.**
The new output is eight RVQ codebooks of frozen mono EnCodec24k at 6 kbps/75 Hz.
The original paper's principal SSL-token + LF0 vocoder is replaced by native
EnCodec decoding, without training a new vocoder. GTSinger's supplied phone
timing defines the controlled acoustic task. Generation receives no target
LF0, target tokens, or audio. The original duration predictor remains trained.

`scripts/setup_official.sh` obtains the pinned dependency. Install the CUDA
PyTorch stack first, then `requirements-toksing.txt` and `requirements-official.txt`.
Set `ESPNET_ROOT` to that checkout and `TORCH_HOME` to the codec weight cache.

```bash
python -m rrq.official_pilot prepare --output artifacts/official-pilot
python -m rrq.official_pilot profile --output artifacts/official-pilot
torchrun --standalone --nproc_per_node=2 -m rrq.official_pilot train \
    --output artifacts/official-pilot --mode parallel --arm baseline
```

This freezes 36 training and 18 held-out validation phrases, with equal
language representation, for 64 short epochs (1,152 optimizer updates).
It is a bounded fitting/generalization diagnostic, not a final experiment.
Native codec roundtrip and reconstruction must pass before training. W&B
records epoch losses, per-language waveform pitch/spectral metrics and all
predetermined reference/codec/generated audition examples. Quality inference
uses the predicted melody, never the teacher-forcing token shortcut.

The separate `prefix` mode adds 2.75M parameters to the 73.31M official-config
predictor. A matched continuation requires a fixed common prefix bank; all
three arms load identical common weights, reset Adam and share sample/mask
schedules. `targets` refuses a failed common synthesis audit. There is no
automatic full-corpus launch; pitch/spectral success alone cannot prove lyric
intelligibility or RRQ benefit. Source review and measured feasibility must
precede a full experiment.

## Historical custom DAC experiment (inactive)

Historical scope: GTSinger singing only; no AIHub audio. This was a custom
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
80/10/10 family split retains each singer in train/valid/test, with at least
three distinct song families per language in every split. Unidentified
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

Before branch training, operational gates require common-model SC to improve
by 10% over both the initial model and a zero-output baseline (SC=1), with
phone/note condition-use gaps, then matched oracle room on the validation panel.
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

Training has epoch/batch tqdm and W&B group gtsinger-rrq-v2 with separate arm
runs. Resume restores model+optimizer and deterministic epoch RNG; incomplete
epochs are replayed. The committed last checkpoint contains a recoverable best
snapshot; best.pt/report writes precede the epoch commit. No optimizer tensors
are loaded onto a GPU for inference.

Isolated rrq.verify_runtime checks maximum source length, real score shapes,
synthetic audio, three objectives, accumulation and two-GPU synchronization.
It saves no research checkpoint and does not establish synthesis quality.

After the protocol is frozen and readiness passes, one explicit launch can run
the complete experiment in an existing tmux shell:

```bash
set -o pipefail
../.venv-toksing/bin/python -u -m rrq.run_once --unlock-test-after-training \
  2>&1 | tee ../logs/gtsinger-experiment.log
```

This runs common, gated targets, all three matched continuations, then one full
test evaluation and W&B comparison publication. It stops on any failure; it does
not retry, tune, change seeds or weaken gates. A lock rejects concurrent launch
and an existing experiment record rejects a second fresh launch. Explicit
`--resume` continues the same experiment and skips completed stages; checkpoint
resume remains epoch-based. `experiment-state.json` records stage and completion.
No test result is used to choose a checkpoint or configuration.

Optional `rrq.profile_budget` measures bounded two-GPU throughput before launch,
including validation and target-generation cost. It discards model weights and
quality results, uses artificial branch labels for timing, and is not an SVS
experiment. Its runtime estimate includes a margin and does not predict convergence.

## Reusable codec cache

The matched runner stores native DAC tokens in `../data/toksing/dac-cache`, keyed
only by source audio SHA256, codec checkpoint SHA256, and the audio encoding
implementation/runtime contract. Split, annotations, sample ID, and training
configuration do not belong to this key. Each experiment stores its small
`alignment-cache` separately; a split-only change reuses alignment too, while
changed phone duration/pitch/phones rebuild it. Singer/language/technique values
are read from the current manifest at each batch. Invalid alignment fails
without deleting or overwriting the audio tokens.

Existing `artifacts/gtsinger-rrq-v1/cache` files remain untouched. Promote them
without loading DAC or encoding audio:

```bash
../.venv-toksing/bin/python -m rrq.migrate_codec_cache \
  --legacy-output artifacts/gtsinger-rrq-v1 \
  --manifest ../data/toksing/gtsinger-prepared/manifest.jsonl \
  --codec ../data/toksing/dac44k.pth \
  --acoustic-cache ../data/toksing/dac-cache \
  --report artifacts/acoustic-cache-migration.json
```

Migration checks the legacy codec, manifest, encoding source and package versions,
compares every token exactly, and verifies every old file stayed byte-identical.
The new default output is `artifacts/gtsinger-rrq-v2`; do not overwrite the old
immutable protocol. For a future prepared split/annotation revision:

```bash
PREPARED=/path/to/revised-prepared OUTPUT=artifacts/revised-split bash run_matched.sh cache
```

`ACOUSTIC_CACHE` can override the shared directory. The completion report counts
`acoustic_encoded`, `acoustic_reused`, and `alignments_rebuilt`. Native token reuse
does not make a stale checkpoint or RRQ target bank valid: these still bind the
complete annotation/split protocol. RRQ target construction needs a continuous
target latent, so that separate stage still runs the encoder; FP32 latents are
not retained in the token store. Codec reconstruction uses the decoder only
when native tokens already exist. A new validation panel may have fewer stored
clean-prefix audits; audited sample count is explicit, never assumed complete.

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
