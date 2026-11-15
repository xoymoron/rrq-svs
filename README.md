# RRQ-SVS

Matched residual re-quantization experiments on a multilingual adaptation of
[Prompt-Singer](https://github.com/cyanbx/Prompt-Singer) (NAACL 2024).
Original Prompt-Singer authors: Yongqi Wang, Ruofan Hu, Rongjie Huang, Zhiqing Hong,
Ruiqi Li, Wenrui Liu, Fuming You, Tao Jin, and Zhou Zhao. Upstream MIT license and
source are retained. This repository does not redistribute datasets or weights.

## Scope and evidence

RRQ recomputes RVQ suffix training targets against a model-predicted first codebook,
using the frozen native codec and target latent. Compare baseline (clean targets),
exposure (predicted coarse + original suffix), and RRQ (predicted coarse + recomputed
suffix) with identical sampling decisions and training budgets.

This is an annotation-conditioned pilot, **not zero-shot singer cloning**. It uses
score note pitches on supplied phoneme durations, a fixed neutral text prompt, and
no reference-audio encoder. Target audio/F0 is not an inference condition. Target
latent is used for training labels and explicitly labeled oracle diagnostics only.

Current evidence: multilingual teacher-forced CE decreases, but generated lyric
intelligibility is inadequate. Oracle latent reconstruction improves with RRQ;
learned RRQ generation improvement is not yet established. Do not interpret
training CE across different target maps as a common quality score.

## Data contract

- GTSinger and M4Singer require waveform, supplied phonemes/lyrics and note annotation.
- Bounded source data: 640 phrases per corpus. All 1,280 are preserved in the common
  manifest; no foreign phones are mapped to UNK or discarded for Mandarin compatibility.
- Primary: 576 non-Chinese GTSinger phrases, eight languages. Split 472/50/54;
  approximately 77.4 minutes of train audio. Chinese GTSinger + M4Singer: 704 auxiliary.
- Provisional normalized-song-title split; not a held-out-singer benchmark.
  Pretraining overlap and translated title aliases remain unverified.
- Original 3,960 vocabulary IDs retained; 365 phone tokens appended in this pilot.
- GTSinger supplied MusicXML and per-take annotations are retained. MIDI note values
  need not be packaged as a `.mid` file. Missing orthographic lyrics are not invented.
- Raw and processed audio are different canonical stages, not disposable duplicates.
  Native raw F0/features are retained for diagnostics; primary inference uses score pitches.

Sources: [GTSinger](https://arxiv.org/abs/2409.13832),
[M4Singer](https://openreview.net/forum?id=qiDmAaG6mP).
Both corpus licenses are CC BY-NC-SA 4.0; preserve their original license files and
additional author terms. Downloaders verify selected members and enforce a 12 GiB
free-space reserve. Additional corpora are not claimed supported or downloaded.

## Environment and preparation

Install the appropriate CUDA Torch first, then:

```bash
python -m pip install -r requirements-multilingual.txt -r requirements-data.txt
python -m rrq.assets --with-backbone
python -m rrq.prepare_data --help
python -m rrq.multilingual --help
python -m rrq.multilingual_run --help
```

`rrq.multilingual_run` now provides only codec/prompt preparation and shared loading.
The superseded embedding-only smoke trainer and greedy rollout CLI were removed.
All current training/generation goes through `rrq.experiment`.

For the existing prepared server data:

```bash
cd ~/SVS/rrq-svs
source ../.venv/bin/activate
python -m rrq.multilingual_run cache --manifest ../data/multilingual/all.jsonl \
  --output ../data/multilingual/codec-cache
python -m rrq.multilingual_run prompt --output artifacts/multilingual-prompt.pt
```

Do not recreate the existing prompt feature: the preparation command deliberately
refuses overwriting it. `rrq.assets` verifies pinned official weights and hashes.
The pipeline uses 16 kHz/50 Hz SoundStream codes and a 24 kHz CodeBigVGAN decoder.
Full Fairseq installation is not required for the current Torch runner; vendored
upstream files remain available for attribution, native controls and compatibility.

## Current matched experiment

For a NEW experiment, choose a new root; do not overwrite an existing protocol:

```bash
python -m rrq.experiment --root artifacts/NEW_RUN init --world-size 2 \
  --bundle ../data/multilingual --assets artifacts/pretrained \
  --prompt artifacts/multilingual-prompt.pt --cache ../data/multilingual/codec-cache
```

Authenticate with `wandb login` outside Git. Then explicitly start/resume:

```bash
python -u -m rrq.experiment_schedule --root artifacts/matched-v4-ddp \
  --wandb-project rrq-svs --wandb-entity oxymoron_music
```

No training or tmux session is automatically started by repository setup.
At cleanup on 2026-11-15, all assistant-created SVS tmux windows were closed.
The current v4 run is paused after a disk-reserve error, not complete.

Protocol:

- Common warmup: 8 epochs; update new global embedding rows, last two global/local
  blocks, linear map and output head. Original input embedding rows remain fixed.
- Common checkpoint: minimum language-macro validation CE.
- Branches: baseline/exposure/RRQ x seeds 17/29/43 x 6 fixed epochs. Update last two
  local blocks, linear map and output head. Global model/input embeddings frozen.
- Two RTX 3090 ranks jointly train one model using NCCL DDP, one unpadded phrase per
  rank: effective batch 2, 236 updates per 472-phrase epoch. Per-phrase mean CE is
  averaged across ranks. Nondivisible train sets fail rather than duplicate/drop.
- Same per-seed shuffle and replacement draws (rho=0.5) across arms. All branch losses
  are suffix-only. Validation is gathered with exact sample coverage checks.
- Generation uses temperature 0.8, no top-k, stable item-keyed random numbers.
  Every arm receives the same frozen common-model coarse sequence (two-pass inference).
- No learned duration/stop prediction: generation uses the annotated frame count.
- Rank 0 writes checkpoints/W&B/tqdm. Parameter hashes match across ranks after the
  first update and each epoch. Validation loss is explicitly labeled by target scope.

`epoch-XX.jsonl` has one optimizer step per row with a `samples` list for both ranks.
The current v4 final checkpoints and failed-branch resume checkpoint remain intact.
Completed branch `last.pt` files were removed only after verifying that `final.pt`
contains identical parameter tensors. Existing completion markers skip those jobs.

## Evaluation and W&B

Local JSON is canonical. Log step loss, pre-clip gradient norm, LR, mixture decisions,
GPU memory, per-language CE, and initial/selected phone-reversal diagnostics.
Generated evaluation includes CER, MR-STFT, F0/voicing, latent MSE, clipping/RMS/peak,
codebook diversity, ASR text and length-cap flags. ASR is not human MOS.

Language-macro CER remains the original headline. Median/p90 phrase CER,
character-weighted CER, language counts and metric support are supplementary;
failed phrases are never selectively excluded. Song-cluster CIs use a song-macro
estimand, explicitly distinct from the language-macro headline.

`paired_audition` uses the same two SHA256-selected sample IDs per language for
reference, codec and generated variants. It is not ranked by output quality.
Final `comparison.json`/`comparison.csv` appear only after all required test runs.
W&B uploads scalar/table/audio evaluation and final report artifacts, not model weights
or the whole raw corpus. API keys must never enter source, protocol JSON or Git.

## Code map and verification

- `rrq/core.py`: frozen native residual re-quantization.
- `rrq/prepare_data.py`, `corpora.py`, `multilingual.py`: corpus-specific preprocessing.
- `rrq/acoustic.py`, `cached_ar.py`: strict checkpoint loading and cached native AR.
- `rrq/experiment*.py`, `distributed.py`, `tracking.py`: current matched experiment.
- `rrq/pipeline.py`, `targets.py`, `training.py`: reusable oracle diagnostic and legacy
  Fairseq adapter utilities (still covered by tests; not the current training CLI).

```bash
python -m pytest rrq_tests -q
```

Historical v1-v3 artifacts and smoke outputs were removed during cleanup; their
measured conclusions are summarized in the parent `SVS_RESEARCH_RECORD.md`.
The one-off greedy-to-sampling checkpoint importer was removed. Git history retains
superseded code. Manuscripts and video sources are separate authored assets.
