# RRQ: TokSing-derived score-to-DAC singing diagnostics

This branch replaces Prompt-Singer with a TokSing-derived score encoder,
length regulator, shared temporal decoder and depth-conditional RVQ head.
It is a PyTorch reimplementation, not a reproduction of the original TokSing
SSL+LF0 vocoder. DAC44k is frozen. Existing rrq.core.requantize, make_arms,
metrics and targets.training_map are reused.

## Scope

Inputs: phonemes, MIDI note pitch, provided phoneme durations, singer/language.
The first mechanism experiment is annotation-conditioned, seen-singer SVS.
No target waveform, latent or extracted F0 enters model.generate. Annotation
inventory includes all supplied symbols (transductive). Original TokSing's
melody-enhancement and learned duration paths are not claimed reproduced.

1. Encode/decode real singing through frozen DAC; measure reconstruction.
2. Train score-to-RVQ predictor; select by language-macro validation CE.
3. Generate real greedy codec predictions from score only.
4. Decode clean codec, full prediction, predicted first codebook + original
   suffix (exposure oracle), and predicted first codebook + RRQ suffix.

Oracle suffixes access target audio: not learned RRQ performance, and greedy
RRQ is not a global upper bound. DAC's normalized lookup AND full-dimensional
output projection are retained. Codec clock is44100/512Hz, not the old50Hz.
Only codec endpoint padding is cropped; no time stretch or gain normalization.

## Server execution

Repository ~/SVS/rrq-svs; fresh environment ~/SVS/.venv-toksing, Python3.10.
Install matching torch/torchaudio2.6.0 CUDA12.4, then:

```bash
~/SVS/.venv-toksing/bin/pip install -r requirements-toksing.txt
~/SVS/.venv-toksing/bin/python -m pytest
# In the svs-toksing diagnostic tmux window:
cd ~/SVS/rrq-svs
bash run_toksing.sh /path/to/reviewed-manifest.jsonl artifacts/new-experiment 2>&1 | tee ~/SVS/logs/toksing-diagnostic.log
# Attach from another terminal:
tmux attach -t svs-toksing
```

Both3090s are used: DDP training and disjoint cache/evaluation shards. Epoch
progress, per-language CE/coarse accuracy/phone-reversal gap, local JSON and
W&B metrics/audio are recorded. Global training sampling is language-uniform
with replacement; validation visits each phrase exactly once. best.pt/last.pt
are tied to manifest and codec hashes. Cache also verifies audio/annotations.
Train cache stores tokens; validation retains exact FP32 latents for oracle.
F0/voicing/MRSTFT are diagnostics, not MOS or proof of intelligibility.

The initial small-data mechanism diagnostic is complete. Its metrics are
historical evidence, not a multilingual SVS quality claim. Acquisition now
retains every original singing asset in all nine GTSinger languages, including
WAV, JSON, TextGrid and MusicXML. Speech takes are excluded. There is no singer,
duration or language cap at acquisition. Chinese minimization is a future
training-sampling policy, not deletion from the source collection.

```bash
~/SVS/.venv-toksing/bin/python -m rrq.toksing_acquire --root ~/SVS/data/gtsinger --output ~/SVS/data/toksing/gtsinger-full.jsonl --workers 8 --wandb
```

The pinned source tree, source sizes and available official LFS SHA256 hashes
are checked. Existing verified raw files are reused. tqdm and W&B show verified
files/bytes and failures; source-inventory.json and download-status.json are
local records. Raw acquisition never depends on training annotation eligibility.
The metadata manifest is an inventory, not a finalized training split.
Title groups, translation aliases and cross-corpus song identity still need audit.

AIHub dataset473 (next-tone guide vocals) acquisition covers all four accessible
archives TL1/TS1/VL1/VS1 (file keys34217-34220). The portal reports92 singers and
157.39h; the actual accessible labels contain3609 recordings and141.87h of
voice_duration. Do not substitute the advertised number for the measured one.
Note timestamps are NOT phoneme alignments. Breath flags override placeholder
MIDI96/84. Overlapping annotation segments are reported, not silently repaired.
The preprocessor defaults to all singers with no duration cap; segmentation and
annotation quality checks are separate from preservation of all source archives.

rrq.acquisition_watch observes the official AIHub downloader PID, publishes byte
growth/rounded-size progress to tqdm and W&B, then checks all ZIP CRCs and
label/audio filename coverage after download/merging exits. A size estimate or
an existing ZIP filename alone never counts as successful acquisition.

Implementation status: frozen DAC, score predictor, DDP diagnostic training,
reconstruction and actual-predicted-prefix RRQ oracle work. The new backbone's
learned exposure/RRQ training arms are not yet connected to the runner; final
multilingual preprocessing/splits and matched training remain future work.
The earlier50-phrase diagnostic gave language-macro MRSTFT-SC0.9772 for original
suffix vs0.3376 for RRQ oracle suffix. First-codebook prediction accuracy was
only3.3%; this is a weak-model mechanism test, not evidence of learned RRQ gains.

## Migration review and history

Legacy source is recoverable at commit00e364edd4cf79cfcf8122efe0815b6e76ebd790.
Before replacement its tests passed92 with one Linux-only skip. The important
safeguards survive: native quantization, frame clocks, exact validation coverage,
language-macro reporting, matched conditions, no oracle-as-generation claim.
Fairseq/T5/SoundStream/CodeBigVGAN source and launch paths are retired. Raw data
are not cleanup targets. Old checkpoints/environment are removed only after
compact historical metrics are preserved and the new environment works.

Old Prompt-Singer pilot:472train/50valid/54test,77.44train minutes, eight primary
languages. Generated multilingual intelligibility was poor. Stored latent
oracle MSE9.977146 ->6.257880 on50/50 validation phrases, versus clean.771707.
This did not establish learned RRQ gains. DAC results must be reported anew.

## Sources and licenses

- TokSing: https://arxiv.org/abs/2406.08416
- Architecture reference: https://github.com/espnet/espnet/blob/master/espnet2/svs/discrete/toksing.py
- DAC: https://github.com/descriptinc/descript-audio-codec (package1.0.0)
- GTSinger: https://arxiv.org/abs/2409.13832
- AIHub473: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=473
- Historical Prompt-Singer: https://github.com/cyanbx/Prompt-Singer

Original repository MIT license retained. No ESPnet source is redistributed.
Dataset/pretrained-model licenses are independent of this repository license.
