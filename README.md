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
bash run_toksing.sh 2>&1 | tee ~/SVS/logs/toksing-diagnostic.log
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

The initial run reuses the existing small GTSinger subset. It is a bounded
mechanism diagnostic, not the tens-hour multilingual experiment. Expand:

```bash
~/SVS/.venv-toksing/bin/python -m rrq.toksing_acquire --root ~/SVS/data/gtsinger-pilot --output ~/SVS/data/toksing/gtsinger-expanded.jsonl --hours-per-singer 1.5
```

Pinned GTSinger metadata supplies score and phoneme timing; eight non-Chinese
languages retained, song-diverse selection capped by duration per singer.
Title-group split is provisional: translation aliases and cross-corpus song
identity need audit before final paper experiments.

AIHub preference is dataset473 (next-tone guide vocals):92 singers,157.39h
singing,1.71h/person average, timed note/lyric/MIDI JSON. Select6-8 adult singers
and balance hours with GTSinger. Note timestamps are NOT phoneme alignments.
Download requires approved account/API access. Neither AIHub corpus is marked
acquired until server files and annotation coverage are actually inspected.

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
