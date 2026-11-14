# Prompt-Singer: Controllable Singing-Voice-Synthesis with Natural Language Prompt
#### Yongqi Wang*, Ruofan Hu*, Rongjie Huang, Zhiqing Hong, Ruiqi Li, Wenrui Liu, Fuming You, Tao Jin, Zhou Zhao 
#### Zhejiang University

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2403.11780) [![Demo](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue.svg?logo=Github)](https://prompt-singer.github.io/)  

contact: [cyanbox@zju.edu.cn](mailto:cyanbox@zju.edu.cn)

This is the PyTorch implementation of Prompt-Singer (NAACL'24), a singing-voice-synthesis model with controllability over speaker gender, volume, and vocal range with natural language prompt. We provide the finetuned FLAN-T5 version of our code and checkpoints.

## Note

**UPD:** We observe that the effect of attribute drop during training may need to be reevaluated, as it could damage the model's control capabilities in some cases. If you find that the control performance is not good when training your own model, you can try modifying [`research/PromptSinger/dataset/t2a_sing_t5_config_dataset.py#L499`](research/PromptSinger/dataset/t2a_sing_t5_config_dataset.py#L499) of the code by setting `p1` and `p2` to 0.0 to turn off attribute dropping. This should still yield good control performance for both single and multiple attributes.

```python
overall_text_prompt = self.choose(gender = gender_label, volume = volume_label, pitch = pitch_label, p1 = 0.0, p2 = 0.0)
```
---

This re-implementated version differs slightly in details with the ones in the paper. We use a 16kHz SoundStream to extract acoustic units of the training data of the transformer (downsampled to 16kHz). The vocoder is still trained to generate 24kHz audio.

The correctness of this open-source version has not been fully verified. Feel free to create an issue if you find any problems.

## Dependencies
We recommend the following environment:
* PyTorch version >= 2.1.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and NCCL
* For faster training install NVIDIA's apex library:
  ```shell
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --no-cache-dir --global-option="--cpp_ext" \
    --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" \
    --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
  ```

* Install this fairseq-based repo and develop locally if you don't have fairseq installed:
  ```shell
  pip install --editable ./
  ```
  If you have already installed fairseq, you can run Prompt-Singer with the existing fairseq by specifying the path to the `research` directory (see infer and training parts).

## Checkpoints
[![Static Badge](https://img.shields.io/badge/🤗Hugging_Face-Model-yellow.svg)](https://huggingface.co/Cyanbox/Prompt-Singer)  
We provide checkpoints for each stage of the finetuned flan-t5 large version model on Hugging Face. In order to download the checkpoints, simply run the following python script:

```python
from huggingface_hub import snapshot_download 
downloaded_path = snapshot_download(repo_id="Cyanbox/Prompt-Singer")
```
This hugging face repo contains checkpoints of the SoundStream, the finetuned FLAN-T5, the transformer backbone and the unit vocoder.

## Inference

### Acoustic token inference

We provide some TSV files with a small batch of samples in `infer_tsv` directory for quick start, together with configuration and dictionary files needed. The TSV files contain the input information needed for controllable SVS, where the file names correspond to the respective control attribute categories, and the prompt sentences included in these files match these attributes. You need to modify `text_encoder_version` and `audio_tokenizer_ckpt_path` in `config.yaml` to the paths on your own machine before inference.

Switch to the `Prompt-Singer` root directory, modify the relevant parameters and paths, then run the following command to generate acoustic units:

```shell
python research/PromptSinger/generate.py  infer_tsv \
 --task t2a_sing_t5_config_task \
 --path <PATH_TO_CKPT_DIR>/prompt-singer-flant5-large-finetuned/checkpoint_last.pt \
 --gen-subset <PREFIX_NAME_OF_TEST_TSV> \
 --batch-size 1 --max-tokens 10000 \
 --max-source-positions 10000 --max-target-positions 10000 \
 --max-len-a 1 --max-len-b 0 \
 --results-path <PATH_TO_OUTPUT_DIR> --user-dir research \
 --fp16  --num-workers 0 
```

The output file would be `<PATH_TO_OUTPUT_DIR>/generate-<PREFIX_NAME_OF_TEST_TSV>.txt`. Note that the model is trained to generate audio up to a maximum length of 13 seconds, and input exceeding this length will be truncated.

### Wavefrom generation

After getting the output txt file from the previous step, switch to the `wave_generation` directory and run the following command for audio generation:

```shell
python infer.py --input_code_file <PATH_TO_OUTPUT_DIR>/generate-<PREFIX_NAME_OF_TEST_TSV>.txt \
 --output_dir <PATH_TO_AUDIO_OUTPUT_DIR> \
 --checkpoint_file <PATH_TO_CKPT_DIR>/vocoder_24k/g_00885000
```

The generated audio will be in `<PATH_TO_AUDIO_OUTPUT_DIR>/syn`.

## Train with your own datasets

Due to copyright and other issues, we do not provide the original SVS datasets here. If you use the same training data as us, we provide the processed manifests in [this link](https://drive.google.com/drive/folders/1q0b-uzSqG1HaQ09bev1kDSvhEGMXgQm8?usp=drive_link), which includes duplicated phonemes, rescaled and averaged f0, and gender labels. We also provide the method for constructing the training data as well as the training command.

### Data Composition

We provide an example of the data composition format in the `data`` directory. 
The training data should consist of the following parts:

* `config.yaml`: configuration file.
  * `word_dict`: filename of the keyword pickle file.
  * `sentence_dict`: filename of the sentence template pickle file.
  * `text_encoder_version`: path to the local FLAN-T5 repo.
  * `audio_tokenizer_ckpt_path`: path to the codec checkpoint.
  * `dict_length`: keep it the same with the length of `dict_uni.txt`
  * `batch_max_frames`: maximum length of audio data for training / inference. 
  * keep other parameters unchanged.
* `dict_uni.txt`: dictionary file for model input.
* `English_word.pkl`: pickle file containig the keywords we use.
* `English_sen.pkl`: pickle file containig the sentence templates we use.
* `train.tsv` and `valid.tsv`: data manifests for training and evaluation.
  * `item_name`: name of data item.
  * `task`: keep it `text_to_acoustic_sing`.
  * `audio_path`: path to the wav file.
  * `gender`: gender lable of the item, should be one of `[male, female]`.
  * `phone`: duplicated phonemes by the durations.
  * `txt`: original lyrics, not used in training.
  * `f0_rescale`: relative value of f0 with the average of its voiced part being 222.
  
    > The `f0_rescale` in the manifests we provides above has a voiced-part mean of 222, which is slightly different from the 230 reported in the paper. However, small changes in its value will not affect model performance as long as it is consistent for all data items.
  * `f0_avg`: average of the voiced part of f0.
  * `n_frames`: length of the acoustic units sequences, need to be pre-calculated.
  * `pitch`: vocal range lable of the item, should be one of `[high, low]`.

The processing methods of phonemes and f0 are provided below.

#### Get aligned (duplicated) phonemes

Most SVS datasets (including m4singer, opensinger, opencpop, and popcs that we use) provide phoneme-level alignment in textgrid format. For speech data, you can obtain textgrids through [MFA (Montreal Forced Aligner)](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps). Upon having the textgrids, use `data_scripts/duplicate_phone.py` for generating duplicated phonemes. The command is:

```shell
python duplicate_phone.py --input-tsv <PATH_TO_INPUT_TSV> --output-tsv <PATH_TO_OUTPUT_TSV> --textgrid-dir <PATH_TO_TEXTGRID_ROOT>
```

where the input tsv should contain `item_name` and `n_frames`, and the textgrid files should be named consistently with `item_name`.

#### F0 extraction

Use `data_scripts/extract_f0.py` for extracting continuous f0 values with the command

```shell
python extract_f0.py --input-tsv <PATH_TO_INPUT_TSV> --output-dir <PATH_TO_OUTPUT_DIR>
```

The input tsv file should contain `audio_path` and `item_name`.

#### F0 rescale and averaging

After finishing f0 extraction, use `data_scripts/rescale_average_f0.py` for calculating average and rescaled f0. The command is

```shell
python rescale_average_f0.py --input-dir <PATH_TO_INPUT_DIR> --output-tsv <PATH_TO_OUTPUT_TSV>
```

The average f0 value can be further used to obtain `pitch` label according to singer gender.

### Training

Use the following command for training:

```shell
fairseq-train --task t2a_sing_t5_config_task \
  <PATH_TO_DATA_DIR> \
  --num-workers 0 \
  --save-dir <PATH_TO_SAVE_DIR> \
  --tensorboard-logdir <PATH_TO_SAVE_DIR> \
  --arch acoustic_lm_global300M_noprefix  \
  --no-epoch-checkpoints \
  --criterion acoustic_language_modeling_cross_entropy \
  --optimizer adam --adam-betas '(0.9, 0.95)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 8000 --sample-break-mode none \
  --update-freq 16 \
  --fp16 \
  --max-tokens 8000 \
  --max-update 500000 \
  --n-ctx 15000 \
  --user-dir <PATH_TO_PROMPT_SINGER>/research
```

## Acknowledgements
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution.

* Fairseq: https://github.com/facebookresearch/fairseq  
* AcademiCodec: https://github.com/yangdongchao/AcademiCodec  
* BigVGAN: https://github.com/NVIDIA/BigVGAN


## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@inproceedings{wang2024prompt,
  title={Prompt-Singer: Controllable Singing-Voice-Synthesis with Natural Language Prompt},
  author={Wang, Yongqi and Hu, Ruofan and Huang, Rongjie and Hong, Zhiqing and Li, Ruiqi and Liu, Wenrui and You, Fuming and Jin, Tao and Zhao, Zhou},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={4780--4794},
  year={2024}
}
```

## Disclaimer
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

---

## RRQ extension (2026-11-15)

This fork adds a **frozen oracle diagnostic and opt-in cached-target training adapter**.
It is not a trained RRQ checkpoint and is not a zero-shot singer-cloning implementation.
Original upstream: `14816cda5aae948cf0f84bd9f5b19b7189154666`.

### Quick start (PowerShell, repository root)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-rrq.txt
.\.venv\Scripts\python.exe -m rrq fetch-assets
.\.venv\Scripts\python.exe -m pytest rrq_tests -q
.\.venv\Scripts\python.exe -m rrq smoke --output artifacts/smoke-new
```

For GPU execution install the appropriate PyTorch build instead. Linux uses `.venv/bin/python`.
The frozen diagnostic needs no Fairseq, T5 or full generator checkpoint. The pinned download is about 794 MB, with SHA256 checks and no silent overwrite of corrupt files. `smoke` uses a synthetic harmonic signal and explicitly **GT-derived perturbed tokens**. It checks wiring only; its metrics are not singing-quality or real-error evidence. A run directory cannot be reused.

### Real model error diagnostic

1. Run the original Prompt-Singer generator in its supported environment (the upstream instructions above), without teacher-forcing acoustic target tokens. This needs the additional generator/T5 assets. They are **not** downloaded by `fetch-assets`.
2. Use fixed prompt text, `volume=None`, original phrase boundaries and matching 50 Hz input conditions. The native recipe consumes aligned phonemes and F0; identify those runs as `annotation_conditioned`, not score-native or zero-shot.
3. Import the exact original `D-ID<TAB>...` output:

```powershell
.\.venv\Scripts\python.exe -m rrq import-codes --input generate-test.txt --output artifacts/predictions --generator CHECKPOINT_HASH_AND_REVISION --seed 0
.\.venv\Scripts\python.exe -m rrq diagnostic --manifest YOUR_MANIFEST.jsonl --output artifacts/diagnostic-new --device cpu
```

Copy the row structure from `rrq_examples/manifest.jsonl`. Paths are relative to the manifest. Match generated D-IDs to target recordings yourself; IDs alone do not establish alignment. `alignment=same_50hz_grid` is an explicit caller attestation, not an automatic proof. Frame mismatch is rejected, never cropped. Use mono phrases 0.2–13 seconds, explicit `gain=1`, and split/crop them before generation. No automatic normalization, DTW or delay fitting is performed. Only codec right padding of less than one frame is applied and reported. Import provenance is user-declared; the tool cannot prove an external token file was honestly generated.

Outputs per phrase: `codec.wav`, `model.wav`, `exposure.wav`, `rrq_oracle.wav`, and `targets.npz`. All use the **actual 24 kHz CodeBigVGAN**, not a substituted native decoder. The report includes native latent MSE, multi-resolution spectral distance, autocorrelation F0/voicing diagnostics, input/model/source hashes, individual results and equal-weight singer summaries with paired deltas against Exposure and the model. F0 is a lightweight diagnostic estimator (60–1500 Hz), not a validated singing-quality evaluator; `null` means no jointly voiced frames. No perceptual superiority or statistical significance is inferred.

### Cached-target training

After a real diagnostic, merge `rrq_examples/training-overlay.yaml` into the original training data config. Run the same upstream `fairseq-train` recipe for `arm: clean`, `arm: exposure`, and `arm: rrq`, from the same starting checkpoint and with the same update/optimizer/data settings. The extension does not choose those budgets for you.

- Train and validation samples must have fixed prompt text, `volume=None`, matching audio hash, cache ID and frame count. All phrases must fit `batch_max_frames`; implicit frame truncation and random crop are disabled in this mode.
- `rho` is a deterministic, **fixed per-sequence subset** chosen from `(seed,item_name)`, independent of arm and loader ordering. It is not resampled per epoch.
- The cache replaces the entire acoustic token map, so context and labels are consistent. The criterion supervises only codebooks 2 and 3, with a separate loss mask that leaves model attention masks unchanged.
- The full generator's parameters remain whatever the original trainer makes trainable. **Shared parameters and changed suffix history can change future first-code predictions.** This implementation does not claim an independently frozen G1 or identical first-code trajectories after fine-tuning. Fixed cache controls training examples, not post-training inference trajectories.
- `synthetic_smoke` cache is rejected for training. Use held-out singer/song splits before generating training/validation/test caches. The code does not manufacture a split or guarantee pretraining non-overlap.
- Unit/adapter tests and real pretrained codec/vocoder smoke were run locally. **Full Fairseq/T5 generator inference and SVS fine-tuning were not run in this implementation environment.** Python/PyTorch compatibility of that separate full upstream stack remains to be validated.

Code map: `rrq/core.py` native requantization; `rrq/io.py` strict import/manifest; `rrq/pipeline.py` four-arm experiment; `rrq/metrics.py` diagnostics; `rrq/training.py` cached training hook; `rrq/cli.py` commands. Research decisions and review history remain in the parent `SVS_RESEARCH_RECORD.md`.
