"""Contract tests using the actual upstream model, not a surrogate architecture."""
from pathlib import Path
import importlib.util
import torch
import pytest

ROOT = Path(__file__).resolve().parents[2] / 'references' / 'UniAudio'


def adapter():
    assert importlib.util.find_spec('rrq.uniaudio') is not None, 'UniAudio adapter is not implemented'
    from rrq import uniaudio
    return uniaudio


def tiny_model():
    api = adapter()
    upstream = api.upstream_model(ROOT)
    torch.manual_seed(17)
    return upstream.MegaByteModel(n_vocab=96, n_ctx=80, n_state=24,
                                 n_head=4, n_layer=2, prefix_lm=False,
                                 na_repeat=3, local_sos=5, global_sos=4).eval()


def test_cached_generation_matches_original_full_forward_without_future_audio():
    api, model = adapter(), tiny_model()
    prefix = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3]])
    codes = api.generate(model, prefix, frames=4, audio_offset=20,
                         book_size=8, mode='greedy')
    assert codes.shape == (3, 1, 4)
    # Recompute EVERY generated decision with upstream full forward and no
    # cache. Its causal shifts must not see the current/future target.
    ids = api.flatten_codes(codes, 20, 8)
    seq = torch.cat([prefix, ids], dim=1)
    logits = model(seq, torch.zeros_like(seq, dtype=torch.bool))
    for t in range(4):
        for q in range(3):
            scores = logits[0, prefix.numel() + t * 3 + q, 20+q*8:20+(q+1)*8]
            assert scores.argmax().item() == codes[q, 0, t].item()
    assert all(not m._forward_hooks for m in model.modules())


def test_local_rrq_intervention_keeps_global_history_and_q0_logits_fixed():
    api, model = adapter(), tiny_model()
    seq = torch.randint(6, 95, (1, 24))
    mask = torch.ones_like(seq, dtype=torch.bool)
    changed = seq.clone(); changed[:, 12::3] = 6
    vanilla = model(seq, mask)
    assert torch.allclose(api.depth_forward(model, seq, seq, mask), vanilla, atol=1e-6)
    intervened = api.depth_forward(model, seq, changed, mask)
    assert torch.allclose(intervened[:, ::3], vanilla[:, ::3], atol=1e-6)
    assert not torch.allclose(intervened[:, 13], vanilla[:, 13])


def test_no_silent_frame_crop_or_unknown_phone():
    api = adapter()
    assert api.native_phone('<AP>', 'Chinese') == 'AP'
    assert api.native_phone('l_zh', 'Chinese') == 'l'
    with pytest.raises(ValueError, match='Mandarin'):
        api.native_phone('a_kr', 'Korean')
    with pytest.raises(ValueError, match='frame'):
        api.generate(tiny_model(), torch.ones((1, 3), dtype=torch.long),
                     frames=0, audio_offset=20, book_size=8)


def test_native_rvq_reuses_existing_rrq_and_keeps_codebooks_frozen():
    api = adapter()
    api.add_upstream_path(ROOT)
    from tools.tokenizer.soundstream.quantization.core_vq_lsx_version import ResidualVectorQuantization
    from rrq.core import requantize
    torch.manual_seed(19)
    rvq = ResidualVectorQuantization(num_quantizers=3, dim=8, codebook_size=16,
                                     kmeans_init=False).eval().requires_grad_(False)
    latent = torch.randn(1, 8, 9)
    clean = rvq.encode(latent)
    before = {k: v.clone() for k, v in rvq.state_dict().items()}
    assert torch.equal(requantize(latent, clean[:1], rvq.layers), clean)
    predicted = (clean[:1] + 1) % 16
    relabeled = requantize(latent, predicted, rvq.layers)
    assert torch.equal(relabeled[:1], predicted)
    assert all(torch.equal(v, rvq.state_dict()[k]) for k, v in before.items())


def test_singing_conditions_match_original_collator():
    # Execute the actual upstream collator class without importing T5/fairseq
    # and other tokenizers that the singing task does not use.
    import ast
    api = adapter()
    path = ROOT / 'UniAudio/utils/dataloader.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Collate_Fn_Factory')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    special = ['<start>', '<end>', '<sing_task>', '<sing_phone_start>', '<sing_phone_end>',
               '<sing_midi_start>', '<sing_midi_end>', '<audio_prompt_start>', '<audio_prompt_end>',
               '<audio_start>', '<audio_end>']
    tokens = special + ['unused'] * (128-len(special))
    tokens += [f'<audio_{i}>' for i in range(3073)]
    tokens += [f'<sing_phone_{i}>' for i in range(68)]
    tokens += [f'<sing_midi_{i}>' for i in range(138)]
    class Identity:
        is_discrete = True
        def tokenize(self, x, *args): return x
    types = ['sing_phone', 'sing_midi', 'audio_prompt', 'audio']
    bias = {t: tokens.index('<audio_0>' if t == 'audio_prompt' else f'<{t}_0>') for t in types}
    collator = namespace['Collate_Fn_Factory']({t: Identity() for t in types}, tokens, bias, 3)
    prompt = torch.randint(0, 1024, (3, 1, 7))
    phones = torch.tensor([1, 2, 3, 4]); midi = torch.tensor([60, 60, 62, 0])
    data = dict(task='sing', sing_phone=phones, sing_midi=midi,
                audio_prompt=api.flatten_codes(prompt, 0)[0], audio=torch.tensor([1,1025,2049]))
    seq, mask, _, _ = collator.splice_sequence(data, types, types, 'audio')
    seq = collator.add_special_tokens(seq, 'global')
    mask = collator.add_special_tokens(mask, 'mask')
    stop = (~mask).sum().item() + 3
    prefix = api.singing_prefix(tokens, phones+bias['sing_phone'], midi, prompt)
    assert torch.equal(seq[:stop], prefix[0])


def test_pilot_prompt_is_other_train_song_and_test_is_untouched():
    assert importlib.util.find_spec('rrq.uniaudio_pilot') is not None, 'pilot not implemented'
    from rrq.uniaudio_pilot import select_panel
    rows = []
    for split, song in [('train','A'), ('valid','B'), ('test','C')]:
        rows.append(dict(id=split, split=split, song_group=song, singer='s1',
                         language='Chinese', phone_durations=[4.0]))
    panel = select_panel(rows, per_singer=2)
    assert len(panel) == 1
    assert panel[0]['target']['id'] == 'valid'
    assert panel[0]['prompt']['id'] == 'train'


def test_half_inference_preserves_upstream_float_layer_norm():
    api = adapter()
    assert hasattr(api, 'inference_precision'), 'mixed precision conversion is missing'
    model = api.inference_precision(tiny_model(), 'cpu', torch.float16)
    prefix = torch.tensor([[1,1,1,2,2,2]])
    output = api.generate(model, prefix, 2, 20, 8, mode='greedy')
    assert output.shape == (3,1,2)
    assert model.g_ln.weight.dtype == torch.float32
    assert model.g_emb.weight.dtype == torch.float16


def test_load_full_official_checkpoint_numeric_reporter(tmp_path):
    import datetime
    import numpy as np
    api = adapter()
    assert hasattr(api, 'load_release'), 'released checkpoint loader is missing'
    path = tmp_path/'checkpoint.pt'
    torch.save(dict(model={'x':torch.arange(3)}, reporter={
        'time':datetime.timedelta(seconds=2), 'loss':np.float64(1.2)}),path)
    loaded = api.load_release(path)
    assert torch.equal(loaded['model']['x'],torch.arange(3))
    assert loaded['reporter']['loss'] == 1.2


def test_failed_pilot_cannot_report_success(tmp_path):
    import json
    from types import SimpleNamespace
    from rrq.uniaudio_pilot import aggregate
    (tmp_path/'panel.json').write_text(json.dumps({'items':[{'target':{'id':'a'}}]}))
    (tmp_path/'a').mkdir()
    (tmp_path/'a/result.json').write_text(json.dumps({'id':'a','status':'failed','error':'dependency'}))
    with pytest.raises(RuntimeError, match='failed'):
        aggregate(SimpleNamespace(output=str(tmp_path),wandb=False))
    assert json.loads((tmp_path/'summary.json').read_text())['successful'] == 0
