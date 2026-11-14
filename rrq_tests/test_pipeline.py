import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from rrq.io import save_prediction
from rrq.pipeline import run_diagnostic
from rrq.upstream import quantizer_class, load_state


class Codec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vq = quantizer_class()(num_quantizers=3, dim=4, codebook_size=1024, kmeans_init=False)
        self.quantizer = type('Quantizer', (), {'vq': self.vq})()
        self.eval().requires_grad_(False)

    def encoder(self, x):
        return x.reshape(1, 1, -1, 320).mean(-1).repeat(1, 4, 1)


class Vocoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eval()

    def forward(self, codes):
        return (codes.float().mean(1, keepdim=True)/1024).repeat_interleave(480, -1)


def fixture(tmp_path, wrong_frames=False):
    torch.manual_seed(8)
    codec = Codec()
    wav = np.sin(2*np.pi*220*np.arange(5120)/16000).astype('float32')*.1
    sf.write(tmp_path/'target.wav', wav, 16000, subtype='FLOAT')
    clean = codec.vq.encode(codec.encoder(torch.from_numpy(wav)[None, None]), n_q=3).squeeze(1).numpy()
    pred = clean.copy(); pred[0, ::5] = (pred[0, ::5]+1)%1024
    if wrong_frames: pred = pred[:, :-1]
    save_prediction(tmp_path/'prediction.npz', pred,
                    {'id':'a', 'origin':'synthetic_smoke', 'frame_rate':50,
                     'generator':'unit-test', 'target_audio_used_for_prediction':True})
    row = {'id':'a','singer':'s','audio':'target.wav','prediction':'prediction.npz',
           'gain':1.,'origin':'synthetic_smoke','alignment':'same_50hz_grid',
           'conditioning':'synthetic'}
    (tmp_path/'manifest.jsonl').write_text(json.dumps(row)+'\n')
    return codec


def test_end_to_end_cache_report_and_all_four_outputs(tmp_path):
    codec = fixture(tmp_path)
    result = run_diagnostic(tmp_path/'manifest.jsonl', tmp_path/'out', codec, Vocoder(), {})
    assert result['origin'] == 'synthetic_smoke'
    assert result['is_research_result'] is False
    assert result['summary']['n_phrases'] == 1
    assert len(list((tmp_path/'out/a').glob('*.wav'))) == 4
    with np.load(tmp_path/'out/a/targets.npz', allow_pickle=False) as data:
        assert set(data.files) == {'clean','predicted','rrq','metadata'}
        assert np.array_equal(data['predicted'][0], data['rrq'][0])
    report = json.loads((tmp_path/'out/report.json').read_text())
    assert report['samples'][0]['clean_prefix_identity']
    with pytest.raises(ValueError):
        run_diagnostic(tmp_path/'manifest.jsonl', tmp_path/'out', codec, Vocoder(), {})


def test_mismatched_frames_fail_without_partial_sample_output(tmp_path):
    codec = fixture(tmp_path, wrong_frames=True)
    with pytest.raises(ValueError, match='frame'):
        run_diagnostic(tmp_path/'manifest.jsonl', tmp_path/'out', codec, Vocoder(), {})
    assert not (tmp_path/'out/report.json').exists()
    assert not (tmp_path/'out/a').exists()


def test_legacy_omegaconf_metadata_loads_with_weights_only(tmp_path):
    from omegaconf import OmegaConf
    p = tmp_path/'legacy.pt'
    torch.save({'codec_model':{'x':torch.ones(2)},'config':OmegaConf.create({'a':[1,2]})},p)
    state = load_state(p)
    assert torch.equal(state['codec_model']['x'],torch.ones(2))


def test_vocoder_loader_freezes_parameters_created_by_weight_norm_removal(tmp_path, monkeypatch):
    from wave_generation import models
    from rrq.upstream import load_vocoder
    class FakeVocoder(torch.nn.Module):
        def __init__(self, config): super().__init__()
        def remove_weight_norm(self): self.weight = torch.nn.Parameter(torch.ones(1))
    monkeypatch.setattr(models, 'CodeBigVGAN', FakeVocoder)
    (tmp_path/'config.json').write_text(json.dumps({'codebook_num':3,'code_num':1024,
                                                  'sampling_rate':24000,'unit_hop_size':480}))
    torch.save({'generator':{}},tmp_path/'weights.pt')
    model = load_vocoder(tmp_path/'weights.pt')
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
