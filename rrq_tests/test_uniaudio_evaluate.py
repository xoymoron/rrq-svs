import importlib.util
import numpy as np
import torch


def module():
    assert importlib.util.find_spec('rrq.uniaudio_evaluate'), 'SVS evaluation is missing'
    from rrq import uniaudio_evaluate
    return uniaudio_evaluate


def test_counterfactual_changes_only_requested_condition():
    mod=module()
    tokens=['pad','<sing_phone_start>','<sing_phone_end>','<sing_midi_start>',
            '<sing_midi_end>','<sing_midi_0>']
    seq=torch.tensor([[1,1,1,20,20,20,21,21,21,2,2,2,3,3,3,65,65,65,5,5,5,4,4,4,99,99,99]])
    phone=mod.condition_variant(seq,tokens,'phone')
    assert phone[0,3:9].tolist()==[21,21,21,20,20,20]
    assert torch.equal(phone[:,9:],seq[:,9:])
    midi=mod.condition_variant(seq,tokens,'midi')
    assert midi[0,15:21].tolist()==[66,66,66,5,5,5]
    assert torch.equal(midi[:,:15],seq[:,:15])
    assert torch.equal(midi[:,21:],seq[:,21:])


def test_metrics_keep_missing_tail_instead_of_cropping():
    a,b=module().pad_pair(np.ones(10),np.ones(7))
    assert len(a)==len(b)==10
    assert (b[7:]==0).all()


def test_gate_requires_every_language_and_is_not_intelligibility_claim():
    mod=module()
    thresholds=dict(within_50_cents_min=.5,voiced_recall_min=.7,voicing_error_max=.2,
                    f0_rmse_cents_max=300.,spectral_convergence_max=.9,
                    phone_ce_gap_min=0.,midi_ce_gap_min=0.)
    good=dict(within_50_cents=.8,voiced_recall=.9,voicing_error=.1,f0_rmse_cents=100.,
              mrstft_spectral_convergence=.5,phone_ce_gap=.1,midi_ce_gap=.1)
    assert mod.operational_gate({'Chinese':good},thresholds,['Chinese'])
    assert not mod.operational_gate({'Chinese':good},thresholds,['Chinese','English'])
    assert not mod.operational_gate({'Chinese':dict(good,phone_ce_gap=-.1)},thresholds,['Chinese'])


def test_chunk_synthesis_uses_annotation_only_and_keeps_final_partial_chunk(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from rrq import uniaudio_data, uniaudio
    mod = module()
    phones, midi = torch.arange(5), torch.tensor([60, 61, 62, 63, 64])
    monkeypatch.setattr(uniaudio_data, 'score_conditions', lambda row, vocab: (phones, midi))
    seen = []
    class Data:
        rows = {'row': {'annotations': 'only'}}
        vocab = {'tokens': ['<audio_0>']}
        recipe = {'config': {'max_frames': 2}}
        def _prefix(self, row, ph, notes, device, ending):
            seen.append((ph.tolist(), notes.tolist()))
            return notes[None]
        def _cached(self, *args):
            raise AssertionError('must not open target cache')
    def generate(model, prefix, frames, audio_offset, **kwargs):
        assert frames <= 2
        return prefix[None].expand(3, 1, frames)
    monkeypatch.setattr(uniaudio, 'generate', generate)
    runtime = SimpleNamespace(device='cpu', amp=nullcontext)
    output = mod.synthesize({'synthesis_chunk_frames': 2}, Data(), 'row', object(), runtime)
    assert output.shape == (3, 1, 5)
    assert output[0, 0].tolist() == [60, 61, 62, 63, 64]
    assert seen == [([0, 1], [60, 61]), ([2, 3], [62, 63]), ([4], [64])]
