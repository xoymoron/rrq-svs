import pytest
import torch
from torch import nn

from rrq.core import requantize


def test_native_dac_clean_and_forced_prefix():
    from rrq.dac_codec import DACLayer
    from dac.nn.quantize import ResidualVectorQuantize
    torch.manual_seed(19)
    q = ResidualVectorQuantize(input_dim=8, n_codebooks=3, codebook_size=16,
                              codebook_dim=4).eval().requires_grad_(False)
    layers = nn.ModuleList([DACLayer(layer) for layer in q.quantizers]).eval()
    z = torch.randn(2, 8, 7)
    native = q(z)[1].permute(1, 0, 2)
    assert torch.equal(requantize(z, native[:1], layers), native)
    prefix = (native[:1] + 3) % 16
    actual = requantize(z, prefix, layers)
    residual = z - q.quantizers[0].out_proj(q.quantizers[0].decode_code(prefix[0]))
    for k in range(1, 3):
        decoded, _, _, ids, _ = q.quantizers[k](residual)
        assert torch.equal(actual[k], ids)
        residual -= decoded
    assert torch.equal(actual[:1], prefix)


def test_frame_clock_uses_codec_hop_and_rejects_misalignment():
    from rrq.toksing_data import frame_condition
    row = dict(phones=['a','b'], phone_durations=[.1,.2],
               score={'ep_pitches':[60,64]}, language='Korean')
    got = frame_condition(row, frames=26, samples=13230, sr=44100, hop=512)
    assert got['phone_index'].tolist() == [0]*9+[1]*17
    assert got['midi'].tolist() == [60]*9+[64]*17
    with pytest.raises(ValueError, match='duration'):
        frame_condition(row, frames=30, samples=18000, sr=44100, hop=512)


def test_model_prefix_causality_and_score_only_generation():
    from rrq.toksing_model import TokSingDAC
    torch.manual_seed(5)
    model = TokSingDAC(10, 2, 2, codebooks=3, entries=8, dim=16,
                       layers=1, heads=2, dropout=0).eval()
    batch = dict(phones=torch.tensor([[1,2]]), midi=torch.tensor([[60,64]]),
                 durations=torch.tensor([[.1,.2]]),
                 frame_index=torch.tensor([[0,0,1,1,1]]),
                 singer=torch.tensor([0]), language=torch.tensor([1]))
    a = torch.zeros(1,3,5,dtype=torch.long)
    b = a.clone(); b[:,0]=1
    logits_a = model(**batch, codes=a)
    logits_b = model(**batch, codes=b)
    assert torch.equal(logits_a[:,0], logits_b[:,0])
    assert not torch.equal(logits_a[:,1], logits_b[:,1])
    b = a.clone(); b[:,2]=2
    assert torch.equal(logits_a, model(**batch,codes=b))
    prediction = model.generate(**batch)
    assert prediction.shape == (1,3,5)
    assert torch.equal(prediction[:,0], logits_a[:,0].argmax(-1))
    model.train(); model(**batch,codes=a).sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_validation_does_not_duplicate_uneven_shards():
    from rrq.toksing_run import shard
    rows = list(range(7))
    assert sorted(shard(rows,0,2)+shard(rows,1,2)) == rows


def test_cache_rejects_changed_score_annotations(tmp_path):
    from rrq.toksing_run import read_cache
    from rrq.toksing_data import annotation_hash
    row=dict(id='test',phones=['a'],phone_durations=[.1],score={'notes':[60]},
             language='Korean',singer='s',split='train')
    (tmp_path/'cache').mkdir()
    torch.save(dict(id='test',annotation_sha256=annotation_hash(row)),tmp_path/'cache/test.pt')
    read_cache(tmp_path,row)
    with pytest.raises(ValueError,match='annotation'):
        read_cache(tmp_path,dict(row,phone_durations=[.2]))


def test_rrq_map_preserves_forced_prefix_and_corrected_context():
    from rrq.targets import training_map
    clean=torch.zeros(9,2,5,dtype=torch.long)
    predicted=torch.ones_like(clean)
    corrected=torch.full_like(clean,2);corrected[0]=predicted[0]
    choice=torch.tensor([True,False])
    rrq=training_map(clean,predicted,corrected,'rrq',choice)
    exposure=training_map(clean,predicted,corrected,'exposure',choice)
    assert torch.equal(rrq[0],exposure[0])
    assert (rrq[1:,0]==2).all() and (exposure[1:,0]==0).all()
    assert torch.equal(rrq[:,1],clean[:,1])
