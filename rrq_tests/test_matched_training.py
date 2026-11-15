import torch
from rrq.targets import training_pair
from rrq.toksing_model import TokSingDAC


def test_corrupted_prefix_is_context_but_never_coarse_supervision():
    clean=torch.zeros(3,2,5,dtype=torch.long)
    predicted=torch.ones_like(clean);corrected=torch.full_like(clean,2);corrected[0]=1
    choice=torch.tensor([True,False])
    e_context,e_target=training_pair(clean,predicted,corrected,'exposure',choice)
    r_context,r_target=training_pair(clean,predicted,corrected,'rrq',choice)
    assert torch.equal(e_context[0],r_context[0])
    assert torch.equal(e_target[0],clean[0]) and torch.equal(r_target[0],clean[0])
    assert torch.equal(r_context[1:],r_target[1:])
    assert torch.equal(e_target,clean)
    assert (r_target[1:,0]==2).all() and (r_target[:,1]==0).all()


def score():
    return dict(phones=torch.tensor([[1,2]]),midi=torch.tensor([[60,64]]),
        durations=torch.tensor([[.1,.2]]),frame_index=torch.tensor([[0,0,1,1]]),
        singer=torch.tensor([0]),language=torch.tensor([0]),techniques=torch.zeros(1,2,6))


def test_fixed_prefix_generation_has_no_target_suffix_argument():
    model=TokSingDAC(3,1,1,codebooks=3,entries=8,dim=16,layers=1,heads=2,dropout=0).eval()
    prefix=torch.full((1,1,4),3,dtype=torch.long)
    result=model.generate_with_prefix(prefix,**score())
    assert result.shape==(1,3,4) and torch.equal(result[:,:1],prefix)
    assert torch.equal(model.generate(**score()),model.generate_with_prefix(None,**score()))


def test_supplied_technique_condition_changes_score_representation():
    torch.manual_seed(1)
    model=TokSingDAC(3,1,1,codebooks=3,entries=8,dim=16,layers=1,heads=2,dropout=0).eval()
    a=score();b=dict(a,techniques=torch.ones(1,2,6))
    assert not torch.equal(model.encode_score(**a),model.encode_score(**b))


def test_matched_schedule_and_mixture_do_not_depend_on_arm_or_input_order():
    from rrq.experiment import epoch_schedule, mixture_choice
    rows=[dict(id=str(i),language='Chinese' if i<4 else 'English',singer='s'+str(i%2)) for i in range(12)]
    a=epoch_schedule(rows,seed=17,epoch=2,world=2,accumulation=2,chinese_weight=.25)
    b=epoch_schedule(list(reversed(rows)),seed=17,epoch=2,world=2,accumulation=2,chinese_weight=.25)
    assert a==b and len(a)%4==0
    assert [mixture_choice(17,2,i,.5) for i in range(len(a))]==[mixture_choice(17,2,i,.5) for i in range(len(b))]


def test_training_vocabulary_does_not_read_held_out_symbols():
    from rrq.toksing_data import inventory,score_batch
    train=dict(phones=['a'],language='English',singer='s')
    vocab=inventory([train])
    row=dict(train,phones=['unseen'],score={'notes':[60]},phone_durations=[.1])
    batch=score_batch(row,{'frame_index':torch.tensor([0])},vocab,'cpu')
    assert batch['phones'][0,0]==vocab['phones'].index('<UNK>')


def test_equal_prefix_frames_keep_native_labels_despite_rounding_difference():
    from rrq.targets import preserve_clean_frames
    clean=torch.zeros(3,1,2,dtype=torch.long)
    predicted=clean.clone();predicted[0,0,1]=1
    raw=torch.full_like(clean,2);raw[0]=predicted[0]
    corrected=preserve_clean_frames(clean,predicted,raw)
    assert torch.equal(corrected[:,:,0],clean[:,:,0])
    assert torch.equal(corrected[:,:,1],raw[:,:,1])


def test_feasibility_gate_requires_generation_and_both_conditions():
    from rrq.experiment import common_quality_gate
    initial={'language_macro':{'generated':{'sc':1.}}}
    cfg=dict(selection_metric='sc',gate_min_condition_gap=.001,gate_relative_sc=.9,gate_min_languages=1)
    best=dict(language_macro={'generated':{'sc':.5}},diagnostics={'English':{'phone_reverse_ce_gap':.01,'note_shift_ce_gap':-.01}})
    assert not common_quality_gate(initial,best,cfg)['passed']
    best['diagnostics']['English']['note_shift_ce_gap']=.01
    assert common_quality_gate(initial,best,cfg)['passed']
