from copy import deepcopy
import pytest

from rrq.gtsinger_prepare import assign_splits, build_families, prepare_row


def item(i, singer, title, language='English'):
    return dict(id=str(i),singer=singer,song_title=title,language=language,
                phones=['a','b'],phone_durations=[.1,.2],score={'ep_pitches':[60,64]},
                techniques={k:[0,1] for k in ('mix_tech','falsetto_tech','breathy_tech','pharyngeal_tech','vibrato_tech','glissando_tech')})


def test_families_join_normalized_titles_aliases_and_duplicate_pcm():
    rows=[item(0,'s1','Héllo!'),item(1,'s2','hello'),item(2,'s1','Other'),item(3,'s2','Translated')]
    rows[1]['pcm_sha256']='same';rows[2]['pcm_sha256']='same'
    fam=build_families(rows,[['Other','Translated']])
    assert len(set(fam.values()))==1


def test_split_is_order_invariant_groups_all_takes_and_covers_singers():
    rows=[item(f'{s}-{j}',s,f'song {j}') for s in ['s1','s2'] for j in range(20)]
    fam=build_families(rows,[])
    split=assign_splits(rows,fam,seed=17)
    assert split==assign_splits(list(reversed(rows)),fam,seed=17)
    for j in range(20):assert split[f's1-{j}']==split[f's2-{j}']
    for singer in ['s1','s2']:
        assert {split[r['id']] for r in rows if r['singer']==singer}=={'train','valid','test'}


def test_alignment_does_not_stretch_or_silently_drop_long_utterances():
    r=item('a','s','song');r['phone_durations']=[15.,17.]
    got=prepare_row(r,dict(samplerate=44100,frames=32*44100,channels=1))
    assert got['phone_durations']==[15.,17.]
    assert got['audio_seconds']==32
    with pytest.raises(ValueError,match='duration'):
        prepare_row(r,dict(samplerate=44100,frames=33*44100,channels=1))


def test_technique_controls_must_be_binary_and_aligned():
    r=item('a','s','song');r['techniques']['mix_tech']=[0,2]
    with pytest.raises(ValueError,match='technique'):
        prepare_row(r,dict(samplerate=44100,frames=13230,channels=1))
