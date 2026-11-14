import io
import numpy as np
import pytest
from rrq import corpora as c
from rrq.data_download import RangeReader, safe_relative


def test_grouping_keeps_singer_and_technique_variants_together():
    a={'item_name':'English#EN-Alto-1#Breathy#Song!#Control_Group#0000',
       'ph':['AH0_en'], 'ph_durs':[1.0], 'wav_fn':'English/a.wav'}
    b=dict(a,item_name='English#EN-Tenor-1#Vibrato#song#Vibrato_Group#0001')
    x,y=c.gtsinger_row(a),c.gtsinger_row(b)
    assert x['song_group']==y['song_group']
    assert x['split']==y['split']
    assert x['id']!=y['id']
    assert x['techniques'] is None
    assert x['phones']==['AH0_en']


def test_m4_preserves_score_and_missing_technique():
    x=c.m4singer_row({'item_name':'Tenor-1#Song#0001', 'phs':['SP','a'],
        'ph_dur':[.2,.8], 'notes':[0,60], 'notes_dur':[.2,.8], 'txt':'a'})
    assert x['singer']=='m4singer:Tenor-1'
    assert x['techniques'] is None
    assert x['score']['notes']==[0,60]
    assert x['source_audio']=='Tenor-1#Song/0001.wav'


def test_frame_mapping_does_not_stretch_duration():
    assert c.frame_phones(['a','b'],[.025,.035],3,.06)==['a','b','b']
    with pytest.raises(ValueError,match='duration'):
        c.frame_phones(['a'],[.8],50,1.)
    with pytest.raises(ValueError): c.frame_phones(['a'],[-1],50,1.)
    with pytest.raises(ValueError): c.frame_phones(['a'],[float('nan')],50,1.)


def test_phone_mapping_never_silently_converts_foreign_phones():
    assert c.model_phone('<SP>','English')=='SP'
    assert c.model_phone('a_zh','Chinese')=='a'
    assert c.model_phone('AH0_en','English')=='AH0_en'
    assert c.model_phone('a_ja','Japanese')=='a_ja'


def test_selection_balanced_reproducible_and_rejects_bad_annotations():
    rows=[dict(id=f'{s}-{i}',singer=s,phones=['a'],phone_durations=[1.],song_group=str(i))
          for s in ['x','y'] for i in range(9)]
    assert c.select_rows(rows,3)==c.select_rows(list(reversed(rows)),3)
    assert len(c.select_rows(rows,3))==6
    with pytest.raises(ValueError): c.select_rows(rows,0)


@pytest.mark.parametrize('p',['../escape','/absolute','C:/escape','a/../escape','a\\b'])
def test_download_path_escape_rejected(p):
    with pytest.raises(ValueError): safe_relative(p)


class Response(io.BytesIO):
    status=206
    def __init__(self,data,start,end,total):
        super().__init__(data); self.headers={'Content-Range':f'bytes {start}-{end}/{total}'}


def test_range_reader_never_accepts_full_body_or_exceeds_budget():
    source=b'0123456789'
    def opener(req,timeout):
        start,end=map(int,req.get_header('Range').removeprefix('bytes=').split('-'))
        return Response(source[start:end+1],start,end,len(source))
    f=RangeReader('https://example.test/data.zip',max_bytes=8,opener=opener)
    f.seek(-3,2); assert f.read()==b'789'
    f.seek(0); assert f.read(2)==b'01'
    with pytest.raises(ValueError,match='budget'): f.read(4)
    def ignored(req,timeout):
        r=Response(source,0,9,10); r.status=200; return r
    with pytest.raises(ValueError,match='206'): RangeReader('https://example.test/x',opener=ignored)


def test_preprocess_preserves_clock_and_rejects_unknown_export(tmp_path):
    import soundfile as sf
    from rrq.prepare_data import process_row
    raw=tmp_path/'raw'; raw.mkdir()
    sf.write(raw/'x.wav',.1*np.sin(2*np.pi*220*np.arange(4800)/16000),16000,subtype='FLOAT')
    row=c.gtsinger_row(dict(item_name='English#EN-Alto-1#Breathy#Song#Control_Group#0000',
        ph=['AH0_en'],ph_durs=[.3],wav_fn='x.wav'))
    out=tmp_path/'processed'; out.mkdir()
    result=process_row(row,raw,out,{'SP','AP','a'},f0_function=lambda x,sr:np.full(16,220.))
    assert result['n_frames']==15
    assert result['prompt_singer_eligible'] is False
    assert result['unknown_phones']==['AH0_en']
    assert result['conditioning']=='annotation_conditioned'
    assert result['techniques'] is None
    assert result['f0_extra_endpoint_frames']==1
    assert sf.info(result['audio']).samplerate==24000
    arrays=np.load(result['features'])
    assert len(arrays['f0_hz'])==15
    assert result['phone_duration_error_seconds']==pytest.approx(0)


def test_voiceless_and_large_f0_mismatch_are_not_exported(tmp_path):
    import soundfile as sf
    from rrq.prepare_data import process_row
    sf.write(tmp_path/'x.wav',np.zeros(4800),16000)
    row=c.m4singer_row(dict(item_name='Tenor-1#Song#0001',phs=['a'],ph_dur=[.3]))
    row['source_audio']='x.wav'
    out=tmp_path/'out'; out.mkdir()
    with pytest.raises(ValueError,match='silent'):
        process_row(row,tmp_path,out,{'a'},f0_function=lambda x,sr:np.zeros(15))
