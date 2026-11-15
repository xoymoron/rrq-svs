import importlib.util
import pytest


def api():
    assert importlib.util.find_spec('rrq.pronunciation'), 'pronunciation evaluator missing'
    from rrq import pronunciation
    return pronunciation


def test_lyric_normalization_removes_annotation_tags_but_not_words():
    mod=api()
    class Converter:
        def convert(self,text):return text.replace('夢','梦')
    assert mod.normalize(['夢','想','<AP>','，','Ａ','！'],Converter())=='梦想a'
    assert mod.normalize('梦 想 a',Converter())=='梦想a'


def test_cer_accounts_for_deletions_insertions_and_empty_output():
    mod=api()
    assert mod.edit_errors('梦想','梦')==1
    assert mod.edit_errors('梦想','梦想呀')==1
    assert mod.edit_errors('梦想','理想')==1
    assert mod.edit_errors('梦想','')==2
    with pytest.raises(ValueError,match='empty'):
        mod.cer('','anything')


def test_bad_reference_asr_cannot_pass_pronunciation_gate():
    mod=api();thresholds={'reference_cer_max':.25,'codec_cer_max':.3,'generated_cer_max':.3,'excess_cer_max':.15}
    good={'reference_cer':.1,'codec_cer':.1,'generated_cer':.2,'generated_excess_cer':.1}
    assert mod.gate(good,thresholds)
    assert not mod.gate(dict(good,reference_cer=.6),thresholds)
    assert not mod.gate(dict(good,generated_cer=.8),thresholds)


def test_codec_diagnostic_policy_still_requires_generated_and_reference_quality():
    mod=api();thresholds={'reference_cer_max':.25,'codec_cer_max':.3,'generated_cer_max':.3,'excess_cer_max':.15}
    good={'reference_cer':.1,'codec_cer':.7,'generated_cer':.2,'generated_excess_cer':.1}
    assert mod.gate(good,thresholds,codec_diagnostic_only=True)
    assert not mod.gate(good,thresholds)
    assert not mod.gate(dict(good,generated_cer=.8),thresholds,codec_diagnostic_only=True)


def test_multilingual_asr_uses_requested_language_and_separate_cache(monkeypatch,tmp_path):
    from types import SimpleNamespace
    mod=api()
    class Recognizer:
        def transcribe(self,path,**kwargs):
            return [SimpleNamespace(text={'zh':'夢想','en':'Hello'}[kwargs['language']])],None
    monkeypatch.setattr(mod,'recognizer',lambda *args:(Recognizer(),{'language':'zh','normalization':'old'}))
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixed audio identity')
    settings={'model_path':'unused','cache_path':str(tmp_path/'cache')}
    en,ce=mod.transcribe(audio,settings,0,language='en')
    zh,cz=mod.transcribe(audio,settings,0,language='zh')
    assert en['normalized']=='hello' and zh['normalized']=='梦想'
    assert en['key']!=zh['key'] and ce['language']=='en' and cz['language']=='zh'
    assert len(list((tmp_path/'cache').glob('*.json')))==2
    assert mod.normalize('夢',language='ja')=='夢'
