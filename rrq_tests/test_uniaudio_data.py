import json
import subprocess

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from test_uniaudio_workflow import api, vocabulary


def annotation(uid, audio, durations=(0.02, 0.02, 0.0)):
    return dict(id=uid, singer='S', language='Chinese', song_title=f'歌 {uid}',
                audio=str(audio), phones=['a_zh', '<SP>', '<AP>'],
                phone_durations=list(durations), score={'ep_pitches': [60, 0, 99]})


def test_frame_centers_use_last_positive_phone_only_within_endpoint_tolerance():
    data = api()
    row = annotation('x', 'unused.wav')
    vocab = data.build_vocabulary([], vocabulary(), {'a': 0, 'AP': 1, 'SP': 2})
    phones, midi = data.frame_conditions(row, 3, 800, vocab)
    assert phones.tolist() == [3201, 3203, 3203]
    assert midi.tolist() == [60, 0, 0]
    assert data.score_conditions(row, vocab)[1].tolist() == [60, 0]
    with pytest.raises(ValueError, match='duration'):
        data.frame_conditions(row, 5, 1600, vocab)
    row['score']['ep_pitches'][0] = 60.5
    with pytest.raises(ValueError, match='MIDI'):
        data.score_conditions(row, vocab)


def test_prepare_real_gtsinger_records_and_use_acoustic_cache(tmp_path, monkeypatch):
    data = api()
    upstream = tmp_path / 'upstream'
    phone_path = upstream / 'UniAudio/tools/tokenizer/Sing/dict_phone.txt'
    phone_path.parent.mkdir(parents=True)
    phone_path.write_text('a 10\nAP 8\nSP 6\n', encoding='utf8')
    git = ['git', '-C', str(upstream)]
    subprocess.run([*git, 'init', '-q'], check=True)
    subprocess.run([*git, 'add', '.'], check=True)
    subprocess.run([*git, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'Fixture'], check=True)
    head = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(data.ua, 'UPSTREAM_COMMIT', head)
    codec = tmp_path / 'codec'
    codec.mkdir()
    (codec / 'model.pth').write_bytes(b'codec fixture')
    (codec / 'config.yaml').write_text(yaml.safe_dump(dict(generator=dict(name='SoundStream',
        config=dict(D=2, sample_rate=16000, ratios=[8, 5, 4, 2], bins=1024)))))
    (tmp_path / 'model.pth').write_bytes(b'checkpoint fixture')
    (tmp_path / 'model.yaml').write_text(yaml.safe_dump(dict(token_list=vocabulary(),
        block_size=100, non_acoustic_repeat=3, prefix_lm=False)))
    (tmp_path / 'aliases.json').write_text('{"groups": []}')
    rows = []
    for index in range(11):
        audio = tmp_path / f'노래 {index}.wav'
        sf.write(audio, np.full(2400, (index + 1) / 64, dtype=np.float32), 48000)
        rows.append(annotation(f'x{index}', audio))
    rows.extend([annotation('bad-timing', rows[0]['audio'], (1.0, 1.0, 0.0)),
                 dict(rows[1], id='zz-duplicate', song_title='Alias', techniques={'extra': 1})])
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(upstream_path='upstream', codec_path='codec',
        source_path='source.jsonl', aliases_path='aliases.json', checkpoint_path='model.pth',
        model_config_path='model.yaml', prepared_path='prepared', cache_path='cache',
        max_frames=2, prompt_frames=2)))
    cfg = data.load_recipe(config, tmp_path)
    report = data.prepare(cfg)
    assert report['included'] == 11
    assert {record['id'] for record in report['rejected']} == {'bad-timing'}
    assert report['exact_duplicate_pcm'] == [{'id': 'zz-duplicate', 'kept': 'x1'}]
    prepared = data.PreparedData(cfg['prepared_path'], cfg['cache_path'])
    row = prepared.rows['x0']
    assert row['samples'] == 800 and row['frames'] == 3
    for item in prepared.rows.values():
        prepared.cache.put(item['source_sha256'], torch.zeros(1, 2, 3),
                           torch.arange(1, 10).reshape(3, 1, 3), 800)
    first, tail = [segment for segment in prepared.segments if segment['row_id'] == 'x0']
    assert (first['start'], first['stop'], tail['start'], tail['stop']) == (0, 2, 2, 3)
    example = prepared.example(tail)
    start = example['prefix_length']
    assert example['sequence'][0, start:].tolist() == [131, 1158, 2185, 11, 11, 11, 2, 2, 2]
    assert example['latent'].shape == (1, 2, 1)
    prompt, frames = prepared.inference_prefix('x0')
    prepared.cache.path(row['source_sha256']).unlink()
    assert torch.equal(prompt, prepared.inference_prefix('x0')[0]) and frames == 2
    with pytest.raises(FileNotFoundError):
        prepared.example(first)
    with pytest.raises(ValueError, match='immutable'):
        data.prepare(dict(cfg, max_frames=3))
    assert data.prepare(cfg)['included'] == 11
