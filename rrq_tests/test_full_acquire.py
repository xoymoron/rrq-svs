from rrq.toksing_acquire import singing_files


def test_full_singing_selection_preserves_all_languages_and_annotations():
    paths = [
        'Chinese/CN-Soprano/Glissando/song/Control_Group/0001.wav',
        'English/EN-Tenor/Breathy/Speech song/Control_Group/0001.musicxml',
        'Korean/KR-Soprano/Vibrato/song/Technique_Group/0001_TextGrid',
        'English/EN-Tenor/Breathy/song/Speech/0001.wav',
        'processed/English/metadata.json',
    ]
    rows = [dict(type='file', path=p, size=200_000_000) for p in paths]
    assert [r['path'] for r in singing_files(rows)] == paths[:3]


def test_full_singing_selection_rejects_unsafe_remote_paths():
    import pytest
    with pytest.raises(ValueError):
        singing_files([dict(type='file', path='English/a/b/c/d/../../escape', size=1)])


def test_archive_progress_does_not_double_count_merging_parts(tmp_path):
    from rrq.acquisition_watch import archive_progress
    (tmp_path / 'TS1.zip.part0').write_bytes(b'1234')
    (tmp_path / 'TS1.zip.part1').write_bytes(b'5678')
    (tmp_path / 'TS1.zip').write_bytes(b'12345678')
    status, archives = archive_progress([tmp_path])
    assert status['received_bytes'] == 8
    assert status['archives_present'] == 1
    assert set(archives) == {'TS1.zip'}
