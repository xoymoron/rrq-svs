import pytest


def test_breath_marker_is_not_a_sung_c7_and_syllables_keep_real_timing():
    from rrq.aihub473 import segment_units
    notes=[dict(start_time='0.2',end_time='0.4',midi_num=96,is_breath=True,lyric=''),
           dict(start_time='0.6',end_time='1.0',midi_num=60,is_breath=False,lyric='가')]
    units,durations,pitches=segment_units(notes,0,1.2)
    assert units==['<SP>','<AP>','<SP>','가','<SP>']
    assert durations==pytest.approx([.2,.2,.2,.4,.2])
    assert pitches==[0,0,0,60,0]


def test_overlapping_notes_are_rejected_not_silently_retimed():
    from rrq.aihub473 import segment_units
    notes=[dict(start_time=0,end_time=.8,midi_num=60,is_breath=False,lyric='가'),
           dict(start_time=.5,end_time=1,midi_num=62,is_breath=False,lyric='나')]
    with pytest.raises(ValueError,match='overlap'):
        segment_units(notes,0,1)
