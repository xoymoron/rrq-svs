import pytest
from rrq.compare_matched import compare


def report(delta):
    return dict(protocol_sha256='same',split='valid',samples=[
        dict(id=str(i),language='English',song_group='song'+str(i//2),
             arms={'generated':{'mrstft_spectral_convergence':1.+delta}}) for i in range(8)])


def test_paired_song_bootstrap_is_reproducible_and_requires_exact_coverage():
    reports={'baseline':report(0),'exposure':report(-.1),'rrq':report(-.2)}
    a=compare(reports,repetitions=100);b=compare(reports,repetitions=100)
    assert a==b
    assert a['rrq_minus_exposure']['delta']==pytest.approx(-.1)
    assert a['rrq_minus_exposure']['ci95']==pytest.approx([-.1,-.1])
    reports['rrq']['samples'].pop()
    with pytest.raises(ValueError,match='coverage'):compare(reports,repetitions=100)
