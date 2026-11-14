import hashlib

import pytest

from rrq.assets import fetch_assets


def test_asset_hash_checked_and_existing_corrupt_file_not_overwritten(tmp_path,monkeypatch):
    import rrq.assets as assets
    data=b'known official fixture'
    files={'codec/test.pt':{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}}
    def download(url,path): path.write_bytes(data)
    monkeypatch.setattr(assets,'download',download)
    fetch_assets(tmp_path,files)
    assert (tmp_path/'codec/test.pt').read_bytes()==data
    (tmp_path/'codec/test.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'): fetch_assets(tmp_path,files)
    assert (tmp_path/'codec/test.pt').read_bytes()==b'corrupt'


def test_bad_download_never_becomes_checkpoint(tmp_path,monkeypatch):
    import rrq.assets as assets
    monkeypatch.setattr(assets,'download',lambda url,path:path.write_bytes(b'bad'))
    with pytest.raises(ValueError): fetch_assets(tmp_path,{'x.pt':{'sha256':'0'*64,'bytes':3}})
    assert not (tmp_path/'x.pt').exists()
