"""Promote verified legacy native tokens without instantiating or running DAC."""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .cache_store import AcousticCache, fingerprint
from .io import safe_id, sha256, write_json
from .toksing_data import read_rows


def migrate(legacy,manifest,codec,shared,report):
    legacy=Path(legacy);manifest=Path(manifest)
    protocol=json.loads((legacy/'protocol.json').read_text())
    expected=dict(protocol);digest=expected.pop('sha256')
    if fingerprint(expected)!=digest:raise ValueError('legacy protocol checksum mismatch')
    store=AcousticCache(shared,sha256(codec))
    if (protocol['codec_sha256']!=store.codec_sha256 or
        protocol['manifest_sha256']!=sha256(manifest)):
        raise ValueError('legacy codec/manifest mismatch')
    for name in ('dac_codec.py','toksing_data.py'):
        if protocol['implementation'][name]!=sha256(Path(__file__).parent/name):
            raise ValueError('legacy audio encoding implementation changed: '+name)
    env=protocol['environment']
    if env['packages']!=store.contract['packages'] or env['cuda']!=store.contract['cuda']:
        raise ValueError('legacy audio encoding environment changed')
    audit_current=protocol['implementation'].get('core.py')==sha256(Path(__file__).with_name('core.py'))
    rows=read_rows(manifest);verified=[];created=0
    for row in tqdm(rows,desc='Promote existing DAC tokens',mininterval=2):
        path=legacy/'cache'/(safe_id(row['id'])+'.pt')
        original=sha256(path);old=torch.load(path,map_location='cpu',weights_only=True)
        if (old['id']!=row['id'] or old['source_sha256']!=row['source_sha256'] or
            old['codec_sha256']!=store.codec_sha256):raise ValueError('legacy token provenance mismatch')
        existed=store.path(row['source_sha256']).exists()
        old['clean_prefix_audit_recipe']=store.audit_recipe if audit_current else None
        new=store.put(row['source_sha256'],old)
        if not torch.equal(old['codes'].long(),new['codes'].long()):raise ValueError('migration changed tokens')
        if sha256(path)!=original:raise ValueError('legacy file changed during migration')
        created+=int(not existed);verified.append((row['id'],original,new['codes_sha256']))
    result=dict(n=len(verified),created=created,reused=len(verified)-created,
        exact_tokens_verified=len(verified),legacy_files_unchanged=len(verified),encoder_calls=0,
        legacy_protocol_sha256=digest,encoding_contract_sha256=store.contract_sha256,
        verification_sha256=fingerprint(verified),acoustic_cache=str(store.root))
    write_json(report,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-output',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--codec',required=True);p.add_argument('--acoustic-cache',required=True)
    p.add_argument('--report',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    print(json.dumps(migrate(a.legacy_output,a.manifest,a.codec,a.acoustic_cache,a.report)))


if __name__=='__main__':main()
