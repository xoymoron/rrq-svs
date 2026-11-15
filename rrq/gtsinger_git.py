"""Materialize the pinned singing inventory from Git/LFS without resolver requests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.parse
from tqdm import tqdm

from .io import write_json, sha256
from .toksing_acquire import GT_REV, BASE, annotation_manifest
from .data_download import safe_relative


def materialize(root, repo):
    root=Path(root).resolve();repo=Path(repo).resolve()
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if revision!=GT_REV:raise ValueError('Git source revision mismatch')
    source=json.loads((root/'source-inventory.json').read_text())
    if source['revision']!=GT_REV:raise ValueError('source inventory revision mismatch')
    files=source['files'];total=sum(x['size'] for x in files);done=0
    # Missing LFS objects are not replaced by pointer text or silently skipped.
    missing=[]
    for entry in files:
        oid=entry.get('lfs',{}).get('oid')
        if oid and not (repo/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid).is_file():missing.append(entry['path'])
    if missing:raise ValueError(f'{len(missing)} LFS objects still missing; wait for successful git lfs fetch')
    git=subprocess.Popen(['git','-C',str(repo),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
        for index,entry in enumerate(tqdm(files,desc='Verify/materialize singing',mininterval=5)):
            rel=safe_relative(entry['path']);dst=root/'raw'/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            oid=entry.get('lfs',{}).get('oid')
            if oid:
                src=repo/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid
                if src.stat().st_size!=entry['size'] or sha256(src)!=oid:raise ValueError('LFS object verification failed: '+entry['path'])
                if dst.exists():
                    if sha256(dst)!=oid:raise ValueError('existing raw waveform/annotation mismatch')
                else:os.link(src,dst)
                checksum=oid
            else:
                git.stdin.write((entry['oid']+'\n').encode());git.stdin.flush()
                header=git.stdout.readline().decode().split()
                if len(header)!=3 or header[1]!='blob':raise ValueError('missing Git blob')
                data=git.stdout.read(int(header[2]));git.stdout.read(1)
                blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                if len(data)!=entry['size'] or blob!=entry['oid']:raise ValueError('Git blob verification failed')
                checksum=hashlib.sha256(data).hexdigest()
                if dst.exists() and sha256(dst)!=checksum:raise ValueError('existing raw file mismatch')
                if not dst.exists():
                    temp=dst.with_name(dst.name+'.part');temp.write_bytes(data);temp.replace(dst)
            write_json(dst.with_name(dst.name+'.download.json'),dict(
                url=BASE+urllib.parse.quote(entry['path'],safe='/'),bytes=entry['size'],sha256=checksum))
            done+=entry['size']
            if index%500==0 or index==len(files)-1:
                write_json(root/'download-status.json',dict(files_verified=index+1,files_total=len(files),
                    bytes_verified=done,bytes_total=total,fraction=done/total,failures=0,
                    complete=index==len(files)-1,transport='pinned Git/LFS batch'))
    finally:
        git.stdin.close();git.stdout.close();git.wait()
    write_json(root/'download-failures.json',[])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--repo',required=True)
    a=p.parse_args();materialize(a.root,a.repo)


if __name__=='__main__':main()
