"""Pinned, hash-verified official codec/vocoder download (about 794 MB)."""
from pathlib import Path
import urllib.request

from .io import sha256, write_json

REVISION = '4a9ad215081865e168df26ea4a54cc78e87378a6'
FILES = {
    'codec/config.yaml': {'bytes':2778,'sha256':'85a37659b7607ce04a544aa39950a5fde628bd26f978cc3e67869f186dfb9069'},
    'codec/ckpt_01135000.pth': {'bytes':291809077,'sha256':'52b341c9dc01b019d126d56858261812ae8c0f92a7497d6842057bbef4ff7697'},
    'vocoder_24k/config.json': {'bytes':1109,'sha256':'7412d3a1cde8bf8c8bbe1624c55f8c4ce09f736966f2d3f5390667ee9dc4df77'},
    'vocoder_24k/g_00885000': {'bytes':502009752,'sha256':'c544a3cddac5416aadcb93c0f041778e59f6f80cafe29800363fbd95e628a145'},
}


def download(url,path):
    with urllib.request.urlopen(url,timeout=60) as source, path.open('wb') as dest:
        while block := source.read(1024*1024):
            dest.write(block)


def fetch_assets(output,files=None):
    output = Path(output).resolve()
    files = FILES if files is None else files
    for rel, expected in files.items():
        dest = (output/rel).resolve()
        if not dest.is_relative_to(output):
            raise ValueError('invalid asset path')
        dest.parent.mkdir(parents=True,exist_ok=True)
        temporary = dest.with_name(dest.name+'.part')
        if not dest.exists():
            if temporary.exists():
                raise ValueError(f'incomplete download exists: {temporary}; remove it explicitly before retry')
            print('Downloading '+rel,flush=True)
            download('https://huggingface.co/Cyanbox/Prompt-Singer/resolve/'+REVISION+'/'+rel,temporary)
            candidate = temporary
        else:
            candidate = dest
        if candidate.stat().st_size != expected['bytes'] or sha256(candidate) != expected['sha256']:
            raise ValueError('asset size/hash mismatch: '+str(candidate))
        if candidate == temporary:
            temporary.replace(dest)
    write_json(output/'provenance.json',{'repo':'Cyanbox/Prompt-Singer','revision':REVISION,'files':files})
