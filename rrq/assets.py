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

BACKBONE_FILES = {
    'prompt-singer-flant5-large-finetuned/checkpoint_last.pt': {'bytes':5324119048,'sha256':'69b23f208c842800b2312b81ff7f6c71bc829269515d603505a3b15d9da50f62'},
    'flant5-large-finetuned/checkpoint-40000/pytorch_model.bin': {'bytes':3132793669,'sha256':'e3bfdee4fd607dbf15bc2c88ccf40cabf53b1ba99592314b0105b19e6165963f'},
    'flant5-large-finetuned/checkpoint-40000/config.json': {'bytes':828,'sha256':'5368d6a085a7ba6d6e51514cec780489469db7cc9386e5a17c110836b046dc04'},
    'flant5-large-finetuned/checkpoint-40000/special_tokens_map.json': {'bytes':2201,'sha256':'5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f'},
    'flant5-large-finetuned/checkpoint-40000/spiece.model': {'bytes':791656,'sha256':'d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86'},
    'flant5-large-finetuned/checkpoint-40000/tokenizer_config.json': {'bytes':2367,'sha256':'27c113258803c4e8a0f9ef7245cc773d28f5a39c0b8a266950b91b9cb39768b9'},
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


def main():
    import argparse
    from .data_download import free_space
    parser=argparse.ArgumentParser(description='Pinned official codec and optional multilingual backbone assets')
    parser.add_argument('--output',type=Path,default=Path('artifacts/pretrained'))
    parser.add_argument('--with-backbone',action='store_true')
    args=parser.parse_args()
    files=dict(FILES,**BACKBONE_FILES) if args.with_backbone else FILES
    args.output.mkdir(parents=True,exist_ok=True)
    remaining=sum(v['bytes'] for k,v in files.items() if not (args.output/k).exists())
    free_space(args.output,remaining)
    fetch_assets(args.output,files)


if __name__=='__main__': main()
