"""Current multilingual asset preparation and shared runtime loading.

Training and generation live exclusively in rrq.experiment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .acoustic import load_acoustic
from .io import sha256, write_json, safe_id, validate_codes
from .multilingual import Vocabulary, read_rows
from .pipeline import prepare_audio
from .upstream import load_codec
from .data_download import free_space

ACOUSTIC='prompt-singer-flant5-large-finetuned/checkpoint_last.pt'
T5='flant5-large-finetuned/checkpoint-40000'


def prompt_cache(args):
    from transformers import T5Tokenizer, T5EncoderModel
    root=args.assets/T5
    out=args.output
    if out.exists(): raise ValueError('prompt cache exists; do not overwrite provenance')
    tokenizer=T5Tokenizer.from_pretrained(root,local_files_only=True)
    model=T5EncoderModel.from_pretrained(root,local_files_only=True).eval().requires_grad_(False).to(args.device)
    tokens=tokenizer('A singer sings.',truncation=True,max_length=77,padding='max_length',return_tensors='pt')['input_ids'].to(args.device)
    # Match upstream: padding tokens are included, no attention_mask is passed.
    with torch.inference_mode(): feature=model(input_ids=tokens).last_hidden_state.cpu()
    if feature.shape!=(1,77,1024) or not torch.isfinite(feature).all(): raise ValueError('invalid T5 feature')
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(feature=feature,prompt='A singer sings.',
                    checkpoint_sha256=sha256(root/'pytorch_model.bin'),upstream_padding_policy=True),out)
    print('PROMPT_CACHE_OK',list(feature.shape),flush=True)


def codec_cache(args):
    rows=read_rows([args.manifest]); root=args.output
    root.mkdir(parents=True,exist_ok=True)
    checkpoint=args.assets/'codec/ckpt_01135000.pth'
    codec=load_codec(checkpoint,args.device); digest=sha256(checkpoint)
    for i,row in enumerate(rows):
        dest=root/(safe_id(row['id'])+'.npz')
        audio_hash=sha256(row['audio'])
        if dest.exists():
            load_codes(dest,row,codec_sha256=digest)
        else:
            pcm,_,_=prepare_audio(row['audio'],1.)
            with torch.inference_mode():
                z=codec.encoder(torch.from_numpy(pcm)[None,None].to(args.device))
                codes=codec.quantizer.vq.encode(z,n_q=3)[:,0].cpu().numpy()
            if codes.shape!=(3,row['n_frames']): raise ValueError('codec frame count mismatch')
            free_space(root,1024**2)
            np.savez_compressed(dest,codes=codes,metadata=np.array(json.dumps(dict(id=row['id'],
                target_audio_sha256=audio_hash,codec_sha256=digest,frame_rate=50))))
        if (i+1)%25==0 or i+1==len(rows): print(f'CODEC_CACHE {i+1}/{len(rows)}',flush=True)
    write_json(root/'report.json',dict(items=len(rows),manifest_sha256=sha256(args.manifest),codec_sha256=digest))


def load_codes(path,row,codec_sha256=None):
    with np.load(path,allow_pickle=False) as f:
        codes=validate_codes(f['codes']).copy(); meta=json.loads(str(f['metadata'].item()))
    if meta['id']!=row['id'] or meta['target_audio_sha256']!=sha256(row['audio']) or meta['frame_rate']!=50:
        raise ValueError('stale codec cache')
    if codec_sha256 is not None and meta['codec_sha256']!=codec_sha256: raise ValueError('codec checkpoint changed')
    if codes.shape!=(3,row['n_frames']): raise ValueError('codec cache frame mismatch')
    return codes


def setup(args):
    vocab=Vocabulary.load(args.bundle/'vocabulary.json')
    info=json.loads((args.bundle/'vocabulary.json').read_text())
    model,old,cfg=load_acoustic(args.assets/ACOUSTIC,len(vocab.tokens),args.device)
    if old!=info['base_size']: raise ValueError('checkpoint/dictionary base vocabulary mismatch')
    if model.global_model.prefix_lm: raise ValueError('expected published causal checkpoint')
    prompt_data=torch.load(args.prompt,map_location='cpu',weights_only=True)
    if prompt_data['checkpoint_sha256']!=sha256(args.assets/T5/'pytorch_model.bin') or prompt_data['prompt']!='A singer sings.' or not prompt_data.get('upstream_padding_policy'):
        raise ValueError('prompt feature provenance differs from official fixed prompt')
    prompt=prompt_data['feature'].to(args.device)
    if prompt.shape!=(1,77,cfg['text_enc_dim']) or not torch.isfinite(prompt).all(): raise ValueError('invalid prompt feature')
    return model,vocab,old,cfg,prompt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--assets',type=Path,default=Path('artifacts/pretrained'))
    sub=p.add_subparsers(dest='command',required=True)
    prompt=sub.add_parser('prompt'); prompt.add_argument('--output',type=Path,required=True)
    cache=sub.add_parser('cache'); cache.add_argument('--manifest',type=Path,required=True); cache.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); torch.manual_seed(1234); torch.set_num_threads(2)
    dict(prompt=prompt_cache,cache=codec_cache)[args.command](args)


if __name__=='__main__': main()
