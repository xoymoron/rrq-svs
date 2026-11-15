"""Isolated two-GPU maximum-length forward/backward check; never a research run."""
import argparse
import json
from pathlib import Path
import time
import os

import numpy as np
import torch
import torch.nn.functional as F

from .core import requantize
from .dac_codec import FrozenDAC
from .distributed import Distributed
from .experiment import model_config, fingerprint
from .io import write_json, sha256
from .targets import training_pair
from .toksing_data import inventory,frame_condition,score_batch
from .toksing_model import TokSingDAC


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    p.add_argument('--codec',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    d=Distributed();d.start()
    try:
        device=torch.device(f'cuda:{d.local_rank}')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rows=[json.loads(x) for x in Path(a.source).read_text().splitlines() if x]
        row=max(rows,key=lambda r:sum(r['phone_durations']))
        cfg=json.loads(Path(a.config).read_text());vocab=inventory(rows)
        if d.world!=cfg['world_size']:raise ValueError('verify the actual requested GPU world size')
        codec=FrozenDAC(a.codec,device);samples=round(sum(row['phone_durations'])*44100)
        # Maximum source length and real score, but synthetic silence exercises memory,
        # numerical kernels and DDP only. No learning-quality inference is permitted.
        z,clean=codec.encode(torch.zeros(1,1,samples,device=device))
        cond=frame_condition(row,clean.shape[-1],samples,44100,512)
        cache={'frame_index':torch.from_numpy(cond['phone_index'])}
        batch=score_batch(row,cache,vocab,device)
        torch.manual_seed(cfg['seed']);base=TokSingDAC(**model_config(vocab,cfg)).to(device).eval()
        with torch.inference_mode():predicted=base.generate(**batch).permute(1,0,2)
        corrected=requantize(z,predicted[:1],codec.layers)
        weights={k:v.cpu().clone() for k,v in base.state_dict().items()};del base,z
        results=[]
        for arm in ('baseline','exposure','rrq'):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
            model=TokSingDAC(**model_config(vocab,cfg)).to(device);model.load_state_dict(weights)
            wrapped=d.wrap(model);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
            context,labels=training_pair(clean,predicted,corrected,arm,torch.ones(1,dtype=torch.bool,device=device))
            assert torch.equal(labels[0],clean[0])
            optimizer.zero_grad(set_to_none=True);model.train();started=time.time()
            for index in range(cfg['gradient_accumulation']):
                from contextlib import nullcontext
                update=index==cfg['gradient_accumulation']-1
                with (nullcontext() if update else wrapped.no_sync()):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        out=wrapped(**batch,codes=context.permute(1,0,2))
                        loss=F.cross_entropy(out.flatten(0,2),labels.permute(1,0,2).flatten())
                    (loss/cfg['gradient_accumulation']).backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();synchronized=d.assert_synced(model)
            result=dict(arm=arm,rank=d.rank,finite=bool(torch.isfinite(loss)),loss=float(loss),grad_norm=float(norm),
                        seconds=time.time()-started,peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/1024**3,
                        gpu_total_GiB=torch.cuda.get_device_properties(device).total_memory/1024**3,
                        synchronized_parameter_sha256=synchronized)
            if not result['finite']:raise ValueError('nonfinite maximum-length verification')
            results.append(result);del model,wrapped,optimizer,out,loss
        results=d.gather(results)
        if d.primary:
            output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
            write_json(output,dict(passed=True,synthetic_audio=True,source_score_id=row['id'],
                max_seconds=samples/44100,max_frames=clean.shape[-1],gradient_accumulation=cfg['gradient_accumulation'],
                config_sha256=fingerprint(cfg),source_manifest_sha256=sha256(a.source),
                implementation={name:sha256(Path(__file__).parent/name) for name in ('toksing_model.py','targets.py','dac_codec.py','core.py','distributed.py')},
                research_training_started=False,results=results))
    finally:d.close()


if __name__=='__main__':main()
