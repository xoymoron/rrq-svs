"""Bounded throughput measurement; discard weights, never select by quality."""
import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .core import requantize
from .dac_codec import FrozenDAC
from .distributed import Distributed
from .experiment import Experiment,epoch_schedule,model_config
from .io import sha256,write_json
from .matched_run import evaluate_panel,seed_epoch
from .targets import training_pair
from .toksing_data import load_audio,score_batch
from .toksing_model import TokSingDAC


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','output','codec','config','report'):p.add_argument('--'+name,required=True)
    p.add_argument('--steps',type=int,default=48)
    a=p.parse_args();torch.set_num_threads(2)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d=Distributed();d.start();device=torch.device(f'cuda:{d.local_rank}')
    try:
        e=Experiment(a.prepared,a.output,a.codec,a.config);e.require_cache()
        if d.world!=2:raise ValueError('profile must use the actual two-GPU setup')
        if list(e.root.glob('*/last.pt')):raise ValueError('profile before the research experiment only')
        cfg=e.config;acc=cfg['gradient_accumulation'];n=(a.steps+4)*acc*d.world
        draws=epoch_schedule(e.train,cfg['seed'],1,d.world,acc,cfg['chinese_sampling_weight'])[:n]
        if len(draws)!=n:raise ValueError('profile sample budget exceeds one epoch')
        timings={}
        for arm in ('baseline','exposure','rrq'):
            torch.manual_seed(cfg['seed']);model=TokSingDAC(**model_config(e.vocab,cfg)).to(device)
            wrapped=d.wrap(model);opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            seed_epoch(cfg['seed'],1,d.rank);model.train();opt.zero_grad(set_to_none=True)
            for i,item_id in enumerate(draws[d.rank::d.world]):
                if i==4*acc:
                    d.barrier();torch.cuda.synchronize(device);start=time.perf_counter()
                row=e.by_id[item_id];c=e.read_cache(row,device);batch=score_batch(row,c,e.vocab,device)
                # Shape-correct artificial labels exercise each loss path only.
                # No oracle score, quality result or checkpoint is retained.
                pred=(c['codes']+1)%1024;corrected=(c['codes']+2)%1024;corrected[0]=pred[0]
                context,labels=training_pair(c['codes'],pred,corrected,arm,torch.tensor([i%2==0],device=device))
                update=(i+1)%acc==0
                with (nullcontext() if update else wrapped.no_sync()):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        out=wrapped(**batch,codes=context.permute(1,0,2))
                        target=labels.permute(1,0,2)
                        loss=(F.cross_entropy(out[:,0].flatten(0,1),target[:,0].flatten())+
                              8*F.cross_entropy(out[:,1:].flatten(0,2),target[:,1:].flatten()))/9
                    (loss/acc).backward()
                if update:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                    opt.step();opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device);d.barrier()
            timings[arm]=(time.perf_counter()-start)/a.steps
            del wrapped,opt,out,loss,model
        # Timed exact validation code. Discard numerical quality measurements.
        model=TokSingDAC(**model_config(e.vocab,cfg)).to(device).eval().requires_grad_(False)
        codec=FrozenDAC(e.codec,device);d.barrier();start=time.perf_counter()
        evaluate_panel(e,d,device,model,codec,e.panels['valid'],ablate=True)
        torch.cuda.synchronize(device);d.barrier();validation=time.perf_counter()-start
        targets=sorted(e.train,key=lambda r:r['id'])[d.rank:128:d.world]
        d.barrier();start=time.perf_counter()
        with torch.inference_mode():
            for row in targets:
                c=e.read_cache(row,device);prediction=model.generate(**score_batch(row,c,e.vocab,device)).permute(1,0,2)
                x=load_audio(row['audio'],codec.sample_rate)
                z,native=codec.encode(torch.from_numpy(x)[None,None].to(device))
                if not torch.equal(native,c['codes']):raise ValueError('native token reproducibility failure')
                requantize(z,prediction[:1],codec.layers)
        torch.cuda.synchronize(device);d.barrier();target_seconds=time.perf_counter()-start
        gathered=d.gather([dict(rank=d.rank,seconds_per_update=timings,validation_seconds=validation,
                               targets_seconds=target_seconds,targets_n=len(targets))])
        if d.primary:
            steps=math.ceil(len(e.train)/(d.world*acc))
            speed={arm:max(r['seconds_per_update'][arm] for r in gathered) for arm in timings}
            val=max(r['validation_seconds'] for r in gathered)
            # Branch validation has an extra decode and diagnostic forwards: budget 2x.
            train=steps*(cfg['common_epochs']*speed['baseline']+cfg['branch_epochs']*sum(speed.values()))
            validation_cost=val*1.125*(cfg['common_epochs']+1+2*3*(cfg['branch_epochs']+1))
            target_cost=max(r['targets_seconds'] for r in gathered)*(len(e.train)+len(e.panels['valid']))/128+val*2
            test_cost=3*val*sum(r['split']=='test' for r in e.rows)/len(e.panels['valid'])
            total=train+validation_cost+target_cost+test_cost
            report=dict(protocol_sha256=e.hash,profiler_sha256=sha256(__file__),measured_steps_per_arm=a.steps,
                research_checkpoint_saved=False,quality_metrics_used=False,steps_per_epoch=steps,ranks=gathered,
                hours=dict(training=train/3600,epoch_validation=validation_cost/3600,targets=target_cost/3600,
                           final_test=test_cost/3600,total=total/3600,with_30_percent_margin=1.3*total/3600),
                limitations='Short throughput sample, synthetic branch labels, no target-bank disk read timing; branch validation budgeted at 2x. Not a convergence estimate or runtime guarantee.')
            write_json(a.report,report);print(report,flush=True)
    finally:d.close()


if __name__=='__main__':main()
