"""SVS-only training around the unmodified official UniAudio generator.

Preparation/preflight never runs an optimizer step. Common and matched branches
are explicit commands. RRQ intervenes only in within-frame residual depth;
temporal history remains clean during teacher-forced training in every arm.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import time
import shutil

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from . import uniaudio as ua
from .core import requantize
from .io import sha256, write_json, safe_id


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def atomic_save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(value,temporary);temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def extend_vocabulary(model,size,seed=17):
    old=model.g_emb.num_embeddings
    if size<old:raise ValueError('cannot truncate pretrained vocabulary')
    if size==old:return model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for name in ('g_emb','l_emb','lm_head'):
            source=getattr(model,name);width=source.weight.shape[1]
            new=nn.Linear(width,size,bias=False) if name=='lm_head' else nn.Embedding(size,width)
            new=new.to(device=source.weight.device,dtype=source.weight.dtype)
            with torch.no_grad():
                new.weight[:old].copy_(source.weight)
                new.weight[old:].copy_(source.weight.float().mean(0).to(source.weight.dtype))
            # Mean-head initialization preserves the learned logit offset. By
            # Jensen, added mass <= (size-old)/old of original mass; random
            # zero-mean rows can instead dominate a pretrained softmax.
            setattr(model,name,new)
    return model


class SVSModel(nn.Module):
    def __init__(self,generator):
        super().__init__();self.generator=generator
        self.generator.continuous_mappings.requires_grad_(False)

    def forward(self,global_sequence,local_sequence,mask):
        return ua.depth_forward(self.generator,global_sequence,local_sequence,mask)


def arm_sequences(sequence,prefix_length,frames,bank,arm,intervene,audio_offset):
    if arm not in ('common','baseline','exposure','rrq'):raise ValueError('unknown arm')
    local=sequence.clone();labels=sequence.clone()
    if arm in ('common','baseline'):return local,labels
    if bank is None:raise ValueError('matched prefix bank required')
    if intervene.shape!=(frames,) or intervene.dtype!=torch.bool:raise ValueError('bad intervention mask')
    pred=bank['predicted'].to(sequence.device).long()
    rrq=bank['rrq'].to(sequence.device).long()
    if pred.shape!=(frames,) or rrq.shape!=(3,frames):raise ValueError('bank frame mismatch')
    if ((pred<0)|(pred>=1024)).any() or ((rrq<0)|(rrq>=1024)).any():raise ValueError('invalid bank codes')
    if not torch.equal(pred,rrq[0]):raise ValueError('RRQ bank prefix changed')
    view=local[:,prefix_length:prefix_length+3*frames].reshape(1,frames,3)
    target=labels[:,prefix_length:prefix_length+3*frames].reshape(1,frames,3)
    view[0,intervene,0]=pred[intervene]+audio_offset
    if arm=='rrq':
        for q in (1,2):
            view[0,intervene,q]=rrq[q,intervene]+audio_offset+q*1024
            target[0,intervene,q]=rrq[q,intervene]+audio_offset+q*1024
    # q0 prediction is always supervised by the native clean q0 target.
    return local,labels


def epoch_batches(items,world,seed,epoch):
    if world<1:raise ValueError('world must be positive')
    values=list(items);random.Random(seed+1000003*epoch).shuffle(values)
    return [values[i:i+world]+[None]*max(0,world-len(values[i:i+world]))
            for i in range(0,len(values),world)]


def ddp_loss(local_sum,global_count,world):
    if global_count<=0:raise ValueError('empty global objective')
    return local_sum*(world/global_count)


def validate_bank(actual,expected):
    if any(actual.get(k)!=v for k,v in expected.items()):
        raise ValueError('prefix bank provenance mismatch')


def protocol(data):
    keys=('manifest_sha256','codec_sha256','codec_config_sha256','encoding_contract',
          'model_config_sha256','vocabulary_sha256','checkpoint_sha256','prompts_sha256',
          'segments_sha256','panels_sha256','implementation_sha256','upstream_commit')
    return fingerprint({'objective':'official-all-token-ce-depth-only-v2-mean-vocab-expansion',
                        'timing':{k:data.recipe['config'][k] for k in ('max_frames','prompt_frames')},
                        **{k:data.recipe[k] for k in keys}})


def native_state(model):
    return {k.replace('._checkpoint_wrapped_module.','.'):v.detach().cpu()
            for k,v in model.state_dict().items()}


def load_model(cfg,data,device,checkpoint=None):
    from .uniaudio_data import _identity
    from .uniaudio_resplit import validate_holdout
    from .uniaudio_initialization import random_generator,validate_initialization
    import yaml
    validate_holdout(cfg,data)
    actual=_identity(cfg)
    if any(data.recipe.get(key)!=value for key,value in actual.items() if key!='config'):
        raise ValueError('runtime source or checkpoint differs from prepared recipe')
    mode=validate_initialization(cfg)
    if mode=='scratch':
        generator=random_generator(cfg['upstream_path'],yaml.safe_load(Path(cfg['model_config_path']).read_text(encoding='utf8')),
                                   data.vocab['tokens'],cfg.get('seed',17),device)
    else:
        generator,_=ua.load_generator(cfg['upstream_path'],cfg['model_config_path'],
                                      cfg['checkpoint_path'],device,dtype=torch.float32)
        extend_vocabulary(generator,len(data.vocab['tokens']),cfg.get('seed',17))
    saved=None
    if checkpoint:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        validate_initialization(cfg,saved)
        if saved['protocol_sha256']!=protocol(data):raise ValueError('checkpoint protocol mismatch')
        generator.load_state_dict(saved['model'],strict=True)
    generator.train().requires_grad_(True)
    return generator,saved


def run_contract(cfg,data,arm,languages,common_hash):
    from .uniaudio_evaluate import evaluation_contract
    return dict(protocol_sha256=protocol(data),arm=arm,languages=languages,
                common_sha256=common_hash,seed=cfg.get('seed',17),
                initialization=cfg.get('initialization','release'),
                lr_schedule=cfg.get('lr_schedule','constant'),warmup_steps=cfg.get('warmup_steps',0),
                synthesis_chunk_frames=cfg.get('synthesis_chunk_frames',0),
                learning_rate=cfg.get('learning_rate',1e-5),gradient_clip=cfg.get('gradient_clip',1.),
                intervention_probability=cfg.get('intervention_probability',.5),
                world_size=int(os.environ.get('WORLD_SIZE',1)),batch_per_gpu=1,
                loss_weighting='global non-padding token mean',
                evaluation_contract=evaluation_contract(cfg),
                evaluation_interval_steps=cfg.get('evaluation_interval_steps',0),
                trainer_sha256=sha256(__file__))


def bank_contract(cfg,data,common):
    return dict(common_sha256=sha256(common),protocol_sha256=protocol(data),seed=cfg.get('seed',17),
                topk=cfg.get('topk',30),temperature=cfg.get('temperature',.8),
                builder_sha256=sha256(__file__))


def language_scope(data,languages):
    available={r['language'] for r in data.rows.values()}
    selected=set(languages or available)
    if not selected<=available:raise ValueError('unknown requested language')
    return sorted(selected)


def validate_gate(gate,expected):
    if not gate.get('operational_gate_passed') or any(gate.get(k)!=v for k,v in expected.items()):
        raise ValueError('common synthesis gate is absent, stale, wrong scope or failed')


def resume_cursor(epoch,batch_index,total_batches):
    if not 0<=batch_index<total_batches:raise ValueError('invalid batch cursor')
    return (epoch+1,0) if batch_index+1==total_batches else (epoch,batch_index+1)


def preserve_qualified(out,metrics):
    target=out/'qualified'
    if not metrics['operational_gate_passed'] or target.exists():return
    if sha256(out/'last.pt')!=metrics['checkpoint_sha256']:raise ValueError('qualified checkpoint changed')
    pending=out/'qualified.pending';pending.mkdir(exist_ok=True)
    shutil.copyfile(out/'last.pt',pending/'common.pt')
    write_json(pending/'evaluation.json',metrics)
    pending.rename(target)


def activation_checkpointing(generator):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
    for layers in (generator.g_layers,generator.l_layers):
        for i in range(len(layers)):
            layers[i]=checkpoint_wrapper(layers[i],checkpoint_impl=CheckpointImpl.NO_REENTRANT)


class Runtime:
    def __init__(self):
        self.rank=int(os.environ.get('RANK',0));self.world=int(os.environ.get('WORLD_SIZE',1))
        self.local_rank=int(os.environ.get('LOCAL_RANK',0))
        self.device=f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.set_device(self.local_rank)
        if self.world>1:dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')

    def barrier(self):
        if self.world>1:dist.barrier()

    def gather(self,value):
        if self.world==1:return [value]
        outputs=[None]*self.world;dist.all_gather_object(outputs,value);return outputs

    def amp(self):
        return torch.autocast('cuda',dtype=torch.bfloat16) if self.device.startswith('cuda') else nullcontext()

    def close(self):
        if dist.is_initialized():dist.destroy_process_group()


def choose_segments(data,split,languages=None):
    selected=[s for s in data.segments if s['split']==split and (not languages or s['language'] in languages)]
    if not selected:raise ValueError('no examples in requested split/languages')
    return selected


def intervention(frames,key,seed,probability,device):
    generator=torch.Generator(device='cpu').manual_seed(int(fingerprint([key,seed])[:15],16))
    return (torch.rand(frames,generator=generator)<probability).to(device)


def read_bank(directory,example,contract,file_hash=None):
    path=Path(directory)/(safe_id(example['id'])+'.pt')
    if file_hash is not None and sha256(path)!=file_hash:raise ValueError('prefix bank content changed')
    item=torch.load(path,map_location='cpu',weights_only=True)
    validate_bank(item,contract)
    if item['id']!=example['id']:raise ValueError('bank id mismatch')
    return item


def preflight(cfg,data,out,runtime,languages=None):
    """Real forward/backward contract checks, with exactly zero optimizer steps."""
    model,_=load_model(cfg,data,runtime.device)
    activation_checkpointing(model)
    wrapper=SVSModel(model)
    if runtime.world>1:wrapper=DDP(wrapper,device_ids=[runtime.local_rank],broadcast_buffers=False,gradient_as_bucket_view=True)
    codec=ua.load_codec(cfg['upstream_path'],cfg['codec_path'],runtime.device)
    selected=choose_segments(data,'train',languages)
    # One example of every selected language per rank; same calls on each rank.
    representatives=[]
    for language in sorted({s['language'] for s in selected}):
        choices=[s for s in selected if s['language']==language]
        representatives.append(choices[runtime.rank%len(choices)])
    before=sha_state(model);results=[];offset=data.vocab['tokens'].index('<audio_0>')
    for segment in representatives:
        ex=data.example(segment,runtime.device);seq=ex['sequence'];mask=torch.zeros_like(seq,dtype=torch.bool)
        with torch.no_grad(),runtime.amp():
            original=model(seq,mask); adapted=ua.depth_forward(model,seq,seq,mask)
            difference=(original-adapted).abs().max().item()
            q0=original[:,ex['prefix_length']:ex['prefix_length']+ex['frames']*3:3,offset:offset+1024].argmax(-1)
        rrq=requantize(ex['latent'],q0[None],codec.quantizer.vq.layers[:3])
        bank={'predicted':q0[0],'rrq':rrq[:,0]}
        if difference>1e-4:raise ValueError('original clean-forward equivalence failed')
        for arm in ('baseline','exposure','rrq'):
            wrapper.zero_grad(set_to_none=True)
            local,labels=arm_sequences(seq,ex['prefix_length'],ex['frames'],bank,arm,
                                      torch.ones(ex['frames'],device=runtime.device,dtype=torch.bool),offset)
            with runtime.amp():
                logits=wrapper(seq,local,mask);loss=F.cross_entropy(logits.transpose(1,2),labels,ignore_index=0)
            loss.backward()
            norm=nn.utils.clip_grad_norm_(wrapper.parameters(),float('inf'),error_if_nonfinite=True)
            if not torch.isfinite(loss):raise ValueError('nonfinite preflight loss')
            results.append(dict(id=ex['id'],language=ex['language'],arm=arm,loss=loss.item(),grad_norm=norm.item(),forward_max_abs=difference))
    if sha_state(model)!=before:raise ValueError('preflight changed model weights')
    merged=runtime.gather(results)
    memory=runtime.gather(torch.cuda.max_memory_allocated(runtime.local_rank) if runtime.device.startswith('cuda') else 0)
    if runtime.rank==0:
        write_json(out,dict(status='passed',optimizer_steps=0,weights_unchanged=True,
                           world_size=runtime.world,protocol_sha256=protocol(data),results=sum(merged,[]),
                           initialization=cfg.get('initialization','release'),initial_weights_sha256=before,
                           trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           peak_allocated_bytes_per_rank=memory,
                           synthesis_quality_passed=False))


def sha_state(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def build_bank(cfg,data,out,runtime,common,languages=None):
    languages=language_scope(data,languages)
    model,saved=load_model(cfg,data,runtime.device,common)
    if saved.get('arm')!='common':raise ValueError('prefix source must be common checkpoint')
    model.eval().requires_grad_(False)
    codec=ua.load_codec(cfg['upstream_path'],cfg['codec_path'],runtime.device)
    contract=bank_contract(cfg,data,common)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'bank.json').exists():validate_bank(json.loads((out/'bank.json').read_text()),contract)
    selected=choose_segments(data,'train',languages);offset=data.vocab['tokens'].index('<audio_0>')
    done=[]
    for segment in tqdm(selected[runtime.rank::runtime.world],desc=f'prefix bank GPU{runtime.rank}'):
        path=out/(safe_id(segment['id'])+'.pt')
        ex=data.example(segment,runtime.device)
        if path.exists():read_bank(out,ex,contract);done.append(segment['id']);continue
        with runtime.amp():logits=model(ex['sequence'],torch.zeros_like(ex['sequence'],dtype=torch.bool))
        scores=logits[0,ex['prefix_length']:ex['prefix_length']+ex['frames']*3:3,offset:offset+1024].float()
        values,indices=scores.topk(cfg.get('topk',30),dim=-1)
        rng=torch.Generator(device=runtime.device).manual_seed(int(fingerprint([segment['id'],contract['seed']])[:15],16))
        draw=torch.multinomial((values/cfg.get('temperature',.8)).softmax(-1),1,generator=rng)
        predicted=indices.gather(1,draw)[:,0]
        relabeled=requantize(ex['latent'],predicted[None,None],codec.quantizer.vq.layers[:3])
        atomic_save(dict(**contract,id=segment['id'],predicted=predicted.cpu().short(),rrq=relabeled[:,0].cpu().short()),path)
        done.append(segment['id'])
    gathered=sum(runtime.gather(done),[])
    if sorted(gathered)!=sorted(s['id'] for s in selected):raise ValueError('prefix bank coverage mismatch')
    if runtime.rank==0:
        hashes={i:sha256(out/(safe_id(i)+'.pt')) for i in sorted(gathered)}
        write_json(out/'bank.json',dict(**contract,ids=sorted(gathered),languages=languages,file_sha256=hashes,
            temporal_history='clean teacher forced',prefix='actual sampled q0',complete=True))


def train(cfg,data,out,runtime,arm,epochs,languages=None,common=None,bank=None,resume=None,wandb=False):
    from .uniaudio_evaluate import evaluate_loaded,evaluation_contract
    from .uniaudio_initialization import learning_rate_at
    languages=language_scope(data,languages)
    if epochs<1:raise ValueError('positive epoch budget required')
    if arm!='common' and not common:raise ValueError('matched arms require the same common checkpoint')
    if arm=='common' and common:raise ValueError('common uses the configured initialization, not a branch checkpoint')
    learning_rate_at(cfg,1)
    if arm!='common':
        gate_path=Path(common).parent/'evaluation.json'
        gate=json.loads(gate_path.read_text()) if gate_path.exists() else {}
        validate_gate(gate,dict(checkpoint_sha256=sha256(common),protocol_sha256=protocol(data),
            languages=languages,thresholds=cfg['gate'],sampling=dict(topk=cfg.get('topk',30),
            temperature=cfg.get('temperature',.8),seed=cfg.get('seed',17)),evaluation_contract=evaluation_contract(cfg)))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not resume:raise ValueError('existing run requires explicit resume')
    model,saved=load_model(cfg,data,runtime.device,resume or common)
    if common and not resume and saved.get('arm')!='common':
        raise ValueError('matched branches require a common-stage checkpoint')
    common_hash=sha256(common) if common else None
    run_settings=run_contract(cfg,data,arm,languages,common_hash)
    run_settings['bank_sha256']=sha256(Path(bank)/'bank.json') if bank else None
    if resume and saved.get('run_contract')!=run_settings:raise ValueError('resume run mismatch')
    if common and saved.get('languages')!=languages:raise ValueError('common/branch language scope mismatch')
    lock=out/'run.json'
    if lock.exists() and json.loads(lock.read_text())!=run_settings:raise ValueError('run configuration changed')
    if runtime.rank==0:write_json(lock,run_settings)
    activation_checkpointing(model);wrapped=SVSModel(model)
    if runtime.world>1:wrapped=DDP(wrapped,device_ids=[runtime.local_rank],broadcast_buffers=False,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=cfg.get('learning_rate',1e-5),betas=(.9,.95),weight_decay=.001,foreach=False)
    step=saved['step'] if resume else 0;start=saved['next_epoch'] if resume else 0
    start_batch=saved.get('next_batch',0) if resume else 0
    if resume:optimizer.load_state_dict(saved['optimizer'])
    selected=choose_segments(data,'train',languages)
    contract=bank_contract(cfg,data,common) if common else None
    if arm in ('exposure','rrq'):
        if not bank:raise ValueError('prefix bank required')
        metadata=json.loads((Path(bank)/'bank.json').read_text());validate_bank(metadata,contract)
        if not metadata.get('complete') or set(metadata['ids'])!={s['id'] for s in selected}:raise ValueError('prefix bank coverage mismatch')
    log=None
    if runtime.rank==0 and wandb:
        import wandb as wb
        log=wb.init(project=cfg.get('wandb_project','rrq-svs'),entity=cfg.get('wandb_entity'),
                    id=saved.get('wandb_run_id') if resume else None,resume='must' if resume and saved.get('wandb_run_id') else None,
                    name=f"uniaudio-185m-{cfg.get('initialization','release')}-{arm}-"+('-'.join(languages)),dir=str(out),config=dict(run_settings,epochs=epochs,
                    train_segments=len(selected),global_batch=runtime.world))
        log.define_metric('step');log.define_metric('*',step_metric='step')
    def save_and_evaluate(next_epoch,next_batch):
        runtime.barrier()
        if runtime.rank==0:
            checkpoint=dict(model=native_state(model),optimizer=optimizer.state_dict(),arm=arm,
                            protocol_sha256=protocol(data),next_epoch=next_epoch,next_batch=next_batch,
                            step=step,languages=languages,common_sha256=common_hash,seed=cfg.get('seed',17),
                            run_contract=run_settings,wandb_run_id=log.id if log else None)
            atomic_save(checkpoint,out/'last.pt')
            write_json(out/'progress.json',{k:v for k,v in checkpoint.items() if k not in ('model','optimizer')})
        runtime.barrier()
        metrics=evaluate_loaded(cfg,data,model,out/'last.pt',out/f'step-{step:06d}',runtime,languages)
        if runtime.rank==0:
            write_json(out/'evaluation.json',metrics)
            if arm=='common':preserve_qualified(out,metrics)
            if log:log_evaluation(log,metrics,next_epoch+next_batch/len(epoch_batches(selected,runtime.world,cfg.get('seed',17),next_epoch)),step)
            print(json.dumps(dict(step=step,gate=metrics['operational_gate_passed'],
                within50=metrics['macro']['within_50_cents'],pronunciation=metrics.get('pronunciation')),ensure_ascii=False),flush=True)
        runtime.barrier()
    if cfg.get('evaluate_before_training'):
        prior=json.loads((out/'evaluation.json').read_text()) if (out/'evaluation.json').exists() else {}
        if not resume or prior.get('checkpoint_sha256')!=sha256(resume):save_and_evaluate(start,start_batch)
    launch_step=step;started=time.monotonic()
    offset=data.vocab['tokens'].index('<audio_0>')
    for epoch in tqdm(range(start,epochs),desc=f'{arm} epochs',disable=runtime.rank!=0):
        batches=epoch_batches(selected,runtime.world,cfg.get('seed',17),epoch)
        skip=start_batch if epoch==start else 0
        bar=tqdm(enumerate(batches[skip:],start=skip),total=len(batches),initial=skip,
                 desc=f'epoch {epoch+1}/{epochs}',disable=runtime.rank!=0,leave=False)
        for batch_index,batch in bar:
            segment=batch[runtime.rank];real=segment is not None
            ex=data.example(segment or selected[0],runtime.device);seq=ex['sequence'];mask=torch.zeros_like(seq,dtype=torch.bool)
            item=read_bank(bank,ex,contract,metadata['file_sha256'][ex['id']]) if arm in ('exposure','rrq') else None
            intervention_mask=intervention(ex['frames'],[ex['id'],epoch],cfg.get('seed',17),cfg.get('intervention_probability',.5),runtime.device)
            local,labels=arm_sequences(seq,ex['prefix_length'],ex['frames'],item,arm,intervention_mask,offset)
            count=torch.tensor(int(labels.ne(0).sum()) if real else 0,device=runtime.device,dtype=torch.float64)
            if runtime.world>1:dist.all_reduce(count)
            optimizer.zero_grad(set_to_none=True)
            with runtime.amp():
                logits=wrapped(seq,local,mask)
                loss_sum=F.cross_entropy(logits.transpose(1,2),labels,ignore_index=0,reduction='sum')*(1 if real else 0)
                loss=ddp_loss(loss_sum,count.item(),runtime.world)
            loss.backward();norm=nn.utils.clip_grad_norm_(wrapped.parameters(),cfg.get('gradient_clip',1.),error_if_nonfinite=True)
            rate=learning_rate_at(cfg,step+1)
            for group in optimizer.param_groups:group['lr']=rate
            optimizer.step();step+=1
            summed=loss_sum.detach().double()
            if runtime.world>1:dist.all_reduce(summed)
            value=(summed/count).item()
            if runtime.rank==0:
                bar.set_postfix(loss=f'{value:.3f}',step=step)
                if step%10==0:
                    progress={'train/loss':value,'train/grad_norm':norm.item(),'train/learning_rate':rate,'epoch':epoch+(batch_index+1)/len(batches),
                              'step':step,'epochs_budget':epochs,'seconds_per_update':(time.monotonic()-started)/(step-launch_step)}
                    write_json(out/'live.json',progress)
                    if log:log.log(progress)
            interval=cfg.get('evaluation_interval_steps',0)
            if batch_index+1==len(batches) or (interval and step%interval==0):
                save_and_evaluate(*resume_cursor(epoch,batch_index,len(batches)))
    if log:log.finish()


def log_evaluation(run,metrics,epoch,step):
    import wandb
    scalars={'epoch':epoch,'step':step}
    scalars.update({'valid/'+k:v for k,v in metrics['macro'].items()})
    scalars.update({'valid/asr_'+k:v for k,v in metrics.get('pronunciation',{}).items()})
    scalars['valid/operational_gate_passed']=int(metrics['operational_gate_passed'])
    for cohort, values in metrics.get('by_cohort',{}).items():
        scalars.update({f'valid/{cohort}/{k}':v for k,v in values['macro'].items()})
    for language,values in metrics['per_language'].items():
        scalars.update({f'valid/{language}/{k}':v for k,v in values.items()})
    for language,values in metrics.get('per_language_pronunciation',{}).items():
        scalars.update({f'valid/{language}/asr_{k}':v for k,v in values.items()})
    for singer,values in metrics.get('per_singer',{}).items():
        scalars.update({f'valid/singer/{singer}/{k}':v for k,v in values['acoustic'].items()})
        scalars.update({f'valid/singer/{singer}/asr_{k}':v for k,v in values['pronunciation'].items()})
    table=wandb.Table(columns=['id','language','singer','cohort','reference','codec','generated','within50','voicing_error',
                               'lyrics','reference_asr','codec_asr','generated_asr','generated_cer'])
    for row in metrics['examples']:
        speech=row.get('pronunciation');texts=[speech['transcripts'][arm]['raw'] for arm in ('reference','codec','generated')] if speech else [None]*3
        table.add_data(row['id'],row['language'],row['singer'],row.get('singer_cohort','seen'),*[wandb.Audio(row['audio'][arm],sample_rate=16000)
            for arm in ('reference','codec','generated')],row['within_50_cents'],row['voicing_error'],
            speech['lyrics'] if speech else None,*texts,speech['metrics']['generated_cer'] if speech else None)
    scalars['audition']=table;run.log(scalars)


def main():
    from .uniaudio_data import load_recipe,PreparedData
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','bank','train'])
    p.add_argument('--config',required=True);p.add_argument('--workspace',required=True)
    p.add_argument('--output',required=True);p.add_argument('--languages',nargs='+')
    p.add_argument('--arm',choices=['common','baseline','exposure','rrq'],default='common')
    p.add_argument('--epochs',type=int);p.add_argument('--common');p.add_argument('--bank');p.add_argument('--resume')
    p.add_argument('--wandb',action='store_true')
    args=p.parse_args();cfg=load_recipe(args.config,args.workspace)
    data=PreparedData(cfg['prepared_path'],cfg['cache_path']);runtime=Runtime();out=Path(args.output)
    if args.stage=='preflight':out.parent.mkdir(parents=True,exist_ok=True)
    try:
        if args.stage=='preflight':preflight(cfg,data,out,runtime,args.languages)
        elif args.stage=='bank':
            if not args.common:p.error('--common required')
            build_bank(cfg,data,out,runtime,args.common,args.languages)
        else:
            if args.epochs is None:p.error('--epochs required; training never starts as preparation')
            train(cfg,data,out,runtime,args.arm,args.epochs,args.languages,args.common,args.bank,args.resume,args.wandb)
    finally:runtime.close()


if __name__=='__main__':main()
