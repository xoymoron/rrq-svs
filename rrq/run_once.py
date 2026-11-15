"""One frozen experiment: common -> gated targets -> matched arms -> final test."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from .io import sha256,write_json


def now():return datetime.now(timezone.utc).isoformat()


def run_pipeline(root,execute,resume=False):
    root=Path(root);path=root/'experiment-state.json'
    protocol=json.loads((root/'protocol.json').read_text())['sha256']
    source=sha256(__file__)
    stages=['common','targets','branches','final-test','publish']
    if path.exists():
        if not resume:raise ValueError('experiment already launched; explicit --resume required, no automatic retry')
        state=json.loads(path.read_text())
        if state['protocol_sha256']!=protocol or state['launcher_sha256']!=source:
            raise ValueError('resume protocol/launcher changed')
        if state['completed']!=stages[:len(state['completed'])]:raise ValueError('invalid completed stage order')
    else:
        if list(root.glob('*/last.pt')):raise ValueError('untracked research checkpoint; refuse new experiment')
        ready=json.loads((root/'readiness.json').read_text())
        if ready['protocol_sha256']!=protocol or not ready['ready_for_common'] or ready['training_started']:
            raise ValueError('current readiness is required before first launch')
        state=dict(protocol_sha256=protocol,launcher_sha256=source,started_utc=now(),completed=[],
                   status='ready',test_policy='one final frozen-model evaluation after all arms; no test-based tuning')
        write_json(path,state)
    for stage in stages[len(state['completed']):]:
        state.update(status='running',stage=stage,updated_utc=now());write_json(path,state)
        print(json.dumps(dict(stage=stage,status='starting',utc=now())),flush=True)
        try:execute(stage)
        except BaseException as error:
            state.update(status='failed',error_type=type(error).__name__,updated_utc=now())
            write_json(path,state)
            raise
        state['completed'].append(stage);state.update(status='between_stages',updated_utc=now());write_json(path,state)
    state.update(status='complete',updated_utc=now());write_json(path,state)


def publish(root):
    import wandb
    protocol=json.loads((root/'protocol.json').read_text())
    comparison=json.loads((root/'comparison-test.json').read_text())
    if comparison['protocol_sha256']!=protocol['sha256']:raise ValueError('comparison protocol mismatch')
    run=wandb.init(project='rrq-svs',entity='oxymoron_music',group=root.name,name=root.name+'/comparison',
                   job_type='final-evaluation',config=protocol,dir=str(root),save_code=False)
    values={}
    for arm in ('baseline','exposure','rrq'):
        report=json.loads((root/arm/'evaluation-test.json').read_text())
        if report['protocol_sha256']!=protocol['sha256'] or report['model_sha256']!=sha256(root/arm/'best.pt'):
            raise ValueError('final evaluation provenance mismatch')
        for metric,value in report['language_macro']['generated'].items():
            if value is not None:values[f'test/{arm}/{metric}']=value
        for lang,arms in report['per_language'].items():
            for metric,value in arms['generated'].items():
                if value is not None:values[f'test/{lang}/{arm}/{metric}']=value
    for other in ('baseline','exposure'):
        result=comparison['rrq_minus_'+other]
        for key,value in [('delta',result['delta']),('ci95_low',result['ci95'][0]),('ci95_high',result['ci95'][1])]:
            values[f'comparison/rrq_minus_{other}/{key}']=value
    run.log(values)
    write_json(root/'comparison-wandb.json',dict(id=run.id,url=run.url));run.finish()


def verify_matched(root):
    """Audit executed budgets/schedules before unlocking any test metrics."""
    import torch
    protocol=json.loads((root/'protocol.json').read_text());epochs=protocol['config']['branch_epochs']
    common=sha256(root/'common/best.pt');steps=[];sequences=[]
    keys=('step','draws','schedule_sha256','shared_predicted_mixture_sha256')
    for arm in ('baseline','exposure','rrq'):
        state=torch.load(root/arm/'last.pt',map_location='cpu',weights_only=True,mmap=True)
        if (state['protocol_sha256']!=protocol['sha256'] or state['arm']!=arm or
            state['initial_checkpoint_sha256']!=common or state['epoch']!=epochs):
            raise ValueError('unmatched initialization/protocol/final epoch')
        steps.append(state['step']);del state
        records=[]
        for epoch in range(1,epochs+1):
            report=json.loads((root/arm/f'epoch-{epoch:03d}.json').read_text())
            if report['protocol_sha256']!=protocol['sha256']:raise ValueError('unmatched epoch protocol')
            records.append({k:report[k] for k in keys})
        sequences.append(records)
    if len(set(steps))!=1 or any(s!=sequences[0] for s in sequences[1:]):
        raise ValueError('unmatched optimizer budget, sample schedule or mixture')
    write_json(root/'matched-training-audit.json',dict(passed=True,protocol_sha256=protocol['sha256'],
        common_sha256=common,branch_epochs=epochs,updates_per_arm=steps[0],epoch_schedules=sequences[0]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='artifacts/gtsinger-rrq-v2');p.add_argument('--resume',action='store_true')
    p.add_argument('--unlock-test-after-training',action='store_true',required=True)
    a=p.parse_args();root=Path(a.output).resolve()
    if shutil.disk_usage(root).free<10*1024**3:raise ValueError('less than 10 GiB free for checkpoints/targets')
    # Holding this descriptor prevents a duplicate launcher on the same experiment.
    import fcntl
    with (root/'experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,OUTPUT=str(root),WANDB_MODE='online',PYTHONUNBUFFERED='1')
        os.environ['WANDB_MODE']='online'
        def execute(stage):
            if stage=='publish':return publish(root)
            if stage=='final-test':verify_matched(root)
            command=['bash','run_matched.sh',stage]
            if stage=='final-test':command.append('--unlock-test')
            subprocess.run(command,env=env,check=True)
        run_pipeline(root,execute,a.resume)


if __name__=='__main__':main()
