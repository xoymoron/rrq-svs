"""Resumable two-GPU experiment pipeline. Run inside tmux; all output is mirrored."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time

from .experiment import ARMS, SEEDS, atomic_json


def execute(root,device,module,arguments,label):
    command=[sys.executable,'-u','-m',module,'--root',str(root),'--device',f'cuda:{device}',*arguments]
    logs=root/'logs'; logs.mkdir(exist_ok=True)
    print('START',label,'GPU',device,flush=True)
    with (logs/(label+'.log')).open('a',encoding='utf-8') as stream:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            stream.write(line); stream.flush(); print(f'[{label}] {line}',end='',flush=True)
        code=process.wait()
    if code: raise RuntimeError(f'{label} exited {code}; see {logs/(label+".log")}')
    print('FINISHED',label,flush=True)


def batches(root,jobs):
    for start in range(0,len(jobs),2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(execute,root,i,module,args,label) for i,(module,args,label) in enumerate(jobs[start:start+2])]
            for future in futures: future.result()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asr',type=Path,default=Path('artifacts/whisper-turbo')); args=p.parse_args()
    root=args.root; started=time.time()
    def stage(name):
        atomic_json(root/'pipeline-status.json',dict(stage=name,state='running',elapsed_seconds=time.time()-started))
        print('PIPELINE_STAGE',name,flush=True)
    try:
        stage('common multilingual adaptation')
        if not (root/'warmup/complete.json').exists(): execute(root,0,'rrq.experiment',['train','--stage','warmup'],'common')
        stage('full common rollouts and initial validation renders')
        batches(root,[('rrq.experiment',['targets'],'targets'),
                      ('rrq.experiment',['render','--variant','initial','--split','valid'],'initial-valid')])
        execute(root,0,'rrq.experiment',['render','--variant','common','--split','valid'],'common-valid')
        stage('multilingual validation audio and ASR assessment')
        execute(root,0,'rrq.experiment_metrics',['--asr',str(args.asr),'--split','valid','--labels','initial','common'],'valid-metrics')
        stage('matched branches: 3 arms x 3 seeds x 6 epochs')
        jobs=[('rrq.experiment',['train','--stage','branch','--arm',arm,'--seed',str(seed)],f'train-{arm}-{seed}')
              for seed in SEEDS for arm in ARMS]
        batches(root,jobs)
        stage('full held-out test generation')
        jobs=[('rrq.experiment',['render','--variant','common','--split','test'],'common-test')]
        jobs += [('rrq.experiment',['render','--variant',arm,'--seed',str(seed),'--split','test'],f'test-{arm}-{seed}')
                 for seed in SEEDS for arm in ARMS]
        batches(root,jobs)
        stage('held-out test metrics')
        labels=['common']+[f'{arm}-{seed}' for seed in SEEDS for arm in ARMS]
        execute(root,0,'rrq.experiment_metrics',['--asr',str(args.asr),'--split','test','--labels',*labels],'test-metrics')
        stage('paired summary')
        from .experiment_summary import summarize
        summarize(root)
        atomic_json(root/'pipeline-status.json',dict(stage='complete',state='complete',elapsed_seconds=time.time()-started))
        print('MATCHED_EXPERIMENT_COMPLETE',str(root),flush=True)
    except BaseException as exc:
        atomic_json(root/'pipeline-status.json',dict(stage='failed',state='failed',error=str(exc),elapsed_seconds=time.time()-started))
        raise


if __name__=='__main__': main()
