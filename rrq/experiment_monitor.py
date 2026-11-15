"""Read-only tmux progress display for the matched experiment."""
import argparse
import json
from pathlib import Path
import time


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return {}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); args=p.parse_args()
    while True:
        print('\033[2J\033[H',end='')
        print('RRQ MATCHED EXPERIMENT |',time.strftime('%Y-%m-%d %H:%M:%S'))
        state=read(args.root/'pipeline-status.json')
        print('Stage:',state.get('stage','common adaptation / initial evaluation'))
        print('Protocol: 472 train / 50 valid / 54 test; 8 languages; 3 paired seeds')
        print('Job                       Epoch      Step       Train CE       State')
        folders=[args.root/'warmup']+[args.root/f'{arm}-{seed}' for seed in (17,29,43) for arm in ('baseline','exposure','rrq')]
        for folder in folders:
            data=read(folder/'status.json'); complete=(folder/'complete.json').exists()
            print(f"{folder.name:25} {str(data.get('epoch','-'))+'/'+str(data.get('epochs','-')):10} "
                  f"{str(data.get('step','-'))+'/'+str(data.get('steps','-')):10} "
                  f"{str(round(data.get('mean_train_ce',0),4)) if data else '-':14} {'complete' if complete else 'running' if data else 'pending'}")
        targets=read(args.root/'targets/status.json')
        print('Target cache:',targets.get('done',0),'/',targets.get('total',576))
        print('Completed metric tables: validation',len(list((args.root/'evaluation/valid').glob('*.json'))),
              '/4; test',len(list((args.root/'evaluation/test').glob('*.json'))),'/12')
        print('Outputs:',args.root.resolve())
        print('Status is not a quality claim. Logs and per-language metrics are retained.')
        if state.get('state') in ('complete','failed'):
            print(json.dumps(state,ensure_ascii=False)); break
        time.sleep(5)


if __name__=='__main__': main()
