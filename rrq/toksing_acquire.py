"""Pinned GTSinger audio acquisition and timed, song-diverse manifest selection."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.parse

from .corpora import gtsinger_row, digest, validate_annotations
from .data_download import download
from .io import write_json

GT_REV='4426c862beed558b7e1cb8a4dce7e8c0c83bb208'
BASE=f'https://huggingface.co/datasets/GTSinger/GTSinger/resolve/{GT_REV}/'
LANGUAGES=['English','French','German','Italian','Japanese','Korean','Russian','Spanish']


def timed_subset(rows,hours):
    groups=defaultdict(list)
    for r in rows:
        seconds=sum(r['phone_durations'])
        if .2<=seconds<=13:groups[r['singer']].append(r)
    result=[]
    for singer,items in sorted(groups.items()):
        songs=defaultdict(list)
        for r in sorted(items,key=lambda x:digest(x['id'])):songs[r['song_group']].append(r)
        queues=[songs[k] for k in sorted(songs,key=digest)];duration=0
        while queues and duration<hours*3600:
            for queue in list(queues):
                row=queue.pop();result.append(row);duration+=sum(row['phone_durations'])
                if not queue:queues.remove(queue)
                if duration>=hours*3600:break
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--hours-per-singer',type=float,default=1.5)
    p.add_argument('--existing-only',action='store_true');a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True);rows=[]
    download(BASE+'dataset_license.md',root/'dataset_license.md',100000)
    for lang in LANGUAGES:
        path=download(BASE+f'processed/{lang}/metadata.json',root/'metadata'/lang/'metadata.json')
        for item in json.loads(path.read_text(encoding='utf-8')):
            if 'Speech' in item['item_name']:continue
            r=gtsinger_row(item);validate_annotations(r)
            parts=item['item_name'].split('#');rel='/'.join(parts[:-1]+[parts[-1]+'.wav'])
            r['source_audio']=rel;r['audio']=str((root/'raw'/rel).resolve())
            r['source_revision']=GT_REV
            if not a.existing_only or Path(r['audio']).is_file():rows.append(r)
    rows=timed_subset(rows,a.hours_per_singer)
    def get(row):
        download(BASE+urllib.parse.quote(row['source_audio'],safe='/'),row['audio'],16*1024**2)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(get,rows)):
            if i%25==0:print(f'GTSinger audio {i+1}/{len(rows)}',flush=True)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    by_singer=defaultdict(float)
    for r in rows:by_singer[r['singer']]+=sum(r['phone_durations'])/3600
    write_json(path.with_suffix('.report.json'),dict(n=len(rows),hours=sum(by_singer.values()),
        per_singer_hours=by_singer,revision=GT_REV,existing_only=a.existing_only,
        note='Published score in pinned metadata; supplied phoneme clock. Not zero-shot.'))


if __name__=='__main__':main()
