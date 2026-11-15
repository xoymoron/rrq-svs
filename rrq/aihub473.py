"""AIHub473 actual note-lyric adapter; never fabricate phoneme alignments.

Audited source contains breath markers with MIDI96 (and84), and overlapping
notes. Breath flags take precedence over their placeholder pitch. Ambiguous
segments are reported and rejected, not shifted. Segments retain real clocks.
"""
import argparse
from collections import defaultdict
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf

from .corpora import digest
from .data_download import free_space
from .io import write_json


def segment_units(notes,start,end):
    if not math.isfinite(start+end) or not 0<=start<end:raise ValueError('invalid segment bounds')
    units=[];durations=[];pitches=[];cursor=start
    for n in sorted(notes,key=lambda x:float(x['start_time'])):
        a,b=float(n['start_time']),float(n['end_time'])
        if not math.isfinite(a+b) or a<0 or b<=a:raise ValueError('invalid note interval')
        if b<=start or a>=end:continue
        a=max(a,start);b=min(b,end)
        if a<cursor-1e-8:raise ValueError('overlapping note annotation')
        if a>cursor:
            units.append('<SP>');durations.append(a-cursor);pitches.append(0)
        breath=n['is_breath'];unit='<AP>' if breath else n['lyric']
        pitch=0 if breath else n['midi_num']
        if not unit or not isinstance(pitch,int) or not 0<=pitch<=127:
            raise ValueError('missing lyric or invalid MIDI')
        units.append(unit);durations.append(b-a);pitches.append(pitch);cursor=b
    if cursor<end:
        units.append('<SP>');durations.append(end-cursor);pitches.append(0)
    if not any(pitches):raise ValueError('no sung notes')
    return units,durations,pitches


def prepare(labels,audio_zips,output,singers,hours=1.5,max_segments=None):
    output=Path(output);(output/'audio').mkdir(parents=True,exist_ok=True)
    annotations={}
    for path in labels:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith('.json'):continue
                raw=z.read(name);label=json.loads(raw)
                filename=label['data_info']['wav_filename']
                if filename in annotations:raise ValueError('duplicate annotation filename')
                annotations[filename]=(label,str(path),name,'valid' if Path(path).name.startswith('VL') else 'train')
    rows=[];rejected=[];used=defaultdict(float);found=set()
    for archive_path in audio_zips:
        with zipfile.ZipFile(archive_path) as z:
            for member in sorted(z.namelist(),key=digest):
                filename=Path(member).name
                if filename not in annotations:continue
                singer=filename.split('_')[1]
                if singer not in singers or used[singer]>=hours*3600:continue
                if filename in found:raise ValueError('duplicate source waveform')
                found.add(filename);label,label_path,label_member,split=annotations[filename]
                info=z.getinfo(member)
                if info.file_size>256*1024**2:raise ValueError('source waveform exceeds memory cap')
                x,sr=sf.read(io.BytesIO(z.read(member)),dtype='float32',always_2d=True)
                if x.shape[1]!=1 or not np.isfinite(x).all():raise ValueError('require finite mono source')
                for index,start_sample in enumerate(range(0,len(x),8*sr)):
                    stop=min(start_sample+8*sr,len(x));start,end=start_sample/sr,stop/sr
                    if end-start<2:continue
                    try:units,durations,pitches=segment_units(label['notes'],start,end)
                    except ValueError as exc:
                        rejected.append(dict(file=filename,start=start,end=end,reason=str(exc)));continue
                    item='aihub473-'+digest(filename+':'+str(index))[:24]
                    path=output/'audio'/(item+'.flac');free_space(path,(stop-start_sample)*4)
                    if not path.exists():sf.write(path,x[start_sample:stop],sr,subtype='PCM_24')
                    rows.append(dict(id=item,corpus='aihub473',singer='aihub473:'+singer,
                        language='Korean',audio=str(path.resolve()),phones=units,
                        phone_durations=durations,score={'notes':pitches},
                        lyrics=''.join(u for u in units if u not in ('<SP>','<AP>')),
                        lyric_unit='provided_note_lyric_NOT_phoneme_alignment',
                        conditioning='score_pitch_gt_note_lyric_duration',
                        split=split,song_group='aihub473-recording:'+Path(filename).stem,
                        split_policy='official-recording-split-composition-overlap-unverified',
                        source_archive=str(archive_path),source_member=member,
                        source_crc32=info.CRC,source_label_archive=label_path,
                        source_label_member=label_member,start_sample=start_sample,stop_sample=stop))
                    used[singer]+=end-start
                    if used[singer]>=hours*3600 or (max_segments and len(rows)>=max_segments):break
                if max_segments and len(rows)>=max_segments:break
        if max_segments and len(rows)>=max_segments:break
    if not rows:raise ValueError('no eligible selected-singer segments; inspect archives')
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    report=dict(segments=len(rows),per_singer_hours={k:v/3600 for k,v in used.items()},
        rejected=rejected,notes='Original note-lyric intervals, not phoneme forced alignment; overlapping segments rejected. Breath marker MIDI96/84 is not sung pitch.')
    write_json(output/'preprocess-report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='rejected'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',nargs='+',required=True);p.add_argument('--audio-zips',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--singers',nargs='+',default=['16','18','36','38','52','60','71','79'])
    p.add_argument('--hours',type=float,default=1.5);p.add_argument('--max-segments',type=int)
    a=p.parse_args();prepare(a.labels,a.audio_zips,a.output,a.singers,a.hours,a.max_segments)


if __name__=='__main__':main()
