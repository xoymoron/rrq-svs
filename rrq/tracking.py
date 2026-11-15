"""Optional native W&B tracking; local JSON remains the canonical metric record."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def run_id(group, protocol_hash, label):
    return hashlib.sha256(f'{group}:{protocol_hash}:{label}'.encode()).hexdigest()[:24]


def audition_ids(rows, per_language=2):
    groups = defaultdict(set)
    for row in rows:
        groups[row['language']].add(row['id'])
    return [item for language in sorted(groups) for item in
            sorted(groups[language], key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:per_language]]


def token_metrics(codes):
    result = {}
    for book, values in enumerate(codes, 1):
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        result.update({f'codebook_{book}_unique': int(len(counts)),
                       f'codebook_{book}_top_fraction': float(probabilities.max()),
                       f'codebook_{book}_entropy_bits': float(-(probabilities * np.log2(probabilities)).sum())})
    return result


def validation_metrics(metrics, prefix='validation'):
    result = {f'{prefix}/language_macro_ce': metrics['language_macro_ce']}
    if metrics.get('loss_scope'):
        result[f'{prefix}/{metrics["loss_scope"]}_ce']=metrics['language_macro_ce']
    result.update({f'{prefix}/language/{language}/ce': value for language, value in metrics['per_language'].items()})
    for key in ('epoch', 'train_ce', 'seconds', 'mixture_draws'):
        if key in metrics:
            result[f'{prefix}/{key}'] = metrics[key]
    gaps = defaultdict(list)
    for row in metrics.get('samples', []):
        if 'reversed_phone_ce' in row:
            gaps[row['language']].append(row['reversed_phone_ce'] - row['ce'])
    if gaps:
        means = {language: float(np.mean(values)) for language, values in gaps.items()}
        result[f'{prefix}/phone_reversal_ce_gap'] = float(np.mean(list(means.values())))
        result.update({f'{prefix}/language/{language}/phone_reversal_ce_gap': value for language, value in means.items()})
    return result


class Tracker:
    def __init__(self, root, label):
        self.root, self.label, self.run = Path(root), label, None

    def __enter__(self):
        project = os.environ.get('WANDB_PROJECT')
        if not project or int(os.environ.get('RANK',0)) != 0:
            return self
        import wandb
        raw = (self.root / 'protocol.json').read_bytes()
        protocol_hash = hashlib.sha256(raw).hexdigest()
        cfg = json.loads(raw)
        config = {k: v for k, v in cfg.items() if k not in ('bundle', 'assets', 'prompt', 'cache', 'split_ids')}
        config.update(split_counts={k: len(v) for k, v in cfg['split_ids'].items()},
                      protocol_sha256=protocol_hash, label=self.label,
                      metric_note='CE is not intelligibility; ASR is not MOS. Audition IDs fixed independent of scores.')
        folder = self.root / 'wandb'
        folder.mkdir(exist_ok=True)
        self.run = wandb.init(project=project, entity=os.environ.get('WANDB_ENTITY'),
            id=run_id(self.root.name, protocol_hash, self.label), resume='allow',
            group=self.root.name, name=f'{self.root.name}/{self.label}',
            job_type='train' if self.label == 'warmup' or self.label.startswith(('baseline-', 'exposure-', 'rrq-')) else 'evaluation',
            config=config, dir=str(folder), mode='online', save_code=False,
            settings=wandb.Settings(init_timeout=90))
        if self.label.startswith(('baseline-', 'exposure-', 'rrq-')):
            arm, seed = self.label.rsplit('-', 1)
            self.run.config.update({'arm': arm, 'training_seed': int(seed), 'loss_scope': 'suffix_only'})
        elif self.label == 'warmup':
            self.run.config.update({'arm': 'common', 'training_seed': 17, 'loss_scope': 'all_codebooks'})
        self.run.define_metric('train/*', step_metric='train/global_step')
        self.run.define_metric('validation/*', step_metric='validation/epoch')
        from .experiment import atomic_json
        atomic_json(folder / f'{self.label}.json', dict(id=self.run.id, url=self.run.url, project=project,
                                                       entity=self.run.entity, protocol_sha256=protocol_hash))
        print('WANDB_RUN', self.run.url, flush=True)
        return self

    def log(self, values):
        if self.run is not None:
            self.run.log(values)

    def summary(self, values):
        if self.run is not None:
            self.run.summary.update(values)

    def __exit__(self, typ, value, traceback):
        if self.run is not None:
            self.run.summary['execution_state'] = 'failed' if typ else 'finished'
            self.run.finish(exit_code=1 if typ else 0)


def publish_evaluation(root, split, labels):
    """Full per-phrase metrics plus a fixed, paired multilingual audition panel."""
    with Tracker(root, f'evaluation-{split}') as tracker:
        if tracker.run is None:
            return
        import wandb
        from .experiment_summary import summarize_rows
        root = Path(root)
        records = {label: json.loads((root / 'evaluation' / split / f'{label}.json').read_text())['samples'] for label in labels}
        reference_ids = {row['id'] for row in records['reference']}
        if any({row['id'] for row in rows} != reference_ids for rows in records.values()):
            raise ValueError('unpaired W&B evaluation samples')
        keys = sorted({key for rows in records.values() for row in rows for key in row['metrics']})
        table = wandb.Table(columns=['label', 'id', 'language', 'singer', 'song_group', 'reference_text',
                                    'asr_text', 'asr_hit_length_cap', *keys])
        for label, rows in records.items():
            summary = summarize_rows(rows)
            tracker.summary({f'{split}/{label}/diagnostics':{
                'asr':summary['asr_diagnostics'],'per_language_asr':summary['per_language_asr'],
                'per_language_counts':summary['per_language_counts'],'metric_support':summary['metric_support']}})
            tracker.log({f'{split}/{label}/{key}': value for key, value in summary['language_macro'].items() if value is not None})
            for language, metrics in summary['per_language'].items():
                tracker.log({f'{split}/{label}/language/{language}/{key}': value for key, value in metrics.items() if value is not None})
            tracker.summary({f'{split}/{label}/asr_cap_fraction': sum(r['asr_hit_length_cap'] for r in rows) / len(rows)})
            for row in rows:
                table.add_data(label, *[row[k] for k in ('id', 'language', 'singer', 'song_group', 'reference_text', 'asr_text', 'asr_hit_length_cap')],
                               *[row['metrics'].get(k) for k in keys])
        selected = audition_ids(records['reference'])
        indexed = {label: {r['id']: r for r in rows} for label, rows in records.items()}
        audio = wandb.Table(columns=['id', 'language', 'singer', 'reference_text', *labels])
        for item in selected:
            row = indexed['reference'][item]
            clips = [wandb.Audio(str(root / 'renders' / split / label / f'{item}.wav'),
                                 caption=f'{label} | {row["language"]} | {item}') for label in labels]
            audio.add_data(item, row['language'], row['singer'], row['reference_text'], *clips)
        tracker.log({f'{split}/per_phrase_metrics': table, f'{split}/paired_audition': audio})
        tracker.summary({'audition_ids': selected, 'audition_selection': 'SHA256 ID order, two per language; no quality filtering'})


def publish_summary(root):
    with Tracker(root, 'comparison') as tracker:
        if tracker.run is None:
            return
        import wandb
        root = Path(root)
        data = json.loads((root / 'comparison.json').read_text())
        tracker.summary(data['test_three_seed_aggregate'])
        artifact = wandb.Artifact(f'{root.name}-comparison', type='evaluation')
        for name in ('comparison.json', 'comparison.csv', 'protocol.json'):
            artifact.add_file(str(root / name))
        tracker.run.log_artifact(artifact)
