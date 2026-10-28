"""Separate multi-corpus denoising study; every noise level uses identical gates.

Defaults are explicitly authored structured corpora, not evidence of natural
language understanding. --corpus-files accepts broader user-supplied text.
Vocabulary and memory caps are explicit; held-out OOVs retain their error mass.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ..coherent import CORE_DISCLOSURE
from ..recurrent_mnist import provenance, write_json
from ..study_utils import cluster_interval, digest, initialize, run_tasks
from ..resources import single_writer


DEFAULT_CORPORA = [Path(__file__).resolve().parents[1] / 'assets' / f'diffusion-{name}.txt'
    for name in ('animals', 'city', 'devices')]


def evaluate_task(task):
    from .evaluation import execute_benchmark
    corpus = Path(task['corpus_file'])
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != task['corpus_sha256']:
        raise ValueError('Corpus snapshot changed')
    target = Path(task['output_dir']) / 'benchmark'
    report = execute_benchmark(corpus_file=str(corpus), seed=task['data_seed'],
        measurement_seed=task['measurement_seed'], shots=task['shots'], examples=task['examples'],
        readout_errors=task['readout_errors'], max_train_records=task['max_train_records'],
        max_vocabulary=task['max_vocabulary'], steps=task['steps'], amplify=task['amplify'],
        output_dir=str(target), resume=(target / 'study-manifest.json').exists(),
        case_start=task.get('case_start', 0), case_stop=task.get('case_stop'),
        batch_size=task.get('case_batch_size', 4), prepared_corpus=task.get('prepared_corpus'),
        prepared_corpus_sha256=task.get('prepared_corpus_sha256'))
    if 'case_start' in task:
        path = target / 'report.json'
        return {'report': str(path), 'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return compact_report(report, task, target / 'report.json')


def compact_report(report, task, path):
    return {'corpus': task['corpus_name'], 'corpus_sha256': task['corpus_sha256'],
        'data_seed': task['data_seed'], 'measurement_seed': task['measurement_seed'],
        'train_count': report['train_count'], 'heldout_count': report['heldout_count'],
        'evaluation_count': report['evaluation_count'], 'independent_sentences': report['independent_heldout_sentence_count'],
        'classical_conditional_baseline': report['classical_conditional_baseline'],
        'matched_results': {mode: {step: {key: value for key, value in row.items()
            if key not in ('native_circuit_sha256', 'resources')} | {
                'cx_shot_weighted_total': row['resources']['cx_shot_weighted_total']}
            for step, row in values.items()} for mode, values in report['matched_results'].items()},
        'report': str(path)}


@single_writer
def study(corpus_files=None, data_seeds=(42, 43, 44), measurement_seeds=(101, 202),
        shots=1024, examples=0, readout_errors=(.01, .05), max_train_records=128,
        max_vocabulary=32, steps=(0, 1, 3), amplify=1, jobs=1, cpu_budget=1,
        output_dir='results/diffusion-study', resume=False, smoke=False, case_batch_size=1):
    files = [Path(path).resolve() for path in (corpus_files or DEFAULT_CORPORA)]
    if case_batch_size < 0:
        raise ValueError('case-batch-size must be nonnegative')
    if (not data_seeds or not measurement_seeds or len(set(data_seeds)) != len(data_seeds) or
            len(set(measurement_seeds)) != len(measurement_seeds) or
            any(s < 0 for s in (*data_seeds, *measurement_seeds))):
        raise ValueError('Nonempty independent unique nonnegative seed lists required')
    if len(set(files)) != len(files) or len({path.name for path in files}) != len(files):
        raise ValueError('Corpus files must have unique paths and filenames for separate summaries')
    configuration = {'corpora': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_metadata_sha256': hashlib.sha256(path.with_suffix('.source.json').read_bytes()).hexdigest() if path.with_suffix('.source.json').exists() else None}
        for path in files], 'data_seeds': list(data_seeds), 'measurement_seeds': list(measurement_seeds),
        'shots': shots, 'examples': examples, 'readout_errors': list(readout_errors),
        'max_train_records': max_train_records, 'max_vocabulary': max_vocabulary,
        'steps': list(steps), 'amplify': amplify, 'smoke': smoke, 'case_batch_size': case_batch_size}
    directory = Path(output_dir).resolve()
    manifest = initialize(directory, configuration, provenance(), resume)
    snapshots = directory / 'corpora'
    snapshots.mkdir(exist_ok=True)
    tasks = []
    for index, path in enumerate(files):
        snapshot = snapshots / f'{index}-{path.name}'
        if not snapshot.exists():
            snapshot.write_bytes(path.read_bytes())
        identity = configuration['corpora'][index]['sha256']
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != identity:
            raise ValueError('Stored corpus snapshot changed')
        attribution = path.with_suffix('.source.json')
        if attribution.exists():
            destination = snapshot.with_suffix('.source.json')
            if not destination.exists():
                destination.write_bytes(attribution.read_bytes())
            if hashlib.sha256(destination.read_bytes()).hexdigest() != configuration['corpora'][index]['source_metadata_sha256']:
                raise ValueError('Corpus attribution metadata changed')
        for seed in data_seeds:
            for measurement_seed in measurement_seeds:
                tasks.append({'id': f'corpus-{index}-d{seed}-m{measurement_seed}',
                    'corpus_name': path.name, 'corpus_file': str(snapshot), 'corpus_sha256': identity,
                    'data_seed': seed, 'measurement_seed': measurement_seed, 'shots': shots,
                    'examples': examples, 'readout_errors': list(readout_errors),
                    'max_train_records': max_train_records, 'max_vocabulary': max_vocabulary,
                    'steps': list(steps), 'amplify': amplify})
                tasks[-1]['source_sha256'] = manifest['source']['source_sha256']
    if case_batch_size > 0:
        from .matched import corpus
        from .aggregate import combine
        shards, prepared_files = [], {}
        for task in tasks:
            key = (task['corpus_sha256'], task['data_seed'])
            if key not in prepared_files:
                vocab, mask, train, heldout, split = corpus(corpus_file=task['corpus_file'], seed=task['data_seed'],
                    max_train_records=max_train_records, max_vocabulary=max_vocabulary)
                cache = directory / 'prepared' / (task['id'].rsplit('-m', 1)[0] + '.json')
                cache.parent.mkdir(exist_ok=True)
                payload = {'vocabulary': vocab, 'mask': mask, 'training': train, 'heldout': heldout, 'split': split,
                    'configuration': {'seed': task['data_seed'], 'max_train_records': max_train_records,
                        'max_vocabulary': max_vocabulary, 'corpus_sha256': task['corpus_sha256']}}
                if cache.exists():
                    if digest(json.loads(cache.read_text())) != digest(payload):
                        raise ValueError('Prepared corpus cache changed on resume')
                else:
                    write_json(cache, payload)
                prepared_files[key] = (cache, hashlib.sha256(cache.read_bytes()).hexdigest(), len(heldout))
            cache, checksum, total_heldout = prepared_files[key]
            size = min(examples or total_heldout, total_heldout)
            for step in steps:
                for start in range(0, size, case_batch_size):
                    shards.append({**task, 'id': task['id'] + f'-t{step}-i{start:06d}',
                        'parent': task['id'], 'steps': [step], 'case_start': start,
                        'case_stop': min(start + case_batch_size, size), 'case_batch_size': case_batch_size,
                        'prepared_corpus': str(cache), 'prepared_corpus_sha256': checksum})
        parts = run_tasks('qcenn_qml.diffusion.study:evaluate_task', shards,
            jobs=jobs, cpu_budget=cpu_budget, directory=directory / 'tasks')
        grouped_parts = defaultdict(list)
        for shard, part in zip(shards, parts, strict=True):
            grouped_parts[shard['parent']].append(part)
        rows = []
        for task in tasks:
            target = directory / 'runs' / task['id']
            report = combine(grouped_parts[task['id']], target)
            rows.append(compact_report(report, task, target / 'report.json'))
    else:
        rows = run_tasks('qcenn_qml.diffusion.study:evaluate_task', tasks,
            jobs=jobs, cpu_budget=cpu_budget, directory=directory / 'tasks')
    grouped = defaultdict(list)
    for row in rows:
        for mode, values in row['matched_results'].items():
            for step, value in values.items():
                grouped[row['corpus'], mode, step].append((row['data_seed'], value))
    summaries = []
    for (corpus, mode, step), values in grouped.items():
        measured = [value['measured_correct_token_probability'] for _, value in values]
        summaries.append({'corpus': corpus, 'mode': mode, 'steps': int(step),
            'measured_correct_token_probability_mean': float(np.mean(measured)),
            'repeated_split_cluster_95': cluster_interval(measured, [seed for seed, _ in values]),
            'matched_classical_channel_probability_mean': float(np.mean([
                value['classical_matched_replacement_probability_mean'] for _, value in values])),
            'top1_accuracy_mean': float(np.mean([value['sequence_top1_accuracy'] for _, value in values])),
            'accepted_shot_fraction': sum(value['accepted_shots'] for _, value in values) / sum(value['total_shots'] for _, value in values),
            'cx_shot_weighted_total': sum(value['cx_shot_weighted_total'] for _, value in values)})
    report = {'experiment': 'multi_corpus_matched_diffusion_study', **CORE_DISCLOSURE,
        'status': 'DONE', 'configuration': configuration, 'runs': rows, 'summaries': summaries,
        'bundled_corpora_are_authored_structured_text': corpus_files is None,
        'all_heldout_windows_requested': examples == 0, 'source': manifest['source'],
        'uncertainty_scope': 'Sentence clustering within runs; repeated data-split clustering across runs. Overlapping splits are not independent corpora.',
        'limitations': ['Bounded-vocabulary raw-memory denoising, not a trained diffusion LLM',
            'Bundled structured text is a reproducible diagnostic; use independent real text for language claims',
            'Paired readout flips do not model quantum gate or device noise']}
    write_json(directory / 'report.json', report)
    lines = ['# Separate matched diffusion study', '',
        '| Corpus | Mode | Steps | Measured target probability | Matched classical channel |',
        '|---|---|---:|---:|---:|']
    for row in summaries:
        lines.append(f'| {row["corpus"]} | {row["mode"]} | {row["steps"]} | '
            f'{row["measured_correct_token_probability_mean"]:.4f} | {row["matched_classical_channel_probability_mean"]:.4f} |')
    lines += ['', report['limitations'][1], '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf8')
    print(f'Diffusion study DONE; report={directory / "report.json"}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-files', nargs='+')
    parser.add_argument('--data-seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--measurement-seeds', type=int, nargs='+', default=[101, 202])
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--examples', type=int, default=0, help='0 evaluates every held-out window')
    parser.add_argument('--readout-errors', type=float, nargs='+', default=[.01, .05])
    parser.add_argument('--max-train-records', type=int, default=128)
    parser.add_argument('--max-vocabulary', type=int, default=32)
    parser.add_argument('--steps', type=int, nargs='+', default=[0, 1, 3])
    parser.add_argument('--amplify', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--case-batch-size', type=int, default=1, help='Independent held-out windows per worker; 0 retains whole-run tasks')
    parser.add_argument('--cpu-budget', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', default='results/diffusion-study')
    options = vars(parser.parse_args(argv))
    if options['smoke']:
        options.update(data_seeds=[42], measurement_seeds=[101], shots=32, examples=2,
            max_train_records=8, max_vocabulary=8, steps=[0, 1], amplify=0)
        if options['corpus_files'] is None:
            options['corpus_files'] = [str(DEFAULT_CORPORA[0])]
    return study(**options)


if __name__ == '__main__':
    main()
