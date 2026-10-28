"""Staged measured MNIST study: convergence, memory, ablation, freeze, fresh test.

All model predictions come from final circuit counts. Classical validation
selection is experimental design, not fitted neural inference or optimization.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .coherent import CORE_DISCLOSURE, _compile_components, decode_postselection, execute
from .mnist_data import load_dataset
from .quantum_memory import memory_circuit
from .recurrent_mnist import classification_metrics, patches_from_images, provenance, write_json
from .study_utils import circuit_digest, cluster_interval, digest, execution_policy, independent_seed, initialize, run_tasks
from .resources import single_writer


PREVIOUS_TESTS = Path(__file__).parent / 'assets' / 'evaluated-mnist-indices.json'


def configuration(rounds=4, memory=1, encoding='amplitude', side=8, damping=0.):
    return dict(rounds=int(rounds), memory=int(memory), encoding=encoding,
        side=int(side), damping=float(damping))


def configuration_id(config):
    return f'm{config["memory"]}-r{config["rounds"]}-{config["encoding"]}{config["side"]}-d{config["damping"]:g}'


def make_splits(train_labels, test_labels, seeds, validation_per_class, test_per_class, exclusions):
    """Nested memory records, fixed validation, and mutually disjoint fresh tests."""
    reserved = set(map(int, exclusions))
    result = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        train, validation, test = [], [], []
        for label in range(10):
            ids = rng.permutation(np.flatnonzero(train_labels == label))
            if len(ids) < 3 + validation_per_class:
                raise ValueError('Insufficient training/validation images')
            train.extend(ids[:3].tolist())
            validation.extend(ids[3:3 + validation_per_class].tolist())
            available = np.array([i for i in np.flatnonzero(test_labels == label) if int(i) not in reserved])
            if len(available) < test_per_class:
                raise ValueError('Insufficient unused test images; lower --test-per-class or number of seeds')
            chosen = rng.permutation(available)[:test_per_class].tolist()
            test.extend(chosen)
            reserved.update(chosen)
        result[str(seed)] = {'training_nested_by_class': train,
            'validation': validation, 'fresh_test': test}
    return result


def budget_shots(native_cx, anchor_cx, base_shots, mode):
    """Hard per-image CX budget; rounding never exceeds the declared budget."""
    if min(native_cx, anchor_cx, base_shots) < 1:
        raise ValueError('Positive native CX and shot counts required')
    if mode == 'equal_shot':
        return base_shots, None
    if mode != 'equal_cnot':
        raise ValueError('Unknown resource comparison')
    budget = anchor_cx * base_shots
    shots = budget // native_cx
    if shots < 1:
        raise ValueError('CX budget cannot pay for a single execution')
    return shots, budget


def _vectors(images, config):
    values = patches_from_images(images, config['side']).reshape(len(images), -1) + 1e-8
    return np.clip(values, 0, 1) if config['encoding'] == 'angle' else values


def memory_indices(labels, memory, seed):
    selected = np.concatenate([np.flatnonzero(labels == label)[:memory] for label in range(10)])
    return np.random.default_rng(independent_seed(seed, memory, 910)).permutation(selected)


def _native(query, keys, labels, config, seed):
    return _compile_components(memory_circuit(query, keys, labels, rounds=config['rounds'],
        encoding=config['encoding'], damping=config['damping'], max_qubits=128), seed)


def _sample(native, shots, seed, path, rounds, truth, cx_budget=None):
    native_hash = circuit_digest(native)
    if path.exists():
        cached = json.loads(path.read_text())
        expected = decode_postselection(cached['counts'], 4, rounds, 10)
        if (cached.get('native_circuit_sha256') != native_hash or cached['measurement_seed'] != seed
                or cached['total_shots'] != shots or cached['truth'] != int(truth)
                or cached['cx_budget'] != cx_budget or any(cached[k] != value for k, value in expected.items())):
            raise ValueError('Cached measurement identity/counts changed: ' + str(path))
        return cached
    samples, _, resources = execute([native], shots, seed, precompiled=True)
    row = {**decode_postselection(samples[0], 4, rounds, 10), 'truth': int(truth),
        'measurement_seed': seed, 'resources': resources, 'cx_budget': cx_budget,
        'native_circuit_sha256': native_hash}
    if cx_budget is not None and resources['cx_shot_weighted_total'] > cx_budget:
        raise RuntimeError('Hard CX budget exceeded')
    write_json(path, row)
    return row


def _measure_grid(images, truth, keys, labels, config, task, directory, *, test=False, anchor_data=None):
    """Compile once per image and reuse identical native gates at every budget."""
    from qiskit import qpy
    directory.mkdir(exist_ok=True, parents=True)
    summaries = defaultdict(list)
    anchor = task['budget_anchor_configuration']
    anchor_images, anchor_keys, anchor_labels = anchor_data
    budgets = task['shot_budgets'] if not test else [task['selected_base_shots']]
    seeds = task['measurement_seeds'] if not test else [task['test_measurement_seed']]
    for local_index, (query, target) in enumerate(zip(images, truth, strict=True)):
        index = local_index + task.get('image_offset', 0)
        compile_seed = independent_seed(task['data_seed'], index, 17)
        timer = perf_counter()
        native = _native(query, keys, labels, config, compile_seed)
        compilation_seconds = perf_counter() - timer
        timer = perf_counter()
        anchored = native if config == anchor else _native(anchor_images[local_index], anchor_keys, anchor_labels, anchor, compile_seed)
        anchor_seconds = perf_counter() - timer
        metadata = directory / f'circuit-{index:05d}.json'
        if not metadata.exists():
            write_json(metadata, {'native_circuit_sha256': circuit_digest(native),
                'classical_model_compilation_seconds': compilation_seconds,
                'classical_budget_anchor_compilation_seconds': anchor_seconds,
                'compilation_amortization': 'one native circuit reused across all modes, budgets and measurement seeds',
                'native_qubits': native.num_qubits, 'native_depth': native.depth()})
        elif json.loads(metadata.read_text())['native_circuit_sha256'] != circuit_digest(native):
            raise ValueError('Compiled circuit changed on resume')
        cx, anchor_cx = int(native.count_ops().get('cx', 0)), int(anchored.count_ops().get('cx', 0))
        if index == 0:
            with (directory / 'example-native.qpy').open('wb') as stream:
                qpy.dump(native, stream)
        for mode in ('equal_shot', 'equal_cnot'):
            for budget in budgets:
                shots, cx_budget = budget_shots(cx, anchor_cx, budget, mode)
                for repetition in seeds:
                    seed = independent_seed(task['data_seed'], repetition, index, 999 if test else 31,
                        0 if mode == 'equal_shot' else 1, budget)
                    path = directory / f'{mode}-b{budget}-s{repetition}-i{index:05d}.json'
                    row = _sample(native, shots, seed, path, config['rounds'], target, cx_budget)
                    summaries[mode, budget, repetition].append(row)
        write_json(directory / 'progress.json', {'completed_images': local_index + 1, 'total_images': len(images), 'global_image_index': index})
        print(f'{directory.name}: measured image {local_index + 1}/{len(images)} at all budgets/seeds', flush=True)
    result = []
    for (mode, budget, repetition), rows in summaries.items():
        predictions = [r['prediction'] for r in rows]
        result.append({'mode': mode, 'base_shots': budget, 'measurement_seed': repetition,
            **classification_metrics(truth, predictions), 'predictions': predictions,
            'truth': np.asarray(truth, int).tolist(),
            'accepted_shots': sum(r['accepted_shots'] for r in rows),
            'total_shots': sum(r['total_shots'] for r in rows),
            'cx_shot_weighted_total': sum(r['resources']['cx_shot_weighted_total'] for r in rows),
            'sampling_seconds': sum(r['resources']['sampling_seconds'] for r in rows),
            'accepted_shots_min': min(r['accepted_shots'] for r in rows),
            'raw_counts_directory': str(directory), 'native_circuit_archive_scope': 'first image; all others reproducible from raw images and task manifest'})
    return result


def merge_grids(grids):
    grouped = defaultdict(list)
    for grid in grids:
        for row in grid:
            grouped[row['mode'], row['base_shots'], row['measurement_seed']].append(row)
    result = []
    for (mode, budget, seed), rows in grouped.items():
        truth = sum((r['truth'] for r in rows), [])
        predicted = sum((r['predictions'] for r in rows), [])
        result.append({'mode': mode, 'base_shots': budget, 'measurement_seed': seed,
            **classification_metrics(truth, predicted), 'predictions': predicted, 'truth': truth,
            **{key: sum(row[key] for row in rows) for key in (
                'accepted_shots', 'total_shots', 'cx_shot_weighted_total', 'sampling_seconds')},
            'accepted_shots_min': min(row['accepted_shots_min'] for row in rows),
            'raw_counts_directories': [row['raw_counts_directory'] for row in rows]})
    return result


def image_task(task):
    """An independent fit/validation image batch, with no test data access."""
    directory, config = Path(task['output_dir']), task['configuration']
    prepared = Path(task['prepared_data'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest() != task['prepared_data_sha256']:
        raise ValueError('Prepared training/validation data changed')
    start, stop = task['image_offset'], task['image_stop']
    with np.load(prepared) as data:
        selected = memory_indices(data['training_labels'], config['memory'], task['data_seed'])
        keys = _vectors(data['training_images'][selected], config)
        labels = data['training_labels'][selected]
        if task['image_kind'] == 'fit':
            rows = []
            for index in range(start, min(stop, len(keys))):
                native = _native(keys[index], keys, labels, config, independent_seed(task['data_seed'], index, 3))
                rows.append(_sample(native, task['fit_shots'], independent_seed(task['data_seed'], index, task['fit_shots'], 7),
                    directory / f'image-{index:03d}.json', config['rounds'], labels[index]))
            return {'rows': rows, 'truth': labels[start:stop].tolist()}
        images, truth = _vectors(data['validation_images'][start:stop], config), data['validation_labels'][start:stop]
        anchor = task['budget_anchor_configuration']
        indices = memory_indices(data['training_labels'], anchor['memory'], task['data_seed'])
        anchor_data = (_vectors(data['validation_images'][start:stop], anchor),
            _vectors(data['training_images'][indices], anchor), data['training_labels'][indices])
    return {'results': _measure_grid(images, truth, keys, labels, config, task, directory / 'validation', anchor_data=anchor_data)}


def evaluate_sharded(tasks, *, jobs, cpu_budget, directory, batch_size):
    """Gate each candidate first, then schedule only its permitted validation."""
    directory = Path(directory)
    results = {t['id']: {'configuration': t['configuration'], 'data_seed': t['data_seed'],
        'fit_gate': [], 'fit_gate_passed': False, 'results': [], 'test_evaluated': False,
        'validation_evaluated': False, 'prepared_data_sha256': t['prepared_data_sha256']} for t in tasks}
    pending = list(tasks)
    for budget in tasks[0]['gate_shot_budgets'] if tasks else []:
        shards = []
        for task in pending:
            for index in range(0, task['configuration']['memory'] * 10, batch_size):
                shards.append({**task, 'id': task['id'] + f'-i{index:05d}', 'parent': task['id'],
                    'image_kind': 'fit', 'fit_shots': budget, 'image_offset': index, 'image_stop': index + batch_size})
        measured = run_tasks('qcenn_qml.mnist_study:image_task', shards, jobs=jobs,
            cpu_budget=cpu_budget, directory=directory / f'fit-{budget}')
        grouped = defaultdict(list)
        for task, row in zip(shards, measured, strict=True):
            grouped[task['parent']].append(row)
        for parent, rows in grouped.items():
            truth = sum((r['truth'] for r in rows), [])
            predictions = [value['prediction'] for row in rows for value in row['rows']]
            metrics = classification_metrics(truth, predictions)
            results[parent]['fit_gate'].append({'shots': budget, **metrics})
            results[parent]['fit_gate_passed'] = metrics['accuracy'] == 1.
        pending = [t for t in pending if not results[t['id']]['fit_gate_passed']]
        if not pending:
            break
    shards = []
    for task in tasks:
        if not results[task['id']]['fit_gate_passed']:
            continue
        with np.load(task['prepared_data']) as data:
            size = len(data['validation_labels'])
        for index in range(0, size, batch_size):
            shards.append({**task, 'id': task['id'] + f'-i{index:05d}', 'parent': task['id'],
                'image_kind': 'validation', 'image_offset': index, 'image_stop': index + batch_size})
    values = run_tasks('qcenn_qml.mnist_study:image_task', shards, jobs=jobs,
        cpu_budget=cpu_budget, directory=directory / 'validation')
    grouped = defaultdict(list)
    for task, row in zip(shards, values, strict=True):
        grouped[task['parent']].append(row['results'])
    for task in tasks:
        row = results[task['id']]
        row.update(results=merge_grids(grouped[task['id']]),
            validation_evaluated=row['fit_gate_passed'], status='DONE' if row['fit_gate_passed'] else 'BLOCKED_FIT_GATE')
    return [results[t['id']] for t in tasks]


def evaluate_task(task):
    directory = Path(task['output_dir'])
    config = task['configuration']
    with np.load(task['prepared_data']) as data:
        train_raw, train_labels = data['training_images'], data['training_labels']
        # Each digit contributes the first m of its three fixed nested records.
        selected = memory_indices(train_labels, config['memory'], task['data_seed'])
        keys, labels = _vectors(train_raw[selected], config), train_labels[selected]
        images, truth = _vectors(data['validation_images'], config), data['validation_labels']
        anchor = task['budget_anchor_configuration']
        anchor_selected = memory_indices(train_labels, anchor['memory'], task['data_seed'])
        anchor_data = (_vectors(data['validation_images'], anchor),
            _vectors(train_raw[anchor_selected], anchor), train_labels[anchor_selected])
    gate = []
    gate_passed = False
    for budget in task['gate_shot_budgets']:
        rows = []
        target = directory / f'fit-{budget}'
        target.mkdir(exist_ok=True)
        for index, (query, label) in enumerate(zip(keys, labels, strict=True)):
            native = _native(query, keys, labels, config, independent_seed(task['data_seed'], index, 3))
            rows.append(_sample(native, budget, independent_seed(task['data_seed'], index, budget, 7),
                target / f'image-{index:03d}.json', config['rounds'], label))
        measured = classification_metrics(labels, [r['prediction'] for r in rows])
        gate.append({'shots': budget, **measured})
        print(f'{configuration_id(config)} tiny retrieval: shots={budget} accuracy={measured["accuracy"]:.3f}', flush=True)
        if measured['accuracy'] == 1.:
            gate_passed = True
            break
    write_json(directory / 'fit-gate.json', {'passed': gate_passed, 'attempts': gate})
    results = _measure_grid(images, truth, keys, labels, config, task, directory / 'validation', anchor_data=anchor_data) if gate_passed else []
    return {'configuration': config, 'data_seed': task['data_seed'], 'fit_gate_passed': gate_passed,
        'fit_gate': gate, 'validation_evaluated': gate_passed, 'test_evaluated': False,
        'results': results, 'prepared_data_sha256': task['prepared_data_sha256'],
        'status': 'DONE' if gate_passed else 'BLOCKED_FIT_GATE'}


def summarize(candidates, agreement_threshold):
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[digest(candidate['configuration'])].append(candidate)
    summaries = []
    for runs in grouped.values():
        config = runs[0]['configuration']
        if not all(r['fit_gate_passed'] for r in runs):
            summaries.append({'configuration': config, 'eligible': False, 'reason': 'tiny retrieval gate failed'})
            continue
        rows = defaultdict(list)
        for run in runs:
            for row in run['results']:
                rows[row['mode'], row['base_shots']].append((run['data_seed'], row))
        for (mode, budget), pairs in rows.items():
            agreements = []
            high_agreements = []
            high_matches = []
            largest_budget = max(b for m, b in rows if m == mode)
            for seed in sorted({seed for seed, _ in pairs}):
                predictions = np.asarray([r['predictions'] for s, r in pairs if s == seed])
                agreements.extend(max(Counter(column).values()) / len(column) for column in predictions.T)
                high = np.asarray([r['predictions'] for s, r in rows[mode, largest_budget] if s == seed])
                high_agreements.extend(max(Counter(column).values()) / len(column) for column in high.T)
                high_matches.extend(Counter(a).most_common(1)[0][0] == Counter(b).most_common(1)[0][0]
                    for a, b in zip(predictions.T, high.T, strict=True))
            accuracy = [r['accuracy'] for _, r in pairs]
            agreement = float(np.mean(agreements))
            high_match = float(np.mean(high_matches))
            eligible = min(agreement, float(np.mean(high_agreements)), high_match) >= agreement_threshold and all(r['accepted_shots_min'] > 0 for _, r in pairs)
            summaries.append({'configuration': config, 'mode': mode, 'base_shots': budget,
                'accuracy_mean': float(np.mean(accuracy)),
                'accuracy_data_seed_cluster_95': cluster_interval(accuracy, [s for s, _ in pairs]),
                'prediction_agreement_mean': agreement,
                'largest_budget_consensus_agreement': high_match,
                'accepted_shot_fraction': sum(r['accepted_shots'] for _, r in pairs) / sum(r['total_shots'] for _, r in pairs),
                'cx_shot_weighted_mean': float(np.mean([r['cx_shot_weighted_total'] for _, r in pairs])),
                'eligible': eligible,
                'reason': 'stable measured validation' if eligible else 'measurement variability or abstention unresolved'})
    return summaries


def choose(summaries, mode='equal_shot'):
    eligible = [r for r in summaries if r.get('eligible') and r.get('mode') == mode]
    return min(eligible, key=lambda r: (-r['accuracy_mean'], r['cx_shot_weighted_mean'],
        configuration_id(r['configuration']), r['base_shots'])) if eligible else None


def test_task(task):
    directory = Path(task['output_dir'])
    config = task['configuration']
    with np.load(task['prepared_data']) as data:
        selected = memory_indices(data['training_labels'], config['memory'], task['data_seed'])
        keys = _vectors(data['training_images'][selected], config)
        labels = data['training_labels'][selected]
        anchor = task['budget_anchor_configuration']
        anchor_selected = memory_indices(data['training_labels'], anchor['memory'], task['data_seed'])
        anchor_keys, anchor_labels = _vectors(data['training_images'][anchor_selected], anchor), data['training_labels'][anchor_selected]
    ids = np.asarray(task['test_indices'], int)
    if task.get('test_cache'):
        cache_path = Path(task['test_cache'])
        if hashlib.sha256(cache_path.read_bytes()).hexdigest() != task['test_cache_sha256']:
            raise ValueError('Frozen test cache changed')
        with np.load(cache_path) as cache:
            start = task.get('image_offset', 0); stop = start + len(ids)
            if not np.array_equal(ids, cache['indices'][start:stop]):
                raise ValueError('Frozen test indices differ from cache')
            raw, truth = cache['images'][start:stop], cache['labels'][start:stop]
    else:
        _, _, test_images, test_labels = load_dataset(task['dataset'], task['data_root'], download=False)
        raw, truth = test_images[ids], test_labels[ids]
    images = _vectors(raw, config)
    anchor_data = (_vectors(raw, anchor), anchor_keys, anchor_labels)
    results = {}
    for name, model in [('selected', config), ('k0_reference', {**config, 'rounds': 1, 'damping': 0.})]:
        # k=0 uses the identical raw encoding/memory and a single overlap filter.
        # Every candidate uses the same ten-record amplitude k=0 budget anchor.
        measured_task = {**task, 'measurement_seeds': [task['test_measurement_seed']]}
        results[name] = _measure_grid(images, truth, keys, labels, model, measured_task,
            directory / name, test=True, anchor_data=anchor_data)
    return {'data_seed': task['data_seed'], 'configuration': config, 'test_indices': ids.tolist(),
        'test_evaluated': True, 'freeze_sha256': task['freeze_sha256'], 'results': results}


@single_writer
def test_frozen(freeze_file, output_dir, *, jobs=1, cpu_budget=1, data_root='data', resume=False, image_batch_size=0):
    freeze_path = Path(freeze_file).resolve()
    freeze = json.loads(freeze_path.read_text())
    payload = {k: v for k, v in freeze.items() if k != 'sha256'}
    if digest(payload) != freeze['sha256'] or freeze['selection_used_test_labels']:
        raise ValueError('Invalid or altered freeze manifest')
    source = provenance()
    if source['source_sha256'] != freeze['source']['source_sha256']:
        raise ValueError('Frozen study source differs from current source')
    if source['packages'] != freeze['source']['packages']:
        raise ValueError('Frozen study package versions differ from current environment')
    if freeze.get('execution_policy', execution_policy()) != execution_policy():
        raise ValueError('Frozen study execution backend policy differs from current environment')
    target = Path(output_dir)
    initialize(target, {'freeze_sha256': freeze['sha256'], 'data_root': str(Path(data_root).resolve()),
        'image_batch_size': image_batch_size}, source, resume)
    tasks = []
    test_images = test_labels = None
    if image_batch_size:
        _, _, test_images, test_labels = load_dataset(freeze['dataset'], data_root, download=False)
    for seed, split in freeze['splits'].items():
        prepared = (freeze_path.parent / split['prepared_data']).resolve()
        if hashlib.sha256(prepared.read_bytes()).hexdigest() != split['prepared_data_sha256']:
            raise ValueError('Frozen memory/validation archive changed')
        base = {'id': 'test-' + seed, 'configuration': freeze['selected']['configuration'],
            'prepared_data': str(prepared), 'data_seed': int(seed), 'dataset': freeze['dataset'],
            'data_root': str(Path(data_root).resolve()), 'test_indices': split['fresh_test'],
            'selected_base_shots': freeze['selected']['base_shots'], 'shot_budgets': [],
            'measurement_seeds': [], 'test_measurement_seed': freeze['test_measurement_seed'],
            'budget_anchor_configuration': freeze['budget_anchor_configuration'],
            'source_sha256': freeze['source']['source_sha256'],
            'freeze_sha256': freeze['sha256']}
        if image_batch_size:
            cache = target / f'test-cache-{seed}.npz'
            cache_receipt = cache.with_suffix('.json')
            ids = np.asarray(split['fresh_test'], int)
            if not cache.exists():
                np.savez_compressed(cache, images=test_images[ids], labels=test_labels[ids], indices=ids)
                write_json(cache_receipt, {'sha256': hashlib.sha256(cache.read_bytes()).hexdigest(), 'freeze_sha256': freeze['sha256']})
            cache_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
            if json.loads(cache_receipt.read_text()) != {'sha256': cache_sha, 'freeze_sha256': freeze['sha256']}:
                raise ValueError('Frozen test cache receipt changed')
            for start in range(0, len(ids), image_batch_size):
                tasks.append({**base, 'id': base['id'] + f'-i{start:05d}', 'image_offset': start,
                    'test_cache': str(cache), 'test_cache_sha256': cache_sha,
                    'test_indices': ids[start:start + image_batch_size].tolist()})
        else:
            tasks.append(base)
    rows = run_tasks('qcenn_qml.mnist_study:test_task', tasks, jobs=jobs, cpu_budget=cpu_budget,
        directory=target / 'tasks')
    if image_batch_size:
        merged = []
        for seed, split in freeze['splits'].items():
            parts = [r for r in rows if r['data_seed'] == int(seed)]
            indices = sum((r['test_indices'] for r in parts), [])
            if indices != split['fresh_test']:
                raise ValueError('Frozen test did not measure every reserved image exactly once')
            merged.append({**{k: v for k, v in parts[0].items() if k != 'results'},
                'test_indices': indices, 'results': {name: merge_grids([r['results'][name] for r in parts]) for name in ('selected', 'k0_reference')}})
        rows = merged
    report = {'experiment': 'frozen_fresh_mnist_test', **CORE_DISCLOSURE, 'status': 'DONE',
        'freeze_sha256': freeze['sha256'], 'test_results': rows, 'source': source,
        'selection_used_test_labels': False, 'previously_evaluated_test_indices_excluded': freeze['excluded_test_indices']}
    summary, paired = [], []
    for mode in ('equal_shot', 'equal_cnot'):
        for name in ('selected', 'k0_reference'):
            values = [next(v for v in row['results'][name] if v['mode'] == mode) for row in rows]
            truth = sum((v['truth'] for v in values), [])
            predictions = sum((v['predictions'] for v in values), [])
            summary.append({'model': name, 'mode': mode, **classification_metrics(truth, predictions),
                'cx_shot_weighted_total': sum(v['cx_shot_weighted_total'] for v in values),
                'total_shots': sum(v['total_shots'] for v in values)})
        differences, groups = [], []
        recurrent_only = reference_only = 0
        for row in rows:
            selected = next(v for v in row['results']['selected'] if v['mode'] == mode)
            reference = next(v for v in row['results']['k0_reference'] if v['mode'] == mode)
            for a, b, truth in zip(selected['predictions'], reference['predictions'], selected['truth'], strict=True):
                recurrent_only += a == truth and b != truth
                reference_only += b == truth and a != truth
                differences.append(int(a == truth) - int(b == truth))
                groups.append(row['data_seed'])
        paired.append({'mode': mode, 'accuracy_difference_selected_minus_k0': float(np.mean(differences)),
            'data_seed_cluster_95': cluster_interval(differences, groups),
            'selected_only_correct': recurrent_only, 'k0_only_correct': reference_only})
    report.update(summary=summary, paired_comparisons=paired)
    write_json(target / 'report.json', report)
    return report


@single_writer
def study(*, dataset='mnist', data_seeds=(42, 43, 44), measurement_seeds=(101, 202, 303),
        shot_budgets=(1024, 4096, 16384), gate_shot_budgets=(1024, 4096, 16384),
        validation_per_class=2, test_per_class=100, damping_values=(.05,), agreement_threshold=.95,
        test_measurement_seed=99991, exclusions_file=None, jobs=1, cpu_budget=1,
        data_root='data', no_download=False, output_dir='results/mnist-study', resume=False,
        phase='all', smoke=False, selection_mode='equal_shot', image_batch_size=0):
    if len(set(data_seeds)) != len(data_seeds) or not data_seeds:
        raise ValueError('Unique nonempty data seeds required')
    if len(set(measurement_seeds)) < 2 or len(set(measurement_seeds)) != len(measurement_seeds) or test_measurement_seed in measurement_seeds:
        raise ValueError('At least two distinct measurement seeds and an independent test seed required')
    if any(int(s) < 0 for s in (*data_seeds, *measurement_seeds, test_measurement_seed)):
        raise ValueError('Nonnegative seeds required')
    if len(set(shot_budgets)) < 2 or any(s < 1 for s in (*shot_budgets, *gate_shot_budgets)):
        raise ValueError('At least two distinct positive convergence shot budgets required')
    if min(validation_per_class, test_per_class) < 1 or not 0 < agreement_threshold <= 1:
        raise ValueError('Invalid split sizes/stability threshold')
    if any(not 0 < d < 1 for d in damping_values):
        raise ValueError('Damping ablations require 0 < damping < 1')
    if selection_mode not in ('equal_shot', 'equal_cnot') or image_batch_size < 0:
        raise ValueError('Invalid selection mode/image batch size')
    args = dict(locals())
    for key in ('resume', 'phase', 'jobs', 'cpu_budget', 'no_download'):
        args.pop(key)
    args['data_root'] = str(Path(data_root).resolve())
    args['output_dir'] = str(Path(output_dir).resolve())
    source = provenance()
    directory = Path(output_dir).resolve()
    manifest = initialize(directory, args, source, resume)
    source = manifest['source']
    timer = perf_counter()
    train_images, train_labels, test_images, test_labels = load_dataset(dataset, data_root, download=not no_download)
    loading_seconds = perf_counter() - timer
    excluded = set()
    if dataset == 'mnist':
        excluded.update(json.loads(PREVIOUS_TESTS.read_text())['indices'])
    if exclusions_file:
        external = json.loads(Path(exclusions_file).read_text())
        if external.get('dataset') != dataset:
            raise ValueError('Exclusion dataset mismatch')
        excluded.update(external['indices'])
    splits = make_splits(train_labels, test_labels, data_seeds, validation_per_class, test_per_class, excluded)
    # No test image or test label is passed to any validation worker.
    del test_images, test_labels
    for seed, split in splits.items():
        path = directory / f'data-seed-{seed}.npz'
        if not path.exists():
            ti, vi = split['training_nested_by_class'], split['validation']
            np.savez_compressed(path, training_images=train_images[ti], training_labels=train_labels[ti],
                validation_images=train_images[vi], validation_labels=train_labels[vi])
        split['prepared_data'] = path.name
        split['prepared_data_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(directory / 'splits.json', {'dataset': dataset, 'splits': splits, 'excluded_test_indices': sorted(excluded)})
    del train_images, train_labels
    completed, stages = [], []
    def run_stage(name, configs):
        seen = {digest(row['configuration']) for row in completed}
        tasks = []
        for config in configs:
            if digest(config) in seen:
                continue
            seen.add(digest(config))
            for seed, split in splits.items():
                tasks.append({'id': configuration_id(config) + '-s' + seed, 'configuration': config,
                    'data_seed': int(seed), 'prepared_data': str(directory / split['prepared_data']),
                    'prepared_data_sha256': split['prepared_data_sha256'],
                    'shot_budgets': sorted(shot_budgets), 'gate_shot_budgets': sorted(gate_shot_budgets),
                    'budget_anchor_configuration': configuration(rounds=1, side=4 if smoke else 8),
                    'measurement_seeds': list(measurement_seeds)})
                tasks[-1]['source_sha256'] = source['source_sha256']
        print(f'STAGE {name}: {len(tasks)} configuration/data-seed jobs', flush=True)
        if image_batch_size:
            completed.extend(evaluate_sharded(tasks, jobs=jobs, cpu_budget=cpu_budget,
                directory=directory / name, batch_size=image_batch_size))
        else:
            completed.extend(run_tasks('qcenn_qml.mnist_study:evaluate_task', tasks, jobs=jobs,
                cpu_budget=cpu_budget, directory=directory / name))
        summaries = summarize(completed, agreement_threshold)
        selected = choose(summaries, selection_mode)
        stage = {'stage': name, 'summaries': summaries, 'selected': selected,
            'selection_mode': selection_mode, 'matched_cnot_selected': choose(summaries, 'equal_cnot'),
            'status': 'DONE' if selected else 'BLOCKED_FIT_OR_MEASUREMENT_STABILITY'}
        stages.append(stage)
        write_json(directory / (name + '.json'), stage)
        write_json(directory / 'report.json', {'experiment': 'staged_measured_mnist_study', **CORE_DISCLOSURE,
            'status': 'VALIDATING', 'stages': stages, 'test_evaluated': False, 'source': source})
        return selected
    side = 4 if smoke else 8
    selected = run_stage('convergence', [configuration(rounds=r, side=side) for r in (1, 2, 4)])
    if selected:
        for memory in (2, 3):
            selected = run_stage(f'memory-{memory}', [configuration(rounds=r, memory=memory, side=side) for r in (1, 2, 4)])
        best = selected['configuration']
        ablations = [configuration(best['rounds'], best['memory'], encoding, resolution, damping)
            for encoding, resolution in [('amplitude', side), ('amplitude', 4), ('angle', 4)]
            for damping in (0., *damping_values)]
        selected = run_stage('encoding-damping', ablations)
    final = {'experiment': 'staged_measured_mnist_study', **CORE_DISCLOSURE, 'stages': stages,
        'dataset': dataset, 'smoke_only': smoke, 'source': source, 'test_evaluated': False,
        'dataset_loading_seconds': loading_seconds, 'logical_qubit_limit': 128,
        'record_order': 'Seeded permutation per memory size; damping does not use class-sorted record addresses',
        'status': 'BLOCKED_FIT_OR_MEASUREMENT_STABILITY', 'selection_used_test_labels': False}
    if selected:
        payload = {'schema': 2, 'dataset': dataset, 'selected': selected, 'splits': splits,
            'excluded_test_indices': sorted(excluded), 'test_measurement_seed': test_measurement_seed,
            'selection_used_test_labels': False, 'source': source, 'execution_policy': execution_policy(),
            'budget_anchor_configuration': configuration(rounds=1, side=side),
            'cx_budget_definition': 'Native CX count of k=0, one record per digit, amplitude encoding, times base shot budget; identical budget across all candidates for each image',
            'selection_mode': selection_mode,
            'selection_rule': f'eligible {selection_mode} validation accuracy, then measured CX cost, then deterministic configuration order; equal-CNOT efficiency is reported separately'}
        frozen = {**payload, 'sha256': digest(payload)}
        freeze_path = directory / 'frozen-configuration.json'
        if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
            raise ValueError('Refusing to overwrite a different frozen configuration')
        write_json(freeze_path, frozen)
        final.update(status='VALIDATED_AND_FROZEN', selected=selected, freeze_sha256=frozen['sha256'])
        if phase == 'all':
            result = test_frozen(freeze_path, directory / 'fresh-test', jobs=jobs, cpu_budget=cpu_budget,
                data_root=data_root, resume=resume, image_batch_size=image_batch_size)
            final.update(status='DONE', test_evaluated=True, test_results=result['test_results'],
                test_summary=result['summary'], paired_comparisons=result['paired_comparisons'])
    write_json(directory / 'report.json', final)
    print(f'MNIST study status: {final["status"]}; report={directory / "report.json"}', flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'validation', 'test'], default='all')
    parser.add_argument('--freeze-file')
    parser.add_argument('--dataset', choices=['mnist', 'digits-proxy'], default='mnist')
    parser.add_argument('--data-seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--measurement-seeds', nargs='+', type=int, default=[101, 202, 303])
    parser.add_argument('--shot-budgets', nargs='+', type=int, default=[1024, 4096, 16384])
    parser.add_argument('--gate-shot-budgets', nargs='+', type=int, default=[1024, 4096, 16384])
    parser.add_argument('--damping-values', nargs='+', type=float, default=[.05])
    parser.add_argument('--agreement-threshold', type=float, default=.95)
    parser.add_argument('--selection-mode', choices=['equal_shot', 'equal_cnot'], default='equal_shot')
    parser.add_argument('--image-batch-size', type=int, default=1, help='Independent images per worker; 0 retains coarse seed/config tasks')
    parser.add_argument('--validation-per-class', type=int, default=2)
    parser.add_argument('--test-per-class', type=int, default=100)
    parser.add_argument('--test-measurement-seed', type=int, default=99991)
    parser.add_argument('--exclusions-file')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--cpu-budget', type=int, default=1)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', default='results/mnist-study')
    options = vars(parser.parse_args(argv))
    if options['smoke']:
        options.update(dataset='digits-proxy', data_seeds=[42], measurement_seeds=[101, 202],
            shot_budgets=[64, 128], gate_shot_budgets=[128], validation_per_class=1, test_per_class=1)
    freeze_file = options.pop('freeze_file')
    if options['phase'] == 'test':
        if not freeze_file:
            parser.error('--phase test requires --freeze-file')
        result = test_frozen(freeze_file, options['output_dir'], jobs=options['jobs'],
            cpu_budget=options['cpu_budget'], data_root=options['data_root'], resume=options['resume'], image_batch_size=options['image_batch_size'])
    else:
        if freeze_file:
            parser.error('--freeze-file is only valid for --phase test')
        result = study(**options)
    if result['status'].startswith('BLOCKED'):
        raise SystemExit(2)
    return result


if __name__ == '__main__':
    main()
