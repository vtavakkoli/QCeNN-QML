"""Combine disjoint measured text batches without treating shots as text cases."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ..recurrent_mnist import write_json
from ..study_utils import cluster_interval


def combine(parts, output_dir):
    reports = []
    for part in parts:
        path = Path(part['report'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != part['report_sha256']:
            raise ValueError('Diffusion batch report changed: ' + str(path))
        reports.append(json.loads(path.read_text()))
    if not reports:
        raise ValueError('No measured diffusion batches')
    first = reports[0]
    for report in reports:
        for key in ('train_count', 'heldout_count', 'vocabulary', 'seeds'):
            if report[key] != first[key]:
                raise ValueError('Cannot combine different corpora, splits, sources or runtimes')
        for key in ('source_sha256', 'packages', 'source_package_version'):
            if report['source'].get(key) != first['source'].get(key):
                raise ValueError('Cannot combine different sources or runtimes')
    grouped, resources = defaultdict(list), defaultdict(list)
    seen = set()
    for report in reports:
        for row in report['cases']:
            identity = row['mode'], row['steps'], row['heldout_window_index']
            if identity in seen:
                raise ValueError('Overlapping diffusion measurement shards')
            seen.add(identity)
            grouped[row['mode'], row['steps']].append(row)
        for mode, steps in report['matched_results'].items():
            for step, values in steps.items():
                resources[mode, int(step)].extend(values['resources']['batches'])
    matched, unique_cases = {}, {}
    expected_ids = None
    native_by_case = {(step, r['heldout_window_index']): r['native_circuit_sha256']
        for (mode, step), rows in grouped.items() if mode == 'noiseless' for r in rows}
    for (mode, step), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r['case_index'])
        ids = [r['heldout_window_index'] for r in rows]
        if expected_ids is None:
            expected_ids = ids
        if ids != expected_ids:
            raise ValueError('Noise modes/steps do not measure identical held-out cases')
        groups = [r['sentence_id'] for r in rows]
        batches = resources[mode, step]
        total = sum(r['total_shots'] for r in rows)
        accepted = sum(r['accepted_shots'] for r in rows)
        costs = {key: sum(r[key] for r in batches) for key in (
            'total_shots', 'compilation_seconds', 'cx_shot_weighted_total', 'sampling_seconds')}
        costs.update(qubits_max=max(r['qubits_max'] for r in batches),
            depth_max=max(r['depth_max'] for r in batches), batches=batches)
        if costs['total_shots'] != total:
            raise ValueError('Diffusion shard shot costs do not match raw counts')
        observed = [r['quantum_target_frequency'] for r in rows]
        correct = [float(r['top1_correct']) for r in rows]
        error = next(r['matched_results'][mode][str(step)]['readout_error'] for r in reports if mode in r['matched_results'] and str(step) in r['matched_results'][mode])
        matched.setdefault(mode, {})[str(step)] = {
            'measured_correct_token_probability': float(np.mean(observed)),
            'correct_token_probability_sentence_cluster_95': cluster_interval(observed, groups, first['seeds']['data']),
            'independent_ideal_probability_mean': float(np.mean([r['analytic_ideal_target_probability'] for r in rows])),
            'classical_matched_replacement_probability_mean': float(np.mean([r['matched_classical_channel_target_probability'] for r in rows])),
            'sequence_top1_accuracy': float(np.mean(correct)),
            'top1_sentence_cluster_95': cluster_interval(correct, groups, first['seeds']['data']),
            'samples': len(rows), 'total_shots': total, 'accepted_shots': accepted,
            'accepted_shot_fraction': accepted / total,
            'invalid_value_shots': sum(r['invalid_value_shots'] for r in rows),
            'compiled_max_qubits': costs['qubits_max'], 'compiled_depth_max': costs['depth_max'],
            'native_circuit_sha256': [r['native_circuit_sha256'] for r in rows],
            'readout_error': error, 'resources': costs}
        for row in rows:
            previous = unique_cases.setdefault(row['heldout_window_index'], row)
            if row['native_circuit_sha256'] != native_by_case.get((step, row['heldout_window_index'])):
                raise ValueError('Noisy run used different native gates')
            for key in ('sentence_id', 'truth', 'classical_direct_target_probability', 'classical_direct_top1_correct'):
                if previous[key] != row[key]:
                    raise ValueError('Diffusion baseline changed across paired measurements')
    cases = sorted(unique_cases.values(), key=lambda r: r['case_index'])
    seed = first['seeds']['data']
    baseline = {'correct_token_probability_mean': float(np.mean([r['classical_direct_target_probability'] for r in cases])),
        'correct_token_probability_sentence_cluster_95': cluster_interval(
            [r['classical_direct_target_probability'] for r in cases], [r['sentence_id'] for r in cases], seed),
        'top1_accuracy': float(np.mean([r['classical_direct_top1_correct'] for r in cases])),
        'unseen_context_cases': sum(not r['context_seen_in_training'] for r in cases),
        'unknown_target_cases': sum(not r['target_in_vocabulary'] for r in cases),
        'unknown_context_cases': sum(not r['context_in_vocabulary'] for r in cases),
        'smoothing': 0., 'used_to_compile_quantum_model': False}
    report = {**first, 'matched_results': matched, 'results': matched[list(matched)[-1]],
        'cases': [r for key in sorted(grouped) for r in grouped[key]],
        'evaluation_count': len(cases), 'evaluated_all_heldout_windows': len(cases) == first['heldout_count'],
        'independent_heldout_sentence_count': len({r['sentence_id'] for r in cases}),
        'classical_conditional_baseline': baseline, 'steps': sorted({step for _, step in grouped}),
        'configuration': {**first['configuration'], 'steps': sorted({step for _, step in grouped}),
            'case_start': 0, 'case_stop': len(cases)},
        'unique_evaluation_contexts': len({tuple(r['context']) if r['context_in_vocabulary'] else None for r in cases}),
        'assembled_from_measured_shards': [p['report'] for p in parts]}
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'report.json', report)
    return report
