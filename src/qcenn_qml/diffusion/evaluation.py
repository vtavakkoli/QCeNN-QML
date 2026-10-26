"""Batched held-out denoising with paired gates, OOV accounting and checkpoints."""
import hashlib
from collections import Counter
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from ..coherent import CORE_DISCLOSURE, _compile_components, decode_postselection, execute
from ..corpus_memory import denoising_memory_circuit
from ..recurrent_mnist import provenance, write_json
from ..study_utils import circuit_digest, cluster_interval, independent_seed, initialize


def noisy_target_probability(distribution, target, bits, error):
    return float(sum(probability * error**((code ^ target).bit_count()) *
        (1 - error)**(bits - (code ^ target).bit_count())
        for code, probability in enumerate(distribution)))


def execute_benchmark(shots=1024, beta=1., eta=.65, steps=(0, 1, 3), examples=8,
        seed=42, readout_error=.05, corpus_kind='text', corpus_file=None, amplify=1,
        output_dir='results/diffusion-matched', measurement_seed=None, readout_errors=None,
        max_train_records=256, max_vocabulary=64, batch_size=4, resume=False):
    configuration = dict(locals())
    from qiskit import qpy
    from .matched import corpus
    for key in ('resume', 'output_dir'):
        configuration.pop(key)
    if corpus_file:
        configuration['corpus_file_sha256'] = hashlib.sha256(Path(corpus_file).read_bytes()).hexdigest()
    errors = tuple(readout_errors) if readout_errors is not None else ((readout_error,) if readout_error else ())
    if (shots < 1 or examples < 0 or batch_size < 1 or not steps or
            any(not 0 < error < .5 for error in errors) or len(set(errors)) != len(errors) or
            not 0 <= beta <= 1 or not 0 <= eta <= 1 or not 0 <= amplify <= 3 or
            len(set(steps)) != len(steps) or any(not isinstance(step, int) or not 0 <= step <= 8 for step in steps)):
        raise ValueError('Invalid diffusion experiment budget/channel')
    measurement_seed = seed if measurement_seed is None else measurement_seed
    directory = Path(output_dir)
    manifest = initialize(directory, configuration, provenance(), resume)
    vocab, mask, train, test, split = corpus(corpus_kind, corpus_file, seed,
        max_train_records=max_train_records, max_vocabulary=max_vocabulary)
    order = np.random.default_rng(seed + 100).permutation(len(test))
    selected = order[:examples] if examples else order
    cases = [test[int(i)] for i in selected]
    if not cases:
        raise ValueError('No held-out evaluation cases')
    details = split.get('heldout_windows', [{'sentence_id': i, 'target_in_vocabulary': True,
        'context_in_vocabulary': True, 'words': [vocab[x] for x in window]} for i, window in enumerate(test)])
    case_details = [details[int(i)] for i in selected]
    bits = max(1, (len(vocab) - 1).bit_length())
    contexts = [int(a) + (int(c) << bits) for a, _, c in train]
    values = [int(b) for _, b, _ in train]
    unused = next(code for code in range(1 << (2 * bits)) if code not in set(contexts))
    queries = [a + (c << bits) if detail['context_in_vocabulary'] else unused
        for (a, _, c), detail in zip(cases, case_details, strict=True)]
    write_json(directory / 'splits.json', {'vocabulary': vocab, 'training_records': train,
        'heldout_records': test, 'evaluated_records': cases,
        'evaluated_window_indices': selected.tolist(), 'evaluated_details': case_details, **split})
    np.savez_compressed(directory / 'raw-memory.npz', contexts=contexts, values=values)
    # Reference arrays are never passed to the quantum circuit constructor.
    table_counts = Counter((a, c, b) for a, b, c in train)
    probabilities = []
    for (a, _, c), detail in zip(cases, case_details, strict=True):
        q = np.array([table_counts[a, c, b] for b in range(len(vocab))], float)
        if not detail['context_in_vocabulary']:
            q[:] = 0
        probabilities.append(q / q.sum() if q.sum() else q)
    baseline_correct = [float(q[b]) if detail['target_in_vocabulary'] else 0.
        for q, (_, b, _), detail in zip(probabilities, cases, case_details, strict=True)]
    groups = [detail['sentence_id'] for detail in case_details]
    baseline = {'correct_token_probability_mean': float(np.mean(baseline_correct)),
        'correct_token_probability_sentence_cluster_95': cluster_interval(baseline_correct, groups, seed),
        'top1_accuracy': float(np.mean([bool(q.sum() and detail['target_in_vocabulary'] and int(q.argmax()) == b)
            for q, (_, b, _), detail in zip(probabilities, cases, case_details, strict=True)])),
        'unseen_context_cases': sum(not q.sum() for q in probabilities),
        'unknown_target_cases': sum(not detail['target_in_vocabulary'] for detail in case_details),
        'unknown_context_cases': sum(not detail['context_in_vocabulary'] for detail in case_details),
        'smoothing': 0., 'used_to_compile_quantum_model': False}
    modes = [('noiseless', 0.)] + [('noisy' if len(errors) == 1 else f'noisy_{error:g}', error) for error in errors]
    all_results = {mode: {} for mode, _ in modes}
    all_rows = []
    checkpoints = directory / 'batches'
    checkpoints.mkdir(exist_ok=True)
    for step in steps:
        observations = {mode: [] for mode, _ in modes}
        costs = {mode: [] for mode, _ in modes}
        fingerprints = []
        for start in range(0, len(cases), batch_size):
            subset = cases[start:start + batch_size]
            timer = perf_counter()
            circuits = [denoising_memory_circuit(contexts, values, queries[start + offset],
                context_bits=2 * bits, value_bits=bits, mask=mask, original=b, steps=step,
                beta=beta, eta=eta, amplify=amplify) for offset, (_, b, _) in enumerate(subset)]
            construction_seconds = perf_counter() - timer
            timer = perf_counter()
            native = [_compile_components(circuit, independent_seed(seed, step, start, 71)) for circuit in circuits]
            compile_seconds = perf_counter() - timer
            identity = [circuit_digest(circuit) for circuit in native]
            fingerprints.extend(identity)
            if start == 0:
                with (directory / f'steps-{step}-native.qpy').open('wb') as stream:
                    qpy.dump(native[0], stream)
            for mode, error in modes:
                path = checkpoints / f'steps-{step}-start-{start:06d}-{mode}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved['native_circuit_sha256'] != identity:
                        raise ValueError('Compiled circuit changed relative to saved counts')
                    samples, cost = saved['counts'], saved['resources']
                else:
                    samples, _, cost = execute(native, shots,
                        independent_seed(measurement_seed, step, start, 73), error, precompiled=True)
                    write_json(path, {'native_circuit_sha256': identity, 'counts': samples, 'resources': cost,
                        'classical_construction_seconds': construction_seconds,
                        'classical_compilation_seconds': compile_seconds})
                costs[mode].append(cost)
                for offset, sample in enumerate(samples):
                    case = start + offset
                    a, b, c = cases[case]
                    detail, q = case_details[case], probabilities[case]
                    decoded = decode_postselection(sample, bits, 1, len(vocab))
                    distribution = np.asarray(decoded['histogram'], float) / max(1, decoded['accepted_shots'])
                    ideal = np.zeros(1 << bits)
                    if q.sum():
                        retained = (1 - eta)**step
                        ideal[mask] += retained * beta
                        ideal[b] += retained * (1 - beta)
                        ideal[:len(vocab)] += (1 - retained) * q
                    known = detail['target_in_vocabulary']
                    observed = float(distribution[b]) if known else 0.
                    ideal_target = noisy_target_probability(ideal, b, bits, error) if known else 0.
                    row = {'mode': mode, 'steps': step, 'case_index': case,
                        'heldout_window_index': int(selected[case]), **detail,
                        'context': [detail['words'][0], detail['words'][2]], 'truth': detail['words'][1],
                        **decoded, 'quantum_distribution': distribution.tolist(),
                        'quantum_target_frequency': observed, 'analytic_ideal_target_probability': float(ideal[b]) if known else 0.,
                        'matched_classical_channel_target_probability': ideal_target,
                        'top1_correct': bool(known and decoded['prediction'] == b),
                        'native_circuit_sha256': identity[offset]}
                    observations[mode].append(row)
                    all_rows.append(row)
                write_json(directory / 'progress.json', {'status': 'RUNNING', 'step': step,
                    'mode': mode, 'completed_cases_at_step': start + len(subset), 'total_cases': len(cases)})
            print(f'diffusion step={step}: paired cases {start + len(subset)}/{len(cases)}', flush=True)
        for mode, error in modes:
            rows, batches = observations[mode], costs[mode]
            observed = [row['quantum_target_frequency'] for row in rows]
            correct = [float(row['top1_correct']) for row in rows]
            accepted = sum(row['accepted_shots'] for row in rows)
            resources = {'total_shots': sum(row['total_shots'] for row in batches),
                'compilation_seconds': sum(row['compilation_seconds'] for row in batches),
                'cx_shot_weighted_total': sum(row['cx_shot_weighted_total'] for row in batches),
                'sampling_seconds': sum(row['sampling_seconds'] for row in batches),
                'qubits_max': max(row['qubits_max'] for row in batches),
                'depth_max': max(row['depth_max'] for row in batches), 'batches': batches}
            all_results[mode][str(step)] = {'measured_correct_token_probability': float(np.mean(observed)),
                'correct_token_probability_sentence_cluster_95': cluster_interval(observed, groups, seed),
                'independent_ideal_probability_mean': float(np.mean([row['analytic_ideal_target_probability'] for row in rows])),
                'classical_matched_replacement_probability_mean': float(np.mean([row['matched_classical_channel_target_probability'] for row in rows])),
                'sequence_top1_accuracy': float(np.mean(correct)),
                'top1_sentence_cluster_95': cluster_interval(correct, groups, seed),
                'samples': len(cases), 'total_shots': shots * len(cases), 'accepted_shots': accepted,
                'accepted_shot_fraction': accepted / (shots * len(cases)),
                'invalid_value_shots': sum(row['invalid_value_shots'] for row in rows),
                'compiled_max_qubits': resources['qubits_max'], 'compiled_depth_max': resources['depth_max'],
                'native_circuit_sha256': fingerprints, 'readout_error': error, 'resources': resources}
            print(f'diffusion steps={step} {mode}: correct-token={np.mean(observed):.4f}, accepted={accepted}', flush=True)
    selected_mode = modes[-1][0]
    report = {'name': 'Raw-corpus quantum-memory masked denoising', **CORE_DISCLOSURE,
        'experiment': 'matched_heldout_text_diffusion', 'configuration': configuration,
        'status': 'nonparametric_quantum_memory_not_a_trained_diffusion_LLM',
        'backend': 'Qiskit Aer MPS, truncation 1e-20', 'vocabulary': vocab, 'mask_token': vocab[mask],
        'train_count': len(train), 'heldout_count': len(test), 'evaluation_count': len(cases),
        'evaluated_all_heldout_windows': len(cases) == len(test),
        'unique_evaluation_contexts': len(set(queries)), 'independent_heldout_sentence_count': len(set(groups)),
        'forward_mask_probability_beta': beta, 'reverse_replacement_probability_eta': eta,
        'readout_bitflip_probability': list(errors) if len(errors) > 1 else (errors[0] if errors else 0.),
        'shots_per_case': shots, 'steps': list(steps), 'seeds': {'data': seed, 'simulator': measurement_seed},
        'classical_conditional_baseline': baseline, 'same_native_circuits_and_cases_for_noise_comparison': True,
        'matched_results': all_results, 'results': all_results[selected_mode], 'cases': all_rows,
        'classical_conditional_tables_in_quantum_gates': False,
        'actual_smol135_weights_on_quantum_device': False, 'source': manifest['source'],
        'unknown_targets_receive_no_accuracy_credit': True,
        'unknown_contexts_abstain_without_aliasing_to_known_contexts': True,
        'uncertainty_scope': 'Sentence-cluster bootstrap; quantum shots are not independent language examples; single-cluster intervals are null',
        'limitations': ['Bounded raw-record memory, not trained neural diffusion or unrestricted language modeling',
            'Training-only bounded vocabulary; unknown targets, unseen contexts and invalid noisy codes retain their error mass',
            'Only declared output readout flips are simulated; no device gate noise or QPU claim',
            'All measured cases are checkpointed; QPY files contain the first native circuit per step']}
    write_json(directory / 'report.json', report)
    write_json(directory / 'progress.json', {'status': 'complete', 'execution_state': 'DONE', 'report': 'report.json'})
    lines = ['# Measured held-out diffusion', '', f'Evaluated {len(cases)}/{len(test)} held-out windows from {len(set(groups))} sentences.', '',
        '| Mode | Steps | Measured target probability | Matched classical channel | Accepted shots |',
        '|---|---:|---:|---:|---:|']
    for mode, values_at_steps in all_results.items():
        for step, value in values_at_steps.items():
            lines.append(f'| {mode} | {step} | {value["measured_correct_token_probability"]:.4f} | '
                f'{value["classical_matched_replacement_probability_mean"]:.4f} | {value["accepted_shot_fraction"]:.4f} |')
    lines += ['', 'Raw quantum memory and numerical gate simulation; no neural training, physical execution or advantage claim.', '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf8')
    return report
