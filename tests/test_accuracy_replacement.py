"""Measured banking, validation isolation, coherent composition and GPU API regressions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qcenn_qml import mnist_study as mnist
from qcenn_qml.mnist_memory import NativeMemory, allocate_shots, decode_banks, preprocess


def test_fixed_centering_removes_translation_and_handles_blank_images():
    images = np.zeros((2, 12, 12), np.uint8)
    images[0, 2:5, 2:4] = 255
    images[1, 6:9, 6:8] = 255
    centered = preprocess(images, 'center')
    assert np.array_equal(centered[0], centered[1])
    assert preprocess(np.zeros((1, 8, 8), np.uint8), 'deskew').sum() == 0
    assert np.array_equal(preprocess(images), images)
    with pytest.raises(ValueError):
        preprocess(images, 'fitted')


def test_larger_nested_memories_reserve_validation_before_any_selection():
    labels = np.repeat(np.arange(10), 30)
    splits = mnist.make_splits(labels, labels, [42], 5, 2, [], memory_per_class=16)['42']
    assert len(splits['training_nested_by_class']) == 160
    assert len(splits['validation']) == 50
    assert not set(splits['training_nested_by_class']) & set(splits['validation'])
    stored = labels[splits['training_nested_by_class']]
    sets = [set(mnist.memory_indices(stored, size, 42)) for size in (1, 2, 3, 4, 8, 16)]
    assert all(a < b for a, b in zip(sets, sets[1:]))
    with pytest.raises(ValueError, match='Insufficient raw'):
        mnist.memory_indices(stored, 17, 42)


def test_nearest_readout_uses_measured_record_counts_instead_of_class_sums():
    # Two weak records of class 0 versus one stronger record of class 1.
    counts = {format(16, '07b'): 30, format(32+16, '07b'): 30,
        format(64+16+1, '07b'): 50, '0000000': 18}
    native = NativeMemory([None], [[0, 0, 1]], 'nearest_record')
    row = decode_banks([counts], native, 1)
    assert row['histogram'][:2] == [60, 50]
    assert row['prediction'] == 1
    assert row['measured_record_peaks_by_class'][:2] == [30, 50]
    assert row['total_shots'] == 128


def test_actual_forty_record_banks_keep_measured_counts_and_hard_budget(tmp_path):
    keys = np.repeat(np.eye(16)[:10], 4, axis=0)
    labels = np.repeat(np.arange(10), 4)
    config = mnist.configuration(rounds=2, memory=4, side=4, readout='nearest_record')
    native = mnist._native(np.eye(16)[7], keys, labels, config, 42)
    assert isinstance(native, NativeMemory) and len(native.circuits) == 2
    assert all(np.bincount(y, minlength=10).tolist() == [2]*10 for y in native.labels)
    shots, budget = allocate_shots(native, 9000, 512, 'equal_cnot')
    assert shots % 2 == 0
    path = tmp_path / 'counts.json'
    row = mnist._sample(native, shots, 42, path, 2, 7, budget)
    assert row['prediction'] == 7 and row['total_shots'] == shots
    assert sum(sum(bank.values()) for bank in row['bank_counts']) == shots
    assert row['resources']['cx_shot_weighted_total'] <= budget
    assert mnist._sample(native, shots, 42, path, 2, 7, budget) == row
    row['bank_counts'][0][next(iter(row['bank_counts'][0]))] += 1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        mnist._sample(native, shots, 42, path, 2, 7, budget)


def test_ninety_five_percent_agreement_accepts_only_numeric_roundoff():
    from test_studies import candidate
    row = candidate()
    row['results'] = []
    for budget in (64, 128):
        for seed in (101, 202, 303):
            predictions = [0]*20
            if seed == 303:
                predictions[:3] = [1]*3
            row['results'].append({'mode': 'equal_shot', 'base_shots': budget,
                'measurement_seed': seed, 'predictions': predictions, 'accuracy': 1.,
                'accepted_shots': 20, 'accepted_shots_min': 1, 'total_shots': 128,
                'cx_shot_weighted_total': 1280})
    assert all(r['eligible'] for r in mnist.summarize([row], .95))
    for r in row['results']:
        if r['measurement_seed'] == 303:
            r['predictions'][3] = 1
    assert not any(r['eligible'] for r in mnist.summarize([row], .95))


def test_shared_native_cache_reuses_compilation_and_rejects_corruption(monkeypatch, tmp_path):
    prepared = tmp_path / 'prepared.npz'; prepared.write_bytes(b'identity-only fixture')
    task = {'measurement_shards': True, 'prepared_data': str(prepared),
        'prepared_data_sha256': hashlib.sha256(prepared.read_bytes()).hexdigest(), 'source_sha256': {'fixture': '1'}}
    config = mnist.configuration(rounds=2, readout='nearest_record')
    query, keys, labels = [1., 0.], np.eye(2), [0, 1]
    native = mnist._native_for_task(query, keys, labels, config, 42, task, 0)
    monkeypatch.setattr(mnist, '_native', lambda *args: pytest.fail('Identical native circuit recompiled'))
    cached = mnist._native_for_task(query, keys, labels, config, 42, task, 0)
    from qcenn_qml.mnist_memory import native_hash
    assert native_hash(cached) == native_hash(native)
    path = next((tmp_path / 'native-cache').glob('*/native.qpy'))
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='archive changed'):
        mnist._native_for_task(query, keys, labels, config, 42, task, 0)


def test_independent_measurement_shards_merge_without_duplicate_images(tmp_path):
    labels = np.repeat(np.arange(10), 3)
    images = np.zeros((30, 4, 4), np.uint8)
    for i, label in enumerate(labels):
        images[i].flat[label] = 255
    prepared = tmp_path / 'prepared.npz'
    np.savez(prepared, training_images=images, training_labels=labels,
        validation_images=images[[0, 21]], validation_labels=labels[[0, 21]])
    task = {'id': 'measured-shards', 'configuration': mnist.configuration(rounds=2, side=4),
        'prepared_data': str(prepared), 'prepared_data_sha256': hashlib.sha256(prepared.read_bytes()).hexdigest(),
        'data_seed': 42, 'gate_shot_budgets': [256], 'shot_budgets': [64, 128],
        'measurement_seeds': [101, 202], 'budget_anchor_configuration': mnist.configuration(1, side=4),
        'measurement_shards': True, 'source_sha256': mnist.provenance()['source_sha256']}
    rows = mnist.evaluate_sharded([task], jobs=2, cpu_budget=2, directory=tmp_path / 'tasks', batch_size=1)
    assert rows[0]['fit_gate_passed']
    assert len(rows[0]['results']) == 8
    assert all(r['truth'] == [0, 7] and r['accuracy'] == 1. for r in rows[0]['results'])
    assert len(list((tmp_path / 'native-cache').glob('*/receipt.json'))) == 4


def test_worker_failure_updates_mnist_report_instead_of_leaving_validating(monkeypatch, tmp_path):
    labels = np.repeat(np.arange(10), 10)
    images = np.zeros((100, 4, 4), np.uint8)
    monkeypatch.setattr(mnist, 'load_dataset', lambda *a, **k: (images, labels, images, labels))
    def fail(*args, **kwargs):
        raise RuntimeError('bounded task timed out')
    monkeypatch.setattr(mnist, 'run_tasks', fail)
    root = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='timed out'):
        mnist.study(dataset='digits-proxy', data_seeds=[42], measurement_seeds=[101, 202],
            shot_budgets=[64, 128], gate_shot_budgets=[128], test_per_class=1, output_dir=root)
    report = json.loads((root / 'report.json').read_text())
    assert report['status'] == 'FAILED' and report['failed_stage'] == 'convergence'
    assert not report['test_evaluated'] and not (root / 'frozen-configuration.json').exists()


def test_new_forward_builder_cannot_use_classical_transformer_predictions(monkeypatch):
    from qcenn_qml import transformer_replacement as model
    monkeypatch.setattr(model, 'reference_forward', lambda *args: pytest.fail('CPU forward used in quantum construction'))
    circuit = model.forward_circuit([0, 1], layers=2)
    assert circuit.metadata['layers'] == 2
    assert not circuit.metadata['intermediate_measurements']
    assert not circuit.metadata['actual_smol_weights']
    operations = [item.operation.name for item in circuit.data]
    first_measure = operations.index('measure')
    assert set(operations[first_measure:]) == {'measure'}
    assert 'reset' not in operations


def test_two_complete_quantum_blocks_embedding_and_head_match_reference():
    from qcenn_qml.coherent import execute
    from qcenn_qml.transformer_replacement import forward_circuit, pack_tokens, reference_forward
    inputs = [2, 3]
    circuit = forward_circuit(inputs, layers=2)
    counts, _, resources = execute([circuit], 8, 42)
    prediction = pack_tokens(reference_forward(inputs, 2))
    expected = pack_tokens(inputs) | (prediction << 4)
    assert counts == [{format(expected, f'0{circuit.num_qubits}b'): 8}]
    assert resources['qubits_max'] == 261


def test_diffusion_is_rejected_by_active_runner_before_any_output(tmp_path):
    from qcenn_qml.research_runner import run
    with pytest.raises(ValueError, match='retired'):
        run(tmp_path / 'retired', only='diffusion')
    assert not (tmp_path / 'retired').exists()


def test_gpu_transport_uses_supported_run_time_selection(monkeypatch, tmp_path):
    from qiskit import QuantumCircuit, qpy
    import qiskit_aer
    from qcenn_qml import gpu_worker
    circuit = QuantumCircuit(2, 2)
    circuit.h(0); circuit.cx(0, 1); circuit.measure(range(2), range(2))
    path = tmp_path / 'bell.qpy'
    with path.open('wb') as stream:
        qpy.dump(circuit, stream)
    original = qiskit_aer.AerSimulator
    def cpu_backend(**options):
        assert 'target_gpus' not in options  # The Aer 0.18.2 constructor rejects it.
        options['device'] = 'CPU'
        return original(**options)
    monkeypatch.setattr(qiskit_aer, 'AerSimulator', cpu_backend)
    monkeypatch.setattr(gpu_worker, 'packages', lambda: {'test': 'cpu API regression; not GPU qualification'})
    request = {'expected_packages': gpu_worker.packages(), 'qpy': str(path),
        'qpy_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'readout_error': 0.,
        'method': 'statevector', 'gpu_id': 0, 'memory_limit_mb': 64, 'shots': 128,
        'seed': 42, 'memory': False, 'result': str(tmp_path / 'answer.json')}
    request_file = tmp_path / 'request.json'; request_file.write_text(json.dumps(request))
    gpu_worker.execute(request_file)
    answer = json.loads(Path(request['result']).read_text())
    assert set(answer['counts'][0]) == {'00', '11'} and sum(answer['counts'][0].values()) == 128
