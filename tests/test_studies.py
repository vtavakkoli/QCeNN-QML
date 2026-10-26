"""Scientific ordering, leakage prevention, hard resource budgets and recovery."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from qcenn_qml import mnist_study as mnist
from qcenn_qml.study_utils import cluster_interval, digest, independent_seed, initialize, run_tasks


def test_nested_memory_and_fresh_test_exclusions_are_fixed_before_selection():
    training = np.repeat(np.arange(10), 20)
    testing = np.repeat(np.arange(10), 30)
    excluded = {0, 1, 35, 299}
    splits = mnist.make_splits(training, testing, [42, 43, 44], 3, 4, excluded)
    seen = set(excluded)
    for row in splits.values():
        assert not set(row['training_nested_by_class']) & set(row['validation'])
        assert not seen & set(row['fresh_test'])
        seen.update(row['fresh_test'])
        for label in range(10):
            nested = np.asarray(row['training_nested_by_class'])[training[row['training_nested_by_class']] == label]
            assert len(nested) == 3
            assert set(nested[:1]) <= set(nested[:2]) <= set(nested[:3])
    assert splits == mnist.make_splits(training, testing, [42, 43, 44], 3, 4, excluded)
    with pytest.raises(ValueError, match='unused test'):
        mnist.make_splits(training, testing, [42], 3, 30, excluded)


def test_memory_address_order_is_seeded_and_exemplar_sets_remain_nested():
    labels = np.repeat(np.arange(10), 3)
    sets = [set(mnist.memory_indices(labels, memory, 42)) for memory in (1, 2, 3)]
    assert sets[0] < sets[1] < sets[2]
    assert len(mnist.memory_indices(labels, 3, 42)) == 30
    assert not np.array_equal(labels[mnist.memory_indices(labels, 3, 42)], labels)


def test_multiple_exemplars_allow_recurrence_to_change_exact_class_rankings():
    from qiskit.quantum_info import Statevector
    from qcenn_qml.quantum_memory import memory_circuit
    similarities = np.array([.51, .51, .95, .01])
    keys = np.column_stack([np.sqrt(similarities), np.sqrt(1 - similarities)])
    winners = []
    for rounds in (1, 4):
        circuit = memory_circuit([1, 0], keys, [0, 0, 1, 1], rounds=rounds, value_bits=1)
        read = [circuit.find_bit(item.qubits[0]).index for item in circuit.data if item.operation.name == 'measure']
        probabilities = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities(read)
        winners.append(int(np.argmax(probabilities[-2:])))
    assert winners == [0, 1]


@pytest.mark.parametrize('memory', [2, 3])
def test_measured_twenty_and_thirty_record_quantum_memories(memory):
    from qcenn_qml.coherent import decode_postselection, execute
    from qcenn_qml.quantum_memory import memory_circuit
    # Explicit orthogonal-vector primitive diagnostic, not real-MNIST evidence.
    keys = np.repeat(np.eye(16)[:10], memory, axis=0)
    labels = np.repeat(np.arange(10), memory)
    circuit = memory_circuit(np.eye(16)[7], keys, labels, rounds=4)
    counts, _, costs = execute([circuit], 256, 42)
    row = decode_postselection(counts[0], 4, 4, 10)
    assert len(keys) == 10 * memory and row['prediction'] == 7
    assert sum(row['histogram']) == row['histogram'][7] == row['accepted_shots']
    assert costs['total_shots'] == 256 and costs['cx_shot_weighted_total'] > 0


def test_equal_cnot_budget_is_a_hard_limit_with_explicit_rounding():
    for gates in (19, 90, 111):
        shots, budget = mnist.budget_shots(gates, 90, 1024, 'equal_cnot')
        assert gates * shots <= budget < gates * (shots + 1)
        assert budget == 90 * 1024
    assert mnist.budget_shots(999, 90, 1024, 'equal_shot') == (1024, None)
    with pytest.raises(ValueError, match='single execution'):
        mnist.budget_shots(100, 1, 2, 'equal_cnot')


def test_independent_measurement_seed_hierarchy_and_cluster_uncertainty():
    assert independent_seed(42, 101, 3) == independent_seed(42, 101, 3)
    assert len({independent_seed(42, seed, 3) for seed in (101, 202, 303)}) == 3
    assert cluster_interval([.3, .7], [42, 42]) is None
    assert cluster_interval([.2, .4, .8], [1, 2, 3]) == cluster_interval([.2, .4, .8], [1, 2, 3])


def candidate(config=None, *, failed=False, unstable=False, disagree_high=False):
    results = []
    for mode in ('equal_shot', 'equal_cnot'):
        for budget in (64, 128):
            for seed in (101, 202):
                predictions = [1 if (unstable and seed == 202) or (disagree_high and budget == 64) else 0]
                results.append({'mode': mode, 'base_shots': budget, 'measurement_seed': seed,
                    'predictions': predictions, 'accuracy': float(predictions == [0]),
                    'accepted_shots': 16, 'accepted_shots_min': 16, 'total_shots': budget,
                    'cx_shot_weighted_total': budget * 10})
    return {'configuration': config or mnist.configuration(), 'data_seed': 42,
        'fit_gate_passed': not failed, 'results': results}


def test_validation_selection_rejects_fit_failure_instability_and_false_plateau():
    assert mnist.choose(mnist.summarize([candidate(failed=True)], .95)) is None
    assert mnist.choose(mnist.summarize([candidate(unstable=True)], .95)) is None
    values = mnist.summarize([candidate(disagree_high=True)], .95)
    assert not any(row['eligible'] for row in values if row['base_shots'] == 64)
    assert mnist.choose(values)['base_shots'] == 128
    assert mnist.choose(mnist.summarize([candidate()], .95))['base_shots'] == 64


def test_gate_failure_prevents_validation_execution(monkeypatch, tmp_path):
    images = np.zeros((30, 4, 4), dtype=np.uint8)
    labels = np.repeat(np.arange(10), 3)
    prepared = tmp_path / 'data.npz'
    np.savez(prepared, training_images=images, training_labels=labels,
        validation_images=images[:10], validation_labels=np.arange(10))
    monkeypatch.setattr(mnist, '_native', lambda *args: None)
    monkeypatch.setattr(mnist, '_sample', lambda *args, **kwargs: {'prediction': -1})
    monkeypatch.setattr(mnist, '_measure_grid', lambda *args, **kwargs: pytest.fail('Validation ran after failed fit'))
    result = mnist.evaluate_task({'output_dir': str(tmp_path), 'prepared_data': str(prepared),
        'configuration': mnist.configuration(side=4), 'budget_anchor_configuration': mnist.configuration(1, side=4),
        'data_seed': 42, 'gate_shot_budgets': [16], 'prepared_data_sha256': 'fixture'})
    assert result['status'] == 'BLOCKED_FIT_GATE'
    assert not result['validation_evaluated'] and not result['test_evaluated']


def test_actual_quantum_grid_preserves_counts_and_both_resource_comparisons(tmp_path):
    keys = np.eye(4)[:2]
    config = mnist.configuration(rounds=2, side=4)
    task = {'data_seed': 42, 'shot_budgets': [32, 64], 'measurement_seeds': [101, 202],
        'budget_anchor_configuration': mnist.configuration(rounds=1, side=4)}
    results = mnist._measure_grid(keys[:1], np.array([0]), keys, np.array([0, 1]),
        config, task, tmp_path, anchor_data=(keys[:1], keys, np.array([0, 1])))
    assert len(results) == 8
    assert {row['mode'] for row in results} == {'equal_shot', 'equal_cnot'}
    assert all(row['accuracy'] == 1. for row in results)
    for path in tmp_path.glob('equal_*.json'):
        row = json.loads(path.read_text())
        assert sum(row['counts'].values()) == row['total_shots']
        assert row['prediction'] == 0
        if row['cx_budget'] is not None:
            assert row['resources']['cx_shot_weighted_total'] <= row['cx_budget']


def test_every_validation_stage_precedes_freeze_and_test(monkeypatch, tmp_path):
    labels = np.repeat(np.arange(10), 10)
    images = np.arange(100 * 16, dtype=np.uint8).reshape(100, 4, 4)
    monkeypatch.setattr(mnist, 'load_dataset', lambda *args, **kwargs: (images, labels, images, labels))
    order = []
    def workers(function, tasks, **kwargs):
        assert function.endswith(':evaluate_task')
        order.append(Path(kwargs['directory']).name)
        output = []
        for task in tasks:
            assert 'test_indices' not in task
            row = candidate(task['configuration'])
            row['data_seed'] = task['data_seed']
            # Distinct measured validation fixture scores select larger memories.
            for value in row['results']:
                value['accuracy'] = .1 * task['configuration']['memory']
            output.append(row)
        return output
    monkeypatch.setattr(mnist, 'run_tasks', workers)
    def frozen_test(freeze_file, output_dir, **kwargs):
        freeze = json.loads(Path(freeze_file).read_text())
        assert freeze['sha256'] == digest({k: v for k, v in freeze.items() if k != 'sha256'})
        assert order == ['convergence', 'memory-2', 'memory-3', 'encoding-damping']
        assert freeze['selected']['configuration']['memory'] == 3
        assert not freeze['selection_used_test_labels']
        order.append('test')
        return {'test_results': [], 'summary': [], 'paired_comparisons': []}
    monkeypatch.setattr(mnist, 'test_frozen', frozen_test)
    result = mnist.study(dataset='digits-proxy', data_seeds=[42], measurement_seeds=[101, 202],
        shot_budgets=[64, 128], gate_shot_budgets=[128], validation_per_class=1,
        test_per_class=1, output_dir=str(tmp_path / 'study'))
    assert order[-1] == 'test' and result['status'] == 'DONE'


def test_altered_freeze_is_rejected_before_test_loading(tmp_path):
    freeze = tmp_path / 'freeze.json'
    freeze.write_text(json.dumps({'sha256': 'wrong', 'selection_used_test_labels': False}))
    with pytest.raises(ValueError, match='altered freeze'):
        mnist.test_frozen(freeze, tmp_path / 'test')


def test_frozen_test_worker_reads_reserved_images_without_reselection(monkeypatch, tmp_path):
    # Orthogonal image fixture verifies the actual quantum test path, not MNIST quality.
    labels = np.repeat(np.arange(10), 3)
    images = np.zeros((30, 4, 4), dtype=np.uint8)
    for index, label in enumerate(labels):
        images[index].flat[label] = 255
    prepared = tmp_path / 'prepared.npz'
    np.savez(prepared, training_images=images, training_labels=labels)
    monkeypatch.setattr(mnist, 'load_dataset', lambda *args, **kwargs: (images, labels, images, labels))
    monkeypatch.setattr(mnist, 'choose', lambda *args: pytest.fail('Frozen testing reselected a configuration'))
    ids = [0, 21]
    result = mnist.test_task({'output_dir': str(tmp_path), 'prepared_data': str(prepared),
        'configuration': mnist.configuration(rounds=2, memory=2, side=4),
        'budget_anchor_configuration': mnist.configuration(rounds=1, side=4),
        'data_seed': 42, 'dataset': 'digits-proxy', 'data_root': str(tmp_path),
        'test_indices': ids, 'selected_base_shots': 256,
        'test_measurement_seed': 99991, 'freeze_sha256': 'unit-test-freeze'})
    assert result['test_indices'] == ids and result['test_evaluated']
    for rows in result['results'].values():
        assert {row['mode'] for row in rows} == {'equal_shot', 'equal_cnot'}
        assert all(row['truth'] == [0, 7] and row['accuracy'] == 1. for row in rows)


def test_scheduler_runs_in_parallel_reuses_completed_work_and_detects_tampering(monkeypatch, tmp_path):
    worker = tmp_path / 'fixture_worker.py'
    worker.write_text('import time\ndef work(task):\n start=time.time(); time.sleep(.3)\n return {"start":start,"end":time.time(),"value":task["value"]}\n')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + ':' + str(Path(mnist.__file__).parents[1]))
    tasks = [{'id': 'a', 'value': 1}, {'id': 'b', 'value': 2}]
    result = run_tasks('fixture_worker:work', tasks, jobs=2, cpu_budget=2, directory=tmp_path / 'tasks')
    assert max(row['start'] for row in result) < min(row['end'] for row in result)
    assert run_tasks('fixture_worker:work', tasks, jobs=2, cpu_budget=2, directory=tmp_path / 'tasks') == result
    (tmp_path / 'tasks/a/result.json').write_text('{}')
    with pytest.raises(ValueError, match='Completed result changed'):
        run_tasks('fixture_worker:work', tasks, jobs=2, cpu_budget=2, directory=tmp_path / 'tasks')


def test_resume_rejects_different_configuration_source_or_runtime(tmp_path):
    source = {'source_sha256': {'module': 'a'}, 'packages': {'qiskit': 'x'}}
    initialize(tmp_path, {'setting': 1}, source)
    initialize(tmp_path, {'setting': 1}, source, resume=True)
    with pytest.raises(FileExistsError):
        initialize(tmp_path, {'setting': 2}, source, resume=True)
    with pytest.raises(FileExistsError):
        initialize(tmp_path, {'setting': 1}, {**source, 'packages': {'qiskit': 'y'}}, resume=True)
