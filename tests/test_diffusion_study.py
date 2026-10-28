"""Held-out lexical leakage, paired gate identity and checkpoint recovery."""
import json
from pathlib import Path

import numpy as np
import pytest

from qcenn_qml.diffusion.matched import corpus, execute_benchmark
from qcenn_qml.diffusion.evaluation import noisy_target_probability


def text_file(tmp_path, lines):
    path = tmp_path / 'corpus.txt'
    path.write_text('\n'.join(lines))
    return path


def test_duplicate_sentences_stay_together_and_training_cap_ignores_heldout(tmp_path):
    units = ['the cat sat near the door.', 'the dog ran near the tree.',
        'the bird sat near the room.', 'the child ran near the park.',
        'the cat ran near the gate.', 'the dog sat near the wall.']
    path = text_file(tmp_path, units * 3)
    vocabulary, _, training, heldout, split = corpus(corpus_file=path, seed=4,
        max_train_records=5, max_vocabulary=8)
    assert len(training) == 5 and len(vocabulary) == 8
    train_groups = {index % len(units) for index in split['train_sentence_ids']}
    test_groups = {index % len(units) for index in split['heldout_sentence_ids']}
    assert not train_groups & test_groups
    assert split['training_windows_before_memory_cap'] > len(training)
    assert split['memory_cap_selected_without_heldout_labels']
    assert corpus(corpus_file=path, seed=4, max_train_records=5, max_vocabulary=8) == (
        vocabulary, 1, training, heldout, split)


def test_unknown_target_never_gets_credit_for_predicting_unknown_code(tmp_path):
    path = text_file(tmp_path, [f'left {word} right.' for word in (
        'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel')])
    report = execute_benchmark(corpus_file=str(path), steps=[1], examples=0,
        shots=64, eta=1., readout_error=.2, output_dir=str(tmp_path / 'run'))
    assert report['evaluated_all_heldout_windows']
    assert report['classical_conditional_baseline']['unknown_target_cases'] == report['evaluation_count']
    assert report['classical_conditional_baseline']['top1_accuracy'] == 0
    assert all(not row['target_in_vocabulary'] and row['quantum_target_frequency'] == 0
        and not row['top1_correct'] for row in report['cases'])


def test_unknown_context_abstains_even_when_unk_code_has_training_matches(tmp_path):
    path = text_file(tmp_path, [f'{word} middle right.' for word in (
        'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel')])
    report = execute_benchmark(corpus_file=str(path), steps=[1], examples=0,
        shots=32, eta=1., readout_error=.1, max_vocabulary=4, output_dir=str(tmp_path / 'run'))
    assert all(row['accepted_shots'] == 0 and row['prediction'] == -1 for row in report['cases'])
    assert report['classical_conditional_baseline']['unseen_context_cases'] == report['evaluation_count']


def test_multiple_noise_levels_share_exact_native_gates_and_resume_without_sampling(monkeypatch, tmp_path):
    from qcenn_qml.diffusion import evaluation
    options = dict(corpus_kind='toy', steps=[0, 1], shots=32, examples=2,
        readout_errors=[.01, .1], measurement_seed=101, output_dir=str(tmp_path / 'run'))
    result = execute_benchmark(**options)
    assert set(result['matched_results']) == {'noiseless', 'noisy_0.01', 'noisy_0.1'}
    for step in ['0', '1']:
        rows = [values[step] for values in result['matched_results'].values()]
        assert all(row['native_circuit_sha256'] == rows[0]['native_circuit_sha256'] for row in rows)
        assert len({row['resources']['cx_shot_weighted_total'] for row in rows}) == 1
    monkeypatch.setattr(evaluation, 'execute', lambda *args, **kwargs: pytest.fail('Completed measurements were repeated'))
    resumed = execute_benchmark(**options, resume=True)
    assert resumed['cases'] == result['cases']


def test_classical_noise_reference_retains_invalid_code_probability():
    probabilities = np.array([.7, .3, 0., 0.])
    noisy = [noisy_target_probability(probabilities, code, 2, .1) for code in range(4)]
    assert sum(noisy) == pytest.approx(1.)
    assert noisy[2] > 0 and noisy[3] > 0
    assert noisy_target_probability(probabilities, 0, 2, 0.) == .7


def test_broader_corpus_snapshot_changes_invalidate_resume(tmp_path):
    path = text_file(tmp_path, ['left alpha right.', 'left bravo right.',
        'left charlie right.', 'left delta right.'])
    options = dict(corpus_file=str(path), steps=[0], shots=8, examples=1,
        readout_error=0, output_dir=str(tmp_path / 'run'))
    execute_benchmark(**options)
    path.write_text(path.read_text() + '\nnew changed sentence.')
    with pytest.raises(FileExistsError):
        execute_benchmark(**options, resume=True)


def test_sharded_study_reassembles_each_case_once_and_keeps_noise_pairs(tmp_path):
    from qcenn_qml.diffusion.study import study, DEFAULT_CORPORA
    report = study(corpus_files=[DEFAULT_CORPORA[0]], data_seeds=[42], measurement_seeds=[101],
        shots=32, examples=3, max_train_records=8, max_vocabulary=8, steps=[0, 1],
        amplify=0, jobs=2, cpu_budget=2, output_dir=tmp_path / 'study')
    assert report['status'] == 'DONE' and len(report['runs']) == 1
    run = json.loads(Path(report['runs'][0]['report']).read_text())
    assert run['evaluation_count'] == 3
    assert len(run['cases']) == 3 * 2 * 3  # Cases x steps x matched noise modes.
    assert len({(r['mode'], r['steps'], r['heldout_window_index']) for r in run['cases']}) == 18
    for mode in run['matched_results'].values():
        for row in mode.values():
            assert row['total_shots'] == 3 * 32 == row['resources']['total_shots']
    unique = {r['heldout_window_index']: r for r in run['cases']}
    assert run['classical_conditional_baseline']['correct_token_probability_mean'] == pytest.approx(
        np.mean([r['classical_direct_target_probability'] for r in unique.values()]))
