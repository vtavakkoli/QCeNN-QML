"""Compact exports preserve model evidence, bound diagnostics and verify receipts."""
import gzip
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest

from qcenn_qml.analysis_archive import build, LOG_BUDGET_BYTES, LOG_TAIL_BYTES


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / 'run with spaces'
    experiment = root / 'experiments/mnist-study'
    write(root / 'runtime.json', {'source': {'git_commit': 'fixture'}, 'cpu_budget': 2})
    write(root / 'study-manifest.json', {'configuration': {'smoke_only': True}})
    write(experiment / 'report.json', {'status': 'BLOCKED_FIT_OR_MEASUREMENT_STABILITY',
        'test_evaluated': False, 'configuration': {'weights_integer_numerators': [[1, -1]]}})
    write(experiment / 'splits.json', {'train': [0, 1], 'validation': [2], 'test': [99]})
    write(experiment / 'frozen-configuration.json', {'selection_used_test_labels': False})
    write(experiment / 'tasks/image-0/result.json', {'counts': {'10000': 100},
        'prediction': 0, 'truth': 0, 'accepted_shots': 100, 'measurement_seed': 101,
        'resources': {'cx_shot_weighted_total': 12300}})
    result = experiment / 'tasks/image-0/result.json'
    write(result.with_name('completion.json'), {'result_sha256': hashlib.sha256(result.read_bytes()).hexdigest()})
    (experiment / 'data-seed-42.npz').write_bytes(b'selected raw-record fixture')
    (experiment / 'test-cache-42.npz').write_bytes(b'public test image cache')
    for name in ('native.qpy', 'teacher.safetensors', 'model_weights.json'):
        with (experiment / name).open('wb') as stream:
            stream.truncate(8 * 1024 * 1024)
    (experiment / 'tasks/image-0/worker.log').write_bytes(b'redundant successful chatter\n' * 100000)
    write(experiment / 'tasks/image-0/progress.json', {'completed': 1})
    for index in range(6):
        log = experiment / f'tasks/failed-{index}/worker.log'
        log.parent.mkdir(parents=True)
        log.write_bytes(('älterer Text\n' * 100000 + f'FINAL_ERROR_{index}\n').encode())
    (root / 'logs').mkdir()
    (root / 'logs/mnist-study.log').write_bytes(b'old output\n' * 200000 + b'FIT_GATE_BLOCKED\n')
    for index in range(3):
        with gzip.open(experiment / f'full-readout-first-s{index}-n100.json.gz', 'wt') as stream:
            json.dump({'full_counts': [{'10000': 100}], 'original_counts': [{'10000': 100}]}, stream)
    write(root / 'status/mnist-study.json', {'execution': 'BLOCKED', 'archive': 'ARCHIVING'})
    return root


def verify_payloads(archive):
    manifest = json.loads(archive.read('archive-manifest.json'))
    for item in manifest['included']:
        payload = archive.read(item['path'])
        assert len(payload) == item['bytes']
        assert hashlib.sha256(payload).hexdigest() == item['sha256']
    return manifest


def test_compact_export_keeps_counts_and_records_without_large_weights_or_logs(run_root):
    output = run_root / 'archives/mnist-study.zip'
    manifest = build(run_root, output, 'mnist-study')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        saved = verify_payloads(archive)
        assert saved == manifest
        names = archive.namelist()
        assert 'experiments/mnist-study/data-seed-42.npz' in names
        assert 'experiments/mnist-study/splits.json' in names
        assert 'experiments/mnist-study/frozen-configuration.json' in names
        row = json.loads(archive.read('experiments/mnist-study/tasks/image-0/result.json'))
        assert row['counts'] == {'10000': 100} and row['measurement_seed'] == 101
        assert row['resources']['cx_shot_weighted_total'] == 12300
        report = json.loads(archive.read('experiments/mnist-study/report.json'))
        assert report['status'].startswith('BLOCKED') and not report['test_evaluated']
        assert report['configuration']['weights_integer_numerators'] == [[1, -1]]
        assert not any(name.endswith(('.qpy', '.safetensors', '.log')) for name in names)
        assert not any('test-cache-' in name or 'model_weights.json' in name for name in names)
        assert len([name for name in names if name.endswith('.json.gz')]) == 1
        tails = [r for r in saved['included'] if r['kind'] == 'log_tail']
        assert all(r['bytes'] <= LOG_TAIL_BYTES for r in tails)
        assert sum(r['bytes'] for r in tails) <= LOG_BUDGET_BYTES
        top = archive.read('diagnostics/logs/mnist-study.log.tail.txt')
        assert top.endswith(b'FIT_GATE_BLOCKED\n')
        assert not any('image-0/worker.log' in r['path'] for r in tails)
        for row in tails:
            archive.read(row['path']).decode('utf8')
    assert output.stat().st_size < 100000
    sidecar = output.with_suffix('.zip.sha256').read_text().split()
    assert sidecar == [hashlib.sha256(output.read_bytes()).hexdigest(), output.name]
    assert (run_root / 'experiments/mnist-study/native.qpy').stat().st_size == 8*1024*1024
    assert (run_root / 'logs/mnist-study.log').stat().st_size > 1024*1024


def test_omitted_large_files_are_not_opened_or_hashed(run_root, monkeypatch):
    original = Path.open
    forbidden = {'native.qpy', 'teacher.safetensors', 'model_weights.json', 'test-cache-42.npz'}
    def guard(path, *args, **kwargs):
        if path.name in forbidden:
            pytest.fail('An excluded binary/cache was read: ' + str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)
    build(run_root, run_root / 'archives/compact.zip', 'mnist-study')


def test_combined_export_has_final_execution_and_no_nested_archives(run_root):
    write(run_root / 'experiments/transformer-replacement/report.json', {'status': 'VERIFIED', 'cases': [1, 2]})
    write(run_root / 'experiments/summary/execution.json', {'jobs': {'summary': 'PENDING'}})
    write(run_root / 'run-state.json', {'state': 'RUNNING'})
    write(run_root / 'analysis-execution.json', {'scientific_execution_outcome': 'DONE_WITH_BLOCKED_TESTS',
        'jobs': {'mnist-study': {'execution': 'BLOCKED'}, 'transformer-replacement': {'execution': 'DONE'}}})
    (run_root / 'archives').mkdir()
    (run_root / 'archives/large-prior.zip').write_bytes(b'not a payload')
    output = run_root / 'archives/analysis.zip'
    build(run_root, output)
    with zipfile.ZipFile(output) as archive:
        saved = verify_payloads(archive)
        assert saved['experiment_names'] == ['mnist-study', 'summary', 'transformer-replacement']
        assert not any(name.endswith('.zip') for name in archive.namelist())
        assert 'run-state.json' not in archive.namelist()
        assert 'experiments/summary/execution.json' not in archive.namelist()
        outcome = json.loads(archive.read('analysis-execution.json'))
        assert outcome['scientific_execution_outcome'] == 'DONE_WITH_BLOCKED_TESTS'
        assert json.loads(archive.read('experiments/transformer-replacement/report.json'))['status'] == 'VERIFIED'


def test_corrupted_task_receipt_prevents_a_successful_archive(run_root):
    write(run_root / 'experiments/mnist-study/tasks/image-0/result.json', {'counts': {'changed': 1}})
    output = run_root / 'archives/bad.zip'
    with pytest.raises(ValueError, match='checksum mismatch'):
        build(run_root, output, 'mnist-study')
    assert not output.exists() and not output.with_suffix('.zip.sha256').exists()


def test_symlinks_cannot_export_files_outside_the_run(run_root, tmp_path):
    external = tmp_path / 'private'; external.mkdir()
    (external / 'secret.json').write_text('outside file content')
    (run_root / 'short-tasks').symlink_to(external, target_is_directory=True)
    (run_root / 'experiments/mnist-study/external').symlink_to(external, target_is_directory=True)
    (run_root / 'experiments/mnist-study/secret.json').symlink_to(external / 'secret.json')
    output = run_root / 'archives/safe.zip'
    build(run_root, output, 'mnist-study')
    with zipfile.ZipFile(output) as archive:
        assert not any('secret.json' in name or name.startswith('short-tasks/') for name in archive.namelist())
        assert any(item['reason'] == 'symlink' for item in verify_payloads(archive)['omitted'])


def test_short_worker_receipts_and_small_auxiliary_audit_are_preserved(run_root):
    short = run_root / 'short-tasks/mnist-study'
    write(short / 'result.json', {'report_sha256': 'fixture'})
    write(short / 'completion.json', {'result_sha256': hashlib.sha256((short / 'result.json').read_bytes()).hexdigest()})
    output = run_root / 'archives/short.zip'
    build(run_root, output, 'mnist-study')
    with zipfile.ZipFile(output) as archive:
        assert 'short-tasks/mnist-study/completion.json' in archive.namelist()
        verify_payloads(archive)


def test_native_zip_failure_does_not_publish_an_archive_or_checksum(run_root, monkeypatch):
    import qcenn_qml.analysis_archive as module
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(9, args[0])
    monkeypatch.setattr(module.subprocess, 'run', fail)
    output = run_root / 'archives/failed.zip'
    with pytest.raises(subprocess.CalledProcessError):
        build(run_root, output)
    assert not output.exists() and not output.with_suffix('.zip.sha256').exists()


def test_archive_writer_lock_rejects_concurrent_export_but_not_stale_filename(run_root):
    output = run_root / 'archives/locked.zip'; output.parent.mkdir()
    with output.with_suffix('.zip.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another exporter'):
            build(run_root, output, 'mnist-study')
    build(run_root, output, 'mnist-study')
    assert output.exists()


def test_combined_archive_failure_prevents_runner_done(monkeypatch, tmp_path):
    import qcenn_qml.research_runner as runner
    original = runner._archive
    monkeypatch.setenv('QCENN_LATEST_FILE', str(tmp_path / 'latest.txt'))
    def archive(root, name=None):
        if name is not None:
            return original(root, name)
        stream = (root / 'logs/analysis-archive.log').open('a')
        process = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(9)'], stdout=stream, stderr=stream)
        return process, stream, time.monotonic()
    monkeypatch.setattr(runner, '_archive', archive)
    root = tmp_path / 'failed-final-zip'
    code = runner.run(root, smoke=True, only='mnist', cpus=2, jobs=2, memory_per_worker_gib=.25)
    assert code == 1
    state = json.loads((root / 'run-state.json').read_text())
    assert state['state'] == 'FAILED' and state['analysis_archive']['state'] == 'FAILED'
    assert (root / 'FAILED').exists() and not (root / 'DONE').exists()
    assert not (root / 'DONE_WITH_BLOCKED_TESTS').exists()
