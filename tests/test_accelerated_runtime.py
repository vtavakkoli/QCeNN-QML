"""Global leases, watchdogs, immutable recovery and one-run process control."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from qcenn_qml.resources import ResourcePool, acquire_files, exclusive_directory, thread_environment
from qcenn_qml.run_control import identity, stop_run
from qcenn_qml.study_utils import run_tasks


def wait_for(predicate, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.05)
    raise AssertionError('Expected process/file state was not reached')


def test_kernel_releases_leases_after_child_exits_even_when_files_remain(tmp_path):
    pool = ResourcePool(tmp_path, 1)
    lease = pool.acquire()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.4)'], pass_fds=lease.descriptors)
    lease.close()
    assert pool.acquire() is None
    child.wait(timeout=5)
    assert list(tmp_path.glob('*.lock'))
    recovered = pool.acquire()
    assert recovered is not None
    recovered.close()


def test_two_coordinators_share_one_cpu_budget_and_bind_workers(monkeypatch, tmp_path):
    worker = tmp_path / 'runtime_fixture.py'
    worker.write_text('''import time,os
def work(task):
 started=time.time(); time.sleep(.4)
 return {'start':started,'end':time.time(),'affinity':len(os.sched_getaffinity(0)), 'rayon':os.environ['RAYON_NUM_THREADS']}
''')
    environment = {**os.environ, **thread_environment(1), 'PYTHONPATH': str(tmp_path) + os.pathsep + str(Path(__file__).parents[1] / 'src'),
        'QCENN_RESOURCE_POOL': str(tmp_path / 'pool'), 'QCENN_POOL_CPUS': '2'}
    program = '''from pathlib import Path
import json,sys
from qcenn_qml.study_utils import run_tasks
r=run_tasks('runtime_fixture:work',[{'id':str(i)} for i in range(4)],jobs=2,cpu_budget=2,directory=Path(sys.argv[1]))
Path(sys.argv[2]).write_text(json.dumps(r))
'''
    processes = [subprocess.Popen([sys.executable, '-c', program, str(tmp_path / name), str(tmp_path / (name+'.json'))],
        env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) for name in ('a', 'b')]
    try:
        for process in processes:
            _, errors = process.communicate(timeout=30)
            assert process.returncode == 0, errors
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill(); process.wait()
    rows = sum((json.loads((tmp_path / (name+'.json')).read_text()) for name in ('a', 'b')), [])
    events = sorted([(r['start'], 1) for r in rows] + [(r['end'], -1) for r in rows])
    active = peak = 0
    for _, change in events:
        active += change; peak = max(peak, active)
    assert peak == 2
    assert all(r['affinity'] == 1 and r['rayon'] == '1' for r in rows)


def test_watchdog_ends_and_preserves_failed_attempts(monkeypatch, tmp_path):
    (tmp_path / 'runtime_fixture.py').write_text('import time\ndef work(task): time.sleep(10); return {}\n')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + str(Path(__file__).parents[1] / 'src'))
    with pytest.raises(RuntimeError, match='Experiment tasks failed'):
        run_tasks('runtime_fixture:work', [{'id': 'timeout'}], jobs=1, cpu_budget=1,
            directory=tmp_path / 'tasks', task_timeout=.6, idle_timeout=10, retries=1)
    status = json.loads((tmp_path / 'tasks/status.json').read_text())
    assert status['tasks']['timeout'] == 'TIMED_OUT' and status['execution_complete']
    assert len(json.loads((tmp_path / 'tasks/timeout/attempt-history.json').read_text())) == 2
    assert not (tmp_path / 'tasks/timeout/completion.json').exists()
    lease = acquire_files([tmp_path / 'tasks/timeout/task.writer.lock'])
    assert lease is not None
    lease.close()


def test_stop_targets_only_the_identified_run_tree(monkeypatch, tmp_path):
    root = tmp_path / 'run'; root.mkdir()
    (tmp_path / 'sleep_model.py').write_text('''import argparse,time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output-dir'); a=p.parse_args()
d=Path(a.output_dir); d.mkdir(parents=True,exist_ok=True); (d/'started').touch()
time.sleep(30)
''')
    spec = root / 'short.json'
    spec.write_text(json.dumps({'root': str(root), 'cpus': 1, 'jobs': 1, 'audit': False,
        'tasks': [{'id': 'sleeper', 'module': 'sleep_model', 'arguments': [], 'report_dir': str(root / 'experiments/sleeper')}]}))
    environment = {**os.environ, **thread_environment(1), 'PYTHONPATH': str(tmp_path) + os.pathsep + str(Path(__file__).parents[1] / 'src')}
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    process = subprocess.Popen([sys.executable, '-m', 'qcenn_qml.research_runner', '--short-spec', str(spec)],
        env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # procfs may expose an outer PID namespace in a container. Match the
        # unique explicit spec path, then retain both visible and signal IDs.
        def find_process():
            for entry in Path('/proc').glob('[0-9]*'):
                record = identity(entry.name)
                if record and str(spec) in record['args']:
                    return record
        record = wait_for(find_process)
        (root / 'runner.identity.json').write_text(json.dumps({**record, 'output_dir': str(root)}))
        wait_for(lambda: (root / 'experiments/sleeper/started').exists())
        receipt = stop_run(root, grace=.5)
        process.wait(timeout=10)
        assert receipt['status'] == 'STOPPED' and receipt['signals']
        assert unrelated.poll() is None
        lock = acquire_files([root / 'short-tasks/sleeper/task.writer.lock'])
        assert lock is not None; lock.close()
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        unrelated.terminate(); unrelated.wait(timeout=5)


def test_leftover_writer_filename_does_not_lock_a_finished_directory(tmp_path):
    root = tmp_path / 'result'
    with exclusive_directory(root):
        with pytest.raises(RuntimeError, match='active writer'):
            with exclusive_directory(root):
                pass
    with exclusive_directory(root):
        pass


def test_runner_resume_restores_data_root_and_reuses_measured_counts(tmp_path):
    import hashlib
    root = tmp_path / 'protocol'
    base_env = {**os.environ, **thread_environment(1), 'QCENN_LATEST_FILE': str(tmp_path / 'latest.txt')}
    base_env.pop('QCENN_DATA_ROOT', None)
    command = [sys.executable, '-m', 'qcenn_qml.research_runner', '--study-only', 'mnist',
        '--output-dir', str(root), '--cpus', '2', '--jobs', '2', '--memory-per-worker-gib', '.25']
    first = subprocess.run([*command, '--smoke'], env={**base_env, 'QCENN_DATA_ROOT': str(tmp_path / 'explicit-cache')},
        capture_output=True, text=True, timeout=120)
    assert first.returncode in (0, 2), first.stdout + first.stderr
    measured = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('experiments/mnist-study/**/image-*.json')}
    assert measured
    restored = subprocess.run([*command, '--resume'], env=base_env, capture_output=True, text=True, timeout=60)
    assert restored.returncode == first.returncode, restored.stdout + restored.stderr
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p, value in measured.items())
    assert json.loads((root / 'runtime.json').read_text())['data_root'] == str(tmp_path / 'explicit-cache')
