"""Resource-shared, resumable studies with explicit terminal states and archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from .recurrent_mnist import provenance, write_json
from .resources import detect_resources, exclusive_directory, inherited_run_descriptors, inherited_worker_descriptors, thread_environment
from .run_control import alive, identity, owned_processes, run_identity, stop_run
from .study_utils import digest, initialize, run_tasks


TERMINAL = {'DONE', 'DONE_WITH_BLOCKED_TESTS', 'FAILED', 'INTERRUPTED'}


def module_task(task):
    command = [sys.executable, '-u', '-m', task['module'], *task['arguments'],
               '--output-dir', task['report_dir']]
    directory = Path(task['report_dir'])
    prior = None
    if directory.exists() and any(directory.iterdir()):
        prior = directory.with_name(directory.name + f'.partial-{time.time_ns()}')
        directory.rename(prior)
    try:
        process = subprocess.Popen(command, pass_fds=inherited_worker_descriptors())
        code = process.wait()
    finally:
        if prior:
            target = directory / 'previous-attempts' / prior.name
            target.parent.mkdir(parents=True, exist_ok=True)
            prior.rename(target)
    report = directory / 'report.json'
    if code or not report.exists():
        raise RuntimeError(f'{task["module"]} exited {code}; expected {report}')
    return {'report': str(report), 'sha256': hashlib.sha256(report.read_bytes()).hexdigest()}


def short_group(spec_file):
    spec = json.loads(Path(spec_file).read_text())
    root, jobs, cpus = Path(spec['root']), spec['jobs'], spec['cpus']
    tasks = spec['tasks']
    failed = False
    try:
        run_tasks('qcenn_qml.research_runner:module_task', tasks, jobs=jobs,
            cpu_budget=cpus, directory=root / spec.get('task_folder', 'short-tasks'))
    except RuntimeError:
        failed = True
    if spec.get('audit'):
        block = root / 'experiments/transformer-block/report.json'
        if block.exists() and json.loads(block.read_text()).get('status') == 'VERIFIED':
            audit = {'id': 'smol-audit', 'module': 'qcenn_qml.smol_feasibility',
                'arguments': ['--transformer-report', str(block), '--attention-report',
                    str(root / 'experiments/attention/report.json')],
                'report_dir': str(root / 'experiments/smol-audit'), 'source_sha256': spec['source_sha256']}
            try:
                run_tasks('qcenn_qml.research_runner:module_task', [audit], jobs=1,
                    cpu_budget=cpus, directory=root / 'audit-tasks')
            except RuntimeError:
                failed = True
        else:
            failed = True
    return int(failed)


def latest_path():
    return Path(os.environ.get('QCENN_LATEST_FILE', 'results/research-studies-latest.txt'))


def resolve_output(value):
    if value:
        return Path(value).resolve()
    latest = latest_path()
    if not latest.exists():
        raise FileNotFoundError('No run found; provide the output directory')
    return Path(latest.read_text().strip()).resolve()


def show_status(directory):
    root = resolve_output(directory)
    path = root / 'STATUS.txt'
    if path.exists():
        print(path.read_text(), end='')
    else:
        print('State: NO_STATUS\nOutput: ' + str(root))
    try:
        record = run_identity(root)
    except RuntimeError as error:
        record = None; print(str(error))
    state_file = root / 'run-state.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get('state') in TERMINAL:
        print('Execution is terminal. Scientific gate status is recorded separately.')
    elif record is None or not alive(record):
        processes = owned_processes(root)
        print('ORPHANED_WORKERS_ACTIVE' if processes else 'STALE: coordinator is not alive; this run has not recorded verified completion.')
        print('Use --stop to clear this identified run, then --resume with its unchanged source/environment.')


def _job(name, module, root, arguments=()):
    return {'id': name, 'module': module, 'arguments': list(arguments),
            'report_dir': str(root / 'experiments' / name)}


def _validate_model(name, path, code):
    if not path.exists():
        return 'FAILED'
    report = json.loads(path.read_text())
    if name == 'mnist-study':
        if code == 2 and report.get('status', '').startswith('BLOCKED'):
            return 'BLOCKED'
        return 'DONE' if code == 0 and report.get('status') == 'DONE' and report.get('test_evaluated') else 'FAILED'
    if name == 'diffusion-study':
        expected = report.get('configuration', {})
        count = len(expected.get('corpora', [])) * len(expected.get('data_seeds', [])) * len(expected.get('measurement_seeds', []))
        return 'DONE' if code == 0 and report.get('status') == 'DONE' and count > 0 and len(report.get('runs', [])) == count else 'FAILED'
    if name in ('transformer-block', 'transformer-replacement'):
        return 'DONE' if code == 0 and report.get('status') == 'VERIFIED' else 'FAILED'
    return 'DONE' if code == 0 else 'FAILED'


def _archive(root, name):
    # Arguments are separate from shell code. Native ZIP/CRC/SHA tools, no
    # serialized Python wrapper and no unverified "DONE" marker.
    script = '''set -euo pipefail
cd -- "$1"
zip -1 -qr "archives/$2.partial.zip" "experiments/$2" "logs/$2.log" "status/$2.json" "commands/$2.txt" runtime.json
unzip -tqq "archives/$2.partial.zip"
mv -- "archives/$2.partial.zip" "archives/$2.zip"
cd archives
sha256sum "$2.zip" > "$2.zip.sha256"
'''
    partial = root / 'archives' / (name + '.partial.zip')
    if partial.exists():
        partial.rename(partial.with_name(name + f'.abandoned-{time.time_ns()}.zip'))
    stream = (root / 'logs' / (name + '-archive.log')).open('a')
    process = subprocess.Popen(['bash', '-c', script, 'qcenn-archive', str(root), name],
        stdout=stream, stderr=subprocess.STDOUT, pass_fds=inherited_run_descriptors(), start_new_session=True)
    return process, stream, time.monotonic()


def run(root, *, smoke=False, only='all', cpus=None, jobs=None, corpus_files=None,
        resume=False, memory_per_worker_gib=4., root_lock=None, text_corpus_file=None, text_new_tokens=2):
    root = Path(root).resolve()
    if only not in ('all', 'mnist', 'transformer') or corpus_files:
        raise ValueError('Diffusion is retired from active studies; select mnist or transformer without corpus files')
    available = detect_resources()
    cpus = int(cpus or os.environ.get('QCENN_CPUS') or available['cpu_budget'])
    if not 1 <= cpus <= available['cpu_budget']:
        raise ValueError('CPU budget exceeds the current process/allocation limit')
    memory_cap = max(1, int(.8 * available['available_memory_bytes'] / (memory_per_worker_gib * 1024**3)))
    jobs = int(jobs or os.environ.get('QCENN_JOBS') or min(cpus, memory_cap))
    if not 1 <= jobs <= cpus:
        raise ValueError('Require 1 <= jobs <= CPU budget')
    # Explicit jobs may lower concurrency, never override the memory estimate.
    jobs = min(jobs, memory_cap)
    text_corpus = Path(text_corpus_file or Path(__file__).parent / 'assets/quantum-sequence.txt').resolve()
    data_root = str(Path(os.environ.get('QCENN_DATA_ROOT', 'data')).resolve())
    diffusion_examples = int(os.environ.get('QCENN_DIFFUSION_EXAMPLES', '256')) if corpus_files and not smoke else None
    configuration = {'smoke': smoke, 'only': only, 'corpus_files': [str(Path(p).resolve()) for p in corpus_files] if corpus_files else None,
        'runner_schema': 2, 'selection_mode': 'equal_shot', 'image_batch_size': 1, 'case_batch_size': 1,
        'diffusion_retired': True, 'mnist_accuracy_profile': True,
        'text_corpus_file': str(text_corpus), 'text_corpus_sha256': hashlib.sha256(text_corpus.read_bytes()).hexdigest(),
        'text_new_tokens': text_new_tokens, 'data_root': data_root, 'diffusion_examples': diffusion_examples}
    manifest = initialize(root, configuration, provenance(), resume)
    for name in ('experiments', 'logs', 'status', 'commands', 'archives'):
        (root / name).mkdir(exist_ok=True)
    (root / 'runner.pid').write_text(str(os.getpid()) + '\n')
    write_json(root / 'runner.identity.json', {**identity(), 'output_dir': str(root)})
    previous_runtime = json.loads((root / 'runtime.json').read_text()) if (root / 'runtime.json').exists() else {}
    pool = Path(previous_runtime.get('cpu_pool') or tempfile.mkdtemp(prefix=f'qcenn-pool-{os.getuid()}-'))
    pool.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **thread_environment(1),
        'QCENN_RESOURCE_POOL': str(pool), 'QCENN_POOL_CPUS': str(jobs),
        'QCENN_RUN_OUTPUT': str(root)}
    if root_lock:
        environment['QCENN_RUN_LOCK_FD'] = str(root_lock.descriptors[0])
    old_environment = os.environ.copy()
    os.environ.update(environment)
    runtime = {'cpu_budget': cpus, 'worker_limit_per_coordinator': jobs, 'memory_worker_estimate_gib': memory_per_worker_gib,
        'resources': available, 'cpu_pool': str(pool), 'python': sys.executable,
        'thread_environment': thread_environment(1), 'configuration': configuration,
        'source': manifest['source'], 'source_root': str(Path(__file__).resolve().parents[2]),
        'gpu_profile_path': os.environ.get('QCENN_GPU_PROFILE'),
        'data_root': data_root,
        'started_at': time.time(), 'resumed': resume,
        'execution_policy': manifest['execution_policy']}
    write_json(root / 'runtime.json', runtime)
    latest = latest_path(); latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(root) + '\n')
    long_jobs, short = [], []
    common = ['--jobs', str(jobs), '--cpu-budget', str(cpus)]
    if resume:
        common.append('--resume')
    small = ['--smoke'] if smoke else []
    if only in ('all', 'mnist'):
        long_jobs.append(_job('mnist-study', 'qcenn_qml.mnist_study', root,
            [*common, *small, '--image-batch-size', '1', '--selection-mode', 'equal_shot',
             '--accuracy-profile', '--validation-per-class', '1' if smoke else '5',
             *([] if smoke else ['--shot-budgets', '4096', '16384', '65536',
                '--gate-shot-budgets', '4096', '16384', '65536']),
             '--data-root', data_root]))
    if only in ('all', 'transformer'):
        long_jobs.append(_job('transformer-replacement', 'qcenn_qml.transformer_replacement', root,
            [*common, *small]))
        short.extend([_job('transformer-block', 'qcenn_qml.quantum_transformer', root, ['--shots', '256']),
            _job('attention', 'qcenn_qml.quantum_attention', root, ['--shots', '256' if smoke else '8192'])])
    if only == 'all':
        args = ['--prompt', 'the cat', '--new-tokens', str(text_new_tokens), '--shots', '128' if smoke else '1024',
            '--corpus-file', str(text_corpus)]
        short.append(_job('text', 'qcenn_qml.smol135', root, args))
    names = [j['id'] for j in [*long_jobs, *short]]
    if short:
        names.append('smol-audit')
    for job in short:
        job['source_sha256'] = manifest['source']['source_sha256']
    names.append('summary')
    states = {name: {'execution': 'PENDING', 'archive': 'PENDING', 'exit_code': None} for name in names}
    active, archives, streams, short_specs = {}, {}, [], {}
    began = time.time()
    overall, exit_code, interrupted = 'RUNNING', 0, False
    def snapshot():
        now = time.time()
        details = []
        mnist_root = root / 'experiments/mnist-study'
        scheduler_files = [root / 'experiments/transformer-replacement/tasks/status.json', root / 'short-tasks/status.json', root / 'text-tasks/status.json', root / 'audit-tasks/status.json',
            mnist_root / 'fresh-test/tasks/status.json']
        for phase in ('convergence', 'memory-2', 'memory-3', 'memory-4', 'memory-8', 'memory-16', 'encoding-damping'):
            scheduler_files.extend(mnist_root / phase / sub / 'status.json' for sub in ('validation', 'fit-128', 'fit-1024', 'fit-4096', 'fit-16384', 'fit-65536'))
        for file in scheduler_files:
            # Only scheduler status (not every per-image progress file).
            try:
                data = json.loads(file.read_text())
                if 'tasks' in data:
                    counts = {state: list(data['tasks'].values()).count(state) for state in set(data['tasks'].values())}
                    details.append({'path': str(file.relative_to(root)), 'counts': counts})
            except (OSError, ValueError):
                pass
        state = {'state': overall, 'execution_complete': overall in TERMINAL,
            'output_dir': str(root), 'started_at': began, 'updated_at': now,
            'elapsed_seconds': now-began, 'jobs': states, 'scheduler_progress': details,
            'cpu_budget': cpus, 'shared_worker_slots': jobs, 'finished_at': now if overall in TERMINAL else None}
        write_json(root / 'run-state.json', state)
        lines = [f'State: {overall}', f'Output: {root}', f'Elapsed: {now-began:.0f}s',
            f'CPU budget: {cpus}; shared worker slots: {jobs}; one thread per quantum worker',
            'GPU verification: ' + str((runtime['execution_policy'].get('gpu') or {}).get('verified_gpu_ids', [])),
            'EXPERIMENT          EXECUTION     ARCHIVE       EXIT']
        for name, row in states.items():
            lines.append(f'{name:<19} {row["execution"]:<13} {row["archive"]:<13} {str(row["exit_code"]):<5}')
            write_json(root / 'status' / (name + '.json'), row)
        for row in details:
            lines.append(f'{row["path"]}: {row["counts"]}')
        lines.append('DONE requires reports, complete requested measurements, native ZIP CRC checks and SHA256 receipts.')
        if overall == 'DONE_WITH_BLOCKED_TESTS':
            lines.append('Execution finished. MNIST scientific gates blocked testing; no test result was invented.')
        target = root / 'STATUS.txt'; temporary = target.with_suffix('.tmp')
        temporary.write_text('\n'.join(lines) + '\n'); temporary.replace(target)
    def launch(name, command, stream_name=None):
        stream = (root / 'logs' / ((stream_name or name) + '.log')).open('a', buffering=1)
        streams.append(stream)
        (root / 'commands' / (name + '.txt')).write_text(shlex.join(command) + '\n')
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            env=environment, pass_fds=inherited_run_descriptors(), start_new_session=True)
        active[name] = process
        if name in states:
            states[name]['execution'] = 'RUNNING'
        print(f'START {name}; log={root / "logs" / ((stream_name or name) + ".log")}', flush=True)
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Runner received signal {signum}')
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for job in long_jobs:
            launch(job['id'], [sys.executable, '-u', '-m', job['module'], *job['arguments'], '--output-dir', job['report_dir']])
        if short:
            groups = [('short-group', [j for j in short if j['id'] != 'text'], 'short-tasks', True),
                      ('text-group', [j for j in short if j['id'] == 'text'], 'text-tasks', False)]
            for group, members, folder, audit in groups:
                if not members:
                    continue
                short_specs[group] = {'root': str(root), 'jobs': jobs, 'cpus': cpus,
                    'tasks': members, 'audit': audit, 'task_folder': folder,
                    'source_sha256': manifest['source']['source_sha256']}
                spec_path = root / (group + '-spec.json')
                write_json(spec_path, short_specs[group])
                launch(group, [sys.executable, '-u', '-m', 'qcenn_qml.research_runner', '--short-spec', str(spec_path)])
        last = heartbeat = 0.
        while active:
            for folder in ('short-tasks', 'text-tasks', 'audit-tasks'):
                path = root / folder / 'status.json'
                if path.exists():
                    progress = json.loads(path.read_text()).get('tasks', {})
                    for name, task_state in progress.items():
                        if name in states and states[name]['execution'] == 'PENDING' and task_state != 'PENDING':
                            states[name]['execution'] = 'RUNNING'
            for name, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[name]
                if name not in short_specs:
                    states[name].update(execution=_validate_model(name, root / 'experiments' / name / 'report.json', code), exit_code=code)
                else:
                    spec = short_specs[name]
                    for job in spec['tasks']:
                        name = job['id']; receipt = root / spec['task_folder'] / name / 'completion.json'
                        passed = receipt.exists()
                        model_log = root / spec['task_folder'] / name / 'worker.log'
                        if model_log.exists():
                            shutil.copy2(model_log, root / 'logs' / (name + '.log'))
                        (root / 'commands' / (name + '.txt')).write_text(shlex.join(
                            [sys.executable, '-u', '-m', job['module'], *job['arguments'], '--output-dir', job['report_dir']]) + '\n')
                        states[name].update(execution=_validate_model(name, Path(job['report_dir']) / 'report.json', 0 if passed else 1), exit_code=0 if passed else 1)
                    if spec['audit']:
                        receipt = root / 'audit-tasks/smol-audit/completion.json'
                        states['smol-audit'].update(execution='DONE' if receipt.exists() else 'FAILED', exit_code=0 if receipt.exists() else 1)
                        audit_log = root / 'audit-tasks/smol-audit/worker.log'
                        if audit_log.exists():
                            shutil.copy2(audit_log, root / 'logs/smol-audit.log')
            for name, row in states.items():
                if name == 'summary':
                    continue
                if row['execution'] in ('DONE', 'BLOCKED', 'FAILED') and row['archive'] == 'PENDING':
                    target = root / 'experiments' / name; target.mkdir(parents=True, exist_ok=True)
                    (root / 'logs' / (name + '.log')).touch(exist_ok=True)
                    (root / 'commands' / (name + '.txt')).touch(exist_ok=True)
                    row['archive'] = 'ARCHIVING'; snapshot()
                    archives[name] = _archive(root, name)
            for name, (process, stream, started) in list(archives.items()):
                if process.poll() is None and time.monotonic()-started > 3600:
                    process.terminate()
                code = process.poll()
                if code is not None:
                    stream.close(); del archives[name]
                    states[name]['archive'] = 'DONE' if code == 0 else 'FAILED'
            if time.monotonic()-last >= 5:
                snapshot(); last = time.monotonic()
            if time.monotonic()-heartbeat >= 30:
                print(f'RUNNING; elapsed={time.time()-began:.0f}s; live coordinators={len(active)}; see {root / "STATUS.txt"}', flush=True)
                heartbeat = time.monotonic()
            if active:
                time.sleep(.2)
        # Archive jobs started when the last coordinator finished, too.
        for name, row in states.items():
            if name != 'summary' and row['archive'] == 'PENDING':
                target = root / 'experiments' / name; target.mkdir(parents=True, exist_ok=True)
                (root / 'logs' / (name + '.log')).touch(exist_ok=True)
                (root / 'commands' / (name + '.txt')).touch(exist_ok=True)
                row['archive'] = 'ARCHIVING'; snapshot(); archives[name] = _archive(root, name)
        while archives:
            for name, (process, stream, started) in list(archives.items()):
                if process.poll() is None and time.monotonic()-started > 3600:
                    process.terminate()
                if process.poll() is not None:
                    stream.close(); del archives[name]
                    states[name]['archive'] = 'DONE' if process.returncode == 0 else 'FAILED'
            snapshot(); time.sleep(.2)
        from .report import build
        sources = {key: str(root / 'experiments' / name / 'report.json') for key, name in (
            ('mnist_study', 'mnist-study'), ('transformer_replacement', 'transformer-replacement'),
            ('transformer_block', 'transformer-block'), ('attention', 'attention'),
            ('smol', 'text'), ('smol_feasibility', 'smol-audit')) if name in states}
        summary = root / 'experiments/summary'
        build(None, sources.pop('smol', None), None, summary, **sources)
        write_json(summary / 'execution.json', {'jobs': states, 'runtime': runtime,
            'logical_CX_cost_scope': 'Completed reported shots; failed simulator/compilation overhead is retained in attempt histories and worker logs.'})
        (root / 'logs/summary.log').write_text('Combined reports and execution status written.\n')
        (root / 'commands/summary.txt').write_text('qcenn_qml.report.build; recorded source paths in combined.json\n')
        states['summary'].update(execution='DONE', archive='ARCHIVING', exit_code=0); snapshot()
        process, stream, _ = _archive(root, 'summary')
        archives['summary'] = (process, stream, time.monotonic())
        try:
            code = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5); code = 124
        stream.close(); archives.pop('summary'); states['summary']['archive'] = 'DONE' if code == 0 else 'FAILED'
        failed = any(r['execution'] == 'FAILED' or r['archive'] != 'DONE' for r in states.values())
        blocked = any(r['execution'] == 'BLOCKED' for r in states.values())
        overall, exit_code = ('FAILED', 1) if failed else (('DONE_WITH_BLOCKED_TESTS', 2) if blocked else ('DONE', 0))
    except KeyboardInterrupt:
        overall, exit_code, interrupted = 'INTERRUPTED', 130, True
    except BaseException:
        overall, exit_code = 'FAILED', 1
        (root / 'runner-error.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        for process in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in active.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        for process, stream, _ in archives.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
            stream.close()
        for stream in streams:
            stream.close()
        for row in states.values():
            if row['execution'] in ('RUNNING', 'PENDING'):
                row['execution'] = 'INTERRUPTED' if interrupted else 'FAILED'
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        snapshot()
        for state in TERMINAL:
            marker = root / state
            if marker.exists():
                marker.unlink()
        write_json(root / overall, {'state': overall, 'exit_code': exit_code,
            'execution_finished': True, 'scientific_all_gates_passed': overall == 'DONE',
            'full_pretrained_smol_implemented': False,
            'finished_at': time.time(), 'report': 'run-state.json'})
        os.environ.clear(); os.environ.update(old_environment)
    print(f'FINISHED state={overall} exit={exit_code}; output={root}', flush=True)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?')
    parser.add_argument('--output-dir')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--replace-run')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', default=None)
    parser.add_argument('--study-only', choices=['all', 'mnist', 'transformer'])
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--memory-per-worker-gib', type=float, default=4.)
    parser.add_argument('--corpus-files', nargs='+')
    parser.add_argument('--text-corpus-file')
    parser.add_argument('--text-new-tokens', type=int)
    parser.add_argument('--short-spec', help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.short_spec:
        return short_group(options.short_spec)
    path = options.output_dir or options.output
    if options.status:
        show_status(path); return 0
    if options.stop:
        print(json.dumps(stop_run(resolve_output(path)), indent=2)); return 0
    from importlib.metadata import version
    from . import __version__
    expected = {'qcenn-qml': __version__, 'qiskit': '2.6.1', 'qiskit-aer': '0.18.2'}
    if any(version(package) != required for package, required in expected.items()):
        parser.error('Accelerated runner requires matching project metadata, Qiskit 2.6.1 and Aer 0.18.2. Run scripts/update_and_run_research.sh; no old experiment was stopped.')
    if not 0 < options.memory_per_worker_gib < float('inf'):
        parser.error('memory-per-worker-gib must be positive and finite')
    if options.resume:
        root = resolve_output(path)
        previous = json.loads((root / 'study-manifest.json').read_text())['configuration']
        os.environ.setdefault('QCENN_DATA_ROOT', previous['data_root'])
        if previous.get('diffusion_examples') is not None:
            os.environ.setdefault('QCENN_DIFFUSION_EXAMPLES', str(previous['diffusion_examples']))
        saved_runtime = json.loads((root / 'runtime.json').read_text())
        saved_policy = saved_runtime['execution_policy']
        for env_key, policy_key in [('QCENN_MPS_READOUT', 'mps_readout'),
                ('QCENN_GPU_MIN_QUBITS', 'gpu_min_qubits'),
                ('QCENN_GPU_MAX_STATEVECTOR_QUBITS', 'gpu_max_statevector_qubits'),
                ('QCENN_GPU_MAX_TENSOR_QUBITS', 'gpu_max_tensor_qubits'),
                ('QCENN_GPU_FRACTION', 'gpu_fraction')]:
            os.environ.setdefault(env_key, str(saved_policy[policy_key]))
        if saved_runtime.get('gpu_profile_path'):
            os.environ.setdefault('QCENN_GPU_PROFILE', saved_runtime['gpu_profile_path'])
        if options.smoke is None:
            options.smoke = previous['smoke']
        if options.study_only is None:
            options.study_only = previous['only']
        if options.corpus_files is None:
            options.corpus_files = previous['corpus_files']
        if options.text_corpus_file is None:
            options.text_corpus_file = previous['text_corpus_file']
        if options.text_new_tokens is None:
            options.text_new_tokens = previous['text_new_tokens']
    else:
        root = Path(path or f'results/research-{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}').resolve()
    cpu_cap = detect_resources()['cpu_budget']
    requested = options.cpus or int(os.environ.get('QCENN_CPUS', cpu_cap))
    if not 1 <= requested <= cpu_cap or (options.jobs is not None and not 1 <= options.jobs <= requested):
        parser.error('Invalid CPU/worker budget for this allocation; no old run was stopped')
    if not options.resume and root.exists() and any(root.iterdir()):
        parser.error('Output already contains evidence; choose a fresh output or --resume')
    if options.text_new_tokens is not None and options.text_new_tokens < 1:
        parser.error('text-new-tokens must be positive')
    with exclusive_directory(root) as lock:
        if options.replace_run:
            print(json.dumps(stop_run(options.replace_run), indent=2), flush=True)
        return run(root, smoke=bool(options.smoke), only=options.study_only or 'all',
            cpus=options.cpus, jobs=options.jobs, corpus_files=options.corpus_files,
            resume=options.resume, memory_per_worker_gib=options.memory_per_worker_gib, root_lock=lock,
            text_corpus_file=options.text_corpus_file, text_new_tokens=options.text_new_tokens if options.text_new_tokens is not None else 2)


if __name__ == '__main__':
    raise SystemExit(main())
