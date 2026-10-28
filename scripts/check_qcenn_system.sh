#!/usr/bin/env bash
# QCeNN-QML — Vahid Tavakkoli (2026). See LICENSE and CITATION.cff.
# Standalone, read-only system inspection; only its own report directory is written.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/check_qcenn_system.sh [--benchmark] [--cpu-budget N]
       [--python PATH] [--repo-root PATH] [--output-dir PATH]
       [--benchmark-seconds N] [--memory-per-worker-gib N]

Run ON the machine/allocation that will execute the studies. Inspection needs
only Python >=3.10; Aer/project imports are isolated and time limited.
--benchmark also compares CPU worker/thread combinations using small measured
MNIST-memory and diffusion circuits, with two repeats. It skips benchmarking
while other QCeNN jobs are present. No datasets are downloaded, no scientific
results are changed, and no running experiment is stopped.

Output: report.txt, report.json, recommended.env, commands.txt, and a ZIP with
SHA256 if the native zip command is installed. This script installs nothing.
--cpu-budget can REDUCE the detected process/allocation limit, never raise it.
The memory-per-worker estimate defaults to 4 GiB; this is not a memory guarantee.
EOF
}
qcenn_check_benchmark=0
qcenn_check_python=${QCENN_PYTHON:-}
qcenn_check_repo=$PWD
qcenn_check_output=
qcenn_check_cpus=0
qcenn_check_seconds=3
qcenn_check_worker_gib=4
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --benchmark) qcenn_check_benchmark=1; shift ;;
    --python|--repo-root|--output-dir|--cpu-budget|--benchmark-seconds|--memory-per-worker-gib)
      (($# >= 2)) || { usage >&2; exit 1; }
      case "$1" in
        --python) qcenn_check_python=$2 ;;
        --repo-root) qcenn_check_repo=$2 ;;
        --output-dir) qcenn_check_output=$2 ;;
        --cpu-budget) qcenn_check_cpus=$2 ;;
        --benchmark-seconds) qcenn_check_seconds=$2 ;;
        --memory-per-worker-gib) qcenn_check_worker_gib=$2 ;;
      esac
      shift 2 ;;
    *) usage >&2; exit 1 ;;
  esac
done
qcenn_check_repo=$(cd -- "$qcenn_check_repo" && pwd -P)
if [[ -z $qcenn_check_python ]]; then
  if [[ -x $qcenn_check_repo/.venv-qcenn/bin/python ]]; then
    qcenn_check_python=$qcenn_check_repo/.venv-qcenn/bin/python
  else qcenn_check_python=$(command -v python3 || command -v python || true); fi
fi
command -v "$qcenn_check_python" >/dev/null 2>&1 || { printf 'Python missing; use --python PATH.\n' >&2; exit 1; }
qcenn_check_python=$(command -v "$qcenn_check_python")
"$qcenn_check_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else "Python >=3.10 required")'
[[ $qcenn_check_cpus =~ ^[0-9]+$ ]] && [[ $qcenn_check_seconds =~ ^[0-9]+$ ]] || { usage >&2; exit 1; }
((qcenn_check_seconds >= 1 && qcenn_check_seconds <= 30)) || { printf 'Benchmark seconds must be 1..30.\n' >&2; exit 1; }
qcenn_check_output=${qcenn_check_output:-$qcenn_check_repo/results/system-check-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p -- "$(dirname -- "$qcenn_check_output")"
mkdir -- "$qcenn_check_output" # Refuse to overwrite an earlier report.
qcenn_check_output=$(cd -- "$qcenn_check_output" && pwd -P)
cat > "$qcenn_check_output/system_probe.py" <<'QCENN_PYTHON'
"""Embedded implementation of check_qcenn_system.sh; standard-library collector."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3
DOC = 'https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html'


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return ''


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def limited_env(threads=1):
    return {**os.environ, 'OMP_NUM_THREADS': str(threads), 'QCENN_THREADS': str(threads),
            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
            'QISKIT_PARALLEL': 'FALSE', 'QISKIT_NUM_PROCS': '1'}


def terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def command(args, timeout=8, env=None):
    if not shutil.which(str(args[0])):
        return {'status': 'unavailable', 'command': args}
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
        return {'status': 'ok' if process.returncode == 0 else 'error',
                'exit_code': process.returncode, 'output': output[:24000]}
    except subprocess.TimeoutExpired:
        terminate(process)
        return {'status': 'timeout', 'output': process.communicate()[0][:24000]}
    except BaseException:
        terminate(process)
        raise


def unescape_mount(text):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), text)


def cgroup_locations(mountinfo, membership):
    """Map this process to mounted v1/v2 controllers, including namespace roots."""
    groups = [line.split(':', 2) for line in membership.splitlines() if line.count(':') == 2]
    found = set()
    for line in mountinfo.splitlines():
        if ' - ' not in line:
            continue
        left, right = line.split(' - ', 1)
        fields, fs = left.split(), right.split()
        if len(fields) < 5 or len(fs) < 3 or fs[0] not in ('cgroup', 'cgroup2'):
            continue
        root, mount = Path(unescape_mount(fields[3])), Path(unescape_mount(fields[4]))
        controllers = set(fs[2].split(','))
        for _, names, group in groups:
            if (fs[0] == 'cgroup2' and names) or (fs[0] == 'cgroup' and not controllers.intersection(names.split(','))):
                continue
            try:
                relative = Path(group).relative_to(root)
            except ValueError:
                # In a cgroup namespace /self/cgroup may already be mount-relative.
                relative = Path(group.lstrip('/'))
            candidate = mount / relative
            if '..' in candidate.parts:
                candidate = mount
            if not candidate.exists():
                candidate = mount
            for directory in (candidate, *candidate.parents):
                if directory != mount and mount not in directory.parents:
                    break
                found.add((fs[0], str(directory)))
                if directory == mount:
                    break
    return sorted(found)


def cgroup_limits(locations):
    rows = []
    for kind, path in locations:
        row = {'kind': kind, 'path': path}
        base = Path(path)
        quota = read(base / 'cpu.max').split()
        if len(quota) == 2 and quota[0] != 'max':
            row['cpu_cores'] = int(quota[0]) / int(quota[1])
        q, p = read(base / 'cpu.cfs_quota_us'), read(base / 'cpu.cfs_period_us')
        if q and p and int(q) > 0:
            row['cpu_cores'] = int(q) / int(p)
        for limit_name, used_name in [('memory.max', 'memory.current'),
                                      ('memory.limit_in_bytes', 'memory.usage_in_bytes')]:
            limit, used = read(base / limit_name), read(base / used_name)
            if limit.isdigit() and int(limit) < (1 << 60):
                row['memory_limit_bytes'] = int(limit)
                row['memory_used_bytes'] = int(used or 0)
        for name in ('cpu.stat', 'cpu.pressure', 'memory.high', 'cpuset.cpus.effective', 'cpuset.cpus'):
            value = read(base / name)
            if value:
                row[name] = value
        rows.append(row)
    return rows


def slurm_node_cpus(value, node_index=0):
    for item in value.split(','):
        match = re.fullmatch(r'(\d+)(?:\(x(\d+)\))?', item.strip())
        if not match:
            return None
        # Index directly: no allocation proportional to a potentially huge repeat.
        count, repeat = int(match[1]), int(match[2] or 1)
        if node_index < repeat:
            return count
        node_index -= repeat
    return None


def allocation_limits(affinity_count, groups, scheduler):
    limits = {'affinity': affinity_count}
    quotas = [r['cpu_cores'] for r in groups if 'cpu_cores' in r]
    if quotas:
        limits['cgroup_quota'] = min(quotas)
    for key in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'PBS_NP', 'NSLOTS', 'LSB_DJOB_NUMPROC'):
        if scheduler.get(key, '').isdigit() and int(scheduler[key]) > 0:
            limits[key] = int(scheduler[key])
    node_id = scheduler.get('SLURM_NODEID', '0')
    node = slurm_node_cpus(scheduler.get('SLURM_JOB_CPUS_PER_NODE', ''), int(node_id) if node_id.isdigit() else 0)
    if node:
        limits['SLURM_JOB_CPUS_PER_NODE'] = node
    cores = min(limits.values())
    return {'limits_cores': limits, 'effective_cores': cores,
            'whole_cpu_budget': max(1, math.floor(cores))}


def process_sample():
    processes = {}
    ticks = os.sysconf('SC_CLK_TCK')
    uptime = float(read('/proc/uptime').split()[0])
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            fields = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            module = args[args.index('-m') + 1] if '-m' in args else ''
            qcenn = module.startswith('qcenn_qml.') or any(Path(a).name in
                ('run_quantum_corrections.sh', 'run_research_studies.sh') for a in args)
            processes[int(entry.name)] = {'name': read(entry / 'comm'), 'state': fields[0],
                'cpu_seconds': (int(fields[11]) + int(fields[12])) / ticks,
                'rss_bytes': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
                'threads': int(fields[17]), 'elapsed_seconds': uptime - int(fields[19]) / ticks,
                'qcenn': qcenn, 'module': module}
        except (OSError, ValueError, IndexError):
            continue
    return processes


def aer_probe(output):
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        circuit = QuantumCircuit(2, 2)
        circuit.h(0); circuit.cx(0, 1); circuit.measure(range(2), range(2))
        backend = AerSimulator()
        result = {'status': 'ok', 'devices': list(backend.available_devices()),
                  'methods': list(backend.available_methods()), 'smoke_tests': {}}
        modes = [('matrix_product_state', 'CPU')]
        if 'GPU' in result['devices']:
            modes += [('statevector', 'GPU'), ('matrix_product_state', 'GPU')]
            if 'tensor_network' in result['methods']:
                modes += [('tensor_network', 'GPU')]
        for method, device in modes:
            key = method + '/' + device
            try:
                options = dict(method=method, device=device, max_parallel_threads=1,
                               max_parallel_experiments=1, max_parallel_shots=1)
                if method == 'matrix_product_state':
                    options['mps_omp_threads'] = 1
                answer = AerSimulator(**options).run(circuit, shots=32, seed_simulator=42).result()
                counts = answer.get_counts() if answer.success else {}
                good = answer.success and sum(counts.values()) == 32 and set(counts) <= {'00', '11'}
                result['smoke_tests'][key] = {'success': bool(good), 'counts': counts,
                    'status': str(answer.status)}
            except Exception as error:
                result['smoke_tests'][key] = {'success': False, 'error': str(error)}
    except Exception as error:
        result = {'status': 'unavailable', 'error': str(error)}
    save(output, result)


def prepare_benchmark(directory):
    """Tiny actual project circuits, unrelated to scientific datasets or scores."""
    import numpy as np
    from qiskit import qpy
    from qcenn_qml.coherent import _compile_components
    from qcenn_qml.quantum_memory import memory_circuit
    from qcenn_qml.corpus_memory import denoising_memory_circuit
    rng = np.random.default_rng(42)
    keys = rng.uniform(.1, 1., (10, 16))
    circuits = [memory_circuit(keys[i], keys, np.arange(10), rounds=2) for i in (0, 1)]
    contexts = [i % 4 + ((i // 4) << 2) for i in range(16)]
    values = [(i // 4 + i) % 3 for i in range(16)]
    circuits += [denoising_memory_circuit(contexts, values, i, context_bits=4,
        value_bits=2, mask=3, original=3, steps=1, beta=1., eta=.65, amplify=1) for i in (0, 1)]
    native = [_compile_components(c, 42) for c in circuits]
    with (directory / 'benchmark.qpy').open('wb') as stream:
        qpy.dump(native, stream)
    save(directory / 'benchmark-circuits.json', [{'qubits': c.num_qubits,
        'native_cx': c.count_ops().get('cx', 0), 'depth': c.depth()} for c in native])


def benchmark_worker(spec_path):
    from qiskit import qpy
    from qcenn_qml.coherent import execute
    import resource
    spec = json.loads(Path(spec_path).read_text())
    with Path(spec['qpy']).open('rb') as stream:
        circuits = qpy.load(stream)
    # Warm-up outside the timed interval; all workers rendezvous before timing.
    execute(circuits, 64, 42, precompiled=True)
    Path(spec['ready']).touch()
    deadline = time.monotonic() + 60
    while not Path(spec['control']).exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Benchmark start barrier timed out')
        time.sleep(.02)
    control = json.loads(Path(spec['control']).read_text())
    time.sleep(max(0, control['start'] - time.monotonic()))
    batches, cpu_started = 0, time.process_time()
    while batches == 0 or time.monotonic() < control['stop']:
        counts, _, _ = execute(circuits, 64, 1000 + spec['index'] * 10000 + batches, precompiled=True)
        if any(sum(c.values()) != 64 for c in counts):
            raise RuntimeError('Benchmark measured-shot accounting failed')
        batches += 1
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save(spec['result'], {'batches': batches, 'measured_shots': batches * 64 * len(circuits),
        'finished': time.monotonic(), 'cpu_seconds': time.process_time() - cpu_started,
        'peak_rss_bytes': peak * (1 if sys.platform == 'darwin' else 1024),
        'last_counts': counts})


def worker_grid(cpus, max_workers):
    divisor = max(n for n in range(1, max_workers + 1) if cpus % n == 0)
    jobs = sorted({1, min(2, max_workers), min(4, max_workers), divisor, max_workers})
    return [(n, max(1, cpus // n)) for n in jobs]


def benchmark(directory, cpus, max_workers, seconds, environment):
    preparation = command([sys.executable, __file__, '--worker', 'prepare', '--output-dir',
        str(directory)], timeout=60, env=environment)
    if preparation['status'] != 'ok':
        return {'status': 'failed_preparation', 'details': preparation}
    rows = []
    for jobs, threads in worker_grid(cpus, max_workers):
        for repeat in range(2):
            print(f'Benchmark: {jobs} workers x {threads} threads; repeat {repeat + 1}/2', flush=True)
            case = directory / f'workers-{jobs}-threads-{threads}-repeat-{repeat}'
            case.mkdir()
            active, streams, specs = [], [], []
            try:
                for index in range(jobs):
                    prefix = case / str(index)
                    spec = {'index': index, 'qpy': str(directory / 'benchmark.qpy'),
                        'ready': str(prefix.with_suffix('.ready')), 'control': str(case / 'start.json'),
                        'result': str(prefix.with_suffix('.json'))}
                    path = prefix.with_suffix('.spec.json'); save(path, spec); specs.append(spec)
                    stream = prefix.with_suffix('.log').open('w'); streams.append(stream)
                    active.append(subprocess.Popen([sys.executable, __file__, '--worker', 'benchmark',
                        '--spec', str(path), '--output-dir', str(directory)], stdout=stream, stderr=subprocess.STDOUT,
                        env={**environment, **{k: v for k, v in limited_env(threads).items()
                            if k in ('OMP_NUM_THREADS', 'QCENN_THREADS')}}, start_new_session=True))
                deadline = time.monotonic() + 60
                while not all(Path(s['ready']).exists() for s in specs):
                    if any(p.poll() is not None for p in active) or time.monotonic() > deadline:
                        raise RuntimeError('Benchmark warm-up failed/timed out; inspect worker logs')
                    time.sleep(.05)
                start = time.monotonic() + .2
                save(case / 'start.json', {'start': start, 'stop': start + seconds})
                deadline = start + seconds + 60
                while any(p.poll() is None for p in active):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Benchmark measurement timed out')
                    time.sleep(.05)
                if any(p.returncode != 0 for p in active):
                    raise RuntimeError('Benchmark worker failed; inspect logs')
                values = [json.loads(Path(s['result']).read_text()) for s in specs]
                elapsed = max(v['finished'] for v in values) - start
                rows.append({'jobs': jobs, 'threads_per_worker': threads, 'repeat': repeat,
                    'elapsed_seconds': elapsed, 'measured_shots': sum(v['measured_shots'] for v in values),
                    'shots_per_second': sum(v['measured_shots'] for v in values) / elapsed,
                    'cpu_cores_used_mean': sum(v['cpu_seconds'] for v in values) / elapsed,
                    'sum_worker_peak_rss_bytes': sum(v['peak_rss_bytes'] for v in values)})
            except Exception as error:
                return {'status': 'failed', 'error': str(error), 'completed_trials': rows}
            finally:
                for process in active:
                    terminate(process)
                for stream in streams:
                    stream.close()
    means = []
    for jobs, threads in worker_grid(cpus, max_workers):
        scores = [r['shots_per_second'] for r in rows if r['jobs'] == jobs]
        means.append({'jobs': jobs, 'threads_per_worker': threads,
                        'shots_per_second': sum(scores) / len(scores)})
    best = max(means, key=lambda r: r['shots_per_second'])
    baseline = next(r['shots_per_second'] for r in means if r['jobs'] == 1)
    return {'status': 'complete', 'trials': rows, 'means_of_two_trials': means, 'best': best,
        'speed_ratio_vs_one_worker': best['shots_per_second'] / baseline,
        'scope': 'Small synthetic project circuits, 64 measured shots per circuit; throughput '
                 'excludes import/compilation/warm-up. Not an end-to-end speedup or accuracy benchmark.'}


def main(args):
    output, repo = Path(args.output_dir).resolve(), Path(args.repo_root).resolve()
    environment = limited_env()
    environment['PYTHONPATH'] = os.pathsep.join([str(repo / 'src'), os.environ.get('PYTHONPATH', '')])
    warnings = []
    scheduler = {k: v for k, v in os.environ.items() if k in (
        'SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE',
        'SLURM_NODEID', 'SLURM_JOB_NUM_NODES', 'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
        'PBS_JOBID', 'PBS_NP', 'LSB_JOBID', 'LSB_DJOB_NUMPROC', 'NSLOTS',
        'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'ZE_AFFINITY_MASK')}
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    locations = cgroup_locations(read('/proc/self/mountinfo'), read('/proc/self/cgroup'))
    groups = cgroup_limits(locations)
    allocation = allocation_limits(len(affinity), groups, scheduler)
    cpus = allocation['whole_cpu_budget']
    if args.cpu_budget:
        if args.cpu_budget > cpus:
            raise ValueError(f'Requested {args.cpu_budget} CPUs exceeds detected allocation limit {cpus}')
        cpus = args.cpu_budget
    meminfo = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in
        read('/proc/meminfo').splitlines() if line.split()[-1] == 'kB'}
    available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
    remaining = [max(0, r['memory_limit_bytes'] - r['memory_used_bytes']) for r in groups if 'memory_limit_bytes' in r]
    if remaining:
        available = min([available, *remaining])
    scheduler_memory = None
    if scheduler.get('SLURM_MEM_PER_NODE', '').isdigit():
        scheduler_memory = int(scheduler['SLURM_MEM_PER_NODE']) * 1024 ** 2
    elif scheduler.get('SLURM_MEM_PER_CPU', '').isdigit():
        scheduler_memory = int(scheduler['SLURM_MEM_PER_CPU']) * 1024 ** 2 * allocation['whole_cpu_budget']
    if scheduler_memory:
        available = min(available, scheduler_memory)
    if scheduler.get('SLURM_JOB_ID') and not any('memory_limit_bytes' in r for r in groups):
        warnings.append('No finite cgroup memory limit visible; verify scheduler memory allocation before large runs.')
    before = process_sample(); started = time.monotonic(); time.sleep(1)
    processes = process_sample(); elapsed = time.monotonic() - started
    for pid, row in processes.items():
        row['pid'] = pid
        row['cpu_cores_recent'] = max(0., (row['cpu_seconds'] - before[pid]['cpu_seconds']) / elapsed) if pid in before else None
    relevant = [r for r in processes.values() if r['qcenn']]
    per_worker = max(args.memory_per_worker_gib * GIB, max([r['rss_bytes'] * 1.25 for r in relevant] or [0]))
    memory_workers = max(1, int(.8 * available // per_worker))
    if available < per_worker:
        warnings.append('Available memory is below the per-worker estimate; even one full worker may not fit.')
    max_workers = min(cpus, memory_workers, 18)
    print(f'Detected {cpus} usable whole CPUs; available memory approximately {available/GIB:.2f} GiB.', flush=True)
    probe_path = output / 'aer.json'
    probe_command = command([sys.executable, __file__, '--worker', 'aer', '--output-dir', str(output)],
                            timeout=60, env=environment)
    aer = json.loads(probe_path.read_text()) if probe_path.exists() else {'status': 'probe_failed', 'details': probe_command}
    supported = aer.get('smoke_tests', {}).get('matrix_product_state/CPU', {}).get('success', False)
    if not supported:
        warnings.append('Aer CPU MPS smoke test failed or dependencies are missing; inspect aer.json/report.json.')
    if relevant:
        warnings.append('Existing QCeNN processes found. Recommendations are a total budget for a NEW allocation; do not add them on top of active jobs.')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'host': platform.node(),
        'platform': platform.platform(), 'python': sys.executable, 'repo_root': str(repo),
        'cpu': {**allocation, 'selected_budget': cpus, 'host_logical_cpus': os.cpu_count(),
            'affinity_cpus': affinity, 'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None,
            'lscpu': command(['lscpu']), 'numa': command(['numactl', '--hardware'])},
        'scheduler': scheduler, 'cgroups': groups,
        'thread_environment': {key: os.environ[key] for key in (
            'QCENN_CPUS', 'QCENN_JOBS', 'QCENN_THREADS', 'OMP_NUM_THREADS', 'OMP_PROC_BIND',
            'OMP_PLACES', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'QISKIT_PARALLEL', 'QISKIT_NUM_PROCS') if key in os.environ},
        'memory': {'host_meminfo_bytes': meminfo, 'available_within_visible_limits_bytes': available,
                   'scheduler_memory_budget_bytes': scheduler_memory,
                   'assumed_worker_bytes': per_worker, 'worker_limit_with_20_percent_reserve': memory_workers},
        'storage': command(['df', '-hP', str(repo), str(output), '/tmp']),
        'pressure': {name: read('/proc/pressure/' + name) for name in ('cpu', 'memory', 'io')},
        'gpu': {'nvidia': command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv']),
                'rocm': command(['rocm-smi', '--showproductname', '--showuse', '--showmeminfo', 'vram']),
                'sycl': command(['sycl-ls'])}, 'aer': aer,
        'packages': {}, 'tools': {name: shutil.which(name) for name in ('zip', 'unzip', 'srun', 'sbatch', 'mpirun', 'numactl', 'nvcc')},
        'qcenn_processes': sorted(relevant, key=lambda r: r['pid']),
        'top_user_cpu_processes': sorted(processes.values(), key=lambda r: r['cpu_cores_recent'] or 0, reverse=True)[:12],
        'source_commit': command(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
        'warnings': warnings, 'aer_documentation': DOC}
    for name in ('qcenn-qml', 'qiskit', 'qiskit-aer', 'qiskit-aer-gpu', 'numpy', 'scipy', 'torch', 'mpi4py'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    tuning = {'status': 'not_requested'}
    if args.benchmark:
        if relevant:
            tuning = {'status': 'skipped_busy', 'reason': 'Other QCeNN jobs are present; keep timing uncontaminated.'}
        elif not supported:
            tuning = {'status': 'skipped_missing_aer'}
        else:
            tuning = benchmark(output, cpus, max_workers, args.benchmark_seconds, environment)
    result['benchmark'] = tuning
    jobs = tuning['best']['jobs'] if tuning['status'] == 'complete' else max_workers
    result['recommendation'] = {'cpu_budget': cpus, 'mnist_jobs': min(jobs, 9),
        'diffusion_jobs': min(jobs, 18), 'basis': 'measured_small_circuits' if tuning['status'] == 'complete' else 'untuned_memory_limited_estimate',
        'existing_runner_outer_jobs': 1, 'current_model_backend': 'CPU matrix_product_state',
        'notes': [
            'Current Aer MPS studies use CPU. GPU availability does not move them to a GPU.',
            'GPU statevector/tensor_network would require a separately validated backend; never use a dense statevector for the 241-qubit toy block.',
            'Use independent workers first; a single MPS circuit may not scale with OpenMP thread count.',
            'Current all-track runner splits resources statically. Outer QCENN_JOBS=1 gives each track the entire budget in sequence, but does not apply the memory-limited worker recommendation.',
            'Direct study commands below can use measured worker counts and run their own configurations/seeds in parallel.',
            'Available resources and estimates are a snapshot; NUMA, shared quotas and concurrent users can affect throughput.',
            'Worker memory can grow sharply with circuit entanglement; synthetic timings do not certify larger workloads.',
            'Fit-before-validation and frozen validation-before-test dependencies must stay sequential.',
            'These models load raw records and execute measured circuits; this diagnostic does not implement neural-weight training.'
        ]}
    env_values = {'QCENN_CPUS': str(cpus), 'QCENN_JOBS': '1', 'QCENN_PYTHON': sys.executable,
        'QCENN_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'QISKIT_PARALLEL': 'FALSE', 'QISKIT_NUM_PROCS': '1'}
    (output / 'recommended.env').write_text('# Total budget for a new run; do not add to an existing allocation.\n' +
        '\n'.join(f'export {key}={shlex.quote(value)}' for key, value in env_values.items()) + '\n')
    commands = ['# Inspect before use. Run only one of these in the allocation at a time.',
        '# New output directories; no scientific settings or existing checkpoints are modified.',
        'cd -- ' + shlex.quote(str(repo)), 'source ' + shlex.quote(str(output / 'recommended.env')),
        '# Direct studies with tuned/estimated worker counts (these commands do not create ZIPs):']
    for name, module, cap in [('mnist', 'qcenn_qml.mnist_study', 9), ('diffusion', 'qcenn_qml.diffusion.study', 18)]:
        commands.append(f'"$QCENN_PYTHON" -u -m {module} --jobs {min(jobs, cap)} --cpu-budget "$QCENN_CPUS" '
            f'--output-dir "results/{name}-system-tuned-$(date +%Y%m%d-%H%M%S)-$$"')
    (output / 'commands.txt').write_text('\n'.join(commands) + '\n')
    save(output / 'report.json', result)
    lines = ['QCeNN-QML SYSTEM REPORT', f"Host: {result['host']}", f"UTC: {result['created_utc']}",
        f'Python: {sys.executable}', f'CPU limits: {allocation["limits_cores"]}',
        f'Budget: {cpus} whole CPUs; host logical CPUs: {os.cpu_count()}',
        f'Available memory within visible limits: {available/GIB:.2f} GiB',
        f'Estimated worker memory: {per_worker/GIB:.2f} GiB; 20% memory reserve',
        f'Aer: {aer.get("status")}; devices: {aer.get("devices", [])}; CPU MPS passed: {supported}',
        'Current model backend: CPU MPS. A detected GPU does not accelerate this path automatically.',
        f'Existing QCeNN processes: {len(relevant)}', 'PID       CORES(last 1s)  RSS GiB  THREADS  MODULE']
    for row in relevant:
        cores = '?' if row['cpu_cores_recent'] is None else f'{row["cpu_cores_recent"]:.2f}'
        lines.append(f'{row["pid"]:<9} {cores:>12} {row["rss_bytes"]/GIB:>9.2f} {row["threads"]:>8}  {row["module"] or row["name"]}')
    lines += [f'Benchmark: {tuning["status"]}']
    if tuning['status'] == 'complete':
        for row in tuning['means_of_two_trials']:
            lines.append(f'  {row["jobs"]} workers x {row["threads_per_worker"]} threads: {row["shots_per_second"]:.1f} measured shots/s')
        lines += [f'Small-circuit timing ratio vs one worker: {tuning["speed_ratio_vs_one_worker"]:.2f}x', tuning['scope']]
    lines += [f'Suggested direct study jobs: MNIST={min(jobs, 9)}, diffusion={min(jobs, 18)}',
        f'Recommendation basis: {result["recommendation"]["basis"]}',
        'Existing runner: QCENN_JOBS=1 assigns the whole budget to one track at a time.',
        'GPU inventory: ' + result['gpu']['nvidia'].get('output', result['gpu']['nvidia']['status']).strip(),
        *['NOTE: ' + text for text in warnings], *result['recommendation']['notes'],
        'See commands.txt and recommended.env. No experiments were started or stopped.']
    (output / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--cpu-budget', type=int, default=0)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark-seconds', type=int, default=3)
    parser.add_argument('--memory-per-worker-gib', type=float, default=4)
    parser.add_argument('--worker', choices=('aer', 'prepare', 'benchmark'))
    parser.add_argument('--spec')
    args = parser.parse_args()
    if not math.isfinite(args.memory_per_worker_gib) or args.memory_per_worker_gib <= 0:
        parser.error('memory-per-worker-gib must be finite and positive')
    if args.worker == 'aer':
        aer_probe(Path(args.output_dir) / 'aer.json')
    elif args.worker == 'prepare':
        prepare_benchmark(Path(args.output_dir))
    elif args.worker == 'benchmark':
        benchmark_worker(args.spec)
    else:
        main(args)
QCENN_PYTHON
qcenn_check_args=(--repo-root "$qcenn_check_repo" --output-dir "$qcenn_check_output"
  --cpu-budget "$qcenn_check_cpus" --benchmark-seconds "$qcenn_check_seconds"
  --memory-per-worker-gib "$qcenn_check_worker_gib")
if ((qcenn_check_benchmark)); then qcenn_check_args+=(--benchmark); fi
"$qcenn_check_python" "$qcenn_check_output/system_probe.py" "${qcenn_check_args[@]}"
if command -v zip >/dev/null && command -v sha256sum >/dev/null; then
  (
    cd -- "$(dirname -- "$qcenn_check_output")"
    qcenn_check_name=$(basename -- "$qcenn_check_output")
    zip -qr "$qcenn_check_name.zip" "$qcenn_check_name"
    sha256sum "$qcenn_check_name.zip" > "$qcenn_check_name.zip.sha256"
  )
  printf '\nShare this report: %s.zip\n' "$qcenn_check_output"
else printf '\nZIP unavailable (native zip/sha256sum missing). Report: %s/report.txt\n' "$qcenn_check_output"; fi
