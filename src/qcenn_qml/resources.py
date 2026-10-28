"""Host resource discovery and kernel-released leases, outside the model core."""
from contextlib import contextmanager
import fcntl
from functools import wraps
import inspect
import json
import math
import os
from pathlib import Path
import re
import time


def text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return ''


def thread_environment(threads=1):
    """Set before importing Qiskit/NumPy, including Qiskit's Rust Rayon pool."""
    return {'QCENN_THREADS': str(threads), 'OMP_NUM_THREADS': str(threads),
        'OMP_MAX_ACTIVE_LEVELS': '1', 'RAYON_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'BLIS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
        'QISKIT_PARALLEL': 'FALSE', 'QISKIT_NUM_PROCS': '1'}


def cgroup_directories():
    groups = [line.split(':', 2) for line in text('/proc/self/cgroup').splitlines()]
    paths = set()
    for line in text('/proc/self/mountinfo').splitlines():
        if ' - ' not in line:
            continue
        left, right = line.split(' - ', 1)
        fields, fs = left.split(), right.split()
        if len(fields) < 5 or len(fs) < 3 or fs[0] not in ('cgroup', 'cgroup2'):
            continue
        decode = lambda s: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), s)
        root, mount = Path(decode(fields[3])), Path(decode(fields[4]))
        for group in groups:
            if len(group) != 3:
                continue
            _, controllers, membership = group
            if ((fs[0] == 'cgroup2' and controllers) or
                    (fs[0] == 'cgroup' and not set(controllers.split(',')).intersection(fs[2].split(',')))):
                continue
            try:
                relative = Path(membership).relative_to(root)
            except ValueError:
                relative = Path(membership.lstrip('/'))
            candidate = mount / relative
            if '..' in candidate.parts or not candidate.exists():
                candidate = mount
            for directory in (candidate, *candidate.parents):
                if directory != mount and mount not in directory.parents:
                    break
                paths.add(directory)
                if directory == mount:
                    break
    return paths


def detect_resources():
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    limits = {'affinity': len(affinity)}
    memory = {}
    for line in text('/proc/meminfo').splitlines():
        words = line.split()
        if len(words) == 3 and words[-1] == 'kB':
            memory[words[0].rstrip(':')] = int(words[1]) * 1024
    available = memory.get('MemAvailable', memory.get('MemFree', 0))
    for path in cgroup_directories():
        quota = text(path / 'cpu.max').split()
        if len(quota) == 2 and quota[0] != 'max':
            limits[str(path)] = int(quota[0]) / int(quota[1])
        q, period = text(path / 'cpu.cfs_quota_us'), text(path / 'cpu.cfs_period_us')
        if q and period and int(q) > 0:
            limits[str(path)] = int(q) / int(period)
        for cap, used in [('memory.max', 'memory.current'), ('memory.limit_in_bytes', 'memory.usage_in_bytes')]:
            value = text(path / cap)
            if value.isdigit() and int(value) < 1 << 60:
                available = min(available, max(0, int(value) - int(text(path / used) or 0)))
    for key in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'NSLOTS', 'PBS_NP', 'LSB_DJOB_NUMPROC'):
        value = os.environ.get(key, '')
        if value.isdigit() and int(value) > 0:
            limits[key] = int(value)
    node_id = int(os.environ.get('SLURM_NODEID', '0'))
    for item in os.environ.get('SLURM_JOB_CPUS_PER_NODE', '').split(','):
        match = re.fullmatch(r'(\d+)(?:\(x(\d+)\))?', item)
        if match:
            repeat = int(match[2] or 1)
            if node_id < repeat:
                limits['SLURM_JOB_CPUS_PER_NODE'] = int(match[1])
                break
            node_id -= repeat
    cpus = max(1, math.floor(min(limits.values())))
    for key, multiplier in [('SLURM_MEM_PER_NODE', 1), ('SLURM_MEM_PER_CPU', cpus)]:
        value = os.environ.get(key, '')
        if value.isdigit() and int(value):
            available = min(available, int(value) * multiplier * 1024 ** 2)
    physical = set()
    for cpu in affinity:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        package, core = text(base / 'physical_package_id'), text(base / 'core_id')
        physical.add((package, core) if package and core else ('cpu', str(cpu)))
    return {'cpu_budget': cpus, 'cpu_limits': limits, 'affinity_cpus': affinity,
        'visible_physical_cores': len(physical), 'available_memory_bytes': available,
        'host_memory_bytes': memory.get('MemTotal'), 'detected_at': time.time()}


class Lease:
    def __init__(self, descriptors, slots):
        self.descriptors, self.slots = descriptors, slots

    def close(self):
        for descriptor in self.descriptors:
            os.close(descriptor)
        self.descriptors = []


def acquire_files(paths, count=1):
    """Nonblocking OS locks. A leftover filename never means a stale lock."""
    acquired, slots = [], []
    for path in paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        acquired.append(fd); slots.append(path.name)
        if len(acquired) == count:
            return Lease(acquired, slots)
    for fd in acquired:
        os.close(fd)
    return None


class ResourcePool:
    def __init__(self, directory, slots):
        self.directory, self.slots = Path(directory), int(slots)
        if self.slots < 1:
            raise ValueError('Pool must have a positive CPU budget')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cursor = os.getpid() % self.slots

    def acquire(self, count=1):
        order = [(self.cursor + i) % self.slots for i in range(self.slots)]
        self.cursor = (self.cursor + 1) % self.slots
        return acquire_files([self.directory / f'cpu-{i:05d}.lock' for i in order], count)


@contextmanager
def exclusive_directory(directory):
    """One coordinator per output; the lock is automatically released on exit."""
    path = Path(directory).resolve()
    lock = acquire_files([path.with_name(path.name + '.writer.lock')])
    if lock is None:
        raise RuntimeError(f'Output has an active writer: {path}; use --status or --stop, not a second writer')
    try:
        os.ftruncate(lock.descriptors[0], 0)
        os.write(lock.descriptors[0], json.dumps({'pid': os.getpid(), 'started_at': time.time()}).encode())
        yield lock
    finally:
        lock.close()


def process_cpu_seconds(pid):
    try:
        fields = text(f'/proc/{pid}/stat').rsplit(') ', 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
    except (IndexError, ValueError):
        return None


def tree_cpu_seconds(proc_pid):
    """Use procfs-visible IDs, including a live model/GPU helper child."""
    total = 0.
    pending = [int(proc_pid)]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        total += process_cpu_seconds(current) or 0.
        children = text(f'/proc/{current}/task/{current}/children')
        pending.extend(int(value) for value in children.split())
    return total


def single_writer(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        with exclusive_directory(bound.arguments['output_dir']):
            return function(*args, **kwargs)
    return guarded


def execution_progress(phase, **details):
    target = os.environ.get('QCENN_PROGRESS_FILE')
    if target:
        # One model call at a time per worker; heartbeat is a different file.
        previous = json.loads(text(target) or '{}')
        row = {**previous, 'phase': phase, 'updated_at': time.time(), **details}
        if phase == 'measured':
            row['completed_measurements'] = previous.get('completed_measurements', 0) + 1
        path = Path(target)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(row))
        temporary.replace(path)


def inherited_run_descriptors():
    value = os.environ.get('QCENN_RUN_LOCK_FD')
    if value:
        try:
            fd = int(value)
            fcntl.fcntl(fd, fcntl.F_GETFD)
            return [fd]
        except (OSError, ValueError):
            pass
    return []


def inherited_worker_descriptors():
    descriptors = set(inherited_run_descriptors())
    for value in os.environ.get('QCENN_INHERITED_FDS', '').split(','):
        try:
            fd = int(value)
            fcntl.fcntl(fd, fcntl.F_GETFD)
            descriptors.add(fd)
        except (OSError, ValueError):
            pass
    return sorted(descriptors)
