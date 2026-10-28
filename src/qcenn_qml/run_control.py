"""Read status and stop one identified experiment tree; never name-based pkill."""
import json
import os
from pathlib import Path
import signal
import time

from .resources import text


def identity(proc_pid=None):
    base = Path('/proc/self') if proc_pid is None else Path('/proc') / str(proc_pid)
    try:
        fields = (base / 'stat').read_text().rsplit(') ', 1)[1].split()
        visible_pid = int((base / 'stat').read_text().split()[0])
        status = (base / 'status').read_text().splitlines()
        namespace = next((line.split()[1:] for line in status if line.startswith('NSpid:')), [str(visible_pid)])
        return {'pid': int(namespace[-1]), 'proc_pid': visible_pid, 'start_ticks': int(fields[19]),
            'parent_proc_pid': int(fields[1]), 'uid': base.stat().st_uid, 'state': fields[0],
            'args': (base / 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1],
            'cwd': str((base / 'cwd').resolve())}
    except (OSError, ValueError, IndexError):
        return None


def alive(record):
    current = identity(record.get('proc_pid', record['pid']))
    return bool(current and current['start_ticks'] == record['start_ticks'] and
        current['pid'] == record['pid'] and current['uid'] == record['uid'] == os.getuid() and current['state'] != 'Z')


def run_identity(directory):
    root = Path(directory).resolve()
    modern = root / 'runner.identity.json'
    if modern.exists():
        record = json.loads(modern.read_text())
        if record.get('output_dir') != str(root):
            raise ValueError('Runner identity belongs to another output directory')
        current = identity(record.get('proc_pid', record['pid']))
        if current and current['start_ticks'] == record['start_ticks']:
            args = current['args']
            if '-m' not in args or args[args.index('-m') + 1] != 'qcenn_qml.research_runner':
                raise RuntimeError('Recorded runner is not a research runner; no signal was sent')
        return record
    marker = root / 'runner.pid'
    if not marker.exists():
        return None
    pid = int(marker.read_text().strip())
    # Old runs have no start-tick receipt: validate owner, script, cwd and
    # process creation time against the atomic runner.pid file instead.
    boot = next((int(line.split()[1]) for line in text('/proc/stat').splitlines()
                 if line.startswith('btime ')), None)
    candidates = [identity(pid)]
    if not candidates[0] or candidates[0]['pid'] != pid:
        candidates = [identity(p.name) for p in Path('/proc').glob('[0-9]*')]
    for record in candidates:
        if not record or record['pid'] != pid or record['uid'] != os.getuid():
            continue
        if not any(Path(arg).name in ('run_quantum_corrections.sh', 'run_research_studies.sh') for arg in record['args']):
            continue
        if Path(record['cwd']) not in root.parents:
            continue
        started = boot + record['start_ticks'] / os.sysconf('SC_CLK_TCK') if boot else None
        if started is not None and abs(marker.stat().st_mtime - started) <= 60:
            return {**record, 'output_dir': str(root), 'legacy_identity_verified': True}
    # A missing/reused PID is not permission to signal the new owner.
    if identity(pid) and identity(pid)['state'] != 'Z':
        raise RuntimeError('Old runner PID exists but its identity could not be verified; no signal was sent')
    return None


def owned_processes(directory):
    root = Path(directory).resolve()
    self_id = identity()['proc_pid']
    records = [r for p in Path('/proc').glob('[0-9]*') if (r := identity(p.name)) and r['uid'] == os.getuid() and r['proc_pid'] != self_id]
    selected = {}
    for record in records:
        args = record['args']
        module = args[args.index('-m') + 1] if '-m' in args and args.index('-m') + 1 < len(args) else ''
        recognized = module.startswith('qcenn_qml.') or any(Path(a).name == 'gpu_worker.py' for a in args)
        if 'qcenn-archive' in args and str(root) in args:
            selected[record['proc_pid']] = record
        if record['cwd'] == str(root) and any(Path(a).name in ('zip', 'unzip', 'sha256sum') for a in args[:1]):
            selected[record['proc_pid']] = record
        if not recognized:
            continue
        for flag in ('--task', '--output-dir', '--run-root', '--short-spec'):
            if flag in args and args.index(flag) + 1 < len(args):
                path = Path(args[args.index(flag) + 1]).resolve()
                if path == root or root in path.parents:
                    selected[record['proc_pid']] = record
    try:
        runner = run_identity(root)
    except RuntimeError:
        runner = None
    if runner and runner['proc_pid'] != self_id and alive(runner):
        selected[runner['proc_pid']] = runner
    changed = True
    while changed:
        changed = False
        for record in records:
            if record['parent_proc_pid'] in selected and record['proc_pid'] not in selected:
                selected[record['proc_pid']] = record; changed = True
    return list(selected.values())


def stop_run(directory, grace=90):
    root = Path(directory).resolve()
    warning = None
    try:
        runner = run_identity(root)
    except RuntimeError as error:
        runner = None; warning = str(error)
    processes = owned_processes(root)
    if not processes:
        return {'status': 'ALREADY_STOPPED', 'output_dir': str(root), 'signals': [], 'identity_note': warning}
    signals = []
    def send(record, sig):
        if alive(record):
            try:
                os.kill(record['pid'], sig)
                signals.append({'pid': record['pid'], 'start_ticks': record['start_ticks'], 'signal': sig})
            except ProcessLookupError:
                pass
    if runner:
        send(runner, signal.SIGTERM)
    else:
        for record in processes:
            send(record, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while any(alive(r) for r in processes) and time.monotonic() < deadline:
        time.sleep(.2)
    # Escalate only the same captured, still-owned identities, including any
    # remaining orphans whose explicit task/output paths belong to this run.
    remaining = {r['proc_pid']: r for r in [*processes, *owned_processes(root)]}
    for record in remaining.values():
        if alive(record):
            send(record, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while any(alive(r) for r in remaining.values()) and time.monotonic() < deadline:
        time.sleep(.1)
    for record in remaining.values():
        send(record, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while any(alive(r) for r in remaining.values()) and time.monotonic() < deadline:
        time.sleep(.1)
    if any(alive(r) for r in remaining.values()):
        raise RuntimeError('Identified workers remain alive after stop; a new run was not started')
    result = {'status': 'STOPPED', 'output_dir': str(root), 'ended_at': time.time(), 'signals': signals}
    (root / 'STOP_RECEIPT.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
