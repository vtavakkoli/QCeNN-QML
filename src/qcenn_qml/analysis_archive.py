"""Compact, checksummed analysis exports; original run files are never removed.

Keep scientific JSON/counts, selected raw memory records and bounded diagnostics.
Large QPY, pretrained weights, datasets and transient progress stay on the host.
Only stdlib is imported: making an archive never imports a model or a backend.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time


LOG_TAIL_BYTES = 32 * 1024
LOG_BUDGET_BYTES = 128 * 1024  # Per experiment, including failed workers.
AUDIT_LIMIT_BYTES = 1024 * 1024
TEXT_SUFFIXES = {'.json', '.jsonl', '.csv', '.tsv', '.md', '.txt', '.yaml', '.yml', '.toml', '.cff', '.sha256'}
SKIP_DIRECTORIES = {'previous-attempts', '__pycache__', 'weights', 'checkpoints', 'datasets', 'MNIST', '.git', 'node_modules'}
ROOT_METADATA = ('runtime.json', 'study-manifest.json', 'requirements-versions.txt',
    'analysis-execution.json', 'run-state.json', 'STATUS.txt', 'runner-error.txt')
README = '''# QCeNN-QML model-analysis archive

QCeNN-QML — Vahid Tavakkoli (2026), https://github.com/vtavakkoli/QCeNN-QML.
Retain attribution and cite the software in public research. The repository's
LICENSE, NOTICE.md and CITATION.cff define the applicable rights and citation.

Reports, measured model counts, predictions/targets, split IDs, seeds, frozen
settings, resource metrics, task receipts and selected raw memory records are
retained. Small fixed model definitions inside JSON reports are retained too.
Diagnostics contain bounded log tails, not complete logs. At most one small
auxiliary full-readout JSON audit per experiment is included.

Large QPY/circuit binaries, pretrained weights, public dataset caches, raw test
image caches, old attempt directories and transient progress are omitted.
Original files remain in the source run directory on the execution machine.
This is an analysis export, not a full checkpoint or standalone circuit replay.

archive-manifest.json lists every included payload's SHA256, omissions and their
reasons. Large omitted files are not read or hashed during export. ZIP CRC and
the external .zip.sha256 protect the finished archive, including this index.
In the combined export, analysis-execution.json contains finalized experiment
states. Per-experiment status snapshots may show their ZIP being built.
Scientific failures/blocked gates remain recorded; an archive is not a success
claim for accuracy, pretrained Smol replacement, QPU execution or advantage.
'''


def _plain_name(value):
    if not value or Path(value).name != value or value in ('.', '..') or any(ord(c) < 32 for c in value):
        raise ValueError('Experiment must be a plain directory name')
    return value


def _walk(directory, root, omitted):
    if directory.is_symlink():
        omitted.append({'path': directory.relative_to(root).as_posix(), 'reason': 'symlink'})
        return
    if not directory.exists():
        return
    for current, directories, files in os.walk(directory, followlinks=False):
        current = Path(current)
        retained = []
        for name in sorted(directories):
            path = current / name
            if path.is_symlink() or name in SKIP_DIRECTORIES or name.startswith('.venv') or '.partial-' in name:
                omitted.append({'path': path.relative_to(root).as_posix(),
                    'reason': 'symlink' if path.is_symlink() else 'non_analysis_directory',
                    'directory_not_scanned': True})
            else:
                retained.append(name)
        directories[:] = retained
        for name in sorted(files):
            yield current / name


def _is_log(path):
    return path.suffix == '.log' or 'logs' in path.parts or path.name in ('runner-error.txt', 'stdout.txt', 'stderr.txt')


def _copy_and_hash(source, target):
    """Copy one complete file snapshot, hashing the bytes actually exported."""
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    with source.open('rb') as incoming, target.open('wb') as outgoing:
        before = os.fstat(incoming.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Analysis payload is not a regular file')
        for block in iter(lambda: incoming.read(1024 * 1024), b''):
            outgoing.write(block); checksum.update(block)
        after = os.fstat(incoming.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Analysis payload changed while exporting: ' + str(source))
    return {'bytes': after.st_size, 'sha256': checksum.hexdigest()}


def _select(path, size, audit_kept, group):
    name = path.name.lower()
    if size > AUDIT_LIMIT_BYTES and re.search(r'(?:^|[-_.])(weights?|checkpoint|optimizer)(?:[-_.]|$)', name):
        return 'large_weight_or_optimizer_payload'
    if name.endswith('.json.gz') and name.startswith('full-readout-first-'):
        if size <= AUDIT_LIMIT_BYTES and not audit_kept[group]:
            audit_kept[group] = True
            return None
        return 'redundant_or_large_auxiliary_readout_audit'
    if name in ('progress.json', 'execution-progress.json') or name.endswith('.heartbeat.json'):
        return 'transient_progress'
    if path.suffix.lower() == '.npz':
        if re.fullmatch(r'data-seed-\d+\.npz|prepared(?:[-_].*)?\.npz|memory-records(?:[-_].*)?\.npz', name):
            return None  # Raw examples defining this nonparametric model matter.
        return 'dataset_cache_or_weight_checkpoint'
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in ('LICENSE', 'NOTICE'):
        return None
    return 'circuit_weight_or_other_binary'


def stage_analysis(root, stage, experiment=None):
    """Build a small tree without mutating, pruning or deleting the original run."""
    root, stage = Path(root).resolve(), Path(stage)
    experiment_root = root / 'experiments'
    if experiment_root.is_symlink() or not experiment_root.is_dir():
        raise ValueError('Expected a real experiments directory under the run root')
    if experiment is not None:
        names = [_plain_name(experiment)]
        if not (experiment_root / experiment).is_dir():
            raise FileNotFoundError('Missing requested experiment: ' + experiment)
    else:
        names = sorted(p.name for p in experiment_root.iterdir() if p.is_dir()
            and '.partial-' not in p.name and p.name not in SKIP_DIRECTORIES)
    omitted, included, candidates = [], [], []
    excluded_tops = {folder for folder in ('short-tasks', 'text-tasks', 'audit-tasks', 'status', 'commands', 'logs')
        if (root / folder).is_symlink()}
    omitted.extend({'path': folder, 'reason': 'symlink', 'directory_not_scanned': True} for folder in sorted(excluded_tops))
    for name in names:
        candidates.extend((p, name) for p in _walk(experiment_root / name, root, omitted))
        # Short module workers retain task receipts/audits outside experiments/.
        for folder in ('short-tasks', 'text-tasks', 'audit-tasks'):
            if folder not in excluded_tops:
                candidates.extend((p, name) for p in _walk(root / folder / name, root, omitted))
        for folder, extension in (('status', '.json'), ('commands', '.txt'), ('logs', '.log')):
            path = root / folder / (name + extension)
            if folder not in excluded_tops and (path.exists() or path.is_symlink()):
                candidates.append((path, name))
        error_log = root / 'logs' / (name + '-archive.log')
        if 'logs' not in excluded_tops and error_log.exists():
            candidates.append((error_log, name))
    candidates.extend((root / name, '_run') for name in ROOT_METADATA if (root / name).exists()
        and (experiment is None or name != 'analysis-execution.json'))
    # Prioritize top-level logs before failed worker tails. Successful worker
    # chatter is redundant with its machine-readable completion and result.
    candidates.sort(key=lambda item: (len(item[0].relative_to(root).parts), item[0].as_posix()))
    audit_kept, log_used, seen = defaultdict(bool), defaultdict(int), set()
    final_execution = root / 'analysis-execution.json'
    has_final_execution = final_execution.is_file() and not final_execution.is_symlink()
    for path, group in candidates:
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        info = path.lstat()
        reason = None
        if any(ord(c) < 32 for c in relative):
            reason = 'unsupported_control_character'
        elif stat.S_ISLNK(info.st_mode):
            reason = 'symlink'
        elif not stat.S_ISREG(info.st_mode):
            reason = 'not_a_regular_file'
        if reason:
            omitted.append({'path': relative, 'bytes': info.st_size, 'reason': reason}); continue
        if _is_log(Path(relative)):
            if path.name == 'worker.log' and (path.parent / 'completion.json').exists():
                omitted.append({'path': relative, 'bytes': info.st_size, 'reason': 'completed_worker_log'}); continue
            budget = min(LOG_TAIL_BYTES, LOG_BUDGET_BYTES-log_used[group])
            if budget <= 0 or info.st_size == 0:
                omitted.append({'path': relative, 'bytes': info.st_size, 'reason': 'log_budget_or_empty'}); continue
            with path.open('rb') as stream:
                stream.seek(max(0, info.st_size-budget))
                tail = stream.read(min(budget, info.st_size))
            source_tail_bytes = len(tail)
            tail = tail.decode('utf8', errors='replace').encode('utf8')
            if len(tail) > budget:
                tail = tail[-budget:].decode('utf8', errors='ignore').encode('utf8')
            destination = 'diagnostics/' + relative + '.tail.txt'
            target = stage / destination; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tail); log_used[group] += len(tail)
            included.append({'path': destination, 'source_path': relative, 'bytes': len(tail),
                'source_bytes': info.st_size, 'source_tail_bytes': source_tail_bytes,
                'truncated': source_tail_bytes < info.st_size, 'encoding': 'utf8_replacement',
                'sha256': hashlib.sha256(tail).hexdigest(), 'kind': 'log_tail'})
            continue
        if experiment is None and has_final_execution and relative in (
                'run-state.json', 'STATUS.txt', 'experiments/summary/execution.json'):
            reason = 'superseded_by_final_analysis_execution'
        else:
            reason = _select(path, info.st_size, audit_kept, group)
        if reason:
            omitted.append({'path': relative, 'bytes': info.st_size, 'reason': reason}); continue
        included.append({'path': relative, **_copy_and_hash(path, stage / relative), 'kind': 'science_or_metadata'})
    # Verify the task receipts against the exported bytes, never stale paths.
    exported = {item['path']: item for item in included}
    for item in included:
        if Path(item['path']).name == 'completion.json':
            receipt = json.loads((stage / item['path']).read_text())
            if 'result_sha256' in receipt:
                result = (Path(item['path']).parent / 'result.json').as_posix()
                if result not in exported or exported[result]['sha256'] != receipt['result_sha256']:
                    raise ValueError('Task completion/result checksum mismatch: ' + item['path'])
    (stage / 'ANALYSIS_README.md').write_text(README)
    included.append({'path': 'ANALYSIS_README.md', 'bytes': len(README.encode()),
        'sha256': hashlib.sha256(README.encode()).hexdigest(), 'kind': 'archive_documentation'})
    manifest = {'format': 'qcenn-qml-analysis-v1', 'experiment_names': names,
        'source_run': str(root), 'created_at_unix': time.time(),
        'original_files_retained_on_host': True, 'full_replay_checkpoint': False,
        'policy': {'log_tail_bytes': LOG_TAIL_BYTES, 'log_budget_bytes_per_experiment': LOG_BUDGET_BYTES,
            'auxiliary_audits_per_experiment': 1, 'auxiliary_audit_max_bytes': AUDIT_LIMIT_BYTES,
            'large_omitted_files_hashed': False},
        'included': included, 'omitted': omitted,
        'included_payload_bytes': sum(item['bytes'] for item in included),
        'known_omitted_file_bytes': sum(item.get('bytes', 0) for item in omitted)}
    (stage / 'archive-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def _build_locked(root, output, experiment):
    partial = output.with_name(output.stem + '.partial.zip')
    if partial.exists():
        partial.rename(partial.with_name(partial.stem + f'.abandoned-{time.time_ns()}.zip'))
    with tempfile.TemporaryDirectory(prefix='.analysis-stage-', dir=output.parent) as folder:
        stage = Path(folder)
        manifest = stage_analysis(root, stage, experiment)
        subprocess.run(['zip', '-6', '-q', '-r', str(partial), *sorted(p.name for p in stage.iterdir())],
            cwd=stage, check=True)
        subprocess.run(['unzip', '-tqq', str(partial)], check=True)
        os.replace(partial, output)
    checksum_file = output.with_suffix('.zip.sha256')
    temporary = checksum_file.with_suffix('.tmp')
    with temporary.open('w') as stream:
        subprocess.run(['sha256sum', output.name], cwd=output.parent, stdout=stream, check=True)
    temporary.replace(checksum_file)
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size,
        'included_files': len(manifest['included']), 'omitted_entries': len(manifest['omitted']),
        'policy': 'model-analysis; original full run retained'}), flush=True)
    return manifest


def build(root, output, experiment=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.suffix != '.zip':
        raise ValueError('Analysis output must end in .zip')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.zip.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another exporter is writing this archive') from error
        return _build_locked(root, output, experiment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--experiment')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    build(args.run_root, args.output, args.experiment)


if __name__ == '__main__':
    main()
