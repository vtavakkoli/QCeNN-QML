"""GPU setup isolation and CPU fallback, without requiring a CUDA machine."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('failure', ['none', 'create', 'install', 'check', 'freeze'])
def test_gpu_setup_failures_keep_cpu_handoff_available(tmp_path, failure):
    script = (Path(__file__).parents[1] / 'scripts/update_and_run_research.sh').read_text()
    start = script.index('  qcenn_gpu_python=')
    end = script.index('  if [[ -f $qcenn_config/gpu-ready.txt ]]; then', start)
    setup = script[start:end]
    worktree = tmp_path / 'worktree'
    binary = worktree / '.venv-qcenn-gpu/bin/python'
    binary.parent.mkdir(parents=True)
    binary.write_text('''#!/usr/bin/env bash
set -eu
[[ $1 == -I ]] || exit 90
shift
[[ $1 == -m && $2 == pip ]] || exit 91
stage=$3
printf '%s\\n' "$stage" >> "$QCENN_TEST_CALLS"
[[ $stage != "$QCENN_TEST_FAILURE" ]] || exit 19
printf 'GPU pip %s succeeded\\n' "$stage"
''')
    binary.chmod(0o755)
    creator = tmp_path / 'base-python'
    creator.write_text('''#!/usr/bin/env bash
[[ $QCENN_TEST_FAILURE != create ]]
''')
    creator.chmod(0o755)
    config = tmp_path / 'config'
    config.mkdir()
    program = '''set -euo pipefail
qcenn_setup_state() { :; }
timeout() { shift; "$@"; }
''' + setup + '\nprintf "CPU_HANDOFF_AVAILABLE\\n"\n'
    environment = {**os.environ, 'qcenn_worktree': str(worktree), 'qcenn_config': str(config),
        'qcenn_base_python': str(creator), 'qcenn_revision': 'test-revision',
        'QCENN_TEST_FAILURE': failure, 'QCENN_TEST_CALLS': str(tmp_path / 'calls')}
    result = subprocess.run(['bash', '-c', program], env=environment,
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CPU_HANDOFF_AVAILABLE' in result.stdout
    assert (config / 'gpu-ready.txt').exists() == (failure == 'none')
    if failure != 'none':
        assert 'GPU SETUP UNAVAILABLE' in result.stdout
    if failure == 'check':
        assert (config / 'gpu-check.log').exists()
        assert 'freeze' not in (tmp_path / 'calls').read_text()


def test_isolated_gpu_pip_check_excludes_cpu_project_metadata(tmp_path):
    environment_root = tmp_path / 'gpu-environment'
    subprocess.run([sys.executable, '-m', 'venv', str(environment_root)], check=True, timeout=60)
    python = environment_root / 'bin/python'
    source = tmp_path / 'cpu-source'
    metadata = source / 'qcenn_qml-0.5.0.dist-info'
    metadata.mkdir(parents=True)
    metadata.joinpath('METADATA').write_text('Metadata-Version: 2.1\nName: qcenn-qml\nVersion: 0.5.0\nRequires-Dist: qcenn-missing-gpu-test-dependency\n')
    environment = {**os.environ, 'PYTHONPATH': str(source)}
    ordinary = subprocess.run([str(python), '-m', 'pip', 'check'], cwd=source, env=environment,
        capture_output=True, text=True, timeout=30)
    assert ordinary.returncode != 0
    assert 'qcenn-missing-gpu-test-dependency' in ordinary.stdout + ordinary.stderr
    isolated = subprocess.run([str(python), '-I', '-m', 'pip', 'check'], cwd=source, env=environment,
        capture_output=True, text=True, timeout=30)
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr
