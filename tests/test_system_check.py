"""Allocation discovery and standalone Bash diagnostics; no scientific training."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/check_qcenn_system.sh'


@pytest.fixture
def probe(tmp_path):
    code = SCRIPT.read_text().split("<<'QCENN_PYTHON'\n", 1)[1].split('\nQCENN_PYTHON\n', 1)[0]
    path = tmp_path / 'probe.py'
    path.write_text(code)
    spec = importlib.util.spec_from_file_location('system_probe_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_v2_quota_and_memory_limits_are_not_lost(probe, tmp_path):
    mount = tmp_path / 'cgroup with spaces'
    leaf = mount / 'job/step'
    leaf.mkdir(parents=True)
    (mount / 'cpu.max').write_text('max 100000')
    (mount / 'job/cpu.max').write_text('250000 100000')
    (leaf / 'cpu.max').write_text('max 100000')
    (mount / 'job/memory.max').write_text(str(8 * probe.GIB))
    (mount / 'job/memory.current').write_text(str(3 * probe.GIB))
    escaped = str(mount).replace(' ', r'\040')
    mounts = f'1 0 0:1 / {escaped} rw - cgroup2 cgroup rw'
    groups = probe.cgroup_locations(mounts, '0::/job/step')
    assert {path for _, path in groups} == {str(mount), str(mount / 'job'), str(leaf)}
    rows = probe.cgroup_limits(groups)
    result = probe.allocation_limits(64, rows, {})
    assert result['effective_cores'] == 2.5
    assert result['whole_cpu_budget'] == 2
    memory = [r for r in rows if 'memory_limit_bytes' in r]
    assert len(memory) == 1 and memory[0]['memory_used_bytes'] == 3 * probe.GIB


def test_v1_controllers_and_namespace_mount_root(probe, tmp_path):
    cpu = tmp_path / 'cpu'
    mem = tmp_path / 'memory'
    cpu.mkdir(); mem.mkdir()
    (cpu / 'cpu.cfs_quota_us').write_text('400000')
    (cpu / 'cpu.cfs_period_us').write_text('100000')
    (mem / 'memory.limit_in_bytes').write_text(str(6 * probe.GIB))
    (mem / 'memory.usage_in_bytes').write_text(str(probe.GIB))
    mounts = f'1 0 0:1 /parent/job {cpu} rw - cgroup cgroup rw,cpu,cpuacct\n2 0 0:2 /parent/job {mem} rw - cgroup cgroup rw,memory'
    rows = probe.cgroup_limits(probe.cgroup_locations(mounts, '2:cpu,cpuacct:/\n3:memory:/'))
    assert probe.allocation_limits(32, rows, {})['whole_cpu_budget'] == 4
    assert any(r.get('memory_limit_bytes') == 6 * probe.GIB for r in rows)


def test_scheduler_affinity_and_worker_grid_respect_limits(probe):
    scheduler = {'SLURM_CPUS_PER_TASK': '10', 'SLURM_JOB_CPUS_PER_NODE': '32(x2),8', 'SLURM_NODEID': '2'}
    assert probe.allocation_limits(64, [], scheduler)['whole_cpu_budget'] == 8
    assert probe.allocation_limits(4, [], scheduler)['whole_cpu_budget'] == 4
    assert probe.allocation_limits(64, [], {'NSLOTS': '3'})['whole_cpu_budget'] == 3
    assert probe.slurm_node_cpus('32(x2),8', 1) == 32
    grid = probe.worker_grid(10, 6)
    assert (5, 2) in grid
    assert all(j <= 6 and j * t <= 10 for j, t in grid)


def test_external_probe_timeout_is_bounded(probe):
    result = probe.command([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=.1)
    assert result['status'] == 'timeout'


def test_standalone_script_reports_and_archives_without_changing_environment(tmp_path):
    output = tmp_path / 'check with spaces'
    env = {**os.environ, 'QCENN_DIAGNOSTIC_SECRET': 'do-not-collect-me'}
    result = subprocess.run(['bash', str(SCRIPT), '--python', sys.executable,
        '--cpu-budget', '1', '--output-dir', str(output)], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=100)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'report.json').read_text())
    assert report['cpu']['selected_budget'] == 1
    assert 'do-not-collect-me' not in (output / 'report.json').read_text()
    assert report['benchmark']['status'] == 'not_requested'
    assert (output / 'recommended.env').exists() and (output / 'commands.txt').exists()
    if report['tools']['zip']:
        with zipfile.ZipFile(str(output) + '.zip') as archive:
            assert archive.testzip() is None
            assert output.name + '/report.json' in archive.namelist()
    again = subprocess.run(['bash', str(SCRIPT), '--python', sys.executable,
        '--output-dir', str(output)], cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert again.returncode != 0


def test_cli_refuses_more_than_allocation_and_invalid_benchmark_budget(tmp_path):
    output = tmp_path / 'too many cpus'
    result = subprocess.run(['bash', str(SCRIPT), '--python', sys.executable,
        '--cpu-budget', '9999999', '--output-dir', str(output)], cwd=ROOT,
        text=True, capture_output=True, timeout=10)
    assert result.returncode != 0 and 'exceeds detected allocation' in result.stderr
    assert not (output / 'recommended.env').exists()
    result = subprocess.run(['bash', str(SCRIPT), '--benchmark-seconds', '0',
        '--python', sys.executable], cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert result.returncode != 0


def test_busy_benchmark_leaves_running_job_untouched(tmp_path):
    existing = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)',
                                 'run_research_studies.sh'])
    try:
        output = tmp_path / 'busy'
        result = subprocess.run(['bash', str(SCRIPT), '--python', sys.executable,
            '--benchmark', '--cpu-budget', '1', '--output-dir', str(output)], cwd=ROOT,
            text=True, capture_output=True, timeout=100)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads((output / 'report.json').read_text())
        assert report['benchmark']['status'] == 'skipped_busy'
        assert existing.poll() is None
        # Some test sandboxes expose /proc in a different PID namespace.
        assert any(r['name'].startswith('python') for r in report['qcenn_processes'])
        assert not (output / 'benchmark.qpy').exists()
    finally:
        existing.terminate()
        existing.wait(timeout=5)
