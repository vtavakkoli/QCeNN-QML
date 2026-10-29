"""Exercise real Git worktree recovery and pre-run updater status."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

UPDATER = Path(__file__).parents[1] / 'scripts/update_and_run_research.sh'


def git(repo, *arguments):
    return subprocess.run(['git', '-C', str(repo), *arguments], check=True,
        capture_output=True, text=True).stdout


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / 'original-project'
    repo.mkdir()
    git(repo, '-c', 'init.defaultBranch=main', 'init', '-q')
    git(repo, 'config', 'user.name', 'QCeNN test')
    git(repo, 'config', 'user.email', 'qcenn-test@example.invalid')
    (repo / 'source.txt').write_text('committed source\n')
    git(repo, 'add', 'source.txt')
    git(repo, 'commit', '-qm', 'Initial fixture')
    revision = git(repo, 'rev-parse', 'HEAD').strip()
    target = tmp_path / f'QCeNN-QML-accelerated-{revision[:12]}'
    return repo, revision, target


def prepare_checkout(source_repo):
    repo, revision, _ = source_repo
    source = UPDATER.read_text()
    start = source.index('qcenn_worktree=')
    end = source.index('qcenn_config=', start)
    return subprocess.run(['bash', '-c', 'set -euo pipefail\n' + source[start:end] + '\nprintf "CHECKOUT_READY\\n"\n'],
        cwd=repo, env={**os.environ, 'qcenn_repo': str(repo), 'qcenn_revision': revision},
        capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize('registered', [False, True])
def test_missing_checkout_recovers_without_pruning_other_worktrees(source_repo, registered):
    repo, revision, target = source_repo
    other = repo.parent / 'unrelated-missing-worktree'
    git(repo, 'worktree', 'add', '--detach', str(other), revision)
    shutil.rmtree(other)
    if registered:
        git(repo, 'worktree', 'add', '--detach', str(target), revision)
        shutil.rmtree(target)
    result = prepare_checkout(source_repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CHECKOUT_READY' in result.stdout
    assert git(target, 'rev-parse', 'HEAD').strip() == revision
    assert (target / 'source.txt').read_text() == 'committed source\n'
    assert str(other) in git(repo, 'worktree', 'list', '--porcelain')
    assert not other.exists()


def test_missing_locked_worktree_is_not_overridden(source_repo):
    repo, revision, target = source_repo
    git(repo, 'worktree', 'add', '--detach', str(target), revision)
    git(repo, 'worktree', 'lock', '--reason', 'keep this registration', str(target))
    shutil.rmtree(target)
    result = prepare_checkout(source_repo)
    assert result.returncode != 0
    assert 'locked' in result.stderr
    assert not target.exists()
    assert 'locked keep this registration' in git(repo, 'worktree', 'list', '--porcelain')


def test_existing_dirty_checkout_is_preserved_and_rejected(source_repo):
    repo, revision, target = source_repo
    git(repo, 'worktree', 'add', '--detach', str(target), revision)
    (target / 'source.txt').write_text('local edits\n')
    result = prepare_checkout(source_repo)
    assert result.returncode != 0
    assert 'Worktree changed' in result.stdout + result.stderr
    assert (target / 'source.txt').read_text() == 'local edits\n'


def test_dangling_symlink_is_not_replaced_with_a_checkout(source_repo):
    _, _, target = source_repo
    target.symlink_to(target.parent / 'not-mounted')
    result = prepare_checkout(source_repo)
    assert result.returncode != 0
    assert target.is_symlink()
    assert not target.exists()


@pytest.mark.parametrize(('action', 'exit_code'), [('status', 0), ('resume', 1), ('stop', 1)])
def test_missing_run_marker_has_explicit_status_and_no_cat_error(source_repo, action, exit_code):
    repo, _, _ = source_repo
    # An empty quoted output variable must not become an implicit unknown run.
    arguments = ['--' + action] + ([''] if action != 'status' else [])
    result = subprocess.run(['bash', str(UPDATER), '--repository', str(repo), *arguments],
        cwd=repo, env={**os.environ, 'QCENN_BASE_PYTHON': sys.executable},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert 'State: NO_RUN_REGISTERED' in result.stdout
    assert 'No such file' not in result.stderr
    if action != 'status':
        assert 'explicit existing output directory' in result.stderr
