"""Publish only legacy compact results, preserving concurrent repository changes.

The calibration artifact is uploaded before this command runs. A failed push
therefore never requires repeating the calibration or discarding its output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


RESULT_FILES = (
    'summary.json',
    'calibrator_comparison.csv',
    'downside_calibrator_comparison.csv',
    'topk_by_fold.csv',
    'current_top50.csv',
)


def _git(repo: Path, *args: str, check: bool = True):
    result = subprocess.run(
        ['git', *args], cwd=repo, capture_output=True, text=True,
        stdin=subprocess.DEVNULL,
    )
    if check and result.returncode:
        raise RuntimeError(f'git {args[0]} failed: {result.stderr.strip()}')
    return result


def publish(repo: Path, source: Path, attempts: int = 3) -> dict:
    repo, source = Path(repo).resolve(), Path(source).resolve()
    if not 1 <= attempts <= 5:
        raise ValueError('Publication attempts must be between one and five')
    if _git(repo, 'branch', '--show-current').stdout.strip() != 'main':
        raise ValueError('Compact results may only be published from a main checkout')
    if _git(repo, 'status', '--porcelain', '--untracked-files=no').stdout.strip():
        raise ValueError('Refusing to publish from a checkout with tracked changes')
    for name in RESULT_FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or unsafe compact result: {name}')
    target = repo / 'published_results'
    if target.is_symlink():
        raise ValueError('Refusing a linked results destination')
    target.mkdir(exist_ok=True)
    for name in RESULT_FILES:
        if (target / name).is_symlink():
            raise ValueError(f'Refusing a linked result destination: {name}')
    for name in RESULT_FILES:
        shutil.copyfile(source / name, target / name)
    paths = [f'published_results/{name}' for name in RESULT_FILES]
    _git(repo, 'add', '--', *paths)
    if not _git(repo, 'diff', '--cached', '--name-only').stdout.strip():
        return {'status': 'NO_LOCAL_RESULT_CHANGES', 'push_attempts': 0}
    _git(
        repo, '-c', 'user.name=github-actions[bot]',
        '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com',
        'commit', '-m', 'Update V9 calibration results [skip ci]',
    )
    for attempt in range(1, attempts + 1):
        _git(repo, 'fetch', 'origin', 'main')
        rebase = _git(repo, 'rebase', 'FETCH_HEAD', check=False)
        if rebase.returncode:
            _git(repo, 'rebase', '--abort', check=False)
            raise RuntimeError('Concurrent changes conflict with compact results; publication aborted. The uploaded calibration artifact is preserved.')
        push = _git(repo, 'push', 'origin', 'HEAD:refs/heads/main', check=False)
        if not push.returncode:
            return {
                'status': 'PUBLISHED', 'push_attempts': attempt,
                'commit': _git(repo, 'rev-parse', 'HEAD').stdout.strip(),
                'files': paths,
            }
        reason = push.stdout + push.stderr
        if not any(marker in reason for marker in ('fetch first', 'non-fast-forward')):
            raise RuntimeError('Compact-results push failed; no force push was attempted. ' + push.stderr.strip())
    raise RuntimeError('Remote main kept advancing; bounded publication retries exhausted. The uploaded calibration artifact is preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(Path.cwd(), args.source), sort_keys=True))


if __name__ == '__main__':
    main()
