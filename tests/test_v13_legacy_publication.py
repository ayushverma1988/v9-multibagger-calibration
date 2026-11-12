"""Legacy publishing must not lose concurrent V13 health or source commits."""
import fnmatch
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import compact_results_publish as publisher


class CompactPublication(unittest.TestCase):
    def setUp(self):
        isolated_git = patch.dict(os.environ, {'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1'})
        isolated_git.start()
        self.addCleanup(isolated_git.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.remote = root / 'remote.git'
        self.run_git(root, 'init', '--bare', '--initial-branch=main', str(self.remote))
        self.writer, self.other = root / 'writer', root / 'other'
        self.run_git(root, 'clone', str(self.remote), str(self.writer))
        self.run_git(self.writer, 'config', 'user.name', 'test')
        self.run_git(self.writer, 'config', 'user.email', 'test@example.invalid')
        (self.writer / 'checkpoint.json').write_text('original checkpoint')
        (self.writer / 'published_results').mkdir()
        for name in publisher.RESULT_FILES:
            (self.writer / 'published_results' / name).write_text('old ' + name)
        self.run_git(self.writer, 'add', '.')
        self.run_git(self.writer, 'commit', '-m', 'seed')
        self.run_git(self.writer, 'push', 'origin', 'main')
        self.run_git(root, 'clone', str(self.remote), str(self.other))
        self.run_git(self.other, 'config', 'user.name', 'other')
        self.run_git(self.other, 'config', 'user.email', 'other@example.invalid')
        self.source = root / 'outputs'
        self.source.mkdir()
        for name in publisher.RESULT_FILES:
            (self.source / name).write_text('new ' + name)
        (self.source / 'private_checkpoint.json').write_text('must never publish')

    def run_git(self, repo, *args):
        return subprocess.run(['git', *args], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()

    def advance(self, filename='checkpoint.json', content='newer V13 checkpoint'):
        (self.other / filename).write_text(content)
        self.run_git(self.other, 'add', filename)
        self.run_git(self.other, 'commit', '-m', 'concurrent repository update')
        self.run_git(self.other, 'push', 'origin', 'main')

    def remote_file(self, path):
        return self.run_git(self.remote, 'show', 'main:' + path)

    def test_stale_checkout_publishes_only_results_and_preserves_new_checkpoint(self):
        self.advance()
        result = publisher.publish(self.writer, self.source)
        self.assertEqual(result['status'], 'PUBLISHED')
        self.assertEqual(self.remote_file('checkpoint.json'), 'newer V13 checkpoint')
        for name in publisher.RESULT_FILES:
            self.assertEqual(self.remote_file('published_results/' + name), 'new ' + name)
            self.assertEqual((self.source / name).read_text(), 'new ' + name)
        changed = self.run_git(self.remote, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'main').splitlines()
        self.assertEqual(set(changed), {'published_results/' + n for n in publisher.RESULT_FILES})

    def test_race_between_fetch_and_push_is_retried_without_force(self):
        original = publisher._git
        pushes = []
        def racing(repo, *args, **kwargs):
            if args[0] == 'push':
                pushes.append(args)
                if len(pushes) == 1:
                    self.advance()
            return original(repo, *args, **kwargs)
        with patch.object(publisher, '_git', side_effect=racing):
            result = publisher.publish(self.writer, self.source)
        self.assertEqual(result['push_attempts'], 2)
        self.assertEqual(self.remote_file('checkpoint.json'), 'newer V13 checkpoint')
        self.assertTrue(all('--force' not in args and '-f' not in args for args in pushes))

    def test_shallow_actions_checkout_preserves_new_remote_changes(self):
        shallow = self.writer.parent / 'shallow'
        self.run_git(self.writer.parent, 'clone', '--depth', '1', self.remote.as_uri(), str(shallow))
        self.assertEqual(self.run_git(shallow, 'rev-parse', '--is-shallow-repository'), 'true')
        self.advance()
        result = publisher.publish(shallow, self.source)
        self.assertEqual(result['status'], 'PUBLISHED')
        self.assertEqual(self.remote_file('checkpoint.json'), 'newer V13 checkpoint')

    def test_conflicting_remote_result_aborts_instead_of_overwriting(self):
        self.advance('published_results/summary.json', 'newer published result')
        remote_head = self.run_git(self.remote, 'rev-parse', 'main')
        with self.assertRaisesRegex(RuntimeError, 'conflict'):
            publisher.publish(self.writer, self.source)
        self.assertEqual(self.run_git(self.remote, 'rev-parse', 'main'), remote_head)
        self.assertEqual(self.remote_file('published_results/summary.json'), 'newer published result')
        self.assertEqual((self.source / 'summary.json').read_text(), 'new summary.json')
        self.assertEqual(self.run_git(self.writer, 'status', '--porcelain', '--untracked-files=no'), '')

    def test_persistent_branch_races_stop_at_retry_budget(self):
        original = publisher._git
        pushes = []
        def racing(repo, *args, **kwargs):
            if args[0] == 'push':
                pushes.append(args)
                if len(pushes) > 1:
                    self.run_git(self.other, 'pull', '--ff-only')
                self.advance(content='new checkpoint ' + str(len(pushes)))
            return original(repo, *args, **kwargs)
        with patch.object(publisher, '_git', side_effect=racing):
            with self.assertRaisesRegex(RuntimeError, 'retries exhausted'):
                publisher.publish(self.writer, self.source, attempts=2)
        self.assertEqual(len(pushes), 2)
        self.assertEqual(self.remote_file('published_results/summary.json'), 'old summary.json')

    def test_permission_errors_are_not_hidden_or_retried(self):
        original = publisher._git
        pushes = []
        def denied(repo, *args, **kwargs):
            if args[0] == 'push':
                pushes.append(args)
                return subprocess.CompletedProcess(args, 1, '', 'permission denied')
            return original(repo, *args, **kwargs)
        with patch.object(publisher, '_git', side_effect=denied):
            with self.assertRaisesRegex(RuntimeError, 'permission denied'):
                publisher.publish(self.writer, self.source)
        self.assertEqual(len(pushes), 1)

    def test_missing_inputs_and_unrelated_tracked_changes_fail_before_publication(self):
        old_head = self.run_git(self.remote, 'rev-parse', 'main')
        (self.source / 'summary.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            publisher.publish(self.writer, self.source)
        (self.source / 'summary.json').write_text('new summary.json')
        (self.writer / 'checkpoint.json').write_text('local private work')
        with self.assertRaisesRegex(ValueError, 'tracked changes'):
            publisher.publish(self.writer, self.source)
        self.assertEqual(self.run_git(self.remote, 'rev-parse', 'main'), old_head)

    def test_identical_results_do_not_create_an_empty_commit(self):
        old_head = self.run_git(self.remote, 'rev-parse', 'main')
        for name in publisher.RESULT_FILES:
            (self.source / name).write_text('old ' + name)
        result = publisher.publish(self.writer, self.source)
        self.assertEqual(result['status'], 'NO_LOCAL_RESULT_CHANGES')
        self.assertEqual(self.run_git(self.remote, 'rev-parse', 'main'), old_head)


class LegacyTrigger(unittest.TestCase):
    def test_v13_changes_do_not_start_the_legacy_backfill(self):
        text = (Path(__file__).parents[1] / '.github/workflows/calibrate.yml').read_text()
        trigger = text.split('jobs:', 1)[0]
        paths = re.findall(r"^\s+- '([^']+)'", trigger, re.M)
        self.assertIn('workflow_dispatch:', trigger)
        for path in ['src/v13_runner.py', 'src/v13_priority_research.py', 'tests/test_v13.py', '.github/workflows/calibrate.yml']:
            self.assertFalse(any(fnmatch.fnmatch(path, pattern) for pattern in paths), path)
        for path in ['src/calibrate_v9_2.py', 'src/fetch_legacy_nse.py', 'config.json', 'requirements.txt']:
            self.assertTrue(any(fnmatch.fnmatch(path, pattern) for pattern in paths), path)


if __name__ == '__main__':
    unittest.main()
