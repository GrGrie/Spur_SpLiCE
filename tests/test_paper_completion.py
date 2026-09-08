"""Cheap launch-safety checks; no data, GPU, W&B, or Slurm required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.tools import paper_completion as p


class PaperCompletionTests(unittest.TestCase):
    def test_diagnosis_reports_mismatch_without_accepting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'outputs' / 'crp_controls_logistic_v1_cluster_seeds12'
            (base / 'graphs').mkdir(parents=True)
            for path in (base / 'waterbirds_train_features.pt',
                         base / 'graphs' / 'crp_graph.json',
                         base / 'graphs' / 'raw_clip_graph.json'):
                path.write_bytes(b'different-artifact')
            config = root / 'scripts' / 'crp_signal_checks.conf'
            config.parent.mkdir()
            config.write_text(json.dumps({'expected_fingerprints': {
                'cache': 'expected-cache', 'crp': 'expected-crp', 'raw_clip': 'expected-raw'}}))
            with patch.object(p, 'ROOT', root), patch.object(p, 'BASE', base), patch.object(p, 'OUT', root / 'diagnosis'):
                report = p.diagnose_artifacts()
                self.assertTrue(all(not item['matches_expected'] for item in report['artifacts'].values()))
                self.assertIn('load_error', report['compatibility'])
                with self.assertRaisesRegex(RuntimeError, 'actual='):
                    p.check_graphs()

    def test_all_launcher_preserves_the_pipeline_and_conditional_dispatch(self):
        script = Path('scripts/paper_all.sbatch').read_text(encoding='utf-8')
        dispatcher = Path('scripts/paper_04_dispatch_missing_core.sbatch').read_text(encoding='utf-8')
        for name in ('paper_00_inventory', 'paper_01_prepare', 'paper_02_direct',
                     'paper_03_graph', 'paper_07_summary', 'paper_04_dispatch_missing_core'):
            self.assertIn(name, script)
        self.assertIn('direct="$(submit "${prepare}"', script)
        self.assertIn('graph="$(submit "${prepare}"', script)
        self.assertIn('afterok:${inventory}', script)
        self.assertIn('missing_core_tasks.txt', dispatcher)
        self.assertIn('--array="${missing}%4"', dispatcher)
        self.assertIn('paper_05_lock_test', dispatcher)
        self.assertIn('paper_06_test', dispatcher)
        self.assertIn('paper_08_test_summary', dispatcher)

    def test_matrix_uses_primary_lambda_and_all_seeds(self):
        rows = p.matrix()
        self.assertEqual(len({(r['seed'], r['arm']) for r in rows}), 20)
        for row in rows:
            if row['arm'].endswith('_kl'):
                expected = 'lambda_0.5' if row['seed'] < 3 else 'crp_lambda05_replication_s34'
                self.assertIn(expected, row['source'])

    def test_existing_core_never_starts_ssl(self):
        # The import inside core only needs this command builder in the skip path.
        import types
        stub = types.ModuleType('scripts.tools.run_crp_controls')
        stub.training_command = lambda *a: self.fail('Should not build a command')
        with patch.dict('sys.modules', {'scripts.tools.run_crp_controls': stub}), \
             patch.object(p, 'checkpoint', return_value=Path('last.pth')), \
             patch.object(p.subprocess, 'run') as launch:
            p.core(0)
            launch.assert_not_called()

    def test_incomplete_meeting_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'output': str(root / 'runs'), 'seeds': [1],
                                          'arms': [{'name': 'crp'}]}))
            existing = root / 'runs/seed1/crp/training/run/args.json'
            existing.parent.mkdir(parents=True)
            existing.write_text('{}')
            with patch.dict(p.CONFIGS, {'graph': config}), patch.object(p, 'run') as launch:
                with self.assertRaisesRegex(RuntimeError, 'Incomplete existing run'):
                    p.meeting('graph', 0)
                launch.assert_not_called()

    def test_finished_meeting_run_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'output': str(root / 'runs'), 'seeds': [1],
                                          'arms': [{'name': 'crp'}]}))
            arm = root / 'runs/seed1/crp'
            p.write(arm / 'completed.json', {'status': 'complete'})
            p.write(arm / 'command.json', {'command': ['--epochs', '500', '--temp', '0.05', '--seed', '1']})
            p.write(arm / 'training/run/probe_features_epoch_500_ds_train_val.json',
                    {'convergence': {'converged': True}})
            with patch.dict(p.CONFIGS, {'graph': config}), patch.object(p, 'run') as launch:
                p.meeting('graph', 0)
                launch.assert_not_called()

    def test_lock_refuses_missing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p.write(root / 'inventory.json', [{'checkpoint': None}])
            with patch.object(p, 'OUT', root), patch.object(p, 'inventory'):
                with self.assertRaisesRegex(RuntimeError, 'all 20'):
                    p.lock()
            self.assertFalse((root / 'final_test_lock.json').exists())

    def test_test_refuses_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'last.pth'
            model.write_bytes(b'changed')
            p.write(root / 'final_test_lock.json', {
                'source_sha256': {}, 'rows': [{'checkpoint': str(model), 'sha256': 'old'}]})
            with patch.object(p, 'OUT', root), patch.object(p, 'run') as launch:
                with self.assertRaisesRegex(RuntimeError, 'checkpoint changed'):
                    p.test(0)
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
