import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.tools import evaluate_submission_checkpoints as tool


def result():
    return tool.read(tool.PROJECT_ROOT / "outputs/reports/paper_evidence/final_test/probes/seed_01/splice_crp_kl.json")


class SubmissionEvaluationTests(unittest.TestCase):
    def test_discovery_uses_all_eight_reported_run_ids(self):
        rows = tool.discover(tool.PROJECT_ROOT / "outputs")
        self.assertEqual(len(rows), 8)
        self.assertEqual({r['wandb_id'] for r in rows}, {v for seeds in tool.RUN_IDS.values() for v in seeds.values()})
        self.assertTrue(all(r['expected_sha256'] for r in rows))

    def test_existing_test_matrix_passes_protocol_checks(self):
        for seed in range(1, 5):
            for arm in tool.CORE:
                tool.validate_result(tool.read(tool.PROJECT_ROOT / f"outputs/reports/paper_evidence/final_test/probes/seed_{seed:02d}/{arm}.json"))

    def test_validation_nonconvergence_and_wrong_group_counts_rejected(self):
        for field, value in [('eval_split', 'val'), ('ssl_epoch', 450), ('train_split', 'train')]:
            payload = result(); payload[field] = value
            with self.assertRaises(ValueError): tool.validate_result(payload)
        payload = result(); payload['convergence']['converged'] = False
        with self.assertRaises(ValueError): tool.validate_result(payload)
        payload = result(); payload['metrics']['Linear val group counts'][0] -= 1
        with self.assertRaises(ValueError): tool.validate_result(payload)

    def test_summary_rejects_missing_and_duplicate_seeds(self):
        rows = [dict(arm=arm, seed=seed, avg=50. + seed, wga=40. + seed, groups=[40.] * 4)
                for arm in (*tool.CORE, *tool.STUDIES) for seed in range(1, 5)]
        report = tool.summarize(rows)
        self.assertEqual(report['summary']['simclr']['avg']['mean'], 52.5)
        self.assertAlmostEqual(report['summary']['simclr']['avg']['sd'], (5 / 3) ** .5)
        self.assertEqual(report['paired_cospro_minus_control']['semantic_splice'][0]['wga'], 0.)
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError): tool.summarize(bad)

    def test_wrong_checkpoint_epoch_or_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pth'
            payload = dict(epoch=500, opt=dict(seed=2, model='resnet18_large', dataset='waterbirds',
                                             epochs=500, temp=.05, batch_size=128, splice_weight=.5),
                           model={'encoder.weight': torch.zeros(1)})
            torch.save(payload, path)
            tool.validate_checkpoint(path, dict(seed=2, arm='semantic_splice'))
            with self.assertRaises(ValueError): tool.validate_checkpoint(path, dict(seed=4, arm='semantic_splice'))
            payload['epoch'] = 475; torch.save(payload, path)
            with self.assertRaises(ValueError): tool.validate_checkpoint(path, dict(seed=2, arm='semantic_splice'))

    def test_collect_writes_paired_and_latex_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = []
            for seed in range(1, 5):
                for arm in tool.CORE:
                    path = tool.PROJECT_ROOT / f'outputs/reports/paper_evidence/final_test/probes/seed_{seed:02d}/{arm}.json'
                    core.append(dict(seed=seed, arm=arm, path=str(path), sha256=tool.sha256_file(path)))
            rows = [dict(seed=seed, arm=arm) for seed in range(1, 5) for arm in tool.STUDIES]
            lock_path = root / 'lock.json'
            tool.atomic_write_json(lock_path, dict(core=core, rows=rows))
            for row in rows:
                path = root / f"seed_{row['seed']:02d}" / row['arm'] / 'probe_features_epoch_500_ds_train_test.json'
                tool.atomic_write_json(path, result())
                tool.atomic_write_json(path.parent / 'receipt.json', dict(lock_sha256=tool.sha256_file(lock_path), result_sha256=tool.sha256_file(path)))
            tool.collect(argparse.Namespace(output_dir=root))
            self.assertEqual(len(tool.read(root / 'results.json')['rows']), 28)
            tex = (root / 'table.tex').read_text(encoding='utf-8')
            self.assertIn(r'\pm', tex)
            self.assertIn(r'\end{tabular}', tex)
            self.assertIn('Groups 00, 01, 10, 11', (root / 'results.md').read_text(encoding='utf-8'))

    def test_execute_calls_only_probe_reuses_result_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / 'last.pth'; checkpoint.write_bytes(b'frozen-encoder')
            metadata = root / 'metadata.csv'; metadata.write_text('fixture')
            row = dict(seed=1, arm='semantic_splice', checkpoint=str(checkpoint), sha256=tool.sha256_file(checkpoint))
            lock = dict(protocol=tool.PROTOCOL, code_sha256=tool.code_hash(), metadata=str(metadata),
                        metadata_sha256=tool.sha256_file(metadata), data_folder=directory, rows=[row])
            tool.atomic_write_json(root / 'lock.json', lock)
            def fake_probe(options):
                self.assertEqual(options.eval_split, 'test')
                self.assertTrue(options.final_test)
                self.assertEqual(options.num_workers, 4)
                self.assertEqual(options.batch_size, 128)
                self.assertEqual(options.train_set_linear_layer, 'ds_train')
                tool.atomic_write_json(Path(options.artifact_dir) / 'probe_features_epoch_500_ds_train_test.json', result())
            args = argparse.Namespace(output_dir=root, seed=1, device='cpu')
            with patch('experiments.spurious_eval.linear_probe.main', side_effect=fake_probe) as probe:
                tool.execute(args); tool.execute(args)
                self.assertEqual(probe.call_count, 1)
            self.assertEqual(checkpoint.read_bytes(), b'frozen-encoder')
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError): tool.execute(args)


if __name__ == '__main__':
    unittest.main()
