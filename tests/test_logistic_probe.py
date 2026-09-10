import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from experiments.spurious_eval.training.logistic_probe import fit_logistic_probe
from scripts.tools.build_crp_baseline_graphs import build_matched_raw_clip_graph
from splice.cospro_training import CrpGraphBatchSampler, validate_teacher_graph


def data():
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(120, 8, generator=generator) * torch.logspace(-2, 3, 8)
    labels = (features[:, 4] + features[:, 5] * 0.2 > 0).long()
    return features, labels


class LogisticProbeTests(unittest.TestCase):
    def test_logistic_matches_independent_solver_and_ignores_eval_labels(self):
        features, labels = data()
        records, info = fit_logistic_probe(features, labels, features[:30], labels[:30], num_classes=2, l2=0.02)
        self.assertEqual(len(records), 10)
        self.assertTrue(info["converged"])
        standardized = ((features.double() - features.double().mean(0)) / features.double().std(0, correction=0)).numpy()
        reference = LogisticRegression(C=2 / (len(features) * 0.02), tol=1e-10, max_iter=5000).fit(
            standardized, labels.numpy()
        )
        agreement = np.mean(reference.predict(standardized[:30]) == records[-1].eval_predictions.numpy())
        self.assertGreater(agreement, 0.99)
        other, other_info = fit_logistic_probe(
            features, labels, features[:30] * 30, 1 - labels[:30], num_classes=2, l2=0.02
        )
        self.assertEqual(info, other_info)
        self.assertTrue(torch.equal(records[-1].train_predictions, other[-1].train_predictions))

    def test_nonconvergence_and_bad_features_are_rejected(self):
        features, labels = data()
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            fit_logistic_probe(features, labels, features, labels, num_classes=2, tolerance=1e-30, max_epochs=10)
        features[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_logistic_probe(features, labels, features, labels, num_classes=2)

    def test_raw_graph_matches_budget(self):
        indices = torch.tensor([[1, 2], [2, 3], [-1, -1], [4, 5], [5, -1], [6, -1], [7, -1], [0, -1]])
        weights = (indices >= 0).float()
        weights /= weights.sum(1, keepdim=True).clamp_min(1)
        sample_ids = [f"toy:{index}" for index in range(8)]
        reference = {
            "artifact": "splice_raw_clip_matched_teacher_graph",
            "graph_version": 1,
            "sample_ids": sample_ids,
            "neighbor_indices": indices,
            "weights": weights,
            "confidence": weights.sum(1),
            "anchor_confidence": weights.sum(1) * 0.03,
            "degree_stats": {"indegree_cap": 3},
        }
        cache = {"sample_ids": sample_ids, "centered_clip": torch.nn.functional.normalize(torch.randn(8, 5), dim=1)}
        raw = build_matched_raw_clip_graph(cache, reference)
        validate_teacher_graph(raw)
        torch.testing.assert_close((raw["neighbor_indices"] >= 0).sum(1), (indices >= 0).sum(1))
        sampler = CrpGraphBatchSampler(indices, weights, 4, torch.Generator().manual_seed(0))
        self.assertEqual(sorted(sum(list(sampler), [])), list(range(8)))

    def test_probe_entry_saves_metrics(self):
        from torch.utils.data import DataLoader, TensorDataset
        from experiments.spurious_eval import linear_probe
        from experiments.spurious_eval.metrics import compute_group_metrics

        features, labels = data()
        metadata = torch.stack((torch.arange(len(labels)) % 2, labels), dim=1)

        class Dataset(TensorDataset):
            def eval(self, predictions, targets, meta):
                return compute_group_metrics(predictions, targets, meta).as_spurssl_dict(), ""

        loader = DataLoader(Dataset(features, labels, metadata), batch_size=40)
        spec = {"config": lambda **kwargs: kwargs, "probe_loaders": lambda *args, **kwargs: (loader, loader), "num_classes": 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                dataset="toy", ckpt=str(root / "last.pth"), device="cpu", num_workers=0,
                spurious_probe=True, probe_solver="logistic"
            )
            with patch.dict(linear_probe.DATASET_REGISTRY, {"toy": spec}), patch.object(
                linear_probe, "build_resnet_encoder", return_value=(torch.nn.Identity(), 8)
            ), patch.object(linear_probe, "load_encoder_checkpoint"):
                result = linear_probe.main(args, supcon_epoch=100)
            self.assertTrue(result["Probe converged"])
            saved = json.loads((root / "probe_features_epoch_100_ds_train_val.json").read_text())
            self.assertEqual(len(saved["group_metrics"]["val"]["accuracy"]), 4)


if __name__ == "__main__":
    unittest.main()
