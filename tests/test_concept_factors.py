"""Concept factors: discovery from SpLiCE codes, F1 conditioned batches and F2 factor distillation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from golden_support import synthetic_splice_cache, train_sample_ids, write_waterbirds_fixture
from test_golden_training import COMMON_ARGS, GRAPH_CONFIG
from cospro.methods.concept_factors import ConceptConditionedBatchSampler, FactorDistillationRegularizer
from cospro.pipeline import build_concept_groups
from cospro.pipeline.concept_factors import (
    FactorConfig,
    build_concept_factors,
    factor_report,
    rows_for_subset,
)
from cospro.tracking.artifacts import PROJECT_ROOT
from cospro.tracking.metrics import canonical_train_metrics
from tests.test_storage_identity import storage_name


def synthetic_inputs():
    sample_ids, y, place = train_sample_ids()
    cache = synthetic_splice_cache(sample_ids, y, place)
    return cache, build_concept_groups(cache, GRAPH_CONFIG)


class FactorDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.cache, self.groups = synthetic_inputs()
        self.factors = build_concept_factors(self.cache, self.groups, FactorConfig(min_correlation=0.3))

    def test_the_correlated_class_and_context_groups_form_the_top_pairs(self):
        pairs = factor_report(self.factors)["pairs"]
        self.assertEqual({tuple(pair["concepts"]) for pair in pairs}, {
            ("water / lake / ocean", "waterbird / gull"), ("forest", "sparrow"),
        })
        for pair in pairs:
            self.assertAlmostEqual(pair["phi"], 0.5, places=6)

    def test_whitening_decorrelates_the_entangled_factors(self):
        names = [" / ".join(factor["concepts"]) for factor in self.factors["factors"]]
        water, bird = names.index("water / lake / ocean"), names.index("waterbird / gull")
        raw = torch.corrcoef(self.factors["targets"]["standardized"].T)[water, bird]
        white = torch.corrcoef(self.factors["targets"]["whitened"].T)[water, bird]
        self.assertGreater(float(raw), 0.4)
        self.assertLess(abs(float(white)), 0.1)
        variance = self.factors["targets"]["whitened"].var(dim=0, unbiased=False)
        self.assertTrue(torch.allclose(variance, torch.ones_like(variance), atol=1e-4))

    def test_near_synonyms_never_pair(self):
        factors = build_concept_factors(
            self.cache, self.groups, FactorConfig(min_correlation=0.3, max_text_similarity=-1.0),
        )
        self.assertEqual(factors["pairs"], [])

    def test_groups_from_another_cache_are_rejected(self):
        other = {**self.groups, "sample_ids": list(reversed(self.groups["sample_ids"]))}
        with self.assertRaises(ValueError):
            build_concept_factors(self.cache, other, FactorConfig())

    def test_rows_follow_the_training_subset(self):
        rows = rows_for_subset(["waterbirds:4", "waterbirds:9"], "waterbirds", [9, 4])
        self.assertEqual(rows.tolist(), [1, 0])
        with self.assertRaises(ValueError):
            rows_for_subset(["waterbirds:4"], "waterbirds", [5])


class ConditionedSamplerTests(unittest.TestCase):
    def active(self):
        active = torch.zeros(100, 2, dtype=torch.bool)
        active[:40, 0] = True
        active[60:, 1] = True
        return active

    def test_every_image_occurs_once_per_epoch(self):
        sampler = ConceptConditionedBatchSampler(self.active(), 16, 0.5, torch.Generator().manual_seed(0))
        for _ in range(3):
            batches = list(sampler)
            self.assertEqual(sorted(index for batch in batches for index in batch), list(range(100)))
            self.assertEqual(len(batches), len(sampler))

    def test_a_conditioned_batch_shows_one_factor(self):
        sampler = ConceptConditionedBatchSampler(self.active(), 16, 1.0, torch.Generator().manual_seed(1))
        first = next(iter(sampler))
        shared = self.active()[first].all(dim=0)
        self.assertTrue(bool(shared.any()))

    def test_zero_fraction_is_plain_shuffling(self):
        sampler = ConceptConditionedBatchSampler(self.active(), 16, 0.0, torch.Generator().manual_seed(2))
        list(sampler)
        self.assertEqual(sampler.last_conditioned_fraction, 0.0)

    def test_the_same_generator_seed_gives_the_same_batches(self):
        first = list(ConceptConditionedBatchSampler(self.active(), 16, 0.5, torch.Generator().manual_seed(3)))
        second = list(ConceptConditionedBatchSampler(self.active(), 16, 0.5, torch.Generator().manual_seed(3)))
        self.assertEqual(first, second)


class FactorDistillationTests(unittest.TestCase):
    def test_loss_follows_the_schedule_and_trains_the_backbone(self):
        targets = torch.randn(6, 3)
        regularizer = FactorDistillationRegularizer(targets, weight=2.0, start_epoch=1, warmup_epochs=2)
        backbone, head = torch.nn.Linear(4, 5), torch.nn.Linear(5, 3)
        embeddings = backbone(torch.randn(8, 4))
        regularizer.set_epoch(1)
        self.assertEqual(float(regularizer(head(embeddings), torch.tensor([0, 1, 2, 3]))), 0.0)
        regularizer.set_epoch(2)
        loss = regularizer(head(embeddings), torch.tensor([0, 1, 2, 3]))
        self.assertAlmostEqual(regularizer.scheduled_weight, 1.0)
        loss.backward()
        self.assertGreater(float(backbone.weight.grad.norm()), 0.0)
        mse = regularizer.last_diagnostics["factor_mse"]
        self.assertAlmostEqual(regularizer.last_diagnostics["factor_explained_variance"], 1 - mse)


class ConceptFactorTrainingTests(unittest.TestCase):
    def test_both_mechanisms_train_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_waterbirds_fixture(root / "datasets")
            cache, groups = synthetic_inputs()
            torch.save(cache, root / "cache.pt")
            (root / "groups.json").write_text(json.dumps(groups), encoding="utf-8")
            run_dir = root / "run"
            env = {
                **os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled",
                "SPUR_SPLICE_SCRATCH_ROOT": str(root / "scratch"), "SPUR_SPLICE_OUTPUT_ROOT": str(root / "outputs"),
                "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT), os.environ.get("PYTHONPATH", "")]),
            }
            command = [
                sys.executable, "-u", str(PROJECT_ROOT / "spur_splice.py"), *COMMON_ARGS,
                "--data_folder", str(root / "datasets"), "--arm", "factors",
                "--artifact_dir", str(run_dir / "training"), "--run_record", str(run_dir / "run.json"),
                "--splice_mode", "concept_factors",
                "--factor_concept_groups", str(root / "groups.json"), "--factor_splice_cache", str(root / "cache.pt"),
                "--factor_min_correlation", "0.3", "--factor_condition_fraction", "0.5",
                "--factor_distill_weight", "1.0", "--factor_start_epoch", "0", "--factor_warmup_epochs", "0",
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
            self.assertIn("water / lake / ocean  <->  waterbird / gull", result.stdout)
            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "complete")
            ssl = [event["values"] for event in record["metrics"] if event.get("stage") == "ssl"]
            self.assertGreater(ssl[-1]["SSL splice loss"], 0.0)
            # The diagnostics reach W&B under method/.
            canonical = canonical_train_metrics(ssl[-1])
            self.assertIn("method/factor_mse", canonical)
            self.assertIn("method/factor_conditioned_batch_fraction", canonical)


class ConceptFactorOptionTests(unittest.TestCase):
    def test_factor_runs_get_their_own_storage_name(self):
        name = storage_name(["--splice_mode", "concept_factors", "--factor_distill_weight", "1"])
        self.assertTrue(name.startswith("waterbirds_s3_concept-factors_e500_"))
        self.assertNotEqual(
            name, storage_name(["--splice_mode", "concept_factors", "--factor_condition_fraction", "0.5"]),
        )


if __name__ == "__main__":
    unittest.main()
