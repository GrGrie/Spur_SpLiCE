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
from cospro.cli import sweep_concept_factors
from cospro.cli.cache_splice_dataset import cache_config_name
from cospro.diagnostics.factor_validity import conditional_uncertainty, diagnose_factors
from cospro.pipeline import CoSpRoAuditConfig, build_concept_groups
from cospro.pipeline.concept_factors import (
    FactorConfig,
    build_concept_factors,
    factor_report,
    rows_for_subset,
)
from cospro.tracking.artifacts import PROJECT_ROOT
from unittest.mock import patch
import argparse
import numpy as np
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


class RedundancyMergeTests(unittest.TestCase):
    def test_synonym_groups_merge_by_image_similarity_and_the_pair_survives(self):
        cache, _ = synthetic_inputs()
        # A strict text threshold leaves the three water synonyms in separate groups.
        split = build_concept_groups(cache, CoSpRoAuditConfig(text_similarity_threshold=0.999, coactivation_threshold=0.35))
        self.assertGreater(len(split["groups"]), 5)
        unmerged = build_concept_factors(cache, split, FactorConfig(min_correlation=0.3))
        merged = build_concept_factors(cache, split, FactorConfig(min_correlation=0.3, merge_similarity=0.9))
        self.assertLess(len(merged["factors"]), len(unmerged["factors"]))
        names = [" / ".join(sorted(factor["concepts"])) for factor in merged["factors"]]
        self.assertIn("lake / ocean / water", names)
        pairs = {tuple(sorted(pair["concepts"])) for pair in factor_report(merged)["pairs"]}
        self.assertTrue(any("gull" in " ".join(pair) and "water" in " ".join(pair) for pair in pairs))


class FactorValidityTests(unittest.TestCase):
    def test_class_and_attribute_factors_are_told_apart_under_correlation(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 4000)
        a = np.where(rng.random(4000) < 0.9, y, 1 - y)
        active = np.stack([(y == 1) & (rng.random(4000) < 0.9), (a == 1) & (rng.random(4000) < 0.9),
                           rng.random(4000) < 0.3], axis=1)
        pairs = [{"factors": [0, 1], "phi": 0.7}, {"factors": [0, 2], "phi": 0.1}]
        diagnosis = diagnose_factors(active, pairs, ["bird", "water", "noise"], y, a)
        self.assertEqual([factor["type"] for factor in diagnosis["factors"]], ["class", "attribute", "neither"])
        self.assertEqual([pair["verdict"] for pair in diagnosis["pairs"]], ["cross", "noise"])
        self.assertEqual(diagnosis["summary"]["pair_precision"], 0.5)

    def test_a_single_column_gives_a_scalar(self):
        y = np.array([0, 0, 1, 1]); a = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(float(conditional_uncertainty(np.array([0, 0, 1, 1], bool), y, a)), 1.0)


class SweepTests(unittest.TestCase):
    def test_the_sweep_writes_groups_factor_sets_and_a_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_waterbirds_fixture(root / "datasets")
            cache, _ = synthetic_inputs()
            name = cache_config_name(argparse.Namespace(
                dataset="waterbirds", splice_model=sweep_concept_factors.SPLICE_MODEL,
                splice_pretrained=sweep_concept_factors.SPLICE_PRETRAINED, splice_vocab="laion",
                splice_vocab_size=10000, splice_l1_penalty=0.25, splice_vocab_file=None, splice_vocab_order=None,
            ))
            cache_path = root / "features" / "waterbirds" / "splice_dataset_cache" / name / "splice_dataset_cache.pt"
            cache_path.parent.mkdir(parents=True)
            torch.save(cache, cache_path)
            with patch.dict(os.environ, {"SPUR_SPLICE_OUTPUT_ROOT": str(root / "outputs")}):
                summary = sweep_concept_factors.main([
                    "--data-folder", str(root / "datasets"), "--datasets", "waterbirds", "--vocabs", "laion",
                    "--feature-root", str(root / "features"), "--text-thresholds", "0.80",
                    "--coactivation-thresholds", "0.30", "--merge-similarities", "0", "0.9",
                ])
            rows = json.loads((summary.parent / "summary.json").read_text(encoding="utf-8"))["rows"]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["attribute_factors"] >= 1 for row in rows))
            folder = root / "outputs" / "shared" / "waterbirds" / "factor_sweep" / "laion" / "text_0p80_coactivation_0p30"
            self.assertTrue((folder / "groups.json").is_file())
            record = json.loads((folder / "factors_merge_0p00.json").read_text(encoding="utf-8"))
            # Both planted pairs join a class factor to an attribute factor; weaker pairs follow them.
            self.assertEqual([pair["verdict"] for pair in record["pairs"][:2]], ["cross", "cross"])


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
