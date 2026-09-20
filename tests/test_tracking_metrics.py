"""The W&B metric contract: canonical keys next to the historical ones."""

from __future__ import annotations

import unittest

import torch

from cospro.tracking import canonical_probe_metrics, canonical_train_metrics, define_wandb_metrics
from experiments.spurious_eval.training.ssl_loop import log_rank_metrics

PROBE_METRICS = {
    "Last linear test acc": 70.0,
    "Last linear test worst-group acc": 42.0,
    "Average over last 10 linear test worst-group acc": 41.0,
    "Linear train worst-group acc": 100.0,
    "Probe epochs": 11,
    "Spurious probe last val worst-group acc": 99.0,
    "Linear val group accuracies": [10.0, 20.0, 0.0],
    "Linear val group counts": [5, 7, 0],
}


class FakeWandbRun:
    def __init__(self) -> None:
        self.logged: dict = {}
        self.summaries: dict[str, str] = {}

    def log(self, payload, step=None):
        self.logged.update(payload)

    def define_metric(self, name, summary=None):
        self.summaries[name] = summary


class MetricContractTests(unittest.TestCase):
    def test_probe_keys_follow_the_evaluation_split(self):
        canonical = canonical_probe_metrics(PROBE_METRICS, split="test")
        self.assertEqual(canonical["probe/test/wga"], 42.0)
        self.assertEqual(canonical["probe/test/wga_avg10"], 41.0)
        self.assertEqual(canonical["probe/train/wga"], 100.0)
        self.assertEqual(canonical["probe/spurious/wga"], 99.0)
        # Empty groups stay out of the per-group series.
        self.assertEqual(canonical["probe/test/group_acc/1"], 20.0)
        self.assertNotIn("probe/test/group_acc/2", canonical)

    def test_method_diagnostics_move_under_one_prefix(self):
        canonical = canonical_train_metrics({
            "SSL train loss": 1.0,
            "SSL splice loss": 0.25,
            "SSL relational scheduled weight": 0.5,
            "SSL relational_cosine_loss": 0.75,
            "SSL la_ssl_upsampled_fraction": 0.3,
        })
        self.assertEqual(canonical["train/loss/total"], 1.0)
        self.assertEqual(canonical["train/loss/method"], 0.25)
        self.assertEqual(canonical["method/scheduled_weight"], 0.5)
        self.assertEqual(canonical["method/cosine_loss"], 0.75)
        self.assertEqual(canonical["method/la_ssl_upsampled_fraction"], 0.3)

    def test_worst_group_accuracy_is_the_run_summary(self):
        run = FakeWandbRun()
        define_wandb_metrics(run, split="val")
        self.assertEqual(run.summaries["probe/val/wga"], "max")
        define_wandb_metrics(None, split="val")

    def test_epoch_logging_sends_both_key_sets(self):
        run = FakeWandbRun()
        args = type("Args", (), {"device": "cpu", "channels_last": False})()
        payload = log_rank_metrics(
            model=None, rank_loader=None, optimizer=type("Opt", (), {"param_groups": [{"lr": 0.01}]})(),
            train_metrics={"loss": 1.0, "simclr_loss": 0.9, "decor_loss": 0.0, "entropy_loss": 0.0,
                           "splice_loss": 0.1, "relational_scheduled_weight": 0.5},
            epoch=3, args=args, wandb_run=run, compute_rank=False,
        )
        self.assertEqual(payload["SSL train loss"], 1.0)
        self.assertEqual(run.logged["SSL train loss"], 1.0)
        self.assertEqual(run.logged["train/loss/total"], 1.0)
        self.assertEqual(run.logged["method/scheduled_weight"], 0.5)


if __name__ == "__main__":
    unittest.main()
