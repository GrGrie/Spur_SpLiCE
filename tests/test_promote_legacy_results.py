import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.promote_legacy_results import build_run_record, infer_identity


class LegacyPromotionTests(unittest.TestCase):
    def test_identity_supports_nested_legacy_studies(self):
        identity = infer_identity(Path(
            "crp_signal_checks_v1/transfer/lambda_0.5/seed2/raw_clip_kl/"
            "training/attempt/run_status.json"
        ))
        self.assertEqual(identity, {
            "study": "crp_signal_checks_v1_transfer_lambda_0_5",
            "seed": 2,
            "arm": "raw_clip_kl",
            "attempt_id": "attempt",
        })

    def test_run_record_embeds_metric_history_and_final_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "study" / "seed1" / "arm" / "training" / "attempt"
            run.mkdir(parents=True)
            (run / "args.json").write_text(json.dumps({"epochs": 25}), encoding="utf-8")
            (run / "run_status.json").write_text(json.dumps({
                "status": "complete", "run_identity": {"wandb": {"id": "abc"}}, "cleanup": {}
            }), encoding="utf-8")
            (run / "probe_features_epoch_25_ds_train_val.json").write_text(json.dumps({
                "ssl_epoch": 25,
                "metrics": {"Average over last 10 linear val acc": 75.0},
                "convergence": {"converged": True},
            }), encoding="utf-8")
            record = build_run_record(run / "run_status.json", root, root)
            self.assertEqual(record["identity"]["study"], "study")
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["metrics"][0]["step"], 25)
            self.assertEqual(record["final_metrics"]["Average over last 10 linear val acc"], 75.0)


if __name__ == "__main__":
    unittest.main()
