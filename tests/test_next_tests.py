from __future__ import annotations

import torch

from splice.concept_distillation import (
    ConceptDistillationRegularizer,
    FrozenConceptTransferSubset,
    _derangement,
)
from scripts.tools.summarize_crp_replication import _gate


def _targets(count: int = 4):
    raw = torch.nn.functional.normalize(torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) + 1, dim=1)
    return {
        "sample_ids": [f"waterbirds:{i}" for i in range(count)],
        "raw": raw,
        "reconstruction": raw.clone(),
        "shuffled_reconstruction": raw.flip(0),
        "valid_mask": torch.ones(count, dtype=torch.bool),
    }


def test_derangement_is_reproducible_and_has_no_fixed_points():
    ids = torch.arange(8)
    first = _derangement(ids, 101)
    assert torch.equal(first, _derangement(ids, 101))
    assert torch.all(first != ids)
    assert sorted(first.tolist()) == ids.tolist()


def test_regularizer_uses_targets_without_gradient_and_supports_zero_rows():
    regularizer = ConceptDistillationRegularizer(_targets(), "reconstruction", 0.1, 0, 0)
    regularizer.set_epoch(1)
    predictions = torch.randn(8, 4, requires_grad=True)
    bank, valid = regularizer.targets_for_indices(torch.tensor([0, 1, 2, 3]), predictions.device)
    loss = regularizer(predictions, torch.cat([bank, bank]), valid)
    loss.backward()
    assert loss.isfinite()
    assert predictions.grad is not None
    assert not bank.requires_grad


def test_transfer_subset_does_not_read_annotations():
    class Full:
        def get_input(self, index):
            return torch.tensor([index])

    class Subset:
        indices = [2, 0]
        dataset = Full()

    subset = FrozenConceptTransferSubset(Subset(), lambda image: (image, image), _targets())
    assert subset[0][1] == 2
    assert subset[1][1] == 0


def test_missing_gate_comparison_is_not_evaluable():
    rows = [
        {"seed": 3, "condition": "splice_crp_kl_lambda0.5", "avg_acc_last10": 52.0, "wga_last10": 43.0},
        {"seed": 3, "condition": "simclr", "avg_acc_last10": 50.0, "wga_last10": 40.0},
    ]
    report = _gate(rows, {})
    assert report["full_pooled_contrasts"]["simclr"]["status"] == "NOT_EVALUABLE"
    assert "simclr" in report["scientific_outcomes"]["not_evaluable"]
