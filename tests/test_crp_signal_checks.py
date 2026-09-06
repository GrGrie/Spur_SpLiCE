import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.tools import run_crp_signal_checks as checks
from scripts.tools import run_downstream_evaluator_diagnostic as diagnostic


def test_saved_order_matches_real_loader_and_alignment(monkeypatch):
    count, seed = 24, 3
    y = torch.arange(count) % 2
    metadata = torch.stack((torch.arange(count) // 2 % 2, y), dim=1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.arange(count), y, metadata), batch_size=5,
        shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    batches = list(loader)
    order = torch.cat([b[0] for b in batches])
    assert torch.equal(order, diagnostic.saved_train_order(count, seed, 5))
    subset = SimpleNamespace(y_array=y, metadata_array=metadata)
    class Subset:
        y_array = subset.y_array
        metadata_array = subset.metadata_array
        def __len__(self):
            return count
    class Dataset:
        metadata_fields = ["background", "y"]
        def get_subset(self, *args, **kwargs):
            return Subset()
    monkeypatch.setitem(diagnostic.DATASET_REGISTRY, "fixture", {
        "config": lambda **kwargs: None, "dataset": lambda root: Dataset(), "num_classes": 2,
        "probe_loaders": lambda *args, **kwargs: [SimpleNamespace(dataset=SimpleNamespace(transform=None))] * 2,
    })
    payload = {"seed": seed, "train": (torch.zeros(count, 2), y[order], metadata[order]),
               "evaluation": (torch.zeros(count, 2), y, metadata)}
    config = {"dataset": "fixture", "data_folder": "unused", "batch_size": 5,
              "train_set_linear_layer": "ds_train", "eval_split": "val"}
    assert diagnostic._validate_ids_and_transforms(config, payload)["passed"]
    payload["train"][1][0] = 1 - payload["train"][1][0]
    assert not diagnostic._validate_ids_and_transforms(config, payload)["passed"]


def test_balanced_subsets_are_nested_and_train_only():
    y = torch.tensor([0] * 20 + [1] * 20)
    m = torch.stack((torch.tensor([0] * 10 + [1] * 10) .repeat(2), y), dim=1)
    small = checks.balanced_indices(y, m, 3, 101)
    large = checks.balanced_indices(y, m, 7, 101)
    assert set(small.tolist()) <= set(large.tolist())
    assert len(set(large.tolist())) == 28
    assert torch.bincount(y[large] * 2 + m[large, 0]).tolist() == [7] * 4
    with pytest.raises(ValueError, match="requested"):
        checks.balanced_indices(y, m, 11, 101)


def test_reconstruction_localizes_frequency_loss_and_handles_zero_rows():
    cache = {"splice_codes": torch.tensor([[1., 0.], [1., 1.], [0., 0.]]),
             "dictionary": torch.eye(2), "centered_clip": torch.tensor([[1., 0.], [1., 1.], [1., 0.]])}
    graph = {"config": {"min_concept_frequency": .5, "max_concept_frequency": 1.},
             "selected_group_ids": [0], "groups": [{"group_id": 0, "concept_indices": [0]}]}
    stages = checks.reconstruction_stages(cache, graph)
    filtered, full = stages["filtered_vs_full"]
    assert filtered[1].tolist() == [1., 0.]
    assert full[1].tolist() == [1., 1.]
    result = checks.fidelity(filtered, full, torch.ones(3, dtype=torch.bool))
    assert result["zero_norm_rows"] == 1
    assert result["coverage_cos09"] == pytest.approx(1 / 3)


def test_sampler_exposure_distinguishes_wrong_and_useful_donors():
    graph = {"neighbor_indices": torch.tensor([[1], [0], [3], [2]]),
             "weights": torch.ones(4, 1), "anchor_confidence": torch.ones(4)}
    y, context = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
    rows = checks.sampler_exposure(graph, y, context, batch_size=2, seed=0, epochs=3)
    assert all(r["visits"] == 3 and r["supported_fraction"] == 1 for r in rows)
    assert all(r["useful_batch_mass_fraction"] == 1 and r["wrong_batch_mass_fraction"] == 0 for r in rows)
    # No construction input changes: relabelling only changes the diagnostic.
    rows = checks.sampler_exposure(graph, context, y, batch_size=2, seed=0, epochs=3)
    assert all(r["wrong_batch_mass_fraction"] == 1 for r in rows)


def test_probe_uses_train_only_fit_and_reports_groups():
    torch.set_num_threads(1)
    y = torch.tensor([0, 0, 1, 1] * 4)
    x = torch.stack((y.float() * 2 - 1, torch.arange(16).float() / 16), dim=1)
    m = torch.stack((torch.tensor([0, 1, 0, 1] * 4), y), dim=1)
    cfg = {"probe_tolerance": 1e-6, "probe_max_epochs": 100}
    result = checks.probe((x, y, m), (x, y, m), .01, cfg)
    changed = checks.probe((x, y, m), (x, 1 - y, m), .01, cfg)
    assert result["converged"] and result["avg"] == 100
    assert changed["avg"] == 0
    assert changed["gradient_max"] == result["gradient_max"]
    assert result["group_0_count"] == 4


def test_transfer_requires_all_completed_diagnostics(tmp_path):
    cfg = {"output": str(tmp_path)}
    with pytest.raises(FileNotFoundError):
        checks.prepare_transfer(cfg)


def test_transfer_mapping(monkeypatch, tmp_path):
    from scripts.tools import run_crp_controls
    calls = []
    monkeypatch.setattr(run_crp_controls, "run_one_arm", lambda path, task: calls.append((path, task)))
    cfg = {"output": str(tmp_path), "transfer_weights": [.2, .5], "transfer_seeds": [1, 2]}
    checks.transfer_task(cfg, 0)
    checks.transfer_task(cfg, 7)
    assert calls == [(tmp_path / 'transfer/lambda_0.2/locked_config.json', 0),
                     (tmp_path / 'transfer/lambda_0.5/locked_config.json', 3)]
    with pytest.raises(ValueError):
        checks.transfer_task(cfg, 8)


def test_launchers_and_config_are_consistent():
    config = json.loads(Path('scripts/crp_signal_checks.conf').read_text())
    assert sum(len(s['seeds']) * len(s['arms']) for s in config['sources']) == 20
    assert config['student_model'] == 'resnet18_large'
    for name in ['crp_probe_sensitivity', 'crp_splice_signal', 'crp_graph_localization',
                 'crp_signal_diagnostics', 'prepare_crp_transfer_weights',
                 'crp_transfer_weights_array', 'summarize_crp_transfer_weights', 'crp_signal_all']:
        content = Path(f'scripts/{name}.sbatch').read_bytes()
        assert b'\r' not in content
        for directive in [b'--partition=informatik-mind', b'--mem=40G', b'--cpus-per-task=5']:
            assert directive in content
        assert b'scripts/crp_signal_checks.conf' in content
    assert len(config['transfer_weights']) * len(config['transfer_seeds']) * 2 == 8


def test_pretrained_resnet18_mapping_is_architecturally_exact():
    # No download: verify checkpoint-key mapping and forward equivalence.
    from torchvision import models
    from experiments.spurious_eval.models.resnet import build_resnet_encoder
    reference = models.resnet18(weights=None).eval()
    encoder, dim = build_resnet_encoder('resnet18_large', load_pretrained_weights=False)
    state = {k.replace('.downsample.', '.shortcut.'): v for k, v in reference.state_dict().items()
             if not k.startswith('fc.')}
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    reference.fc = torch.nn.Identity()
    torch.set_num_threads(1)
    with torch.inference_mode():
        x = torch.randn(2, 3, 64, 64)
        assert dim == 512
        torch.testing.assert_close(encoder(x), reference(x))


def test_signal_stage_writes_results_and_finishes_wandb(monkeypatch, tmp_path):
    y = torch.tensor([0, 0, 1, 1] * 2)
    m = torch.stack((torch.tensor([0, 1, 0, 1] * 2), y), dim=1)
    x = torch.stack((y.float() * 2 - 1, torch.arange(8).float()), dim=1)
    split = {"sample_ids": list(range(8)), "y": y, "metadata": m,
             "features": {"clip": x, "splice_codes": x + 2}}
    payload = {"train": split, "val": split, "ds_train_ids": list(range(8)), "preprocessing": {}}
    monkeypatch.setattr(checks, 'locked_inputs', lambda config: ({}, {}))
    monkeypatch.setattr(checks, 'extract_signal_features', lambda *args: payload)
    finished = []
    class Run:
        id, url = 'fixture', 'fixture-url'
        def log(self, row):
            pass
        def log_artifact(self, artifact):
            pass
        def finish(self, **kwargs):
            finished.append(kwargs)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        init=lambda **kwargs: Run(), Table=lambda **kwargs: None,
        Artifact=lambda *args, **kwargs: SimpleNamespace(add_file=lambda path: None)))
    cfg = {"output": str(tmp_path), "signal_per_group": [1], "subset_seeds": [0],
           "signal_l2_grid": [.01], "probe_tolerance": 1e-6, "probe_max_epochs": 100,
           "sources": [], "expected_fingerprints": {}, "wandb_project": 'fixture',
           "wandb_entity": 'fixture', "wandb_group": 'fixture'}
    checks.run_stage(cfg, 'signal')
    result = checks.read_json(tmp_path / 'signal/completed.json')
    assert result['status'] == 'complete'
    assert result['wandb']['rows'] == 6
    assert finished == [{}]
    checks.run_stage(cfg, 'signal')
    assert finished == [{}]  # completed stages do not duplicate W&B runs
    with pytest.raises(ValueError, match='Configuration changed'):
        checks.run_stage({**cfg, 'signal_l2_grid': [.02]}, 'signal')
