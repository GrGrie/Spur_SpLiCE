import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from experiments.spurious_eval.training.logistic_probe import fit_logistic_probe
from scripts.tools.build_crp_baseline_graphs import build_matched_raw_clip_graph
from scripts.tools.run_crp_controls import training_command, ARMS
from splice.crp import CrpAuditConfig, _AuditGeometry, _relation_geometry, _residual_splice_similarity, topk_neighbors
from splice.crp_training import validate_teacher_graph, CrpGraphBatchSampler


def data():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(120, 8, generator=g) * torch.logspace(-2, 3, 8)
    y = (x[:, 4] + x[:, 5] * 0.2 > 0).long()
    return x, y


def test_logistic_converges_matches_independent_solver_and_ignores_eval_labels():
    x, y = data()
    records, info = fit_logistic_probe(x, y, x[:30], y[:30], num_classes=2, l2=0.02)
    assert len(records) == 10 and info["converged"]
    assert max(r.gradient_max for r in records) <= 1e-6
    # Binary sklearn uses a single logit; two penalized softmax rows induce l2/2.
    z = ((x.double() - x.double().mean(0)) / x.double().std(0, correction=0)).numpy()
    ref = LogisticRegression(C=2 / (len(x) * 0.02), tol=1e-10, max_iter=5000).fit(z, y.numpy())
    assert np.mean(ref.predict(z[:30]) == records[-1].eval_predictions.numpy()) > 0.99
    other, other_info = fit_logistic_probe(x, y, x[:30] * 30, 1-y[:30], num_classes=2, l2=0.02)
    assert info == other_info
    assert torch.equal(records[-1].train_predictions, other[-1].train_predictions)


def test_nonconvergence_is_not_a_success_and_bad_features_rejected():
    x, y = data()
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_logistic_probe(x, y, x, y, num_classes=2, tolerance=1e-30, max_epochs=10)
    x[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        fit_logistic_probe(x, y, x, y, num_classes=2)


def test_raw_graph_matches_budget_and_sampler_only_uses_same_graph():
    n = 8
    ix = torch.tensor([[1,2], [2,3], [-1,-1], [4,5], [5,-1], [6,-1], [7,-1], [0,-1]])
    w = (ix >= 0).float(); w /= w.sum(1, keepdim=True).clamp_min(1)
    ids = [f"toy:{i}" for i in range(n)]
    ref = {"artifact":"splice_raw_clip_matched_teacher_graph", "graph_version":1,
           "sample_ids":ids,"neighbor_indices":ix,"weights":w,"confidence":w.sum(1),
           "anchor_confidence":w.sum(1)*0.03,"degree_stats":{"indegree_cap":3}}
    cache = {"sample_ids":ids,"centered_clip":torch.nn.functional.normalize(torch.randn(n,5),dim=1)}
    raw = build_matched_raw_clip_graph(cache,ref)
    validate_teacher_graph(raw)
    assert torch.equal((raw['neighbor_indices']>=0).sum(1),(ix>=0).sum(1))
    assert torch.equal(raw['anchor_confidence'], ref['anchor_confidence'])
    assert raw['degree_stats']['maximum_indegree'] <= 3
    sampler = CrpGraphBatchSampler(ix,w,4,torch.Generator().manual_seed(0))
    assert sorted(sum(list(sampler),[])) == list(range(n))


def test_sparse_geometry_matches_direct_residual_cosine():
    torch.manual_seed(1)
    codes = torch.rand(40, 50); codes[codes < 0.8] = 0
    features = torch.nn.functional.normalize(torch.randn(40, 12),dim=1)
    config = CrpAuditConfig(projected_neighbors=3)
    geometry = _AuditGeometry(features, topk_neighbors(features,3)[0], codes)
    basis = torch.eye(12)[:,:2]
    result = _relation_geometry(geometry,basis,config,[1,2])
    direct = _residual_splice_similarity(codes,result['anchors'],result['neighbours'],[1,2])
    assert torch.allclose(result['residual_splice_similarity'],direct,atol=1e-6)


def test_control_commands_parse_and_preserve_ds_train(tmp_path):
    import json, sys, spur_splice
    config=json.loads(Path('scripts/run_crp_controls.conf').read_text())
    command=training_command(config,0,'simclr',None,tmp_path)
    with patch.object(sys,'argv',['spur_splice.py',*command[3:]]):
        args=spur_splice.parse_args()
    assert args.linear_probe_solver=='logistic' and args.train_set_linear_layer=='ds_train'
    assert args.use_wandb
    sampler=training_command(config,0,'crp_sampler_only',Path('graph.json'),tmp_path)
    regularized=training_command(config,0,'splice_crp_kl',Path('graph.json'),tmp_path)
    assert sampler[sampler.index('--crp_teacher_graph')+1] == regularized[regularized.index('--crp_teacher_graph')+1]
    assert sampler[sampler.index('--splice_weight')+1]=='0'


def test_probe_entry_saves_converged_last_ten_metrics(tmp_path):
    from experiments.spurious_eval import linear_probe
    from torch.utils.data import DataLoader, TensorDataset
    from experiments.spurious_eval.metrics import compute_group_metrics
    x,y=data()
    metadata=torch.stack((torch.arange(len(y))%2,y),dim=1)
    class Dataset(TensorDataset):
        def eval(self,pred,labels,meta):
            return compute_group_metrics(pred,labels,meta).as_spurssl_dict(), ''
    dataset=Dataset(x,y,metadata)
    loader=DataLoader(dataset,batch_size=40)
    args=argparse.Namespace(dataset='toy',ckpt=str(tmp_path/'last.pth'),device='cpu',
                            num_workers=0,spurious_probe=True,probe_solver='logistic')
    spec={'config':lambda **kw:kw,'probe_loaders':lambda *a,**kw:(loader,loader),'num_classes':2}
    with patch.dict(linear_probe.DATASET_REGISTRY,{'toy':spec}), \
         patch.object(linear_probe,'build_resnet_encoder',return_value=(torch.nn.Identity(),8)), \
         patch.object(linear_probe,'load_encoder_checkpoint'):
        result=linear_probe.main(args,supcon_epoch=100)
    assert result['Probe converged']
    assert result['Probe gradient max']<=1e-6
    assert result['Spurious probe converged']
    assert (tmp_path/'probe_features_epoch_100_ds_train_val.json').exists()
    assert (tmp_path/'probe_features_epoch_100_ds_train_val.pt').exists()


def test_posthoc_graph_mass_and_coverage_include_unsupported_sources():
    from scripts.tools.crp_posthoc_diagnostics import group_graph_diagnostics
    graph = {"neighbor_indices": torch.tensor([[2, 3], [-1, -1], [0, -1], [0, -1]]),
             "weights": torch.tensor([[0.75, 0.25], [0., 0.], [1., 0.], [1., 0.]]),
             "anchor_confidence": torch.tensor([0.2, 0., 0.5, 0.8])}
    result = group_graph_diagnostics(graph, [0, 0, 0, 1], [0, 0, 1, 0])
    group = result['target=0,context=0']
    assert group['supported_source_fraction'] == 0.5
    useful = group['relations']['same_target_cross_context']
    assert useful['source_fraction_with_donor'] == 0.5
    assert useful['edge_fraction'] == 0.5
    assert useful['confidence_weighted_mass_fraction'] == pytest.approx(0.75)
    assert useful['confidence_weighted_mass_per_source'] == pytest.approx(0.075)
