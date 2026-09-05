import argparse
import json
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
from splice.crp_safe_graph import build_safe_crp_graph, validate_safe_crp_graph
from scripts.tools.evaluate_crp_control_checkpoints import discover_blocks, build_paired_summary


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
    raw_sampler=training_command(config,0,'raw_clip_sampler_only',Path('raw.json'),tmp_path)
    raw_kl=training_command(config,0,'raw_clip_kl',Path('raw.json'),tmp_path)
    assert raw_sampler[raw_sampler.index('--crp_teacher_graph')+1] == raw_kl[raw_kl.index('--crp_teacher_graph')+1]
    assert raw_sampler[raw_sampler.index('--splice_weight')+1]=='0'


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
    saved = json.loads((tmp_path/'probe_features_epoch_100_ds_train_val.json').read_text())
    assert len(saved['group_metrics']['val']['accuracy']) == 4
    assert len(saved['group_metrics']['val']['count']) == 4


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


def test_checkpoint_discovery_requires_paired_seed_blocks(tmp_path):
    output = tmp_path / 'output'
    arms = ['simclr', 'crp_sampler_only', 'raw_clip_kl', 'splice_crp_kl']
    for arm in arms:
        path = output / 'seed0' / arm / 'training' / 'run'
        path.mkdir(parents=True)
        (path / 'epoch_50.pth').write_bytes(b'checkpoint')
    blocks = discover_blocks(output, [0, 1], arms, [50])
    assert blocks[0]['status'] == 'complete'
    assert blocks[1]['status'] == 'incomplete'
    assert blocks[1]['missing_arms'] == arms
    rows = [
        {'seed': 0, 'ssl_epoch': 50, 'arm': 'simclr', 'avg_acc_last10': 1, 'wga_last10': 2, 'best_group_last10': 3},
        {'seed': 0, 'ssl_epoch': 50, 'arm': 'raw_clip_kl', 'avg_acc_last10': 2, 'wga_last10': 4, 'best_group_last10': 5},
    ]
    summary = build_paired_summary(rows)
    assert summary['raw_clip_kl']['avg_acc_last10']['paired_deltas_vs_simclr'] == [
        {'seed': 0, 'ssl_epoch': 50, 'delta': 1}
    ]


def test_safe_graph_preserves_raw_structure_and_marks_bounded_replacement():
    n = 6
    ids = [f'toy:{i}' for i in range(n)]
    generator = torch.Generator().manual_seed(2)
    clip = torch.nn.functional.normalize(torch.randn(n, 5, generator=generator), dim=1)
    cache = {
        'cache_version': 1, 'sample_ids': ids, 'clip_embeddings': clip,
        'image_mean': torch.zeros(5), 'splice_codes': torch.ones(n, 2),
        'dictionary': torch.eye(2, 5), 'vocabulary': ['a', 'b'],
    }
    raw = {
        'artifact': 'splice_raw_clip_matched_teacher_graph', 'graph_version': 1,
        'sample_ids': ids, 'neighbor_indices': torch.tensor([[1, 2], [0, 2], [0, 1], [0, 1], [0, 1], [0, 1]]),
        'weights': torch.tensor([[.6, .4]] * n), 'confidence': torch.ones(n),
        'anchor_confidence': torch.ones(n), 'degree_stats': {'indegree_cap': 10},
    }
    crp = {
        'artifact': 'splice_crp_v3_teacher_graph', 'graph_version': 3,
        'sample_ids': ids, 'neighbor_indices': torch.tensor([[3, 4], [3, 4], [3, 4], [2, 4], [2, 3], [2, 3]]),
        'weights': torch.tensor([[.6, .4]] * n), 'confidence': torch.ones(n),
        'anchor_confidence': torch.ones(n), 'group_ids': torch.tensor([[0, 0]] * n),
        'intervention_gains': torch.tensor([[.2, .1]] * n),
        'edge_confidences': torch.tensor([[.9, .8]] * n),
        'degree_stats': {'indegree_cap': 10},
    }
    safe = build_safe_crp_graph(cache, crp, raw, {'max_replacement_weight': .7})
    validate_safe_crp_graph(safe, ids)
    assert torch.equal(safe['weights'], raw['weights'])
    assert torch.equal(safe['anchor_confidence'], raw['anchor_confidence'])
    assert int((safe['edge_source'] == 2).sum()) == n
    assert torch.all(safe['group_ids'][safe['edge_source'] == 2] >= 0)
    assert torch.all(safe['neighbor_indices'][safe['edge_source'] == 2] != 1)
