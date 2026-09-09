from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from splice.concept_group_search import group_metrics, propose, select_groups, split_rows, subset
from splice.crp import CrpAuditConfig, group_concepts, run_frozen_audit, validate_feature_cache
from splice.crp_training import validate_teacher_graph
from scripts.tools.visualize_concept_group_search import relation_curve, sample_pairs
from scripts.tools import visualize_concept_group_search as visual
from splice.graph_io import save_graph_json


def fixture():
    generator = torch.Generator().manual_seed(90)
    dictionary = torch.eye(8)[:6]
    dictionary[1] = F.normalize(dictionary[0] + .1 * dictionary[1], dim=0)
    codes = torch.zeros(24, 6)
    codes[::2, 0] = 1
    codes[1::2, 1] = 1
    codes[:, 2:] = (torch.rand(24, 4, generator=generator) > .65).float()
    return validate_feature_cache({"cache_version": 1,
        "sample_ids": [f"waterbirds:{i}" for i in range(24)],
        "clip_embeddings": F.normalize(torch.randn(24, 8, generator=generator), dim=1),
        "image_mean": torch.zeros(8), "splice_codes": codes,
        "dictionary": dictionary, "vocabulary": ["lake", "pond", "bird", "tree", "cloud", "rock"]})


class GroupSearchTests(unittest.TestCase):
    def test_semantic_alternatives_do_not_need_coactivation(self):
        cache = fixture()
        config = CrpAuditConfig(text_similarity_threshold=.8, coactivation_threshold=.35)
        original = group_concepts(cache['splice_codes'], cache['dictionary'], cache['vocabulary'], config)
        self.assertNotIn([0, 1], original)
        self.assertIn([0, 1], propose(cache))
        self.assertEqual(group_metrics(cache, [0, 1])['coactivation_mean'], 0)
        self.assertAlmostEqual(group_metrics(cache, [0, 1])['effective_members'], 2)

    def test_split_is_disjoint_reproducible_and_id_based(self):
        ids = fixture()['sample_ids']
        left, right = split_rows(ids)
        self.assertFalse(set(left) & set(right))
        self.assertEqual(set(left) | set(right), set(range(len(ids))))
        reordered = list(reversed(ids))
        other, _ = split_rows(reordered)
        self.assertEqual({ids[i] for i in left}, {reordered[i] for i in other})

    def test_pool_bounded_unique_and_complete_link(self):
        cache = fixture()
        groups = propose(cache, max_composites=3)
        self.assertLessEqual(len(groups), 3)
        self.assertEqual(len(groups), len({tuple(g) for g in groups}))
        for group in groups:
            self.assertTrue(2 <= len(group) <= 4)
            self.assertGreaterEqual(group_metrics(cache, group)['text_min'], .65 - 1e-6)

    def test_sparse_policy_rejects_one_member_disguised_as_group(self):
        group = {'selected': True, 'group_id': 0, 'concept_indices': [0, 1], 'null_excess_score': 1.}
        metric = {'support': .3, 'removed_energy_p95': .2, 'size': 2, 'effective_members': 1.1,
                  'max_member_mass_share': .98, 'removed_energy_mean': .1, 'rank': 2}
        self.assertEqual(select_groups([group], [metric], 'semantic'), [0])
        self.assertEqual(select_groups([group], [metric], 'compact'), [])
        metric['removed_energy_p95'] = .8
        self.assertEqual(select_groups([group], [metric], 'semantic'), [])

    def test_explicit_candidates_use_canonical_audit_and_validate(self):
        cache = fixture()
        config = CrpAuditConfig(null_trials=2, projected_neighbors=3, max_selected_groups=2)
        graph = run_frozen_audit(cache, config, candidate_groups=[[0, 1], [0]])
        self.assertEqual([g['concept_indices'] for g in graph['groups']], [[0, 1], [0]])
        self.assertEqual(graph['groups'][0]['basis_rank'], 2)
        validate_teacher_graph(graph, cache['sample_ids'])
        with self.assertRaisesRegex(ValueError, 'duplicate groups'):
            run_frozen_audit(cache, config, candidate_groups=[[0, 1], [1, 0]])
        with self.assertRaisesRegex(ValueError, 'valid concept indices'):
            run_frozen_audit(cache, config, candidate_groups=[[99]])

    def test_explicit_matching_groups_reproduce_default_audit(self):
        cache = fixture()
        config = CrpAuditConfig(null_trials=2, projected_neighbors=3)
        groups = group_concepts(cache['splice_codes'], cache['dictionary'], cache['vocabulary'], config)
        default = run_frozen_audit(cache, config)
        explicit = run_frozen_audit(cache, config, candidate_groups=groups)
        self.assertTrue(torch.equal(default['weights'], explicit['weights']))
        self.assertEqual(default['groups'], explicit['groups'])

    def test_annotation_boundary_remains_enforced(self):
        cache = fixture()
        cache['labels'] = torch.zeros(24)
        with self.assertRaisesRegex(ValueError, 'annotation'):
            run_frozen_audit(cache, CrpAuditConfig(), candidate_groups=[[0, 1]])

    def test_pair_sampling_and_chance_reference(self):
        graph = {'neighbor_indices': torch.tensor([[(i+1)%24] for i in range(24)]),
                 'weights': torch.ones(24, 1), 'anchor_confidence': torch.linspace(.01, .24, 24)}
        pairs = sample_pairs(graph)
        self.assertEqual(pairs, sample_pairs(graph))
        self.assertTrue(all(p['row'] != p['donor'] for p in pairs))
        self.assertEqual({p['band'] for p in pairs}, {'low', 'middle', 'high', 'random'})
        curve = relation_curve(graph, torch.arange(24)%2, torch.arange(24)%3)
        self.assertEqual(sum(b['edges'] for b in curve), 24)
        self.assertTrue(all(b['same_target'] == 0 for b in curve))

    def test_html_and_rating_pipeline_on_synthetic_images(self):
        from PIL import Image
        import csv
        from types import SimpleNamespace
        cache = fixture()
        graph = run_frozen_audit(cache, CrpAuditConfig(null_trials=2, projected_neighbors=3), candidate_groups=[[0, 1]])
        graph['groups'][0]['selected'] = True
        graph.update(neighbor_indices=torch.tensor([[(i+1)%24] for i in range(24)]),
                     weights=torch.ones(24, 1), anchor_confidence=torch.linspace(.01, .24, 24))
        dataset = SimpleNamespace(y_array=torch.arange(24)%2, metadata_array=(torch.arange(24)%3)[:, None],
                                  get_input=lambda i: Image.new('RGB', (80, 60), (int(i)*10, 80, 120)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.json'
            save_graph_json(graph, reference)
            with patch.object(visual, 'OUT', root), patch.object(visual, 'REFERENCE', reference), \
                 patch.object(visual, 'inputs', return_value=(cache, graph)), \
                 patch('experiments.spurious_eval.datasets.waterbirds.WaterbirdsDataset', return_value=dataset), \
                 patch('scripts.tools.crp_posthoc_diagnostics.diagnose_fixed_graphs', return_value={}):
                visual.render(0, root)
                dest = root / 'visual/baseline'
                self.assertIn('Failure case', (dest / 'index.html').read_text(encoding='utf-8'))
                self.assertNotIn('mass=', (dest / 'blind_pairs.html').read_text(encoding='utf-8'))
                with (dest / 'ratings.csv').open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    row['shared_object_0_2'] = '1'
                    row['shared_background_0_2'] = '2'
                with (dest / 'ratings.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                visual.summarize_ratings(0)
                result = visual.read(dest / 'human_ratings_summary.json')
                self.assertEqual(result['ratings']['shared_object_0_2']['high_minus_random'], 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
