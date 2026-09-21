import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import pytest

# Figure rendering needs the optional paper dependencies (pip install -e .[paper]).
pytest.importorskip("reportlab")

from tools.paper.build_submission_figure import graph_mass, select_examples, enrich, build
from tools.paper.select_graph_panels import stratify, relation_key, discover_panels


def fixture(root):
    from PIL import Image
    metadata = [dict(img_filename=f'{i}.png', y=str(y), place=str(a), split='0')
                for i, (y, a) in enumerate([(0, 0), (0, 0), (0, 1), (1, 0), (1, 1)])]
    graph = dict(sample_ids=[f'waterbirds:{i}' for i in range(5)],
                 neighbor_indices=[[2], [3], [-1], [-1], [-1]], weights=[[1.], [1.], [0.], [0.], [0.]],
                 anchor_confidence=[1., .1, 0., 0., 0.],
                 groups=[dict(group_id=0, concepts=['Fixture concept'], activation_difference_threshold=.01)],
                 config=dict(min_intervention_gain=.001, residual_splice_similarity_threshold=.25,
                             use_residual_splice_gate=True, graph_top_k=1, max_indegree=10))
    pairs = []
    for i, j, retained, gain in [(0, 2, True, .2), (1, 3, True, .1), (1, 2, False, -.1), (0, 3, False, -.2)]:
        pairs.append(dict(row=i, column=j, left_id=f'waterbirds:{i}', right_id=f'waterbirds:{j}',
                          retained=retained, gain=gain, raw_similarity=.4, projected_similarity=.4 + gain,
                          activation_contrast=.1, residual_similarity=.7, final_edge_weight=1. if retained else 0.,
                          null_calibrated_confidence=.1 if retained else 0.))
    panels = dict(groups=[dict(group_id=0, concepts=['Fixture concept'], pairs=pairs)])
    with (root / 'metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata[0])); writer.writeheader(); writer.writerows(metadata)
    for i in range(5):
        Image.new('RGB', (80, 60), (40 * i, 100, 180)).save(root / f'{i}.png')
    return metadata, graph, panels


class SubmissionFigureTests(unittest.TestCase):
    def test_mass_uses_anchor_confidence_not_edge_counts(self):
        with tempfile.TemporaryDirectory() as d:
            metadata, graph, _ = fixture(Path(d))
            mass = graph_mass(graph, metadata, Path(d))
            self.assertAlmostEqual(mass['00']['percent']['cross_background'], 100 / 1.1)
            self.assertAlmostEqual(mass['00']['percent']['wrong_target'], 10 / 1.1)
            self.assertIsNone(mass['11']['percent']['wrong_target'])

    def test_stale_panel_and_nontraining_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); metadata, graph, panels = fixture(root)
            panels['groups'][0]['pairs'][0]['retained'] = False
            with self.assertRaises(ValueError): enrich(panels, graph, metadata, root)
            metadata[0]['split'] = '2'
            with self.assertRaises(ValueError): graph_mass(graph, metadata, root)

    def test_missing_examples_are_explicit_not_substituted(self):
        self.assertTrue(all(item['pair'] is None for item in select_examples([])))

    def test_full_render_preserves_input_and_records_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); metadata, graph, panels = fixture(root)
            artifacts = root / 'artifacts'; graphs = artifacts / 'shared/waterbirds/graphs'; graphs.mkdir(parents=True)
            for name in ('crp_graph', 'raw_clip_graph'):
                (graphs / f'{name}.json').write_text(json.dumps(graph))
            source = root / 'panels.json'; source.write_text(json.dumps(panels)); original = source.read_bytes()
            output = root / 'figure'
            build(argparse.Namespace(artifact_root=artifacts, dataset_root=root, panels=source, output_dir=output))
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue((output / 'graph_evidence.pdf').stat().st_size > 1000)
            self.assertTrue((output / 'details/concept_panels.pdf').is_file())
            evidence = json.loads((output / 'evidence.json').read_text())
            self.assertEqual(len(evidence['examples']), 4)
            self.assertEqual(sum(r['pair'] is not None for r in evidence['examples']), 2)
            with self.assertRaises(FileExistsError):
                build(argparse.Namespace(artifact_root=artifacts, dataset_root=root, panels=source, output_dir=output))

    def test_strata_cover_inverse_relation_and_diverse_sources(self):
        pairs = []
        for retained in (True, False):
            for i, left in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                for j, right in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                    pairs.append(dict(row=i, column=j+4, left_id=str(i), right_id=str(j+4),
                                      retained=retained, gain=10-i, source_group=list(left),
                                      stratum=relation_key(left, right)))
        slots = stratify(pairs)
        self.assertEqual(len(slots), 8)
        self.assertTrue(all(s['pair'] for s in slots))
        self.assertEqual(len({tuple(s['pair']['source_group']) for s in slots[:4]}), 4)
        self.assertEqual(slots[2]['pair']['stratum'], 'different_y_same_a')
        self.assertEqual(slots[2]['candidate_count'], 4)

    def test_cache_discovery_finds_pairs_outside_old_manifest(self):
        import torch
        from splice.cospro import CoSpRoAuditConfig, build_teacher_graph, build_concept_groups
        torch.manual_seed(42)
        n, k = 24, 4
        codes = torch.rand(n, k)
        cache = dict(cache_version=1, sample_ids=[f'waterbirds:{i}' for i in range(n)],
                     clip_embeddings=torch.nn.functional.normalize(codes + .1, dim=1),
                     splice_codes=codes, dictionary=torch.eye(k), image_mean=torch.zeros(k),
                     vocabulary=[f'concept{i}' for i in range(k)])
        metadata = [dict(y=str((i//2)%2), place=str(i%2), split='0') for i in range(n)]
        config = CoSpRoAuditConfig(projected_neighbors=5, null_trials=2, null_quantile=0.,
                                min_coverage=0., max_concept_frequency=1.,
                                max_selected_groups=4, text_similarity_threshold=.99)
        graph = build_teacher_graph(cache, build_concept_groups(cache, config), config, device='cpu')
        # Selection can be empty for a random fixture; keep a real audited group
        # but no final edges, exercising non-retained candidate reconstruction.
        from cospro.pipeline.graph_io import save_graph_json
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'graph.json'
            save_graph_json(graph, path)
            graph = json.loads(path.read_text())
            from PIL import Image
            root = Path(d)
            rows = [{**m, 'img_filename': f'{i}.png'} for i, m in enumerate(metadata)]
            with (root/'metadata.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            for i in range(n):
                Image.new('RGB', (80, 60), (i*10, 100, 180)).save(root/f'{i}.png')
            graphs = root/'artifacts/shared/waterbirds/graphs'
            graphs.mkdir(parents=True)
            for name in ('crp_graph', 'raw_clip_graph'):
                (graphs/f'{name}.json').write_text(json.dumps(graph))
            cache_path = root/'cache.pt'
            torch.save(cache, cache_path)
            output = root/'figure'
            build(argparse.Namespace(artifact_root=root/'artifacts', dataset_root=root,
                                     splice_dataset_cache=cache_path, panels=None, output_dir=output))
            evidence = json.loads((output/'evidence.json').read_text())
            self.assertTrue(all(s['pair'] for s in evidence['examples']))
            detailed = json.loads((output/'details/concept_panels.json').read_text())
            self.assertEqual(len(detailed['groups']), k)
            self.assertTrue(all(len(g['slots']) == 8 for g in detailed['groups']))
        if any(g['selected'] for g in graph['groups']):
            self.assertTrue(discover_panels(cache, graph, metadata)['groups'])
        for group in graph['groups']:
            group['selected'] = True
        graph['weights'] = [[0.]*len(r) for r in graph['weights']]
        panels = discover_panels(cache, graph, metadata)
        self.assertEqual(len(panels['groups']), k)
        self.assertTrue(any(s['pair'] for g in panels['groups'] for s in g['slots'][4:]))
        cache['sample_ids'] = cache['sample_ids'][::-1]
        with self.assertRaisesRegex(ValueError, 'IDs/order'):
            discover_panels(cache, graph, metadata)


if __name__ == '__main__':
    unittest.main()
