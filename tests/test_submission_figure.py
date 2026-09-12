import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.build_submission_figure import graph_mass, select_examples, enrich, build


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
            self.assertTrue(all(r['pair'] for r in evidence['examples']))
            with self.assertRaises(FileExistsError):
                build(argparse.Namespace(artifact_root=artifacts, dataset_root=root, panels=source, output_dir=output))


if __name__ == '__main__':
    unittest.main()
