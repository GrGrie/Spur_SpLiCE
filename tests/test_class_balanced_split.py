"""cb_train balances the classes and keeps each class's spurious-attribute mix from train."""

from __future__ import annotations

import unittest

import torch

from cospro.data.base import class_balanced_indices


def waterbirds_like(counts=(3498, 184, 56, 1057), held_out=20):
    """Train groups with Waterbirds' counts (group = place + 2 * y) plus some non-train samples."""

    labels, places = [], []
    for group, count in enumerate(counts):
        labels += [group // 2] * count
        places += [group % 2] * count
    labels, places = torch.tensor(labels), torch.tensor(places)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))
    keep = torch.cat([torch.ones(len(labels), dtype=torch.bool), torch.zeros(held_out, dtype=torch.bool)])
    return torch.cat([labels[order], torch.zeros(held_out, dtype=torch.long)]), \
        torch.cat([places[order], torch.zeros(held_out, dtype=torch.long)]), keep


class ClassBalancedSplitTests(unittest.TestCase):
    def test_waterbirds_counts_give_equal_classes_with_train_correlation(self):
        labels, places, keep = waterbirds_like()
        indices = class_balanced_indices(labels, places, keep)
        groups = places[indices] + 2 * labels[indices]
        self.assertEqual(torch.bincount(groups, minlength=4).tolist(), [1057, 56, 56, 1057])
        self.assertTrue(keep[indices].all())
        self.assertTrue(torch.equal(indices, indices.sort().values))
        self.assertEqual(len(indices.unique()), len(indices))

    def test_selection_is_deterministic_and_takes_the_first_samples_per_group(self):
        labels, places, keep = waterbirds_like()
        indices = class_balanced_indices(labels, places, keep)
        self.assertTrue(torch.equal(indices, class_balanced_indices(labels, places, keep)))
        landbird_on_land = torch.nonzero((labels == 0) & (places == 0) & keep).reshape(-1)
        chosen = indices[(labels[indices] == 0) & (places[indices] == 0)]
        self.assertTrue(torch.equal(chosen, landbird_on_land[:1057]))

    def test_rounding_sums_to_the_smaller_class(self):
        labels, places, keep = waterbirds_like(counts=(7, 3, 2, 3), held_out=0)
        indices = class_balanced_indices(labels, places, keep)
        self.assertEqual(torch.bincount(labels[indices]).tolist(), [5, 5])
        self.assertEqual(torch.bincount(places[indices][labels[indices] == 0], minlength=2).tolist(), [4, 1])


if __name__ == "__main__":
    unittest.main()
