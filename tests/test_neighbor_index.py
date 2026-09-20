"""The approximate neighbour index, held to the exact one.

The audit's whole question is which images stop being neighbours once a concept subspace is
projected out, so an approximate index that finds different neighbours changes the method. These
tests fix what both indexes promise and how far the LSH index may drift from the exact answer.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from cospro.pipeline.neighbors import (
    NEIGHBOR_INDEXES,
    ExactNeighbors,
    LshNeighbors,
    build_index,
    resolve_backend,
    topk_neighbors,
)

SAMPLES = 400
DIMENSIONS = 32
CLUSTERS = 20
NEIGHBOURS = 5


def clustered_features(seed: int = 0) -> torch.Tensor:
    """Points around ``CLUSTERS`` centres, which is the structure a concept subspace induces."""

    generator = torch.Generator().manual_seed(seed)
    centres = torch.randn(CLUSTERS, DIMENSIONS, generator=generator)
    assignment = torch.arange(SAMPLES) % CLUSTERS
    features = centres[assignment] + 0.25 * torch.randn(SAMPLES, DIMENSIONS, generator=generator)
    return F.normalize(features, dim=1)


def recall(exact: torch.Tensor, approximate: torch.Tensor) -> float:
    """The fraction of exact neighbours the approximate index also found."""

    found = sum(len(set(row.tolist()) & set(other.tolist())) for row, other in zip(exact, approximate))
    return found / float(exact.numel())


class NeighborIndexContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = clustered_features()

    def test_both_backends_are_registered_under_their_option_value(self):
        self.assertEqual(sorted(NEIGHBOR_INDEXES), ["exact", "lsh"])
        self.assertIs(NEIGHBOR_INDEXES["exact"], ExactNeighbors)
        self.assertIs(NEIGHBOR_INDEXES["lsh"], LshNeighbors)

    def test_auto_switches_to_the_approximate_index_on_large_datasets(self):
        self.assertEqual(resolve_backend("auto", 19_999, 20_000), "exact")
        self.assertEqual(resolve_backend("auto", 20_000, 20_000), "lsh")
        self.assertEqual(resolve_backend("exact", 10**9, 20_000), "exact")
        self.assertIsInstance(build_index("auto", 10, ann_threshold=20_000), ExactNeighbors)
        self.assertIsInstance(build_index("auto", 50_000, ann_threshold=20_000), LshNeighbors)
        with self.assertRaisesRegex(ValueError, "Unknown neighbour backend"):
            build_index("annoy", 10)

    def test_every_index_excludes_the_query_and_sorts_by_similarity(self):
        for index in (ExactNeighbors(chunk_size=64), LshNeighbors(chunk_size=64, seed=1)):
            with self.subTest(backend=index.name):
                indices, similarities = index.search(self.features, NEIGHBOURS)
                self.assertEqual(indices.shape, (SAMPLES, NEIGHBOURS))
                self.assertEqual(similarities.shape, (SAMPLES, NEIGHBOURS))
                rows = torch.arange(SAMPLES).unsqueeze(1)
                self.assertFalse(bool((indices == rows).any()), "a row is never its own neighbour")
                self.assertTrue(bool((similarities[:, :-1] >= similarities[:, 1:]).all()))
                for row in range(0, SAMPLES, 37):
                    self.assertEqual(len(set(indices[row].tolist())), NEIGHBOURS, "no duplicates")

    def test_reported_similarities_are_the_true_cosine_similarities(self):
        for index in (ExactNeighbors(chunk_size=64), LshNeighbors(chunk_size=64, seed=1)):
            with self.subTest(backend=index.name):
                indices, similarities = index.search(self.features, NEIGHBOURS)
                rows = torch.arange(SAMPLES).unsqueeze(1).expand_as(indices)
                expected = (self.features[rows] * self.features[indices]).sum(dim=2)
                torch.testing.assert_close(similarities, expected, rtol=1e-5, atol=1e-6)

    def test_both_indexes_are_deterministic(self):
        for index in (ExactNeighbors(chunk_size=64), LshNeighbors(chunk_size=64, seed=5)):
            with self.subTest(backend=index.name):
                first, _ = index.search(self.features, NEIGHBOURS)
                second, _ = index.search(self.features, NEIGHBOURS)
                torch.testing.assert_close(first, second)

    def test_the_approximate_index_recovers_the_exact_neighbourhood(self):
        exact, _ = ExactNeighbors(chunk_size=64).search(self.features, NEIGHBOURS)
        approximate, _ = LshNeighbors(chunk_size=64, seed=1, tables=8, bucket_size=32).search(
            self.features, NEIGHBOURS,
        )
        self.assertGreaterEqual(recall(exact, approximate), 0.9, "LSH must find the same neighbourhood")

    def test_a_tiny_dataset_still_needs_two_samples(self):
        with self.assertRaisesRegex(ValueError, "At least two samples"):
            topk_neighbors(self.features[:1], NEIGHBOURS)
        indices, _ = topk_neighbors(self.features[:3], 10)
        self.assertEqual(indices.shape, (3, 2), "k is capped at the number of other samples")


if __name__ == "__main__":
    unittest.main()
