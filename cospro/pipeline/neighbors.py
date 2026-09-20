"""Nearest neighbours in the projected embedding space.

The audit asks the same question of every concept group: which images stop being neighbours once
the group's subspace is projected out. Two indexes answer it. ``ExactNeighbors`` compares every
pair in chunks and is the reference; ``LshNeighbors`` buckets by random hyperplanes and reranks the
candidates, which is what makes a dataset of hundreds of thousands of images tractable. Both are
deterministic given a seed, and ``tests/test_neighbor_index.py`` holds the approximate one to the
exact one.
"""

from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn.functional as F


def orthonormal_basis(directions: torch.Tensor, tolerance: float = 1e-6) -> torch.Tensor:
    """Return a column basis, dropping near-collinear dictionary directions."""

    if directions.ndim != 2 or directions.numel() == 0:
        raise ValueError("directions must be a non-empty rank-2 tensor.")
    _, singular_values, right_vectors = torch.linalg.svd(directions.float(), full_matrices=False)
    cutoff = max(float(singular_values.max()) * tolerance, tolerance)
    rank = int((singular_values > cutoff).sum().item())
    if rank == 0:
        raise ValueError("Concept directions do not span a numerically stable subspace.")
    return right_vectors[:rank].T.contiguous()


def project_out(centered_embeddings: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project full centered CLIP vectors away from a concept subspace."""

    residual = centered_embeddings - (centered_embeddings @ basis) @ basis.T
    norms = residual.norm(dim=1)
    if torch.any(norms <= 1e-12):
        raise ValueError("Projection removed an entire image embedding.")
    return F.normalize(residual, dim=1)


def _exact_topk_neighbors(
    features: torch.Tensor, k: int, chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices, similarities = [], []
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        similarity = features[start:stop] @ features.T
        local_rows = torch.arange(stop - start, device=features.device)
        similarity[local_rows, torch.arange(start, stop, device=features.device)] = -torch.inf
        values, neighbours = similarity.topk(k, dim=1)
        indices.append(neighbours)
        similarities.append(values)
    return torch.cat(indices), torch.cat(similarities)


def _lsh_topk_neighbors(
    features: torch.Tensor,
    k: int,
    chunk_size: int,
    *,
    tables: int,
    bucket_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate cosine neighbours using deterministic SimHash buckets.

    Each random-hyperplane table supplies ``k`` candidates per row. Candidates
    are then reranked with their exact cosine similarities. Extending small
    buckets in hash-sorted order guarantees enough non-self candidates even
    for an unlucky hash partition.
    """

    n_samples, dimensions = features.shape
    bits = max(1, min(30, int(round(math.log2(max(2, n_samples / bucket_size))))))
    powers = (2 ** torch.arange(bits, dtype=torch.int64, device=features.device)).view(1, -1)
    table_indices = []
    for table in range(tables):
        generator = torch.Generator(device=features.device)
        generator.manual_seed(int(seed) + 1_000_003 * table)
        planes = torch.randn(
            dimensions, bits, generator=generator, device=features.device, dtype=features.dtype,
        )
        codes = (((features @ planes) >= 0).to(torch.int64) * powers).sum(dim=1)
        order = torch.argsort(codes, stable=True)
        sorted_codes = codes[order]
        boundaries = torch.cat((
            torch.zeros(1, dtype=torch.long, device=features.device),
            torch.where(sorted_codes[1:] != sorted_codes[:-1])[0] + 1,
            torch.tensor([n_samples], dtype=torch.long, device=features.device),
        )).cpu().tolist()
        candidates = torch.empty((n_samples, k), dtype=torch.long, device=features.device)
        for boundary_index in range(len(boundaries) - 1):
            start, stop = boundaries[boundary_index], boundaries[boundary_index + 1]
            # Small buckets borrow adjacent entries in deterministic hash order.
            needed = max(k + 1, bucket_size)
            extra = max(0, needed - (stop - start))
            pool_start = max(0, start - extra // 2)
            pool_stop = min(n_samples, stop + extra - (start - pool_start))
            pool_start = max(0, pool_start - max(0, needed - (pool_stop - pool_start)))
            query_rows = order[start:stop]
            pool_rows = order[pool_start:pool_stop]
            for offset in range(0, len(query_rows), chunk_size):
                rows = query_rows[offset:offset + chunk_size]
                similarity = features[rows] @ features[pool_rows].T
                similarity.masked_fill_(rows.view(-1, 1) == pool_rows.view(1, -1), -torch.inf)
                candidates[rows] = pool_rows[similarity.topk(k, dim=1).indices]
        table_indices.append(candidates)

    candidates = torch.cat(table_indices, dim=1).sort(dim=1).values
    final_indices, final_similarities = [], []
    row_ids = torch.arange(n_samples, device=features.device)
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        rows = row_ids[start:stop]
        candidate_rows = candidates[start:stop]
        similarity = (features[rows].unsqueeze(1) * features[candidate_rows]).sum(dim=2)
        similarity.masked_fill_(candidate_rows == rows.view(-1, 1), -torch.inf)
        duplicate = torch.zeros_like(candidate_rows, dtype=torch.bool)
        duplicate[:, 1:] = candidate_rows[:, 1:] == candidate_rows[:, :-1]
        similarity.masked_fill_(duplicate, -torch.inf)
        values, positions = similarity.topk(k, dim=1)
        final_indices.append(candidate_rows.gather(1, positions))
        final_similarities.append(values)
    return torch.cat(final_indices), torch.cat(final_similarities)


def topk_neighbors(
    features: torch.Tensor,
    k: int,
    chunk_size: int = 512,
    *,
    backend: str = "exact",
    ann_threshold: int = 20_000,
    ann_tables: int = 8,
    ann_bucket_size: int = 512,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine neighbours via exact chunks or a scalable PyTorch LSH index."""

    n_samples = len(features)
    if n_samples < 2:
        raise ValueError("At least two samples are required to construct relations.")
    k = min(k, n_samples - 1)
    resolved = "lsh" if backend == "auto" and n_samples >= ann_threshold else backend
    resolved = "exact" if resolved == "auto" else resolved
    if resolved == "exact":
        return _exact_topk_neighbors(features, k, chunk_size)
    if resolved == "lsh":
        return _lsh_topk_neighbors(
            features, k, chunk_size, tables=ann_tables, bucket_size=ann_bucket_size, seed=seed,
        )
    raise ValueError(f"Unknown neighbour backend: {backend!r}")
