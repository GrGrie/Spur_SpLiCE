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


def _exact_search(features: torch.Tensor, k: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def _lsh_search(
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


class NeighborIndex:
    """The neighbourhood query the audit makes, in one shape for every backend.

    ``search`` returns the ``k`` nearest rows of ``features`` for every row, excluding the row
    itself, plus their cosine similarities, both sorted by descending similarity.
    """

    #: The ``--neighbor-backend`` value that selects this index.
    name: ClassVar[str] = ""

    def __init__(self, *, chunk_size: int = 512, seed: int = 0, **options) -> None:
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.options = options

    def search(self, features: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def provenance(self) -> dict:
        """What the teacher graph records about how its neighbours were found."""

        return {"backend": self.name, "approximate": False}


NEIGHBOR_INDEXES: dict[str, type[NeighborIndex]] = {}


def register_index(cls: type[NeighborIndex]) -> type[NeighborIndex]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must declare a backend name.")
    if cls.name in NEIGHBOR_INDEXES:
        raise ValueError(f"Neighbour backend {cls.name!r} is registered twice.")
    NEIGHBOR_INDEXES[cls.name] = cls
    return cls


@register_index
class ExactNeighbors(NeighborIndex):
    """Every pair, in chunks. The reference every approximate index is compared with."""

    name = "exact"

    def search(self, features: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _exact_search(features, k, self.chunk_size)


@register_index
class LshNeighbors(NeighborIndex):
    """SimHash buckets plus an exact rerank, for datasets too large to compare pairwise."""

    name = "lsh"

    def search(self, features: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _lsh_search(
            features, k, self.chunk_size,
            tables=int(self.options.get("tables", 8)),
            bucket_size=int(self.options.get("bucket_size", 512)),
            seed=self.seed,
        )

    def provenance(self) -> dict:
        return {
            "backend": self.name,
            "approximate": True,
            "tables": int(self.options.get("tables", 8)),
            "bucket_size": int(self.options.get("bucket_size", 512)),
        }


def resolve_backend(backend: str, n_samples: int, ann_threshold: int) -> str:
    """The backend a request selects. ``auto`` picks LSH once the dataset is large enough."""

    if backend != "auto":
        return backend
    return "lsh" if n_samples >= ann_threshold else "exact"


def build_index(
    backend: str,
    n_samples: int,
    *,
    chunk_size: int = 512,
    ann_threshold: int = 20_000,
    ann_tables: int = 8,
    ann_bucket_size: int = 512,
    seed: int = 0,
) -> NeighborIndex:
    """The index a configuration selects, already resolved from ``auto``."""

    resolved = resolve_backend(backend, n_samples, ann_threshold)
    index_class = NEIGHBOR_INDEXES.get(resolved)
    if index_class is None:
        raise ValueError(f"Unknown neighbour backend: {backend!r}")
    return index_class(chunk_size=chunk_size, seed=seed, tables=ann_tables, bucket_size=ann_bucket_size)


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
    """Cosine neighbours through the index a backend name selects."""

    n_samples = len(features)
    if n_samples < 2:
        raise ValueError("At least two samples are required to construct relations.")
    index = build_index(
        backend, n_samples, chunk_size=chunk_size, ann_threshold=ann_threshold,
        ann_tables=ann_tables, ann_bucket_size=ann_bucket_size, seed=seed,
    )
    return index.search(features, min(k, n_samples - 1))
