import torch
import torch.nn as nn

from CoBalT.model import (
    CoBalTDiscoveryModel,
    ConceptDictionary,
    SemanticGrouping,
    align_to_source_overlap,
    concept_assignment_confidence,
)
from CoBalT.sampler import ConceptBalancedSampler, inferred_worst_group_accuracy
from CoBalT.spatial import (
    SpatialSpliceDiscoveryModel,
    TokenSlotAttention,
    _sclip_attention,
    aggregate_spatial_evidence,
)


def test_semantic_grouping_is_normalized_over_slots():
    grouping = SemanticGrouping(num_slots=4, dim=8)
    slots, attention, _ = grouping(torch.randn(3, 8, 5, 5), temperature=0.1)
    assert slots.shape == (3, 4, 8)
    assert attention.shape == (3, 4, 5, 5)
    assert torch.allclose(attention.sum(dim=1), torch.ones(3, 5, 5), atol=1e-6)


def test_concept_assignment_confidence_measures_slot_separation():
    attention = torch.zeros(2, 2, 2, 2)
    attention[0, 0] = 0.5
    attention[0, 1] = 0.5
    attention[1, 0] = 0.9
    attention[1, 1] = 0.1
    confidence = concept_assignment_confidence(attention)
    assert torch.allclose(confidence, torch.tensor([0.5, 0.9]))


def test_identity_overlap_alignment_preserves_map():
    maps = torch.rand(2, 4, 6, 6)
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    aligned = align_to_source_overlap(maps, boxes, torch.zeros(2, dtype=torch.bool), boxes)
    assert torch.allclose(aligned, maps, atol=1e-5)


def test_dictionary_assignments_and_update_are_finite():
    dictionary = ConceptDictionary(size=3, dim=4, momentum=0.9)
    slots = torch.randn(2, 3, 4)
    active = torch.ones(2, 3, dtype=torch.bool)
    probabilities = dictionary.student_probabilities(slots, temperature=0.1)
    dictionary.update(slots, active)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.isfinite(dictionary.codes).all()


def test_concept_balanced_sampler_uses_inverse_class_frequency():
    concepts = torch.zeros(4, 1, dtype=torch.long)
    labels = torch.tensor([0, 0, 0, 1])
    sampler = ConceptBalancedSampler(concepts, labels, sampling_lambda=1.0, num_samples=6000, seed=7)
    sampled_labels = labels[torch.tensor(list(sampler))]
    minority_fraction = sampled_labels.float().mean().item()
    assert 0.72 < minority_fraction < 0.78


def test_inferred_worst_group_counts_multi_concept_membership():
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])
    concepts = torch.tensor([[0, 2], [0, 3], [1, 2], [1, 3]])
    assert inferred_worst_group_accuracy(predictions, labels, concepts) == 0.0


def test_discovery_forward_and_teacher_update_smoke():
    model = CoBalTDiscoveryModel(
        num_slots=3,
        codebook_size=4,
        slot_dim=8,
        hidden_dim=16,
        backbone="resnet18",
        pretrained=False,
    )
    model.train()
    views = torch.randn(2, 2, 3, 64, 64)
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 0.8, 0.8], [0.2, 0.2, 1.0, 1.0]],
            [[0.0, 0.1, 0.9, 1.0], [0.1, 0.0, 1.0, 0.9]],
        ]
    )
    flips = torch.tensor([[False, True], [True, False]])
    before = next(model.teacher_encoder.parameters()).detach().clone()
    output = model(views, boxes, flips)
    output.loss.backward()
    with torch.no_grad():
        next(model.student_encoder.parameters()).add_(0.01)
    model.update_teacher()
    after = next(model.teacher_encoder.parameters()).detach()
    assert torch.isfinite(output.loss)
    assert not torch.equal(before, after)


def test_token_slots_aggregate_native_clip_width_before_projection():
    grouping = TokenSlotAttention(num_slots=3, dim=6)
    slots, attention, _ = grouping(torch.randn(2, 4, 6), (2, 2), temperature=0.1)
    assert slots.shape == (2, 3, 6)
    assert attention.shape == (2, 3, 2, 2)
    assert torch.allclose(attention.sum(dim=1), torch.ones(2, 2, 2), atol=1e-6)


def test_sclip_attention_reuses_frozen_qkv_shapes():
    attention = nn.MultiheadAttention(8, 2, batch_first=True)
    output = _sclip_attention(attention, torch.randn(3, 5, 8))
    assert output.shape == (3, 5, 8)
    assert torch.isfinite(output).all()


def test_spatial_evidence_is_sparse_named_dictionary_evidence():
    regions = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), dim=-1)
    dictionary = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    indices, evidence, confidence = aggregate_spatial_evidence(
        regions,
        torch.tensor([[0.75, 0.25]]),
        dictionary,
        concepts_per_region=1,
        confidence=torch.tensor([0.8]),
    )
    assert indices.shape == evidence.shape == (1, 2)
    assert set(indices[0].tolist()) == {0, 1}
    assert torch.all(evidence >= 0)
    assert torch.allclose(confidence, torch.tensor([0.8]))


def test_spatial_slot_discovery_trains_only_slot_queries():
    class FakeFrozenEncoder(nn.Module):
        native_dim = 4
        output_dim = 3

        def __init__(self):
            super().__init__()
            self.frozen_marker = nn.Parameter(torch.ones(()), requires_grad=False)
            self.register_buffer(
                "projection",
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
            )

        @torch.no_grad()
        def native_patches(self, images):
            pooled = torch.nn.functional.adaptive_avg_pool2d(images, (2, 2))
            fourth = pooled.mean(dim=1, keepdim=True)
            return torch.cat((pooled, fourth), dim=1).flatten(2).transpose(1, 2), (2, 2)

        def project(self, vectors):
            return torch.nn.functional.normalize(vectors @ self.projection, dim=-1)

    model = SpatialSpliceDiscoveryModel(
        FakeFrozenEncoder(), torch.eye(3), num_slots=2, semantic_weight=0.5
    )
    views = torch.randn(2, 2, 3, 8, 8)
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]] * 2] * 2)
    flips = torch.zeros(2, 2, dtype=torch.bool)
    output = model(views, boxes, flips)
    output.loss.backward()
    assert model.student_grouping.slots.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert torch.isfinite(output.loss)
