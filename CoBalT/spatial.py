"""CRPv4 spatial evidence in the frozen SpLiCE vocabulary.

The legacy CoBalT reproduction in :mod:`CoBalT.model` intentionally remains
unchanged.  This module implements the CRPv4 branch: one frozen CLIP visual
encoder, optional trainable slots in the native visual width, and CLIP's own
frozen normalization/projection before matching the SpLiCE dictionary.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from CoBalT.model import active_slots, attention_distillation, concept_assignment_confidence


FeatureSource = Literal["vanilla", "sclip"]


def _sclip_attention(attention: nn.MultiheadAttention, tokens: torch.Tensor) -> torch.Tensor:
    """Apply SCLIP's query-query plus key-key attention to batch-first tokens."""

    if attention.in_proj_weight is None:
        raise ValueError("SCLIP requires a visual attention layer with a combined QKV projection.")
    batch, token_count, width = tokens.shape
    heads = attention.num_heads
    head_width = width // heads
    query, key, value = F.linear(
        tokens, attention.in_proj_weight, attention.in_proj_bias
    ).chunk(3, dim=-1)

    def split_heads(value_tensor: torch.Tensor) -> torch.Tensor:
        return value_tensor.view(batch, token_count, heads, head_width).transpose(1, 2)

    query, key, value = map(split_heads, (query, key, value))
    scale = head_width**-0.5
    query_attention = F.softmax((query @ query.transpose(-2, -1)) * scale, dim=-1)
    key_attention = F.softmax((key @ key.transpose(-2, -1)) * scale, dim=-1)
    # The reference SCLIP implementation sums the two row-stochastic maps.
    attended = (query_attention + key_attention) @ value
    attended = attended.transpose(1, 2).reshape(batch, token_count, width)
    return attention.out_proj(attended)


class FrozenClipPatchEncoder(nn.Module):
    """Expose native patch tokens and the frozen CLIP visual projection."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        feature_source: FeatureSource = "vanilla",
    ) -> None:
        super().__init__()
        if feature_source not in {"vanilla", "sclip"}:
            raise ValueError(f"Unsupported spatial feature source {feature_source!r}.")
        import open_clip

        clip = open_clip.create_model(model_name, pretrained=pretrained)
        visual = clip.visual
        required = ("_embeds", "transformer", "ln_post", "proj", "patch_size")
        missing = [name for name in required if not hasattr(visual, name)]
        if missing:
            raise ValueError(
                "CRPv4 spatial extraction requires an OpenCLIP VisionTransformer; "
                f"missing visual attributes: {missing}."
            )
        self.visual = visual.eval()
        self.visual.requires_grad_(False)
        self.feature_source = feature_source
        self.native_dim = int(visual.ln_post.normalized_shape[0])
        projection = visual.proj
        self.output_dim = int(projection.shape[1]) if projection is not None else self.native_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        return self

    @torch.no_grad()
    def native_patches(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        visual = self.visual
        tokens = visual._embeds(images)
        blocks = visual.transformer.resblocks
        if self.feature_source == "vanilla":
            tokens = visual.transformer(tokens)
        else:
            for block in blocks[:-1]:
                tokens = block(tokens)
            final = blocks[-1]
            normalized = final.ln_1(tokens)
            tokens = tokens + final.ls_1(_sclip_attention(final.attn, normalized))
            tokens = tokens + final.ls_2(final.mlp(final.ln_2(tokens)))
        patch_tokens = tokens[:, 1:]
        patch_height = images.shape[-2] // int(visual.patch_size[0])
        patch_width = images.shape[-1] // int(visual.patch_size[1])
        if patch_tokens.shape[1] != patch_height * patch_width:
            raise ValueError("CLIP patch-token count does not match the input patch grid.")
        return patch_tokens.float(), (patch_height, patch_width)

    def project(self, native_vectors: torch.Tensor) -> torch.Tensor:
        projected = self.visual.ln_post(native_vectors.to(self.visual.ln_post.weight.dtype))
        if self.visual.proj is not None:
            projected = projected @ self.visual.proj
        return F.normalize(projected.float(), dim=-1)


class TokenSlotAttention(nn.Module):
    """Competitive learned slots over native CLIP patch tokens."""

    def __init__(self, num_slots: int, dim: int) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.normal_(self.slots, std=0.02)

    def forward(
        self,
        patches: torch.Tensor,
        grid: tuple[int, int],
        temperature: float,
        center: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if patches.ndim != 3:
            raise ValueError("patches must have shape [batch, patches, native_dim].")
        logits = torch.einsum(
            "nd,bpd->bnp", F.normalize(self.slots, dim=-1), F.normalize(patches, dim=-1)
        )
        centered = logits if center is None else logits - center.view(1, -1, 1)
        attention = F.softmax(centered / temperature, dim=1)
        normalizer = attention.sum(dim=2).clamp_min(1e-6)
        slot_vectors = torch.einsum("bnp,bpd->bnd", attention, patches) / normalizer.unsqueeze(-1)
        maps = attention.reshape(patches.shape[0], self.slots.shape[0], *grid)
        return slot_vectors, maps, logits.reshape(patches.shape[0], self.slots.shape[0], *grid)


def projected_slot_consistency(
    student: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if not valid.any():
        return student.sum() * 0.0
    cosine = (F.normalize(student[valid], dim=-1) * F.normalize(teacher.detach()[valid], dim=-1)).sum(-1)
    return (1.0 - cosine).mean()


@dataclass
class SpatialDiscoveryOutput:
    loss: torch.Tensor
    distillation: torch.Tensor
    semantic_consistency: torch.Tensor
    concept_agreement: torch.Tensor


class SpatialSpliceDiscoveryModel(nn.Module):
    """Label-free slots whose outputs stay in the frozen SpLiCE concept space."""

    def __init__(
        self,
        encoder: FrozenClipPatchEncoder,
        dictionary: torch.Tensor,
        num_slots: int = 4,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.07,
        teacher_momentum: float = 0.99,
        center_momentum: float = 0.9,
        semantic_weight: float = 1.0,
    ) -> None:
        super().__init__()
        dictionary = torch.as_tensor(dictionary, dtype=torch.float32)
        if dictionary.ndim != 2 or dictionary.shape[1] != encoder.output_dim:
            raise ValueError("SpLiCE dictionary and projected CLIP slots must share a dimension.")
        self.encoder = encoder
        self.register_buffer("dictionary", F.normalize(dictionary, dim=-1), persistent=False)
        self.student_grouping = TokenSlotAttention(num_slots, encoder.native_dim)
        self.teacher_grouping = copy.deepcopy(self.student_grouping).requires_grad_(False)
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.teacher_momentum = teacher_momentum
        self.center_momentum = center_momentum
        self.semantic_weight = semantic_weight
        self.register_buffer("teacher_center", torch.zeros(num_slots))

    def _group(self, images: torch.Tensor, teacher: bool = False):
        patches, grid = self.encoder.native_patches(images)
        grouping = self.teacher_grouping if teacher else self.student_grouping
        temperature = self.teacher_temperature if teacher else self.student_temperature
        center = self.teacher_center if teacher else None
        slots, attention, logits = grouping(patches, grid, temperature, center)
        return slots, self.encoder.project(slots), attention, logits

    def forward(
        self, views: torch.Tensor, boxes: torch.Tensor, flips: torch.Tensor
    ) -> SpatialDiscoveryOutput:
        if views.ndim != 5 or views.shape[1] != 2:
            raise ValueError("Expected views with shape [batch, 2, channels, height, width].")
        _, s1_projected, s1_attention, _ = self._group(views[:, 0])
        _, s2_projected, s2_attention, _ = self._group(views[:, 1])
        with torch.no_grad():
            _, t1_projected, t1_attention, t1_logits = self._group(views[:, 0], teacher=True)
            _, t2_projected, t2_attention, t2_logits = self._group(views[:, 1], teacher=True)

        valid_12 = active_slots(s1_attention) & active_slots(t2_attention)
        valid_21 = active_slots(s2_attention) & active_slots(t1_attention)
        distillation = 0.5 * (
            attention_distillation(
                s1_attention, t2_attention, boxes[:, 0], boxes[:, 1], flips[:, 0], flips[:, 1]
            )
            + attention_distillation(
                s2_attention, t1_attention, boxes[:, 1], boxes[:, 0], flips[:, 1], flips[:, 0]
            )
        )
        semantic_consistency = 0.5 * (
            projected_slot_consistency(s1_projected, t2_projected, valid_12)
            + projected_slot_consistency(s2_projected, t1_projected, valid_21)
        )
        loss = distillation + self.semantic_weight * semantic_consistency
        with torch.no_grad():
            student_concepts = (s1_projected @ self.dictionary.T).argmax(dim=-1)
            teacher_concepts = (t2_projected @ self.dictionary.T).argmax(dim=-1)
            concept_agreement = (
                student_concepts[valid_12].eq(teacher_concepts[valid_12]).float().mean()
                if valid_12.any()
                else torch.zeros((), device=views.device)
            )
            batch_center = torch.cat((t1_logits, t2_logits), dim=0).mean(dim=(0, 2, 3))
            self.teacher_center.mul_(self.center_momentum).add_(
                batch_center, alpha=1.0 - self.center_momentum
            )
        return SpatialDiscoveryOutput(loss, distillation, semantic_consistency, concept_agreement)

    @torch.no_grad()
    def update_teacher(self) -> None:
        for student_parameter, teacher_parameter in zip(
            self.student_grouping.parameters(), self.teacher_grouping.parameters()
        ):
            teacher_parameter.mul_(self.teacher_momentum).add_(
                student_parameter, alpha=1.0 - self.teacher_momentum
            )

    @torch.no_grad()
    def spatial_evidence(
        self, images: torch.Tensor, concepts_per_region: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, projected, attention, _ = self._group(images, teacher=True)
        return aggregate_spatial_evidence(
            projected,
            attention.mean(dim=(2, 3)),
            self.dictionary,
            concepts_per_region,
            concept_assignment_confidence(attention),
        )


@torch.no_grad()
def aggregate_spatial_evidence(
    regions: torch.Tensor,
    coverage: torch.Tensor,
    dictionary: torch.Tensor,
    concepts_per_region: int,
    confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sparse per-image concept evidence from patches or learned slots."""

    if concepts_per_region <= 0:
        raise ValueError("concepts_per_region must be positive.")
    dictionary = F.normalize(dictionary.float(), dim=-1)
    similarities = (F.normalize(regions.float(), dim=-1) @ dictionary.T).clamp_min(0)
    per_region = min(concepts_per_region, dictionary.shape[0])
    values, indices = similarities.topk(per_region, dim=-1)
    weighted = values * coverage.unsqueeze(-1)
    dense = torch.zeros(
        regions.shape[0], dictionary.shape[0], device=regions.device, dtype=weighted.dtype
    )
    dense.scatter_add_(1, indices.flatten(1), weighted.flatten(1))
    output_count = min(regions.shape[1] * per_region, dictionary.shape[0])
    evidence, concept_indices = dense.topk(output_count, dim=1)
    if confidence is None:
        confidence = torch.ones(regions.shape[0], device=regions.device)
    return concept_indices.cpu(), evidence.cpu(), confidence.float().clamp(0, 1).cpu()


@torch.no_grad()
def patchwise_spatial_evidence(
    encoder: FrozenClipPatchEncoder,
    dictionary: torch.Tensor,
    images: torch.Tensor,
    concepts_per_region: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patches, _ = encoder.native_patches(images)
    projected = encoder.project(patches)
    coverage = torch.full(
        projected.shape[:2], 1.0 / projected.shape[1], device=projected.device
    )
    return aggregate_spatial_evidence(projected, coverage, dictionary, concepts_per_region)
