from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialResNet(nn.Module):
    """ImageNet ResNet truncated before global pooling."""

    def __init__(self, name: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            network = models.resnet50(weights=weights)
            self.out_dim = 2048
        elif name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            network = models.resnet18(weights=weights)
            self.out_dim = 512
        else:
            raise ValueError("Only resnet50 and resnet18 discovery backbones are supported.")
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layers = nn.Sequential(network.layer1, network.layer2, network.layer3, network.layer4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(self.stem(images))


class SpatialProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class SlotPredictor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        shape = slots.shape
        return self.network(slots.reshape(-1, shape[-1])).reshape(shape)


class SemanticGrouping(nn.Module):
    """Equations 7--8: competition over slots and attention-weighted pooling."""

    def __init__(self, num_slots: int, dim: int) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.normal_(self.slots, std=0.02)

    def forward(
        self,
        patches: torch.Tensor,
        temperature: float,
        center: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.einsum(
            "nd,bdhw->bnhw", F.normalize(self.slots, dim=-1), F.normalize(patches, dim=1)
        )
        centered = logits if center is None else logits - center.view(1, -1, 1, 1)
        attention = F.softmax(centered / temperature, dim=1)
        normalizer = attention.sum(dim=(2, 3), keepdim=False).clamp_min(1e-6)
        slot_vectors = torch.einsum("bnhw,bdhw->bnd", attention, patches) / normalizer.unsqueeze(-1)
        return slot_vectors, attention, logits


class ConceptDictionary(nn.Module):
    """EMA vector-quantized concept dictionary from equations 1--4."""

    def __init__(self, size: int, dim: int, momentum: float = 0.9) -> None:
        super().__init__()
        initial = F.normalize(torch.randn(size, dim), dim=-1)
        self.register_buffer("codes", initial)
        self.momentum = momentum

    def distances(self, slots: torch.Tensor) -> torch.Tensor:
        slots = F.normalize(slots, dim=-1)
        codes = F.normalize(self.codes, dim=-1)
        return torch.sum((slots.unsqueeze(-2) - codes.view(1, 1, *codes.shape)) ** 2, dim=-1)

    def student_probabilities(self, slots: torch.Tensor, temperature: float) -> torch.Tensor:
        return F.softmax(-self.distances(slots) / temperature, dim=-1)

    def teacher_assignments(self, slots: torch.Tensor) -> torch.Tensor:
        return self.distances(slots).argmin(dim=-1)

    @torch.no_grad()
    def update(self, teacher_slots: torch.Tensor, active: torch.Tensor) -> None:
        assignments = self.teacher_assignments(teacher_slots)
        normalized = F.normalize(teacher_slots, dim=-1)
        for code_index in range(self.codes.shape[0]):
            selected = active & assignments.eq(code_index)
            if selected.any():
                batch_mean = normalized[selected].mean(dim=0)
                self.codes[code_index].mul_(self.momentum).add_(
                    batch_mean, alpha=1.0 - self.momentum
                )
                self.codes[code_index].copy_(F.normalize(self.codes[code_index], dim=0))


def active_slots(attention: torch.Tensor) -> torch.Tensor:
    winners = attention.argmax(dim=1)
    one_hot = F.one_hot(winners, num_classes=attention.shape[1])
    return one_hot.sum(dim=(1, 2)).gt(0)


def _overlap_box(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(first[:, :2], second[:, :2])
    right_bottom = torch.minimum(first[:, 2:], second[:, 2:])
    right_bottom = torch.maximum(right_bottom, left_top + 1e-6)
    return torch.cat((left_top, right_bottom), dim=1)


def align_to_source_overlap(
    maps: torch.Tensor,
    view_boxes: torch.Tensor,
    flips: torch.Tensor,
    overlap: torch.Tensor,
) -> torch.Tensor:
    """Inverse-crop attention maps onto a shared source-image overlap grid."""

    batch, _, height, width = maps.shape
    y_fraction = (torch.arange(height, device=maps.device, dtype=maps.dtype) + 0.5) / height
    x_fraction = (torch.arange(width, device=maps.device, dtype=maps.dtype) + 0.5) / width
    source_y = overlap[:, 1, None] + y_fraction[None] * (overlap[:, 3] - overlap[:, 1])[:, None]
    source_x = overlap[:, 0, None] + x_fraction[None] * (overlap[:, 2] - overlap[:, 0])[:, None]
    local_y = (source_y - view_boxes[:, 1, None]) / (
        view_boxes[:, 3] - view_boxes[:, 1]
    ).clamp_min(1e-6)[:, None]
    local_x = (source_x - view_boxes[:, 0, None]) / (
        view_boxes[:, 2] - view_boxes[:, 0]
    ).clamp_min(1e-6)[:, None]
    local_x = torch.where(flips[:, None], 1.0 - local_x, local_x)
    grid_x = local_x[:, None, :].expand(batch, height, width)
    grid_y = local_y[:, :, None].expand(batch, height, width)
    grid = torch.stack((2.0 * grid_x - 1.0, 2.0 * grid_y - 1.0), dim=-1)
    return F.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def attention_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    student_box: torch.Tensor,
    teacher_box: torch.Tensor,
    student_flip: torch.Tensor,
    teacher_flip: torch.Tensor,
) -> torch.Tensor:
    overlap = _overlap_box(student_box, teacher_box)
    aligned_student = align_to_source_overlap(student, student_box, student_flip, overlap)
    aligned_teacher = align_to_source_overlap(teacher, teacher_box, teacher_flip, overlap)
    aligned_student = aligned_student / aligned_student.sum(dim=1, keepdim=True).clamp_min(1e-6)
    aligned_teacher = aligned_teacher / aligned_teacher.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return -(aligned_teacher.detach() * aligned_student.clamp_min(1e-8).log()).sum(dim=1).mean()


def slot_contrastive_loss(
    predicted_student: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query = F.normalize(predicted_student[valid], dim=-1)
    key = F.normalize(teacher.detach()[valid], dim=-1)
    if query.shape[0] < 2:
        return predicted_student.sum() * 0.0
    logits = query @ key.t() / temperature
    labels = torch.arange(query.shape[0], device=query.device)
    return F.cross_entropy(logits, labels) * (2.0 * temperature)


@dataclass
class DiscoveryOutput:
    loss: torch.Tensor
    distillation: torch.Tensor
    contrastive: torch.Tensor
    vector_quantization: torch.Tensor
    code_usage: torch.Tensor


class CoBalTDiscoveryModel(nn.Module):
    def __init__(
        self,
        num_slots: int = 4,
        codebook_size: int = 8,
        slot_dim: int = 32,
        hidden_dim: int = 1024,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.07,
        contrastive_temperature: float = 0.2,
        teacher_momentum: float = 0.99,
        codebook_momentum: float = 0.9,
        center_momentum: float = 0.9,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.contrastive_temperature = contrastive_temperature
        self.teacher_momentum = teacher_momentum
        self.center_momentum = center_momentum

        self.student_encoder = SpatialResNet(backbone, pretrained=pretrained)
        self.student_projector = SpatialProjector(
            self.student_encoder.out_dim, hidden_dim=hidden_dim, out_dim=slot_dim
        )
        self.student_grouping = SemanticGrouping(num_slots, slot_dim)
        self.predictor = SlotPredictor(slot_dim, hidden_dim)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_projector = copy.deepcopy(self.student_projector)
        self.teacher_grouping = copy.deepcopy(self.student_grouping)
        for module in (self.teacher_encoder, self.teacher_projector, self.teacher_grouping):
            module.requires_grad_(False)
        self.dictionary = ConceptDictionary(codebook_size, slot_dim, codebook_momentum)
        self.register_buffer("teacher_center", torch.zeros(num_slots))

    def _student(self, images: torch.Tensor):
        patches = self.student_projector(self.student_encoder(images))
        return self.student_grouping(patches, self.student_temperature)

    @torch.no_grad()
    def _teacher(self, images: torch.Tensor):
        patches = self.teacher_projector(self.teacher_encoder(images))
        return self.teacher_grouping(patches, self.teacher_temperature, self.teacher_center)

    def _vq_loss(
        self, student_slots: torch.Tensor, teacher_slots: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        probabilities = self.dictionary.student_probabilities(
            student_slots, self.student_temperature
        )
        assignments = self.dictionary.teacher_assignments(teacher_slots.detach())
        losses = F.nll_loss(
            probabilities.clamp_min(1e-8).log().reshape(-1, probabilities.shape[-1]),
            assignments.reshape(-1),
            reduction="none",
        ).view_as(assignments)
        return (losses * valid).sum() / valid.sum().clamp_min(1)

    def forward(
        self, views: torch.Tensor, boxes: torch.Tensor, flips: torch.Tensor
    ) -> DiscoveryOutput:
        if views.ndim != 5 or views.shape[1] != 2:
            raise ValueError("Expected views with shape [batch, 2, channels, height, width].")
        student_one = self._student(views[:, 0])
        student_two = self._student(views[:, 1])
        teacher_one = self._teacher(views[:, 0])
        teacher_two = self._teacher(views[:, 1])

        s1_slots, s1_attention, _ = student_one
        s2_slots, s2_attention, _ = student_two
        t1_slots, t1_attention, t1_logits = teacher_one
        t2_slots, t2_attention, t2_logits = teacher_two
        valid_12 = active_slots(s1_attention) & active_slots(t2_attention)
        valid_21 = active_slots(s2_attention) & active_slots(t1_attention)

        distillation = 0.5 * (
            attention_distillation(
                s1_attention,
                t2_attention,
                boxes[:, 0],
                boxes[:, 1],
                flips[:, 0],
                flips[:, 1],
            )
            + attention_distillation(
                s2_attention,
                t1_attention,
                boxes[:, 1],
                boxes[:, 0],
                flips[:, 1],
                flips[:, 0],
            )
        )
        contrastive = 0.5 * (
            slot_contrastive_loss(
                self.predictor(s1_slots), t2_slots, valid_12, self.contrastive_temperature
            )
            + slot_contrastive_loss(
                self.predictor(s2_slots), t1_slots, valid_21, self.contrastive_temperature
            )
        )
        vector_quantization = 0.5 * (
            self._vq_loss(s1_slots, t2_slots, valid_12)
            + self._vq_loss(s2_slots, t1_slots, valid_21)
        )
        loss = distillation + contrastive + vector_quantization

        with torch.no_grad():
            teacher_slots = torch.cat((t1_slots, t2_slots), dim=0)
            teacher_active = torch.cat((active_slots(t1_attention), active_slots(t2_attention)), dim=0)
            self.dictionary.update(teacher_slots, teacher_active)
            assignments = self.dictionary.teacher_assignments(teacher_slots)
            code_usage = torch.bincount(
                assignments[teacher_active], minlength=self.dictionary.codes.shape[0]
            ).float()
            batch_center = torch.cat((t1_logits, t2_logits), dim=0).mean(dim=(0, 2, 3))
            self.teacher_center.mul_(self.center_momentum).add_(
                batch_center, alpha=1.0 - self.center_momentum
            )
        return DiscoveryOutput(loss, distillation, contrastive, vector_quantization, code_usage)

    @torch.no_grad()
    def update_teacher(self) -> None:
        pairs = (
            (self.student_encoder, self.teacher_encoder),
            (self.student_projector, self.teacher_projector),
            (self.student_grouping, self.teacher_grouping),
        )
        for student, teacher in pairs:
            for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters()):
                teacher_parameter.mul_(self.teacher_momentum).add_(
                    student_parameter, alpha=1.0 - self.teacher_momentum
                )

    @torch.no_grad()
    def infer_concepts(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slots, attention, _ = self._teacher(images)
        assignments = self.dictionary.teacher_assignments(slots)
        active = active_slots(attention)
        return assignments.masked_fill(~active, -1), attention
