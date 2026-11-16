"""Supervised training pieces of the concept-curves project: data, classifier, one epoch, evaluation."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from cospro.data.transforms import _RecordedFlip, _RecordedResizedCrop
from cospro.metrics import compute_group_metrics
from cospro.models.resnet import build_resnet_encoder

#: Section 5.3 of docs/PROJECT_PLAN.md.
TRAINING_DEFAULTS = {
    "resnet18_large": {"epochs": 100, "learning_rate": 0.05, "weight_decay": 5e-4, "early_epochs": 20},
    "resnet50_pretrained": {"epochs": 30, "learning_rate": 0.001, "weight_decay": 1e-4, "early_epochs": 5},
}
FALLBACK_DEFAULTS = TRAINING_DEFAULTS["resnet18_large"]
CROP_SCALE = (0.5, 1.0)


class TrainViews(Dataset):
    """One augmented view per training image with its row, class label and crop box: (view, row, y, box).

    ``box`` is [5]: top, left, height and width of the crop as fractions of the image and the flip flag,
    the layout ``warp_maps`` reads. ``subset`` is a split of a ``SpuriousDataset``.
    """

    def __init__(self, subset, image_size: int, mean, std) -> None:
        self.source = subset.dataset
        self.indices = [int(index) for index in subset.indices]
        self.labels = subset.y_array.long()
        self.crop = _RecordedResizedCrop(transforms.RandomResizedCrop(image_size, scale=CROP_SCALE))
        self.flip = _RecordedFlip(transforms.RandomHorizontalFlip())
        self.finish = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        image = self.flip(self.crop(self.source.get_input(self.indices[row])))
        box = torch.tensor([*self.crop.box, float(self.flip.flipped)], dtype=torch.float32)
        return self.finish(image), row, int(self.labels[row]), box


class PlainImages(Dataset):
    """Unaugmented images of a split with their row, class label and metadata: (image, row, y, metadata)."""

    def __init__(self, subset, image_size: int, mean, std) -> None:
        self.source = subset.dataset
        self.indices = [int(index) for index in subset.indices]
        self.labels = subset.y_array.long()
        self.metadata = subset.metadata_array
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean, std),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        return self.transform(self.source.get_input(self.indices[row])), row, int(self.labels[row]), self.metadata[row]


class AuditBatches:
    """A loader of ``PlainImages`` narrowed to the (images, rows, labels) the audit reads."""

    def __init__(self, loader) -> None:
        self.loader = loader

    def __iter__(self):
        for images, rows, labels, _ in self.loader:
            yield images, rows, labels


class Classifier(nn.Module):
    """A ResNet encoder with a linear class head."""

    def __init__(self, name: str, num_classes: int) -> None:
        super().__init__()
        self.encoder, width = build_resnet_encoder(name)
        self.head = nn.Linear(width, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images))


def train_epoch(model, loader, optimizer, scaler, device, amp: bool,
                transform_batch: Callable | None = None) -> dict[str, float]:
    """One epoch of cross-entropy training. ``transform_batch(views, rows, boxes)`` returns the views to
    train on and a boolean per image saying whether it was changed."""

    model.train()
    loss_sum = correct = seen = changed = 0.0
    for views, rows, labels, boxes in loader:
        views, labels = views.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if transform_batch is not None:
            with torch.no_grad():
                views, swapped = transform_batch(views, rows, boxes.to(device))
            changed += float(swapped.sum())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and str(device).startswith("cuda")):
            logits = model(views)
            loss = F.cross_entropy(logits.float(), labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach()) * len(labels)
        correct += float((logits.argmax(dim=1) == labels).sum())
        seen += len(labels)
    return {"loss": loss_sum / seen, "accuracy": correct / seen, "swapped_fraction": changed / seen}


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> dict:
    """Worst-group accuracy, average accuracy and the accuracy of every (class, attribute) group."""

    was_training = model.training
    model.eval()
    predictions, labels, metadata = [], [], []
    try:
        for images, _, batch_labels, batch_metadata in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=amp and str(device).startswith("cuda")):
                predictions.append(model(images.to(device)).argmax(dim=1).cpu())
            labels.append(torch.as_tensor(batch_labels))
            metadata.append(batch_metadata)
    finally:
        model.train(was_training)
    metrics = compute_group_metrics(torch.cat(predictions), torch.cat(labels), torch.cat(metadata))
    return {
        "wga": float(metrics.worst_group),
        "acc": float(metrics.average),
        "group_acc": [round(float(value), 4) for value in metrics.group_accuracy],
        "group_counts": [int(value) for value in metrics.group_counts],
    }
