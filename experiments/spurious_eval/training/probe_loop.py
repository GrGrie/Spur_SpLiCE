from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.spurious_eval.metrics import topk_accuracy
from experiments.spurious_eval.models.resnet import LinearClassifier


def make_feature_loader(feature_dataset: TensorDataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    return DataLoader(
        feature_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=loader_generator,
    )


def extract_features(encoder: torch.nn.Module, loader: DataLoader, device: torch.device) -> TensorDataset:
    encoder.eval()
    features, labels, metadata = [], [], []
    with torch.no_grad():
        for images, batch_labels, batch_metadata in loader:
            images = images.to(device, non_blocking=True)
            batch_features = encoder(images)
            features.append(batch_features.cpu())
            labels.append(batch_labels.cpu())
            metadata.append(batch_metadata.cpu())
    return TensorDataset(torch.cat(features, dim=0), torch.cat(labels, dim=0), torch.cat(metadata, dim=0))


def train_one_epoch(
    feature_loader: DataLoader,
    classifier: LinearClassifier,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    classifier.train()
    total_loss = 0.0
    total_seen = 0
    total_acc = 0.0
    all_predictions, all_labels, all_metadata = [], [], []

    for features, labels, metadata in feature_loader:
        features = features.to(device)
        labels = labels.to(device)
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        acc1 = topk_accuracy(outputs, labels, topk=(1,))[0].item()
        total_loss += loss.item() * batch_size
        total_acc += acc1 * batch_size
        total_seen += batch_size
        all_predictions.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
        all_metadata.append(metadata.cpu())

    return (
        total_loss / total_seen,
        total_acc / total_seen,
        torch.cat(all_predictions),
        torch.cat(all_labels),
        torch.cat(all_metadata),
    )


def validate(
    feature_loader: DataLoader,
    classifier: LinearClassifier,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    classifier.eval()
    total_loss = 0.0
    total_seen = 0
    total_acc = 0.0
    all_predictions, all_labels, all_metadata = [], [], []

    with torch.no_grad():
        for features, labels, metadata in feature_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            batch_size = labels.shape[0]
            acc1 = topk_accuracy(outputs, labels, topk=(1,))[0].item()
            total_loss += loss.item() * batch_size
            total_acc += acc1 * batch_size
            total_seen += batch_size
            all_predictions.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
            all_metadata.append(metadata.cpu())

    return (
        total_loss / total_seen,
        total_acc / total_seen,
        torch.cat(all_predictions),
        torch.cat(all_labels),
        torch.cat(all_metadata),
    )
