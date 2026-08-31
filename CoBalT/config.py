from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PaperConfig:
    dataset: str
    num_classes: int
    discovery_epochs: int
    classifier_epochs: int
    sampling_lambda: float
    num_slots: int = 4
    codebook_size: int = 8
    slot_dim: int = 32
    projector_hidden_dim: int = 1024
    student_temperature: float = 0.1
    teacher_temperature: float = 0.07
    contrastive_temperature: float = 0.2
    teacher_momentum: float = 0.99
    codebook_momentum: float = 0.9
    center_momentum: float = 0.9
    discovery_lr: float = 2e-4
    discovery_weight_decay: float = 5e-4
    discovery_batch_size: int = 128
    classifier_lr: float = 1e-4
    classifier_momentum: float = 0.9
    classifier_weight_decay: float = 0.1
    classifier_batch_size: int = 128
    image_size: int = 224

    def as_dict(self) -> dict:
        return asdict(self)


_CONFIGS = {
    "waterbirds": PaperConfig(
        dataset="waterbirds",
        num_classes=2,
        discovery_epochs=50,
        classifier_epochs=300,
        sampling_lambda=2.0,
    ),
    "celeba": PaperConfig(
        dataset="celeba",
        num_classes=2,
        discovery_epochs=20,
        classifier_epochs=60,
        sampling_lambda=1.0,
    ),
}


def paper_config(dataset: str) -> PaperConfig:
    canonical = dataset.lower()
    if canonical not in _CONFIGS:
        raise ValueError(
            f"This repository currently reproduces the paper on {sorted(_CONFIGS)}; got {dataset!r}."
        )
    return _CONFIGS[canonical]
