from experiments.spurious_eval.datasets.celeba import CELEBA_SPEC
from experiments.spurious_eval.datasets.spur_cifar10 import SPUR_CIFAR10_SPEC
from experiments.spurious_eval.datasets.waterbirds import WATERBIRDS_SPEC


DATASET_REGISTRY = {
    "CelebA": CELEBA_SPEC,
    "celebA": CELEBA_SPEC,
    "celeba": CELEBA_SPEC,
    "spur_cifar10": SPUR_CIFAR10_SPEC,
    "waterbirds": WATERBIRDS_SPEC,
}


def get_dataset_spec(name: str) -> dict:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}. Choices: {sorted(DATASET_REGISTRY)}") from exc
