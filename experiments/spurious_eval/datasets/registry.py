from experiments.spurious_eval.datasets.celeba import CELEBA_SPEC
from experiments.spurious_eval.datasets.spur_cifar10 import SPUR_CIFAR10_SPEC
from experiments.spurious_eval.datasets.waterbirds import WATERBIRDS_SPEC


# Artifact sample IDs include the dataset name, so every producer and consumer
# must agree on one spelling. Keep aliases at the CLI boundary, but always
# return the canonical lower-case name before an artifact is written.
CANONICAL_DATASET_REGISTRY = {
    "celeba": CELEBA_SPEC,
    "spur_cifar10": SPUR_CIFAR10_SPEC,
    "waterbirds": WATERBIRDS_SPEC,
}
DATASET_ALIASES = {
    "celeba": "celeba",
    "spur-cifar10": "spur_cifar10",
    "spur_cifar10": "spur_cifar10",
    "waterbirds": "waterbirds",
}


def canonical_dataset_name(name: str) -> str:
    normalized = str(name).strip().lower()
    try:
        return DATASET_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset: {name}. Choices: {sorted(CANONICAL_DATASET_REGISTRY)}"
        ) from exc


# Backward compatible for code that indexes the registry directly with a
# historical CelebA spelling. New command-line parsers canonicalize first.
DATASET_REGISTRY = {
    **CANONICAL_DATASET_REGISTRY,
    "CelebA": CELEBA_SPEC,
    "celebA": CELEBA_SPEC,
}


def get_dataset_spec(name: str) -> dict:
    return CANONICAL_DATASET_REGISTRY[canonical_dataset_name(name)]
