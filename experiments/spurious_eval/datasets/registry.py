from experiments.spurious_eval.datasets.waterbirds import WATERBIRDS_SPEC


DATASET_REGISTRY = {
    "waterbirds": WATERBIRDS_SPEC,
}


def get_dataset_spec(name: str) -> dict:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}. Choices: {sorted(DATASET_REGISTRY)}") from exc
