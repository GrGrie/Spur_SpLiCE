from __future__ import annotations


def resolve_evaluation_split(requested_split: str | None, final_test: bool) -> str:
    """Resolve a leakage-safe evaluation split.

    Validation is the development default. Access to the test split requires
    an explicit final-evaluation declaration so an old command cannot silently
    turn a hyperparameter sweep into test-set selection.
    """

    if final_test:
        if requested_split not in {None, "test"}:
            raise ValueError("--final_test cannot be combined with an explicit validation split.")
        return "test"
    if requested_split == "test":
        raise ValueError(
            "Test evaluation is reserved for a locked final configuration. "
            "Use the validation default during development, or pass --final_test explicitly."
        )
    return requested_split or "val"


def resolve_probe_mode(requested_mode: str | None, final_test: bool) -> str:
    """Use periodic validation curves, but evaluate test only at the final epoch."""

    if final_test:
        if requested_mode not in {None, "final"}:
            raise ValueError("--final_test requires --linear_probe_mode final.")
        return "final"
    return requested_mode or "periodic"
