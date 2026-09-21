"""Compare two frozen SpLiCE dataset caches field by field.

The cache is the root of every later artifact, so a code change that touches how it is built is
checked by building it twice and comparing: tensors must be equal element for element, sample IDs
and the vocabulary must be equal in order. Provenance is compared unless ``--ignore-provenance``
says the two caches were built from different, equivalent dictionaries on purpose.

    python -m scripts.tools.compare_splice_caches BASELINE.pt CANDIDATE.pt --report report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from splice.artifacts import atomic_write_json, sha256_file

TENSOR_FIELDS = ("clip_embeddings", "image_mean", "splice_codes", "dictionary")
LIST_FIELDS = ("sample_ids", "vocabulary")


def compare_splice_caches(baseline: dict, candidate: dict, *, ignore_provenance: bool = False) -> dict[str, Any]:
    """Field-by-field equality of two caches, with the largest difference of every tensor."""

    fields: dict[str, Any] = {}
    for name in TENSOR_FIELDS:
        left, right = baseline.get(name), candidate.get(name)
        if left is None or right is None or tuple(left.shape) != tuple(right.shape):
            fields[name] = {"equal": False, "reason": "missing or shape differs",
                            "shapes": [None if left is None else list(left.shape),
                                       None if right is None else list(right.shape)]}
            continue
        difference = (left.float() - right.float()).abs()
        fields[name] = {
            "equal": bool(torch.equal(left, right)),
            "shape": list(left.shape),
            "max_abs_difference": float(difference.max()) if difference.numel() else 0.0,
        }
    for name in LIST_FIELDS:
        left, right = list(baseline.get(name) or []), list(candidate.get(name) or [])
        mismatch = next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), None)
        fields[name] = {
            "equal": left == right,
            "lengths": [len(left), len(right)],
            "first_mismatch": mismatch,
        }
    fields["cache_version"] = {"equal": baseline.get("cache_version") == candidate.get("cache_version")}
    provenance_equal = baseline.get("provenance") == candidate.get("provenance")
    fields["provenance"] = {
        "equal": provenance_equal,
        "checked": not ignore_provenance,
        "baseline": baseline.get("provenance"),
        "candidate": candidate.get("provenance"),
    }
    checked = [value["equal"] for name, value in fields.items() if name != "provenance"]
    if not ignore_provenance:
        checked.append(provenance_equal)
    return {"equal": all(checked), "fields": fields}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--ignore-provenance", action="store_true",
                        help="Compare representations only, for caches built from equivalent dictionaries.")
    parser.add_argument("--label", default="", help="A name for this comparison in the report.")
    parser.add_argument("--report", type=Path, help="Append-free JSON report of this comparison.")
    args = parser.parse_args(argv)

    load = lambda path: torch.load(path, map_location="cpu", weights_only=True)  # noqa: E731
    result = compare_splice_caches(load(args.baseline), load(args.candidate),
                                   ignore_provenance=args.ignore_provenance)
    result.update({
        "label": args.label,
        "baseline": {"path": str(args.baseline), "sha256": sha256_file(args.baseline)},
        "candidate": {"path": str(args.candidate), "sha256": sha256_file(args.candidate)},
    })
    if args.report:
        atomic_write_json(args.report, result)
    verdict = "EQUAL" if result["equal"] else "DIFFERENT"
    differing = [name for name, value in result["fields"].items()
                 if not value["equal"] and (name != "provenance" or not args.ignore_provenance)]
    print(f"[compare] {args.label or 'caches'}: {verdict}" + (f" (differs in {differing})" if differing else ""))
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
