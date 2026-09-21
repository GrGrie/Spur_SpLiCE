"""Compare golden training reports from several commits run on one machine.

The golden training snapshot is per operating system, but its probe accuracies come from 16
validation images and flip with the CPU's floating-point kernels, so a snapshot taken on one node
can fail on another with unchanged code. Running several commits on the same node separates the two
causes: reports that agree bit for bit mean the code did not change the numbers.

    python -m tools.maintenance.compare_golden_reports REPORT_DIR --reference HEAD_REPORT.json \\
        [--snapshot tests/golden/training_runs.linux.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def numbers(run: dict) -> dict[str, float]:
    """Every number of one training mode, keyed by a readable path."""

    flat = {f"final_metrics/{key}": value for key, value in run["final_metrics"].items()}
    for epoch, losses in enumerate(run["ssl_losses"], start=1):
        flat.update({f"ssl_losses/epoch {epoch}/{key}": value for key, value in losses.items()})
    return flat


def differences(left: dict, right: dict, *, rel: float = 0.0, abs_: float = 0.0) -> list[str]:
    found = []
    for arm in sorted(set(left["runs"]) | set(right["runs"])):
        if arm not in left["runs"] or arm not in right["runs"]:
            found.append(f"{arm}: present in only one report")
            continue
        a, b = numbers(left["runs"][arm]), numbers(right["runs"][arm])
        for key in sorted(set(a) | set(b)):
            x, y = a.get(key), b.get(key)
            if x is None or y is None:
                found.append(f"{arm}/{key}: missing on one side")
            elif x != y and not (isinstance(x, (int, float)) and isinstance(y, (int, float))
                                 and math.isclose(x, y, rel_tol=rel, abs_tol=abs_) and (rel or abs_)):
                found.append(f"{arm}/{key}: {x!r} vs {y!r}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", type=Path, help="Directory of <commit>.json golden reports.")
    parser.add_argument("--reference", type=Path, required=True, help="The report the others are compared with.")
    parser.add_argument("--snapshot", type=Path, help="A committed snapshot to check the reference against.")
    args = parser.parse_args(argv)

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    identical = True
    for path in sorted(args.reports.glob("*.json")):
        if path.resolve() == args.reference.resolve():
            continue
        found = differences(json.loads(path.read_text(encoding="utf-8")), reference)
        identical &= not found
        verdict = "bit-identical" if not found else f"{len(found)} numbers differ"
        print(f"[golden-compare] {path.stem} vs {args.reference.stem}: {verdict}")
        for line in found[:10]:
            print(f"[golden-compare]    {line}")
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        found = differences(snapshot, reference, rel=1e-3, abs_=1e-4)
        print(f"[golden-compare] snapshot {args.snapshot.name} vs {args.reference.stem}: "
              + ("within tolerance" if not found else f"{len(found)} numbers outside tolerance"))
        for line in found[:20]:
            print(f"[golden-compare]    {line}")
    print("[golden-compare] " + ("SAME NUMBERS: every commit agrees bit for bit on this node" if identical
                                 else "CODE CHANGED NUMBERS: see the lines above"))
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
