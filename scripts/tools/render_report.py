"""Render a small HTML report from a JSON report specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from splice.reporting import render_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="JSON with title and sections fields")
    parser.add_argument("output", help="Destination .html file")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    path = render_report(str(spec["title"]), spec["sections"], args.output)
    print(path)


if __name__ == "__main__":
    main()
