"""Materialize the public Waterbirds mirror into this repository's expected layout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SPLITS = (("train", 0), ("validation", 1), ("test", 2))
REPOSITORY = "grodino/waterbirds"


def _complete_dataset(root: Path) -> bool:
    metadata_path = root / "metadata.csv"
    if not metadata_path.is_file():
        return False
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows) and all((root / row["img_filename"]).is_file() for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    destination = args.output_root.resolve() / "waterbirds"
    if _complete_dataset(destination):
        print(f"[INFO] Reusing complete Waterbirds dataset at {destination}")
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional 'datasets' package to download Waterbirds.") from exc

    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, int | str]] = []
    image_id = 0
    for split_name, split_id in SPLITS:
        split = load_dataset(REPOSITORY, split=split_name)
        split_directory = destination / "images" / split_name
        split_directory.mkdir(parents=True, exist_ok=True)
        for row_index, row in enumerate(split):
            relative_path = Path("images") / split_name / f"{row_index:06d}.png"
            output_path = destination / relative_path
            if not output_path.is_file():
                row["image"].convert("RGB").save(output_path, format="PNG")
            records.append(
                {
                    "img_id": image_id,
                    "img_filename": relative_path.as_posix(),
                    "y": int(row["label"]),
                    "place": int(row["place"]),
                    "split": split_id,
                }
            )
            image_id += 1
            if (row_index + 1) % 500 == 0:
                print(f"[INFO] Materialized {split_name}: {row_index + 1}/{len(split)}", flush=True)

    temporary_metadata = destination / "metadata.csv.tmp"
    with temporary_metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("img_id", "img_filename", "y", "place", "split"),
        )
        writer.writeheader()
        writer.writerows(records)
    temporary_metadata.replace(destination / "metadata.csv")
    if not _complete_dataset(destination):
        raise RuntimeError(f"Waterbirds materialization is incomplete at {destination}")
    print(f"[INFO] Waterbirds ready: {len(records)} images at {destination}")


if __name__ == "__main__":
    main()
