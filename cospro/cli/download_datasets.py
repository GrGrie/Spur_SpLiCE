"""Download Waterbirds and CelebA into the layout the dataset adapters read, and prove they match.

Both datasets are verified against the exact files the paper was computed from, so a reproduction
starts from the same metadata:

Waterbirds  the official Group DRO archive from Stanford. The archive and its ``metadata.csv`` are
            checked against their SHA-256; the metadata hash equals the one recorded by the paper's
            cluster runs.
CelebA      the official aligned images and annotation files. Every file is checked against the MD5
            that torchvision ships for the official release, then the annotations are converted to
            the CSV layout the adapter reads, byte for byte as used in the paper, which the SHA-256
            of the two CSV files confirms.

The official CelebA files live on Google Drive, which rate-limits large downloads. When the
automatic download is refused, download them by hand from the CelebA site, put
``img_align_celeba.zip``, ``list_attr_celeba.txt`` and ``list_eval_partition.txt`` in one directory
and pass it as ``--celeba-archive-dir``; the files are verified the same way.

    python -m cospro.cli.download_datasets --data-folder ~/Datasets
    python -m cospro.cli.download_datasets --data-folder ~/Datasets --datasets celeba \\
        --celeba-archive-dir ~/Downloads/celeba
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

WATERBIRDS_URL = "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz"
WATERBIRDS_ARCHIVE_SHA256 = "56c51b77f7d17283ac97003d973e9107e8af9ceebffcb0c8be14d024f846f951"
WATERBIRDS_ARCHIVE_ROOT = "waterbird_complete95_forest2water2"
WATERBIRDS_METADATA_SHA256 = "2f023b9dc371f6fbb82c7cb810ff0f03e19c54c8055a2e37517ae7bb56644362"
WATERBIRDS_SPLIT_COUNTS = {0: 4795, 1: 1199, 2: 5794}

# The CSV files the CelebA adapter reads, as the paper's runs used them.
CELEBA_ATTRIBUTES_SHA256 = "67be7fa3c3a2a58013c78a6312ef26edfcb139109d5e8bf9727acc286566a286"
CELEBA_PARTITION_SHA256 = "62b4e7ba17c331076e4bd867cef693fae43835b5d6de4ef87108ae05cb7e0a95"
CELEBA_IMAGE_COUNT = 202_599
CELEBA_SPLIT_COUNTS = {0: 162_770, 1: 19_867, 2: 19_962}
CELEBA_OFFICIAL_FILES = ("img_align_celeba.zip", "list_attr_celeba.txt", "list_eval_partition.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> Path:
    """Fetch ``url`` into ``destination`` through a temporary file, reporting progress."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[download] {url}", flush=True)
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        received, reported = 0, -1
        for chunk in iter(lambda: response.read(1 << 20), b""):
            handle.write(chunk)
            received += len(chunk)
            percent = int(100 * received / total) if total else -1
            if percent >= reported + 10:
                reported = percent
                print(f"[download]   {received / 2**20:.0f} MiB" + (f" ({percent}%)" if total else ""), flush=True)
    temporary.replace(destination)
    return destination


def require_hash(path: Path, expected: str, *, algorithm: str = "sha256") -> None:
    actual = (sha256_file if algorithm == "sha256" else md5_file)(path)
    if actual != expected:
        raise RuntimeError(
            f"{path} has {algorithm} {actual}, expected {expected}. "
            "Delete it and download it again; a partial or altered file cannot reproduce the paper."
        )
    print(f"[verify] {path.name}: {algorithm} matches", flush=True)


def write_manifest(directory: Path, payload: dict) -> None:
    payload = {**payload, "verified_at": datetime.now(timezone.utc).isoformat()}
    (directory / "dataset_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# --- Waterbirds ------------------------------------------------------------------------------


def waterbirds_is_ready(directory: Path) -> bool:
    metadata = directory / "metadata.csv"
    return metadata.is_file() and sha256_file(metadata) == WATERBIRDS_METADATA_SHA256


def check_waterbirds(directory: Path) -> None:
    import pandas as pd

    metadata = pd.read_csv(directory / "metadata.csv")
    counts = {int(split): int(count) for split, count in metadata["split"].value_counts().items()}
    if counts != WATERBIRDS_SPLIT_COUNTS:
        raise RuntimeError(f"Waterbirds split sizes {counts} differ from {WATERBIRDS_SPLIT_COUNTS}.")
    missing = [name for name in metadata["img_filename"] if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Waterbirds is missing {len(missing)} images, for example {missing[0]}.")
    print(f"[verify] waterbirds: {len(metadata)} images, splits {counts}", flush=True)


def prepare_waterbirds(data_folder: Path, downloads: Path, keep_archives: bool) -> Path:
    directory = data_folder / "waterbirds"
    if waterbirds_is_ready(directory):
        print(f"[waterbirds] already prepared at {directory}", flush=True)
        check_waterbirds(directory)
        return directory
    if directory.exists():
        raise RuntimeError(f"{directory} exists but its metadata.csv is not the paper's; move it away first.")

    archive = downloads / f"{WATERBIRDS_ARCHIVE_ROOT}.tar.gz"
    if not (archive.is_file() and sha256_file(archive) == WATERBIRDS_ARCHIVE_SHA256):
        download(WATERBIRDS_URL, archive)
    require_hash(archive, WATERBIRDS_ARCHIVE_SHA256)

    staging = data_folder / f".{WATERBIRDS_ARCHIVE_ROOT}.extracting"
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[waterbirds] extracting into {directory}", flush=True)
    with tarfile.open(archive, "r:gz") as bundle:
        # The data filter refuses absolute paths and links out of the target where Python has it.
        if hasattr(tarfile, "data_filter"):
            bundle.extractall(staging, filter="data")
        else:
            bundle.extractall(staging)
    (staging / WATERBIRDS_ARCHIVE_ROOT).rename(directory)
    staging.rmdir()

    require_hash(directory / "metadata.csv", WATERBIRDS_METADATA_SHA256)
    check_waterbirds(directory)
    write_manifest(directory, {
        "dataset": "waterbirds", "source": WATERBIRDS_URL,
        "archive_sha256": WATERBIRDS_ARCHIVE_SHA256, "metadata_sha256": WATERBIRDS_METADATA_SHA256,
    })
    if not keep_archives:
        archive.unlink()
    return directory


# --- CelebA ----------------------------------------------------------------------------------


def celeba_official_md5() -> dict[str, str]:
    """The MD5 of every official CelebA file, as torchvision ships them."""

    from torchvision.datasets import CelebA

    return {filename: md5 for _, md5, filename in CelebA.file_list}


def celeba_is_ready(directory: Path) -> bool:
    attributes, partition = directory / "list_attr_celeba.csv", directory / "list_eval_partition.csv"
    return (
        attributes.is_file() and partition.is_file()
        and sha256_file(attributes) == CELEBA_ATTRIBUTES_SHA256
        and sha256_file(partition) == CELEBA_PARTITION_SHA256
    )


def whitespace_rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_crlf_csv(path: Path, rows: list[list[str]]) -> None:
    """The CSV dialect the paper's CelebA files use: CRLF line ends, no newline after the last row."""

    path.write_bytes("\r\n".join(",".join(row) for row in rows).encode("utf-8"))


def convert_celeba_annotations(official: Path, directory: Path) -> None:
    """Rewrite the official whitespace-separated annotations as the adapter's CSV files."""

    attributes = whitespace_rows(official / "list_attr_celeba.txt")
    count, names, samples = int(attributes[0][0]), attributes[1], attributes[2:]
    if count != len(samples) or any(len(row) != len(names) + 1 for row in samples):
        raise RuntimeError("list_attr_celeba.txt does not have the official shape.")
    write_crlf_csv(directory / "list_attr_celeba.csv", [["image_id", *names], *samples])
    partition = whitespace_rows(official / "list_eval_partition.txt")
    write_crlf_csv(directory / "list_eval_partition.csv", [["image_id", "partition"], *partition])


def check_celeba(directory: Path) -> None:
    partition = [line.split(",") for line in (directory / "list_eval_partition.csv").read_text(
        encoding="utf-8").splitlines()[1:]]
    counts: dict[int, int] = {}
    for _, split in partition:
        counts[int(split)] = counts.get(int(split), 0) + 1
    if counts != CELEBA_SPLIT_COUNTS:
        raise RuntimeError(f"CelebA split sizes {counts} differ from {CELEBA_SPLIT_COUNTS}.")
    images = directory / "img_align_celeba"
    missing = [image for image, _ in partition if not (images / image).is_file()]
    if missing:
        raise RuntimeError(f"CelebA is missing {len(missing)} aligned images, for example {missing[0]}.")
    print(f"[verify] celeba: {len(partition)} images, splits {counts}", flush=True)


def fetch_official_celeba(archive_dir: Path | None, downloads: Path) -> Path:
    """A directory holding the three official files, downloading them when none was given."""

    if archive_dir is not None:
        return archive_dir
    target = downloads / "celeba-official"
    from torchvision.datasets import CelebA

    print("[celeba] downloading the official files from Google Drive through torchvision", flush=True)
    try:
        # torchvision downloads every official file, checks its MD5 and unpacks the images into
        # <root>/celeba; the unpacked copy is left there and the zip is used below.
        CelebA(str(target), split="all", target_type="attr", download=True)
    except Exception as error:  # Google Drive quotas, a missing gdown, no network
        raise RuntimeError(
            f"The automatic CelebA download failed ({error}). Download "
            f"{', '.join(CELEBA_OFFICIAL_FILES)} from the CelebA website into one directory and "
            "pass it as --celeba-archive-dir. torchvision needs the 'gdown' package for Google Drive."
        ) from error
    return target / "celeba"


def prepare_celeba(data_folder: Path, downloads: Path, archive_dir: Path | None, keep_archives: bool) -> Path:
    directory = data_folder / "celeba"
    if celeba_is_ready(directory):
        print(f"[celeba] already prepared at {directory}", flush=True)
        check_celeba(directory)
        return directory

    official = fetch_official_celeba(archive_dir, downloads)
    expected = celeba_official_md5()
    for filename in CELEBA_OFFICIAL_FILES:
        path = official / filename
        if not path.is_file():
            raise FileNotFoundError(f"The official CelebA file {filename} is missing from {official}.")
        require_hash(path, expected[filename], algorithm="md5")

    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "img_align_celeba").is_dir():
        print(f"[celeba] unpacking the aligned images into {directory}", flush=True)
        with zipfile.ZipFile(official / "img_align_celeba.zip") as bundle:
            bundle.extractall(directory)
    convert_celeba_annotations(official, directory)
    require_hash(directory / "list_attr_celeba.csv", CELEBA_ATTRIBUTES_SHA256)
    require_hash(directory / "list_eval_partition.csv", CELEBA_PARTITION_SHA256)
    check_celeba(directory)
    write_manifest(directory, {
        "dataset": "celeba", "source": "official CelebA release (img_align_celeba, attributes, partition)",
        "official_md5": {name: expected[name] for name in CELEBA_OFFICIAL_FILES},
        "attributes_sha256": CELEBA_ATTRIBUTES_SHA256, "partition_sha256": CELEBA_PARTITION_SHA256,
    })
    if not keep_archives and archive_dir is None:
        shutil.rmtree(downloads / "celeba-official", ignore_errors=True)
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-folder", type=Path, required=True,
                        help="The directory passed to training as --data_folder or DATA_FOLDER.")
    parser.add_argument("--datasets", nargs="+", choices=("waterbirds", "celeba"),
                        default=["waterbirds", "celeba"])
    parser.add_argument("--celeba-archive-dir", type=Path,
                        help="A directory with the official CelebA files downloaded by hand.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep the downloaded archives.")
    args = parser.parse_args(argv)

    data_folder = args.data_folder.expanduser().resolve()
    downloads = data_folder / ".downloads"
    prepared = {}
    if "waterbirds" in args.datasets:
        prepared["waterbirds"] = prepare_waterbirds(data_folder, downloads, args.keep_archives)
    if "celeba" in args.datasets:
        prepared["celeba"] = prepare_celeba(data_folder, downloads, args.celeba_archive_dir, args.keep_archives)
    if downloads.is_dir() and not any(downloads.iterdir()):
        downloads.rmdir()
    for name, directory in prepared.items():
        print(f"[ready] {name}: {directory}")
    print(f"[ready] train with --data_folder {data_folder} (or DATA_FOLDER={data_folder})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
