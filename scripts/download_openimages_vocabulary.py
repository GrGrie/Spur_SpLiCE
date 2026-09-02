"""Download the Open Images V7 class-label vocabulary without downloading images."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splice.splice import download_openimages_vocabulary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-root",
        default=None,
        help="Cache root; defaults to ~/.cache/splice.",
    )
    args = parser.parse_args()
    print(download_openimages_vocabulary(args.download_root))


if __name__ == "__main__":
    main()
