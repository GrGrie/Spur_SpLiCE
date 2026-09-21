"""The dataset downloader: what it accepts as the paper's data and how it rewrites CelebA.

The downloads themselves are too large for a unit test, so these tests pin the decisions around
them: a file is accepted only with the expected hash, a prepared dataset is recognised and left
alone, and the CelebA annotations are rewritten in the exact CSV dialect the paper's files use.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cospro.cli import download_datasets as tool


class HashCheckTests(unittest.TestCase):
    def test_a_file_is_accepted_only_with_its_expected_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.bin"
            path.write_bytes(b"waterbirds")
            tool.require_hash(path, tool.sha256_file(path))
            tool.require_hash(path, tool.md5_file(path), algorithm="md5")
            with self.assertRaisesRegex(RuntimeError, "cannot reproduce the paper"):
                tool.require_hash(path, "0" * 64)

    def test_a_waterbirds_folder_counts_as_prepared_only_with_the_paper_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertFalse(tool.waterbirds_is_ready(folder), "no metadata yet")
            (folder / "metadata.csv").write_text("img_id,img_filename,y,split,place\n", encoding="utf-8")
            self.assertFalse(tool.waterbirds_is_ready(folder), "some other metadata")

    def test_an_unrelated_waterbirds_folder_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "waterbirds").mkdir()
            (data / "waterbirds" / "metadata.csv").write_text("mine\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "move it away first"):
                tool.prepare_waterbirds(data, data / ".downloads", keep_archives=False)


class CelebAConversionTests(unittest.TestCase):
    def convert(self, attribute_lines: list[str], partition_lines: list[str]) -> Path:
        directory = Path(self.directory.name)
        official, output = directory / "official", directory / "celeba"
        official.mkdir()
        output.mkdir()
        (official / "list_attr_celeba.txt").write_text("\n".join(attribute_lines) + "\n", encoding="utf-8")
        (official / "list_eval_partition.txt").write_text("\n".join(partition_lines) + "\n", encoding="utf-8")
        tool.convert_celeba_annotations(official, output)
        return output

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_the_official_annotations_become_the_papers_csv_dialect(self):
        output = self.convert(
            ["2", "Blond_Hair Male ", "000001.jpg -1  1", "000002.jpg  1 -1"],
            ["000001.jpg 0", "000002.jpg 2"],
        )
        self.assertEqual(
            (output / "list_attr_celeba.csv").read_bytes(),
            b"image_id,Blond_Hair,Male\r\n000001.jpg,-1,1\r\n000002.jpg,1,-1",
            "CRLF line ends and no newline after the last row, as in the paper's files",
        )
        self.assertEqual(
            (output / "list_eval_partition.csv").read_bytes(),
            b"image_id,partition\r\n000001.jpg,0\r\n000002.jpg,2",
        )

    def test_annotations_with_the_wrong_shape_are_refused(self):
        with self.assertRaisesRegex(RuntimeError, "official shape"):
            self.convert(["3", "Blond_Hair Male", "000001.jpg -1 1"], ["000001.jpg 0"])
        self.directory.cleanup()
        self.directory = tempfile.TemporaryDirectory()
        with self.assertRaisesRegex(RuntimeError, "official shape"):
            self.convert(["1", "Blond_Hair Male", "000001.jpg -1"], ["000001.jpg 0"])

    def test_the_official_md5_table_names_every_file_the_downloader_needs(self):
        table = tool.celeba_official_md5()
        for filename in tool.CELEBA_OFFICIAL_FILES:
            self.assertRegex(table[filename], r"^[0-9a-f]{32}$")


if __name__ == "__main__":
    unittest.main()
