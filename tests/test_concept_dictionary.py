"""Concept dictionaries: a file dictionary for ablations, and the bundled ones left exactly as they were.

The dictionary tensor feeds every cache, concept group, teacher graph and training run, so the
bundled vocabularies must keep producing the same tensor under the same names, and a file
dictionary must be embedded by the same code. A deterministic stand-in for the CLIP text encoder
makes that checkable without model weights; ``scripts/verify_concept_dictionary.sbatch`` repeats
the check on the cluster with the real encoder.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cospro.pipeline.dictionary import (
    BUNDLED_ORDER,
    ConceptDictionary,
    read_word_file,
    resolve_dictionary,
    select_words,
)
from scripts.tools.cache_splice_dataset import cache_config_name, cache_provenance
from third_party.splice import splice as splice_library
from cospro.tracking.artifacts import PROJECT_ROOT

EMBEDDING_DIM = 512  # the bundled ViT-B-32 image mean is 512-dimensional
_WEIGHTS = torch.randn(3, EMBEDDING_DIM, generator=torch.Generator().manual_seed(0))


class FakeTextEncoder(torch.nn.Module):
    """A deterministic stand-in for the CLIP text tower: equal words give equal embeddings."""

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.sin(tokens.float() @ _WEIGHTS) + 0.01 * tokens.float().sum()


def fake_tokenizer(word: str) -> torch.Tensor:
    codes = [ord(character) for character in word] or [0]
    return torch.tensor([[sum(codes) % 997, len(codes), codes[0]]])


FAKE_OPEN_CLIP = types.SimpleNamespace(
    create_model=lambda *arguments, **keywords: FakeTextEncoder(),
    get_tokenizer=lambda name: fake_tokenizer,
)


def dictionary_tensor(home: Path, **keywords) -> torch.Tensor:
    """The dictionary ``splice.load`` builds, with its embedding cache under ``home``."""

    with patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home)}), \
            patch.dict(sys.modules, {"open_clip": FAKE_OPEN_CLIP}):
        model = splice_library.load("open_clip:ViT-B-32", device="cpu", **keywords)
    return model.dictionary.detach().cpu()


def write_words(directory: Path, name: str, lines: list[str]) -> Path:
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cache_arguments(**overrides) -> argparse.Namespace:
    values = dict(
        dataset="waterbirds", output_root=Path("/features"), splice_model="open_clip:ViT-B-32",
        splice_pretrained="laion2b_s34b_b79k", splice_vocab="openimages_v7", splice_vocab_size=-1,
        splice_l1_penalty=0.25,
    )
    return argparse.Namespace(**{**values, **overrides})


class WordFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_every_non_blank_line_is_a_concept(self):
        path = write_words(self.root, "words.txt", ["water", "", "  forest  ", "#", "##"])
        self.assertEqual(read_word_file(path), ["water", "forest", "#", "##"], "LAION has '#' and '##'")

    def test_a_copy_of_a_bundled_vocabulary_reads_back_exactly(self):
        for kind in ("openimages_v7", "laion"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    read_word_file(PROJECT_ROOT / "data" / "vocab" / f"{kind}.txt"),
                    splice_library.get_vocabulary(kind, -1),
                )

    def test_a_repeated_concept_is_refused(self):
        path = write_words(self.root, "words.txt", ["water", "forest", "water"])
        with self.assertRaisesRegex(ValueError, "repeats concepts \\['water'\\]"):
            read_word_file(path)

    def test_an_empty_or_missing_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, "lists no concepts"):
            read_word_file(write_words(self.root, "empty.txt", ["", "   "]))
        with self.assertRaises(FileNotFoundError):
            read_word_file(self.root / "missing.txt")

    def test_a_size_keeps_the_head_or_the_tail(self):
        words = ["a", "b", "c", "d"]
        self.assertEqual(select_words(words, 2, "head"), ["a", "b"])
        self.assertEqual(select_words(words, 2, "tail"), ["c", "d"])
        self.assertEqual(select_words(words, -1, "tail"), words)
        with self.assertRaisesRegex(ValueError, "Unknown dictionary order"):
            select_words(words, 2, "frequency")


class ResolveDictionaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_the_bundled_dictionaries_keep_their_names_and_words(self):
        for kind in ("openimages_v7", "laion"):
            with self.subTest(kind=kind):
                dictionary = resolve_dictionary(kind, size=300)
                self.assertEqual(dictionary.token, kind, "existing cache directories stay valid")
                self.assertEqual(list(dictionary.words), splice_library.get_vocabulary(kind, 300))
                self.assertEqual(dictionary.order, BUNDLED_ORDER[kind])
                self.assertEqual(dictionary.splice_load_arguments(), {"vocabulary": kind, "vocabulary_size": 300})

    def test_a_bundled_dictionary_refuses_file_options(self):
        with self.assertRaisesRegex(ValueError, "bundled"):
            resolve_dictionary("laion", path=self.root / "words.txt")
        with self.assertRaisesRegex(ValueError, "always keeps its tail"):
            resolve_dictionary("laion", order="head")
        with self.assertRaisesRegex(ValueError, "Unknown concept dictionary"):
            resolve_dictionary("wordnet")

    def test_a_file_dictionary_needs_its_path(self):
        with self.assertRaisesRegex(ValueError, "needs the path"):
            resolve_dictionary("file")

    def test_a_file_dictionary_is_named_by_its_content(self):
        first = resolve_dictionary("file", path=write_words(self.root, "ablation.txt", ["water", "forest"]))
        again = resolve_dictionary("file", path=write_words(self.root, "copy.txt", ["water", "forest"]))
        edited = resolve_dictionary("file", path=write_words(self.root, "ablation.txt", ["water", "sky"]))
        self.assertTrue(first.token.startswith("file-ablation-"))
        self.assertEqual(first.sha256, again.sha256, "the same words on another machine compare equal")
        self.assertNotEqual(first.token, edited.token, "an edited file gets a new name")
        self.assertEqual(first.source, "ablation.txt", "no machine-local path enters the provenance")
        arguments = first.splice_load_arguments()
        self.assertEqual(arguments["words"], ["water", "forest"])
        self.assertEqual(arguments["dictionary_id"], f"{first.token}_2")


class DictionaryTensorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_a_file_holding_the_bundled_words_gives_the_bundled_tensor(self):
        bundled = dictionary_tensor(self.root / "named", vocabulary="openimages_v7", vocabulary_size=400)
        words = splice_library.get_vocabulary("openimages_v7", -1)
        path = write_words(self.root, "openimages_copy.txt", words)
        dictionary = resolve_dictionary("file", path=path, size=400, order="head")
        from_file = dictionary_tensor(self.root / "file", **dictionary.splice_load_arguments())
        self.assertTrue(torch.equal(bundled, from_file), "one embedding path for every dictionary")

    def test_the_bundled_tensor_is_reproduced_from_its_cache(self):
        home = self.root / "home"
        first = dictionary_tensor(home, vocabulary="laion", vocabulary_size=200)
        cached = list((home / ".cache" / "splice" / "embeddings").glob("*laion_200_embeddings.pt"))
        self.assertEqual(len(cached), 1, "the historical embedding cache name is kept")
        self.assertTrue(torch.equal(first, dictionary_tensor(home, vocabulary="laion", vocabulary_size=200)))

    def test_editing_a_file_never_reuses_stale_embeddings(self):
        home = self.root / "home"
        path = self.root / "ablation.txt"
        write_words(self.root, "ablation.txt", ["water", "forest", "sky"])
        first = dictionary_tensor(home, **resolve_dictionary("file", path=path).splice_load_arguments())
        write_words(self.root, "ablation.txt", ["water", "forest", "sand"])
        second = dictionary_tensor(home, **resolve_dictionary("file", path=path).splice_load_arguments())
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(len(list((home / ".cache" / "splice" / "embeddings").glob("*file-ablation-*"))), 2)

    def test_an_explicit_word_list_needs_a_cache_name(self):
        with self.assertRaisesRegex(ValueError, "dictionary_id"):
            dictionary_tensor(self.root / "home", vocabulary="file", words=["water"])


class CacheIdentityTests(unittest.TestCase):
    def test_bundled_caches_keep_their_directory_and_provenance(self):
        arguments = cache_arguments()
        dictionary = resolve_dictionary("openimages_v7", size=-1)
        self.assertEqual(
            cache_config_name(arguments),
            "cache_v1__model_open_clip_ViT-B-32__pretrained_laion2b_s34b_b79k__vocab_openimages_v7_all__l1_0p25",
        )
        self.assertEqual(sorted(cache_provenance(arguments, dictionary)), sorted([
            "dataset", "split", "splice_model", "splice_pretrained", "splice_vocab", "splice_vocab_size",
            "splice_l1_penalty",
        ]), "an existing cache still matches what the pipeline expects")

    def test_a_file_cache_records_the_dictionary_it_was_built_from(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_words(Path(directory), "ablation.txt", ["water", "forest"])
            arguments = cache_arguments(splice_vocab="file", splice_vocab_file=path, splice_vocab_order="head")
            dictionary = resolve_dictionary("file", path=path)
            self.assertIn(f"vocab_{dictionary.token}_all", cache_config_name(arguments))
            provenance = cache_provenance(arguments, dictionary)
        self.assertEqual(provenance["concept_dictionary"]["sha256"], dictionary.sha256)
        self.assertEqual(provenance["concept_dictionary"]["word_count"], 2)


class BundledResourcesTests(unittest.TestCase):
    def test_the_bundled_vocabularies_and_mean_ship_with_the_repository(self):
        for relative in ("data/vocab/laion.txt", "data/vocab/openimages_v7.txt",
                         "data/means/open_clip_ViT-B-32_image.pt"):
            with self.subTest(resource=relative):
                self.assertTrue((PROJECT_ROOT / relative).is_file(), "reproduction needs no extra download")


if __name__ == "__main__":
    unittest.main()
