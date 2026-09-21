"""Concept dictionaries: the words SpLiCE decomposes every image into.

The dictionary decides which concepts a teacher graph can talk about, so it is part of a result's
identity. Three kinds exist:

``laion``          the SpLiCE LAION vocabulary, bundled in ``data/vocab/laion.txt``. The file runs
                   from rare to frequent concepts, so a size keeps its tail.
``openimages_v7``  the Open Images V7 class names, bundled in ``data/vocab/openimages_v7.txt``. A
                   size keeps the head, in the official order.
``file``           any UTF-8 text file with one concept per line, for ablations. Blank lines are
                   skipped and a duplicate concept is an error; every other line is a concept,
                   including ones that start with ``#`` (LAION has ``#`` and ``##``). ``order``
                   says whether a size keeps the ``head`` or the ``tail``. A copy of a bundled
                   vocabulary therefore reproduces it exactly.

A file dictionary is identified by its content: the SHA-256 of its selected words names the SpLiCE
cache directory and the embedding cache, so editing the file can never reuse stale embeddings. The
two bundled kinds keep their historical names, so every existing cache stays valid.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUNDLED_KINDS = ("laion", "openimages_v7")
DICTIONARY_KINDS = (*BUNDLED_KINDS, "file")
ORDERS = ("head", "tail")
#: How a size selects the words of each bundled vocabulary, as the vendored SpLiCE loader does.
BUNDLED_ORDER = {"laion": "tail", "openimages_v7": "head"}


@dataclass(frozen=True)
class ConceptDictionary:
    """The words of one dictionary and where they came from."""

    kind: str
    words: tuple[str, ...]
    #: The requested size; zero or less keeps every word.
    size: int
    order: str
    source: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256("\n".join(self.words).encode("utf-8")).hexdigest()

    @property
    def token(self) -> str:
        """The dictionary's name in cache directories and embedding caches."""

        if self.kind in BUNDLED_KINDS:
            return self.kind
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(self.source).stem).strip("_") or "words"
        return f"file-{stem}-{self.sha256[:12]}"

    def splice_load_arguments(self) -> dict[str, Any]:
        """The keyword arguments that make ``splice.load`` embed exactly these words."""

        if self.kind in BUNDLED_KINDS:
            return {"vocabulary": self.kind, "vocabulary_size": self.size}
        return {
            "vocabulary": "file",
            "vocabulary_size": self.size,
            "words": list(self.words),
            "dictionary_id": f"{self.token}_{len(self.words)}",
        }

    def provenance(self) -> dict[str, Any]:
        """What a cache records about this dictionary beyond its kind and size."""

        return {
            "kind": self.kind,
            "source": self.source,
            "order": self.order,
            "size": self.size,
            "word_count": len(self.words),
            "sha256": self.sha256,
        }


def select_words(words: list[str], size: int, order: str) -> list[str]:
    if order not in ORDERS:
        raise ValueError(f"Unknown dictionary order {order!r}; use one of {list(ORDERS)}.")
    if size <= 0:
        return list(words)
    return list(words[-size:] if order == "tail" else words[:size])


def read_word_file(path: str | Path) -> list[str]:
    """One concept per line; blank lines are skipped and duplicates are refused."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Concept dictionary file not found: {path}")
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word:
            words.append(word)
    if not words:
        raise ValueError(f"Concept dictionary file lists no concepts: {path}")
    seen: set[str] = set()
    duplicates = sorted({word for word in words if word in seen or seen.add(word)})
    if duplicates:
        raise ValueError(f"Concept dictionary file repeats concepts {duplicates[:5]}: {path}")
    return words


def resolve_dictionary(
    kind: str, *, size: int = -1, path: str | Path | None = None, order: str | None = None,
) -> ConceptDictionary:
    """The dictionary a kind, a size and, for a file, its path and order describe."""

    if kind not in DICTIONARY_KINDS:
        raise ValueError(f"Unknown concept dictionary {kind!r}; use one of {list(DICTIONARY_KINDS)}.")
    if kind in BUNDLED_KINDS:
        if path is not None:
            raise ValueError(f"The {kind} dictionary is bundled; a file path applies only to kind 'file'.")
        if order is not None and order != BUNDLED_ORDER[kind]:
            raise ValueError(f"The {kind} dictionary always keeps its {BUNDLED_ORDER[kind]}.")
        from third_party.splice import splice as splice_library

        words = splice_library.get_vocabulary(kind, size)
        return ConceptDictionary(
            kind=kind, words=tuple(words), size=size, order=BUNDLED_ORDER[kind], source=kind,
        )
    if path is None:
        raise ValueError("A file dictionary needs the path of its word file.")
    order = order or "head"
    words = select_words(read_word_file(path), size, order)
    return ConceptDictionary(
        # The file name is recorded for readers; the SHA-256 of the words is the identity, so a
        # cache built on another machine from the same file compares equal.
        kind="file", words=tuple(words), size=size, order=order, source=Path(path).name,
    )
