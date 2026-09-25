"""Whether a concept names surroundings or the object itself, from the dictionary alone.

The audit keeps the groups whose removal moves neighbours the most, which is the object itself
wherever the object dominates the representation: the Open Images audit of MetaShift selects cat
and dog breeds, the CelebA audit selects hair colours and the LAION audit of Waterbirds selects
raven, whale and flying. Removing those directions teaches the student to confuse the classes.

This module scores a concept by what its words denote, using the text encoder that built the
dictionary and nothing else: no image, no annotation and no property of the dataset. The score is
how much closer the concept sits to a description of surroundings than to a description of an
object, so a selection rule can drop the most object-like candidates.

The gate carries one assumption, which a paper using it has to state: the target is the object and
the spurious factor is its context. That holds for Waterbirds and MetaShift. It fails for CelebA,
where the target is hair colour and the spurious factor is gender, both attributes of the same
person, and the scores there separate nothing.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

#: What a concept of the surroundings sounds like.
CONTEXT_PROMPTS = (
    "a place or scene",
    "the background of a photo",
    "furniture or a room",
    "an outdoor landscape",
    "weather or time of day",
)
#: What a concept of the depicted object sounds like.
OBJECT_PROMPTS = (
    "a photo of an animal",
    "a kind of animal",
    "a photo of a person",
    "an object in the foreground",
)
PROMPT_TEMPLATE = "a photo of {}"


def _encode(texts: Sequence[str], model, tokenizer) -> torch.Tensor:
    device = next(model.parameters()).device
    with torch.no_grad():
        embeddings = model.encode_text(tokenizer(list(texts)).to(device)).float().cpu()
    return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def context_preference(words: Sequence[str], model, tokenizer) -> list[float]:
    """Per word: similarity to the context prompts minus similarity to the object prompts."""

    context = _encode(CONTEXT_PROMPTS, model, tokenizer)
    objects = _encode(OBJECT_PROMPTS, model, tokenizer)
    encoded = _encode([PROMPT_TEMPLATE.format(word) for word in words], model, tokenizer)
    return ((encoded @ context.T).max(dim=1).values - (encoded @ objects.T).max(dim=1).values).tolist()


def group_scores(groups: Sequence[Mapping], model, tokenizer) -> dict[str, float]:
    """``group_id`` -> the mean context preference of the concepts the group holds."""

    words = [word for group in groups for word in group["concepts"]]
    scores = dict(zip(words, context_preference(words, model, tokenizer)))
    return {str(group["group_id"]): sum(scores[word] for word in group["concepts"]) / len(group["concepts"])
            for group in groups}
