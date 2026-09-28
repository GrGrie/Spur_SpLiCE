"""Show the top SpLiCE concepts of images as one self-contained HTML page.

Runs locally: CLIP embeds each image, the embedding is centred on the dictionary's image mean and
SpLiCE decomposes it into sparse non-negative concept weights, exactly as the cache stage does. The
page lists every image beside its strongest concepts.

    python -m cospro.cli.show_splice_concepts photo.jpg other.png
    python -m cospro.cli.show_splice_concepts --random 12 --dataset waterbirds --data-folder ~/Datasets
    python -m cospro.cli.show_splice_concepts --random 8 --dataset spur_cifar10 --vocab laion --top 32

With ``--random`` the images come from the dataset's split (train by default) and each card also
names the image's class and spurious attribute, read from the dataset metadata for display only.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from cospro.data.registry import canonical_dataset_name, dataset_class, dataset_names
from cospro.pipeline.dictionary import DICTIONARY_KINDS, resolve_dictionary

DEFAULT_VOCABULARY_SIZES = {"laion": 10000, "openimages_v7": -1}
THUMBNAIL_SIZE = 320


@dataclass
class Entry:
    """One image of the page with its concepts."""

    title: str
    image: Image.Image
    concepts: list[tuple[str, float]]
    active: int
    reconstruction: float
    details: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path, help="Image files to decompose.")
    parser.add_argument("--random", type=int, default=0, metavar="N", help="Take N random images of --dataset.")
    parser.add_argument("--dataset", type=canonical_dataset_name, choices=dataset_names())
    parser.add_argument("--data-folder", default=os.environ.get("DATA_FOLDER", ""),
                        help="Dataset root for --random; defaults to $DATA_FOLDER.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random image choice.")
    parser.add_argument("--top", type=int, default=64, help="Concepts shown per image.")
    parser.add_argument("--vocab", default="openimages_v7", choices=DICTIONARY_KINDS)
    parser.add_argument("--vocab-size", type=int, help="Default: 10000 for laion, all for openimages_v7.")
    parser.add_argument("--vocab-file", type=Path, help="Word file of a 'file' dictionary.")
    parser.add_argument("--l1-penalty", type=float, default=0.25)
    parser.add_argument("--model", default="open_clip:ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("splice_concepts.html"))
    args = parser.parse_args(argv)
    if not args.images and not args.random:
        parser.error("give image paths or --random N")
    if args.random and not (args.dataset and args.data_folder):
        parser.error("--random needs --dataset and --data-folder (or $DATA_FOLDER)")
    if args.top < 1:
        parser.error("--top must be positive")
    return args


def top_concepts(code: torch.Tensor, vocabulary: list[str], count: int) -> list[tuple[str, float]]:
    """The ``count`` largest positive weights of one sparse code, strongest first."""

    values, indices = torch.sort(code, descending=True)
    return [(vocabulary[int(index)], float(value)) for value, index in zip(values[:count], indices[:count])
            if float(value) > 0]


def dataset_images(args: argparse.Namespace) -> list[tuple[str, Image.Image, list[str]]]:
    """Random images of the dataset split with their sample id, class and spurious attribute."""

    adapter_class = dataset_class(args.dataset)
    adapter = adapter_class(str(args.data_folder))
    indices = [int(index) for index in adapter.get_subset(args.split, transform=None).indices]
    chosen = random.Random(args.seed).sample(indices, min(args.random, len(indices)))
    metadata = adapter.metadata_array
    fields = list(getattr(adapter, "_metadata_fields", []))
    names = getattr(adapter, "_metadata_map", None) or {}

    def label(column: int, value: int) -> str:
        field_name = fields[column] if column < len(fields) else f"field {column}"
        values = names.get(field_name, [])
        return f"{field_name} = {values[value] if value < len(values) else value}"

    images = []
    for index in chosen:
        row = metadata[index]
        details = [label(adapter_class.target_metadata_index, int(row[adapter_class.target_metadata_index])),
                   label(adapter_class.spurious_metadata_index, int(row[adapter_class.spurious_metadata_index]))]
        images.append((f"{args.dataset}:{index} ({args.split})", adapter.get_input(index).convert("RGB"), details))
    return images


def decompose(images: list[Image.Image], args: argparse.Namespace):
    """CLIP embeddings centred on the image mean and their SpLiCE codes, as the cache stage computes them."""

    from third_party import splice

    size = args.vocab_size if args.vocab_size is not None else DEFAULT_VOCABULARY_SIZES.get(args.vocab, -1)
    dictionary = resolve_dictionary(args.vocab, size=size, path=args.vocab_file)
    device = torch.device(args.device)
    preprocess = splice.get_preprocess(args.model, pretrained=args.pretrained)
    model = splice.load(
        args.model, **dictionary.splice_load_arguments(), device=device, pretrained=args.pretrained,
        l1_penalty=args.l1_penalty, return_weights=True,
    ).eval()
    codes, reconstructions = [], []
    for start in range(0, len(images), args.batch_size):
        batch = torch.stack([preprocess(image) for image in images[start:start + args.batch_size]]).to(device)
        with torch.inference_mode():
            embedding = F.normalize(model.clip.encode_image(batch).float(), dim=1)
            centred = F.normalize(embedding - model.image_mean, dim=1)
            code = model.decompose(centred)
            rebuilt = code @ model.dictionary
        codes.append(code.cpu())
        reconstructions.append(F.cosine_similarity(rebuilt, centred, dim=1).cpu())
    return torch.cat(codes), torch.cat(reconstructions), list(dictionary.words), dictionary.token


def image_data_url(image: Image.Image) -> str:
    """A JPEG thumbnail as a data URL; small images are enlarged without smoothing."""

    image = image.copy()
    if max(image.size) < THUMBNAIL_SIZE:
        scale = THUMBNAIL_SIZE // max(image.size)
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


STYLE = """
:root { --bg: #f7f7f5; --card: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #e4e4df;
        --bar: #3f6fd8; --bar-bg: #e9eefb; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #151515; --card: #1f1f1e; --text: #ececea; --muted: #9a9a94; --line: #2f2f2d;
          --bar: #7fa2f0; --bar-bg: #26304a; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 24px 24px 8px; }
h1 { margin: 0 0 4px; font-size: 20px; }
header p { margin: 0; color: var(--muted); }
main { display: grid; gap: 16px; padding: 16px 24px 32px; }
.card { display: grid; grid-template-columns: minmax(0, 320px) minmax(0, 1fr); gap: 20px;
        background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
.card img { width: 100%; max-width: 320px; border-radius: 6px; image-rendering: pixelated; }
.meta h2 { margin: 12px 0 4px; font-size: 14px; word-break: break-all; }
.meta p { margin: 2px 0; color: var(--muted); font-size: 13px; }
ol { margin: 0; padding: 0; list-style: none; columns: 2 260px; column-gap: 24px; }
li { break-inside: avoid; display: grid; grid-template-columns: 2.2em minmax(0, 1fr) 4.2em;
     align-items: center; gap: 6px; padding: 2px 0; }
.rank { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.name { position: relative; padding: 1px 6px; border-radius: 4px; overflow: hidden; white-space: nowrap;
        text-overflow: ellipsis; background: var(--bar-bg); }
.name span { position: relative; }
.fill { position: absolute; inset: 0 auto 0 0; background: var(--bar); opacity: 0.28; }
.weight { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
@media (max-width: 700px) { .card { grid-template-columns: 1fr; } }
"""


def render_page(entries: list[Entry], subtitle: str) -> str:
    """The whole page: one card per image, concepts as bars scaled to the image's strongest weight."""

    cards = []
    for entry in entries:
        strongest = max((weight for _, weight in entry.concepts), default=1.0) or 1.0
        rows = "".join(
            f'<li><span class="rank">{rank}</span><span class="name" title="{html.escape(name)}">'
            f'<i class="fill" style="width:{100 * weight / strongest:.1f}%"></i><span>{html.escape(name)}</span></span>'
            f'<span class="weight">{weight:.3f}</span></li>'
            for rank, (name, weight) in enumerate(entry.concepts, start=1)
        )
        details = "".join(f"<p>{html.escape(line)}</p>" for line in entry.details)
        cards.append(
            f'<section class="card"><div class="meta"><img alt="" src="{image_data_url(entry.image)}">'
            f"<h2>{html.escape(entry.title)}</h2>{details}"
            f"<p>{entry.active} active concepts, reconstruction cosine {entry.reconstruction:.3f}</p></div>"
            f"<ol>{rows or '<li>No active concept.</li>'}</ol></section>"
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>SpLiCE concepts</title><style>{STYLE}</style></head><body>"
        f"<header><h1>SpLiCE concepts</h1><p>{html.escape(subtitle)}</p></header>"
        f"<main>{''.join(cards)}</main></body></html>"
    )


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    sources = [(str(path), Image.open(path).convert("RGB"), []) for path in args.images]
    if args.random:
        sources += dataset_images(args)
    codes, reconstructions, vocabulary, token = decompose([image for _, image, _ in sources], args)
    entries = [
        Entry(title=title, image=image, concepts=top_concepts(code, vocabulary, args.top),
              active=int((code > 0).sum()), reconstruction=float(reconstruction), details=details)
        for (title, image, details), code, reconstruction in zip(sources, codes, reconstructions)
    ]
    subtitle = (f"{len(entries)} images; dictionary {token} ({len(vocabulary)} concepts); "
                f"{args.model} {args.pretrained}; L1 penalty {args.l1_penalty:g}; top {args.top}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_page(entries, subtitle), encoding="utf-8")
    print(f"[results] SpLiCE concept page: {args.output.resolve()}")
    return args.output


if __name__ == "__main__":
    main()
