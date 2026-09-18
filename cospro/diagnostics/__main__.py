"""Command line for CoSpRo diagnostics.

Evaluate on the cluster (it reads the SpLiCE cache from scratch) and render locally:

    python -m cospro.diagnostics evaluate --dataset waterbirds \\
        --sweep-dir outputs/shared/waterbirds/graphs/concept_groups \\
        --graph cospro=outputs/shared/waterbirds/graphs/crp_graph.json \\
        --graph raw_clip=outputs/shared/waterbirds/graphs/raw_clip_graph.json \\
        --reference raw_clip --splice-dataset-cache CACHE.pt --data-folder DATASETS \\
        --output outputs/reports/cospro_diagnostics/waterbirds/diagnostics.json

    python -m cospro.diagnostics dashboard outputs/reports/cospro_diagnostics/waterbirds/diagnostics.json \\
        --data-folder DATASETS --output diagnostics.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cospro.diagnostics.dashboard import render_file
from cospro.diagnostics.evaluate import evaluate, write_record
from splice.settings import data_folder


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {value!r}.")
    return name, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m cospro.diagnostics", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("evaluate", help="Compute metrics and write the JSON record.")
    run.add_argument("--dataset", required=True)
    run.add_argument("--sweep-dir", type=Path, help="Directory of grouping configurations (concept_groups/).")
    run.add_argument("--graph", type=_named_path, action="append", default=[], metavar="NAME=PATH",
                     help="A teacher graph to describe in full; repeat for baselines.")
    run.add_argument("--reference", help="Graph name whose edges the sweep graphs are compared with.")
    run.add_argument("--splice-dataset-cache", type=Path,
                     help="Frozen SpLiCE cache; enables coherence, coverage, stability and AUC metrics.")
    run.add_argument("--data-folder", type=Path, default=data_folder(),
                     help="Dataset root; enables post-hoc label metrics (defaults to $DATA_FOLDER).")
    run.add_argument("--no-labels", action="store_true", help="Compute the label-free tier only.")
    run.add_argument("--bootstrap-trials", type=int, default=0, help="Regroup N image subsets per configuration.")
    run.add_argument("--output", type=Path, required=True)

    render = commands.add_parser("dashboard", help="Render the HTML dashboard from a JSON record.")
    render.add_argument("record", type=Path)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--data-folder", type=Path, default=data_folder(), help="Enables the edge thumbnail gallery.")
    render.add_argument("--gallery-graph", action="append", help="Limit the gallery to these graph names.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "evaluate":
        if args.reference and args.reference not in dict(args.graph):
            raise SystemExit(f"--reference {args.reference!r} must name one of the --graph entries.")
        record = evaluate(
            args.dataset,
            sweep_dir=args.sweep_dir,
            graphs=dict(args.graph),
            reference_graph=args.reference,
            cache_path=args.splice_dataset_cache,
            data_folder=None if args.no_labels else args.data_folder,
            bootstrap_trials=args.bootstrap_trials,
        )
        print(f"[diagnostics] wrote {write_record(record, args.output)}", flush=True)
    else:
        output = render_file(args.record, args.output, data_folder=args.data_folder, gallery_graphs=args.gallery_graph)
        print(f"[diagnostics] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
