"""Build the results book: one readable Markdown page per dataset and method from every run record.

Scans ``outputs/seeds/**/run.json`` and writes ``outputs/reports/results/<dataset>/<method>.md`` plus
``outputs/reports/results/README.md``. A page lists every study arm of one method on one dataset with
the hyperparameters that tell the arms apart, the seeds and the validation results, and repeats the
SimCLR arms of the same dataset as the reference. ``scripts/run_training.sbatch`` rebuilds the book
after every run; locally:

    python -m cospro.cli.build_results_book

Numbers are validation-probe numbers. ``val WGA`` averages the last four periodic probes when the
record carries them and falls back to the last probe otherwise (marked *). Locked test results live in
the paper registry (``paper/paper_results.json``), not here.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

from cospro.cli.summarize_study import last_ssl_value, probe_trajectory
from cospro.tracking.artifacts import resolve_output_root

METHOD_TITLES = {
    "simclr": "SimCLR",
    "cospro": "CoSpRo (teacher graph: sampler and relational KL)",
    "latetvg": "LateTVG",
    "cospro_latetvg": "CoSpRo + LateTVG",
    "la_ssl": "LA-SSL",
    "concept_factors": "Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks)",
    "concept_transfer": "Frozen concept transfer (direct distillation of CLIP/SpLiCE targets)",
}


def option(config: dict, name: str) -> Any:
    """A training option under its current name or the historical CRP name."""

    if config.get(name) is not None:
        return config[name]
    return config.get(name.replace("cospro_", "crp_"))


def method_of(config: dict) -> str:
    mode = config.get("splice_mode")
    latetvg = bool(config.get("latetvg_prune_rate"))
    if config.get("la_ssl"):
        return "la_ssl"
    if mode in ("cospro_relational", "crp_relational"):
        return "cospro_latetvg" if latetvg else "cospro"
    if mode == "concept_factors":
        return "concept_factors"
    if mode == "frozen_concept_distill":
        return "concept_transfer"
    return "latetvg" if latetvg else "simclr"


def graph_name(config: dict) -> str:
    path = str(option(config, "cospro_teacher_graph") or "")
    if not path:
        return ""
    parts = path.replace("\\", "/").split("/")
    stem = parts[-1].removesuffix(".json")
    # Generated graphs are all called teacher_graph.json; their grouping folder names them.
    if stem == "teacher_graph":
        grouping = next((part for part in parts if part.startswith("text_")), "")
        return f"groups {grouping}".strip()
    return stem


def groups_name(config: dict) -> str:
    path = str(config.get("concept_factor_groups") or "").replace("\\", "/")
    if not path:
        return "default"
    parts = path.split("/")
    vocabulary = "laion" if any("laion" in part for part in parts) else "openimages"
    grouping = next((part for part in parts if part.startswith("text_")), "")
    return f"{vocabulary} {grouping}".strip()


def number(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:g}" if abs(value) < 1e4 else f"{value:.{digits}g}"
    return str(value)


# Columns that tell the arms of each method apart: (header, value of a run's config).
Column = tuple[str, Callable[[dict], Any]]
COMMON: list[Column] = [
    ("epochs", lambda c: c.get("epochs")),
    ("batch", lambda c: c.get("batch_size")),
    ("lr", lambda c: c.get("learning_rate")),
    ("wd", lambda c: c.get("weight_decay")),
    ("temp", lambda c: c.get("temp")),
    ("probe", lambda c: f"{c.get('linear_probe_solver', '')}/{c.get('train_set_linear_layer', '')}"),
]
METHOD_COLUMNS: dict[str, list[Column]] = {
    "simclr": [],
    "cospro": [
        ("graph", graph_name),
        ("KL weight", lambda c: c.get("splice_weight")),
        ("KL temp", lambda c: option(c, "cospro_temperature")),
        ("SimCLR weight", lambda c: c.get("simclr_weight")),
    ],
    "latetvg": [("prune", lambda c: c.get("latetvg_prune_rate")), ("layers", lambda c: c.get("latetvg_layers"))],
    "cospro_latetvg": [
        ("graph", graph_name),
        ("KL weight", lambda c: c.get("splice_weight")),
        ("prune", lambda c: c.get("latetvg_prune_rate")),
        ("layers", lambda c: c.get("latetvg_layers")),
    ],
    "la_ssl": [
        ("eta", lambda c: c.get("la_ssl_eta")),
        ("gamma", lambda c: c.get("la_ssl_gamma")),
        ("quantile", lambda c: c.get("la_ssl_quantile")),
    ],
    "concept_factors": [
        ("F1 fraction", lambda c: c.get("factor_condition_fraction")),
        ("F2 weight", lambda c: c.get("factor_distill_weight")),
        ("targets", lambda c: c.get("factor_targets")),
        ("merge", lambda c: c.get("factor_merge_similarity")),
        ("min freq", lambda c: c.get("factor_min_frequency")),
        ("F2 weighting", lambda c: c.get("factor_sample_weighting")),
        ("F2 cross-fit", lambda c: c.get("factor_cross_fit")),
        ("F3 weight", lambda c: c.get("factor_block_weight")),
        ("F3 context", lambda c: c.get("factor_block_context_weight") if c.get("factor_block_weight") else None),
        ("F3 presence", lambda c: c.get("factor_block_presence") if c.get("factor_block_weight") else None),
        ("start", lambda c: c.get("factor_start_epoch")),
        ("groups", groups_name),
        ("factors", lambda c: c.get("concept_factor_count")),
    ],
    "concept_transfer": [
        ("targets", lambda c: c.get("concept_transfer_target_kind")),
        ("alpha", lambda c: c.get("concept_transfer_alpha_max")),
    ],
}


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("seeds/*/seed_*/*/*/run.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = record.get("config") or {}
        study, _, arm = path.relative_to(root / "seeds").parts[:3]
        final = record.get("final_metrics") or {}
        tail = probe_trajectory(record)[-4:]
        runs.append({
            "study": study,
            "arm": arm,
            "seed": config.get("seed"),
            "status": record.get("status"),
            "dataset": config.get("dataset") or "unknown",
            "method": method_of(config),
            "config": config,
            "wga_last4": statistics.mean(point["wga"] for point in tail) if tail else None,
            "wga_last": final.get("Last linear val worst-group acc"),
            "acc_last4": statistics.mean(point["acc"] for point in tail) if tail else None,
            "acc_last": final.get("Last linear val acc"),
            "explained_variance": last_ssl_value(record, "SSL relational_factor_explained_variance"),
            "record": path.relative_to(root).as_posix(),
        })
    return runs


def spread(values: list[float]) -> str:
    if not values:
        return ""
    text = f"{statistics.mean(values):.1f}"
    return text + (f" ± {statistics.stdev(values):.1f}" if len(values) > 1 else "")


def arm_rows(runs: list[dict], method: str) -> list[dict]:
    """One row per study arm and hyperparameter setting, aggregated over seeds."""

    columns = COMMON + METHOD_COLUMNS[method]
    buckets: dict[tuple, list[dict]] = {}
    for run in runs:
        key = (run["study"], run["arm"], *(number(value(run["config"])) for _, value in columns))
        buckets.setdefault(key, []).append(run)
    rows = []
    for key, members in buckets.items():
        complete = [member for member in members if member["status"] == "complete"]
        fallback = [m for m in complete if m["wga_last4"] is None and m["wga_last"] is not None]
        wga = [m["wga_last4"] if m["wga_last4"] is not None else m["wga_last"] for m in complete
               if (m["wga_last4"] if m["wga_last4"] is not None else m["wga_last"]) is not None]
        accuracy = [m["acc_last4"] if m["acc_last4"] is not None else m["acc_last"] for m in complete
                    if (m["acc_last4"] if m["acc_last4"] is not None else m["acc_last"]) is not None]
        explained = [m["explained_variance"] for m in complete if m["explained_variance"] is not None]
        rows.append({
            "study": key[0],
            "arm": key[1],
            "settings": list(key[2:]),
            "seeds": sorted({m["seed"] for m in complete if m["seed"] is not None}),
            "other": len(members) - len(complete),
            "wga": spread(wga) + ("*" if fallback else ""),
            "wga_mean": statistics.mean(wga) if wga else None,
            "accuracy": spread(accuracy),
            "explained": f"{statistics.mean(explained):.2f}" if explained else "",
        })
    rows.sort(key=lambda row: (row["study"], -(row["wga_mean"] if row["wga_mean"] is not None else -1)))
    return rows


def table(rows: list[dict], method: str) -> list[str]:
    headers = ["study", "arm", *[name for name, _ in COMMON + METHOD_COLUMNS[method]], "seeds",
               "val WGA", "val acc"]
    if method == "concept_factors":
        headers.append("factor expl. var.")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        seeds = ",".join(map(str, row["seeds"])) + (f" (+{row['other']} unfinished)" if row["other"] else "")
        cells = [row["study"], row["arm"], *row["settings"], seeds, row["wga"], row["accuracy"]]
        if method == "concept_factors":
            cells.append(row["explained"])
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return lines


def method_page(dataset: str, method: str, runs: list[dict], reference: list[dict]) -> str:
    lines = [f"# {METHOD_TITLES[method]} on {dataset}", "",
             "Validation probe. `val WGA` and `val acc` average the last four periodic probes over the seeds "
             "(± standard deviation over seeds; * marks rows that fall back to the last probe). Rows are grouped "
             "by study and sorted by val WGA inside it. Compare rows only within one probe protocol (`probe`).", ""]
    lines += table(arm_rows(runs, method), method)
    if method != "simclr" and reference:
        lines += ["", "## SimCLR reference on this dataset", ""]
        lines += table(arm_rows(reference, "simclr"), "simclr")
    lines += ["", "Run records: `outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`.", ""]
    return "\n".join(lines)


def index_page(pages: dict[tuple[str, str], list[dict]]) -> str:
    lines = ["# Results book", "",
             "One page per dataset and method, built from every run record by "
             "`python -m cospro.cli.build_results_book`. Best arm: highest mean val WGA among arms with at "
             "least two finished seeds.", "",
             "| dataset | method | studies | runs | best arm (≥ 2 seeds) | val WGA | val acc |",
             "|---|---|---|---|---|---|---|"]
    for (dataset, method), runs in sorted(pages.items()):
        rows = [row for row in arm_rows(runs, method) if len(row["seeds"]) >= 2 and row["wga_mean"] is not None]
        best = max(rows, key=lambda row: row["wga_mean"]) if rows else None
        studies = len({run["study"] for run in runs})
        link = f"[{METHOD_TITLES[method]}]({dataset}/{method}.md)"
        lines.append(
            f"| {dataset} | {link} | {studies} | {len(runs)} | "
            + (f"{best['study']} / {best['arm']} | {best['wga']} | {best['accuracy']} |" if best else " | | |")
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    root = resolve_output_root()
    runs = load_runs(root)
    pages: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        pages.setdefault((run["dataset"], run["method"]), []).append(run)
    destination = root / "reports" / "results"
    for (dataset, method), members in pages.items():
        reference = pages.get((dataset, "simclr"), [])
        path = destination / dataset / f"{method}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(method_page(dataset, method, members, reference), encoding="utf-8")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "README.md").write_text(index_page(pages), encoding="utf-8")
    print(f"[results] Results book of {len(runs)} runs: {destination / 'README.md'}")
    return destination / "README.md"


if __name__ == "__main__":
    main()
