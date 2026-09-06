"""Staged Waterbirds diagnostics and a locked, small KL-weight experiment.

All labelled computations are downstream/post-hoc. No diagnostic metric selects
teacher edges, graph weights, SSL settings or an SSL stopping epoch.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import tarfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.spurious_eval.datasets.registry import get_dataset_spec
from experiments.spurious_eval.metrics import compute_group_metrics
from experiments.spurious_eval.training.logistic_probe import fit_logistic_probe
from scripts.tools.run_downstream_evaluator_diagnostic import (
    _find_probe_artifacts, _reproduce_saved_probe, _check_checkpoint_loading,
    _write_json, saved_train_order,
)
from splice.crp import validate_feature_cache
from splice.crp_training import CrpGraphBatchSampler, validate_teacher_graph
from splice.graph_io import graph_fingerprint, load_graph_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No results for {path}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def log_numeric_tree(run, prefix, value):
    """Expose nested post-hoc graph metrics as chartable W&B scalar keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            log_numeric_tree(run, f"{prefix}/{key}", item)
    elif isinstance(value, (int, float)):
        run.log({prefix: value})


def balanced_indices(y, metadata, per_group, seed):
    """Nested, without-replacement train-only subsamples; no validation access."""
    groups = y.long() * 2 + metadata[:, 0].long()
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for group in range(4):
        candidates = torch.where(groups == group)[0]
        if len(candidates) < per_group:
            raise ValueError(f"Group {group} has {len(candidates)} rows, requested {per_group}")
        selected.append(candidates[torch.randperm(len(candidates), generator=generator)[:per_group]])
    return torch.cat(selected).sort().values


def probe(train, evaluation, l2, config):
    x, y, metadata = train
    vx, vy, vm = evaluation
    records, convergence = fit_logistic_probe(
        x, y, vx, vy, num_classes=2, l2=l2,
        tolerance=config["probe_tolerance"], max_epochs=config["probe_max_epochs"],
    )
    metrics = [compute_group_metrics(r.eval_predictions, vy, vm) for r in records]
    train_metrics = compute_group_metrics(records[-1].train_predictions, y, metadata)
    result = {
        "l2": l2, "train_count": len(y), "feature_dim": x.shape[1],
        "avg": float(np.mean([m.average for m in metrics]) * 100),
        "wga": float(np.mean([m.worst_group for m in metrics]) * 100),
        "train_avg": float(train_metrics.average * 100),
        "converged": convergence["converged"], "gradient_max": convergence["gradient_max"],
    }
    for i, (acc, count) in enumerate(zip(metrics[-1].group_accuracy, metrics[-1].group_counts)):
        result[f"group_{i}_accuracy"] = float(acc * 100) if count else None
        result[f"group_{i}_count"] = int(count)
    return result


def source_runs(config):
    for entry in config["sources"]:
        for seed in entry["seeds"]:
            for arm in entry["arms"]:
                yield seed, arm, Path(entry["output"]) / f"seed{seed}" / arm / "training"


def locked_inputs(config):
    source = Path(config["source_output"])
    identity = read_json(source / "cache_identity.json")
    cache_path = Path(identity["path"])
    expected = config["expected_fingerprints"]
    if graph_fingerprint(cache_path) != expected["cache"]:
        raise ValueError("Source cache changed; use a separately versioned study")
    cache = validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=True))
    graphs = {}
    for name in ("crp", "raw_clip"):
        path = source / "graphs" / f"{name}_graph.json"
        if graph_fingerprint(path) != expected[name]:
            raise ValueError(f"Source graph changed: {name}")
        graphs[name] = validate_teacher_graph(load_graph_json(path), cache["sample_ids"])
    return cache, graphs


def stage_probes(config, out, run):
    rows, checks = [], []
    for seed, arm, root in source_runs(config):
        result_path, features = _find_probe_artifacts(root, config["ssl_epoch"])
        args = read_json(result_path.parent / "args.json")
        if args["model"] != "resnet18_large":
            raise ValueError(f"Non-ResNet18_large student: {result_path}")
        local_config = {**config, "batch_size": args["batch_size"], "num_classes": 2,
                        "train_set_linear_layer": "ds_train", "eval_split": "val"}
        check = _reproduce_saved_probe(result_path, features, local_config)
        check.update(seed=seed, arm=arm, feature_fingerprint=graph_fingerprint(features),
                     checkpoint_loading=_check_checkpoint_loading(root, local_config))
        checks.append(check)
        _write_json(out / "integrity.json", {"checks": checks, "passed": False})
        if not check["matches_existing"] or not check["alignment"]["passed"]:
            raise RuntimeError(f"Saved probe integrity failed for seed {seed}/{arm}")
        payload = torch.load(features, map_location="cpu", weights_only=True)
        train, evaluation = payload["train"], payload["evaluation"]
        # Undo the historical shuffle so subsample seeds select the same image
        # positions across all SSL seeds and arms (ds_train membership is fixed).
        inverse = torch.argsort(saved_train_order(len(train[1]), payload["seed"], args["batch_size"]))
        train = tuple(t[inverse] for t in train)
        designs = [("original_ds_train", None, 0)]
        designs += [(f"balanced_{n}_per_group", n, s)
                    for n in config["probe_per_group"] for s in config["subset_seeds"]]
        for design, n, subset_seed in designs:
            indices = torch.arange(len(train[1])) if n is None else balanced_indices(train[1], train[2], n, subset_seed)
            for l2 in config["probe_l2_grid"]:
                row = {"seed": seed, "arm": arm, "design": design, "subset_seed": subset_seed,
                       **probe(tuple(t[indices] for t in train), evaluation, l2, config)}
                rows.append(row)
                run.log(row)
        write_csv(out / "results.csv", rows)
    _write_json(out / "integrity.json", {"checks": checks, "passed": True,
                "limitations": ["Legacy IDs inferred from seeded loader; no explicit IDs in old tensors",
                                "14/28-per-group subsets test smaller sample budgets, not >224 student examples",
                                "Deleted checkpoints cannot be checked or used for new feature extraction"]})
    return rows


def reconstruction_stages(cache, graph):
    codes, dictionary = cache["splice_codes"], cache["dictionary"]
    full = codes @ dictionary
    frequency = (codes > 0).float().mean(0)
    cfg = graph["config"]
    active = torch.where((frequency >= cfg["min_concept_frequency"]) &
                         (frequency <= cfg["max_concept_frequency"]))[0]
    filtered = codes[:, active] @ dictionary[active]
    grouped = torch.zeros_like(full)
    selected = torch.zeros_like(full)
    selected_ids = set(graph["selected_group_ids"])
    for group in graph["groups"]:
        ids = group["concept_indices"]
        prototype = F.normalize(dictionary[ids].mean(0), dim=0)
        part = codes[:, ids].sum(1, keepdim=True) * prototype
        grouped += part
        if group["group_id"] in selected_ids:
            selected += part
    # Prototypes are diagnostic approximations; CRP uses full subspace projection.
    return {"full_vs_clip": (full, cache["centered_clip"]),
            "filtered_vs_full": (filtered, full), "grouped_vs_filtered": (grouped, filtered),
            "grouped_vs_full": (grouped, full), "selected_prototypes_vs_full": (selected, full)}


def fidelity(left, right, mask):
    left, right = left[mask], right[mask]
    valid = (left.norm(dim=1) > 1e-12) & (right.norm(dim=1) > 1e-12)
    cos = F.cosine_similarity(left, right, dim=1)
    return {"count": len(left), "zero_norm_rows": int((~valid).sum()),
            "mean_valid": float(cos[valid].mean()) if valid.any() else None,
            **{f"q{q:g}_valid": float(torch.quantile(cos[valid], q)) if valid.any() else None
               for q in [0.05, 0.25, 0.5, 0.95]},
            "coverage_cos08": float(((cos >= .8) & valid).float().mean()),
            "coverage_cos09": float(((cos >= .9) & valid).float().mean())}


def sampler_exposure(graph, y, contexts, batch_size, seed, epochs):
    """Replay sampler only; report full mass retained AND batch-renormalized mass.

    This is a deterministic simulation, not an exact historical RNG replay.
    Labels are read only after the batches have been formed.
    """
    sampler = CrpGraphBatchSampler(graph["neighbor_indices"], graph["weights"], batch_size,
                                   torch.Generator().manual_seed(seed))
    totals = {g: dict(visits=0, supported=0, retained_mass=0., batch_mass=0.,
                     useful_mass=0., wrong_mass=0.) for g in range(4)}
    for _ in range(epochs):
        batches = list(sampler)
        assert sorted(i for b in batches for i in b) == list(range(len(y)))
        for batch in batches:
            members = set(batch)
            for i in batch:
                stat = totals[int(y[i]) * 2 + int(contexts[i])]
                stat["visits"] += 1
                edges = [(int(j), float(w)) for j, w in zip(graph["neighbor_indices"][i], graph["weights"][i])
                         if int(j) in members and float(w) > 0]
                mass = sum(w for _, w in edges)
                if not mass:
                    continue
                q = float(graph["anchor_confidence"][i])
                stat["supported"] += 1
                stat["retained_mass"] += q * mass
                stat["batch_mass"] += q
                stat["useful_mass"] += q * sum(w for j, w in edges if y[i] == y[j] and contexts[i] != contexts[j]) / mass
                stat["wrong_mass"] += q * sum(w for j, w in edges if y[i] != y[j]) / mass
    return [{"group": g, "sampler_seed": seed, "epochs": epochs, **v,
             "supported_fraction": v["supported"] / v["visits"] if v["visits"] else None,
             "useful_batch_mass_fraction": v["useful_mass"] / v["batch_mass"] if v["batch_mass"] else None,
             "wrong_batch_mass_fraction": v["wrong_mass"] / v["batch_mass"] if v["batch_mass"] else None}
            for g, v in totals.items()]


def stage_graph(config, out, run):
    from scripts.tools.crp_posthoc_diagnostics import diagnose_fixed_graphs
    cache, graphs = locked_inputs(config)
    stages = reconstruction_stages(cache, graphs["crp"])
    # All representations and graphs are fixed before annotations are loaded.
    dataset = get_dataset_spec(config["dataset"])["dataset"](config["data_folder"])
    indices = [int(s.rsplit(":", 1)[1]) for s in cache["sample_ids"]]
    y, metadata = dataset.y_array[indices], dataset.metadata_array[indices]
    groups = y * 2 + metadata[:, 0]
    rows = []
    for name, (left, right) in stages.items():
        for group in [-1, 0, 1, 2, 3]:
            mask = torch.ones(len(y), dtype=torch.bool) if group == -1 else groups == group
            row = {"stage": name, "group": group, **fidelity(left, right, mask)}
            rows.append(row)
            run.log(row)
    write_csv(out / "reconstruction.csv", rows)
    posthoc = diagnose_fixed_graphs(graphs, config["dataset"], config["data_folder"])
    _write_json(out / "posthoc_graphs.json", posthoc)
    log_numeric_tree(run, "posthoc", posthoc)
    exposures = []
    for name, graph in graphs.items():
        for seed in config["sampler_seeds"]:
            exposures += [{"graph": name, **row} for row in sampler_exposure(
                graph, y, metadata[:, 0], config["batch_size"], seed, config["sampler_epochs"])]
    write_csv(out / "sampler_exposure.csv", exposures)
    for row in exposures:
        run.log({f"sampler/{key}": value for key, value in row.items()})
    _write_json(out / "report.json", {"passed": True, "fingerprints": config["expected_fingerprints"],
        "usage": "posthoc_only_not_for_selection", "sampler": "simulation, not historical minibatch exposure",
        "prototype_warning": "CRP projects full dictionary subspaces; prototype fidelity is not an eligibility gate"})
    return rows


def extract_signal_features(config, out, cache):
    """One frozen teacher view per image; immutable discovery cache is not edited."""
    import splice
    from experiments.spurious_eval.models.resnet import build_resnet_encoder
    from torchvision import transforms, models
    feature_path = out / "downstream_signal_features.pt"
    if feature_path.exists():
        saved = torch.load(feature_path, map_location="cpu", weights_only=True)
        if saved.get("fingerprints") != config["expected_fingerprints"]:
            raise ValueError("Stale downstream signal features")
        return saved
    provenance = cache["provenance"]
    teacher = splice.load(provenance["splice_model"], provenance["splice_vocab"],
                          provenance["splice_vocab_size"], device=config["device"],
                          pretrained=provenance["splice_pretrained"],
                          l1_penalty=provenance["splice_l1_penalty"], return_weights=True).eval()
    if (not torch.allclose(teacher.dictionary.detach().cpu(), cache["dictionary"], atol=1e-5) or
            not torch.allclose(teacher.image_mean.detach().cpu().reshape(-1), cache["image_mean"], atol=1e-6)):
        raise ValueError("Teacher dictionary/mean differs from the locked discovery model")
    clip_transform = splice.get_preprocess(provenance["splice_model"], pretrained=provenance["splice_pretrained"])
    # Same resize/crop pixels as CLIP; only encoder-specific normalization differs.
    if not isinstance(clip_transform.transforms[-1], transforms.Normalize):
        raise ValueError("Expected CLIP preprocessing to end in Normalize")
    student_transform = transforms.Compose(list(clip_transform.transforms[:-1]) +
        [transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    torch.manual_seed(config["diagnostic_encoder_seed"])
    random_encoder, _ = build_resnet_encoder("resnet18_large", load_pretrained_weights=False)
    pretrained_encoder, _ = build_resnet_encoder("resnet18_large", load_pretrained_weights=False)
    weights = models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    # torchvision's downsample and this repository's shortcut are equivalent.
    weights = {k.replace(".downsample.", ".shortcut."): v for k, v in weights.items() if not k.startswith("fc.")}
    pretrained_encoder.load_state_dict(weights, strict=True)
    encoders = {"random_resnet18_large": random_encoder, "imagenet_resnet18_large": pretrained_encoder}
    for encoder in encoders.values():
        encoder.to(config["device"]).eval()
    dataset = get_dataset_spec(config["dataset"])["dataset"](config["data_folder"])
    result = {}
    for split in ["train", "val"]:
        ids = list(map(int, dataset.get_subset(split, transform=None).indices))
        chunks = {name: [] for name in ["clip", "splice_codes", "splice_reconstruction", *encoders]}
        for offset in range(0, len(ids), config["teacher_batch_size"]):
            batch_ids = ids[offset:offset + config["teacher_batch_size"]]
            images = [dataset.get_input(i) for i in batch_ids]
            cx = torch.stack([clip_transform(im) for im in images]).to(config["device"])
            sx = torch.stack([student_transform(im) for im in images]).to(config["device"])
            with torch.inference_mode():
                clip = F.normalize(teacher.clip.encode_image(cx).float(), dim=1)
                centered = F.normalize(clip - teacher.image_mean, dim=1)
                codes = teacher.decompose(centered)
                chunks["clip"].append(centered.cpu())
                chunks["splice_codes"].append(codes.cpu())
                chunks["splice_reconstruction"].append((codes @ teacher.dictionary).cpu())
                for name, encoder in encoders.items():
                    chunks[name].append(encoder(sx).cpu())
            print(f"Signal features {split}: {min(offset + len(batch_ids), len(ids))}/{len(ids)}", flush=True)
        result[split] = {"sample_ids": ids, "features": {k: torch.cat(v) for k, v in chunks.items()},
                         "y": dataset.y_array[ids], "metadata": dataset.metadata_array[ids]}
    result["artifact"] = "downstream_labelled_signal_features_v1_NOT_A_DISCOVERY_CACHE"
    result["ds_train_ids"] = list(map(int, dataset.get_subset("ds_train", transform=None).indices))
    result["preprocessing"] = {"clip": repr(clip_transform), "resnet18_large": repr(student_transform)}
    result["fingerprints"] = config["expected_fingerprints"]
    temporary = feature_path.with_suffix(".pt.tmp")
    torch.save(result, temporary)
    temporary.replace(feature_path)
    return result


def stage_signal(config, out, run):
    cache, _ = locked_inputs(config)
    payload = extract_signal_features(config, out, cache)
    train, val = payload["train"], payload["val"]
    positions = {sid: i for i, sid in enumerate(train["sample_ids"])}
    ds = torch.tensor([positions[sid] for sid in payload["ds_train_ids"]])
    designs = [("original_ds_train", 0, ds), ("full_train_unbalanced", 0, torch.arange(len(train["y"])))]
    designs += [(f"resampled_{n}_per_group", seed, balanced_indices(train["y"], train["metadata"], n, seed))
                for n in config["signal_per_group"] for seed in config["subset_seeds"]]
    _write_json(out / "subsets.json", [{"design": name, "subset_seed": seed,
        "sample_ids": [train["sample_ids"][i] for i in ix.tolist()]} for name, seed, ix in designs])
    rows = []
    for name, features in train["features"].items():
        for design, seed, indices in designs:
            for l2 in config["signal_l2_grid"]:
                row = {"representation": name, "design": design, "subset_seed": seed,
                       **probe((features[indices], train["y"][indices], train["metadata"][indices]),
                               (val["features"][name], val["y"], val["metadata"]), l2, config)}
                rows.append(row)
                run.log(row)
        write_csv(out / "results.csv", rows)
    # Complementarity diagnostic: IDs matched to legacy saved student features.
    # Teacher uses deterministic CLIP views; legacy students use seeded probe views.
    for seed, arm, root in source_runs(config):
        result_path, features = _find_probe_artifacts(root, config["ssl_epoch"])
        saved = torch.load(features, map_location="cpu", weights_only=True)
        args = read_json(result_path.parent / "args.json")
        order = saved_train_order(len(ds), saved["seed"], args["batch_size"])
        st, sv = saved["train"], saved["evaluation"]
        if not (torch.equal(st[1], train["y"][ds[order]]) and torch.equal(st[2], train["metadata"][ds[order]])
                and torch.equal(sv[1], val["y"]) and torch.equal(sv[2], val["metadata"])):
            raise ValueError("Student/teacher fusion alignment failed")
        for fusion in [False, True]:
            tx, vx = st[0], sv[0]
            if fusion:
                tx = torch.cat([tx, train["features"]["splice_codes"][ds[order]]], dim=1)
                vx = torch.cat([vx, val["features"]["splice_codes"]], dim=1)
            row = {"representation": "student_plus_splice" if fusion else "saved_student",
                   "design": "legacy_views_complementarity", "seed": seed, "arm": arm,
                   **probe((tx, st[1], st[2]), (vx, sv[1], sv[2]), config["probe_l2"], config)}
            rows.append(row)
            run.log(row)
        write_csv(out / "results.csv", rows)
    _write_json(out / "report.json", {"passed": True, "preprocessing": payload["preprocessing"],
        "limitations": ["Frozen comparisons share CLIP spatial preprocessing; legacy students use original probe views",
                        "Fusion requires the teacher at inference and is not a standalone SSL-student improvement",
                        "Full unbalanced train changes both sample count and distribution",
                        "Random and ImageNet ResNet18_large are diagnostic baselines, not matched SSL runs"],
        "fingerprints": config["expected_fingerprints"]})
    return rows


def transfer_configs(config):
    return [Path(config["output"]) / "transfer" / f"lambda_{weight:g}" / "locked_config.json"
            for weight in config["transfer_weights"]]


def prepare_transfer(config):
    from scripts.tools.lock_crp_followup import lock_config
    from scripts.tools.run_crp_controls import prepare_locked_experiment
    for stage in ["probes", "signal", "graph"]:
        status = read_json(Path(config["output"]) / stage / "completed.json")
        if status.get("config") != config:
            raise ValueError(f"Stale diagnostic stage: {stage}")
    locked_inputs(config)
    source = read_json(Path(config["source_output"]) / "experiment.json")
    if source["model"] != "resnet18_large":
        raise ValueError("Transfer source must be ResNet18_large")
    for weight, path in zip(config["transfer_weights"], transfer_configs(config)):
        template = {**source, "protocol": "crp_controls_followup_v1", "relational_weight": weight,
                    "seeds": config["transfer_seeds"], "arms": ["splice_crp_kl", "raw_clip_kl"],
                    "output": str(path.parent), "num_workers": config["num_workers"],
                    "keep_checkpoints": True, "delete_checkpoints_after_training": True,
                    "retain_probe_artifacts_every": 100,
                    "wandb_group": f"{config['wandb_group']}_lambda_{weight:g}"}
        path.parent.mkdir(parents=True, exist_ok=True)
        template_path = path.parent / "template.json"
        if path.exists():
            previous = read_json(path)
            if any(previous.get(k) != v for k, v in template.items() if k not in
                   {"cache", "prebuilt_graph_paths", "prebuilt_graph_fingerprints", "locked_artifact_fingerprints"}):
                raise ValueError("Transfer output already belongs to another configuration")
        _write_json(template_path, template)
        lock_config(template_path, Path(config["source_output"]), path)
        prepare_locked_experiment(path)


def transfer_task(config, task):
    from scripts.tools.run_crp_controls import run_one_arm
    size = len(config["transfer_seeds"]) * 2
    paths = transfer_configs(config)
    if not 0 <= task < len(paths) * size:
        raise ValueError("Transfer task ID outside configured grid")
    return run_one_arm(paths[task // size], task % size)


def summarize_transfer(config, out, run):
    from scripts.tools.run_crp_controls import summarize_experiment
    rows = []
    for weight, path in zip(config["transfer_weights"], transfer_configs(config)):
        summarize_experiment(path)
        for seed in config["transfer_seeds"]:
            for arm in ["splice_crp_kl", "raw_clip_kl"]:
                record = read_json(path.parent / f"seed{seed}" / arm / "completed.json")
                rows.append({"seed": seed, "arm": arm, "weight": weight,
                             "avg": record["avg_acc_last10"], "wga": record["wga_last10"]})
    for seed, arm, root in source_runs(config):
        if seed not in config["transfer_seeds"]:
            continue
        result, _ = _find_probe_artifacts(root, config["ssl_epoch"])
        metrics = read_json(result)["metrics"]
        rows.append({"seed": seed, "arm": arm, "weight": 2. if arm.endswith("_kl") else 0.,
                     "avg": metrics["Linear val acc"], "wga": metrics["Linear val worst-group acc"]})
    write_csv(out / "results.csv", rows)
    contrasts = []
    for row in rows:
        if not row["arm"].endswith("_kl"):
            continue
        control = "crp_sampler_only" if row["arm"] == "splice_crp_kl" else "raw_clip_sampler_only"
        for baseline in [control, "simclr"]:
            match = next(r for r in rows if r["seed"] == row["seed"] and r["arm"] == baseline)
            contrasts.append({**row, "control": baseline, "delta_avg": row["avg"] - match["avg"],
                              "delta_wga": row["wga"] - match["wga"]})
        if row["arm"] == "splice_crp_kl":
            match = next(r for r in rows if r["seed"] == row["seed"] and r["arm"] == "raw_clip_kl"
                         and r["weight"] == row["weight"])
            contrasts.append({**row, "control": "raw_clip_kl", "delta_avg": row["avg"] - match["avg"],
                              "delta_wga": row["wga"] - match["wga"]})
    write_csv(out / "paired_contrasts.csv", contrasts)
    return rows


def run_stage(config, stage):
    import wandb
    out = Path(config["output"]) / stage
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "config.json"
    if manifest.exists() and read_json(manifest) != config:
        raise ValueError("Configuration changed; choose a new output root")
    _write_json(manifest, config)
    if (out / "completed.json").exists():
        print(f"Reusing completed stage {stage}")
        return
    run = wandb.init(project=config["wandb_project"], entity=config["wandb_entity"],
                     group=config["wandb_group"], name=f"{config['wandb_group']}_{stage}",
                     job_type=stage, tags=["diagnostic", "val_only", "resnet18_large"],
                     config={**config, "python": platform.python_version(), "torch": str(torch.__version__)},
                     mode="online", dir=str(out))
    try:
        rows = {"probes": stage_probes, "signal": stage_signal, "graph": stage_graph,
                "transfer_summary": summarize_transfer}[stage](config, out, run)
        # Tables and compact artifacts include categorical axes and post-hoc reports.
        for csv_path in out.glob("*.csv"):
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                columns = next(reader)
                table_rows = []
                for row in reader:
                    converted = []
                    for value in row:
                        try:
                            converted.append(float(value))
                        except ValueError:
                            converted.append(value or None)
                    table_rows.append(converted)
                run.log({csv_path.stem: wandb.Table(columns=columns, data=table_rows)})
        artifact = wandb.Artifact(f"{config['wandb_group']}-{stage}", type="diagnostic-report")
        for path in list(out.glob("*.json")) + list(out.glob("*.csv")):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        identity = {"id": run.id, "url": run.url, "rows": len(rows)}
        run.finish()
        _write_json(out / "completed.json", {"config": config, "wandb": identity, "status": "complete"})
    except BaseException:
        run.finish(exit_code=1)
        raise


def package_review(config):
    """Small portable reports, without weights, feature tensors or W&B internals."""
    root = Path(config["output"])
    temporary = root / f".review_bundle_{uuid.uuid4().hex}.tmp"
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_file() and not path.is_symlink() and "wandb" not in relative.parts and path.suffix in {".json", ".csv", ".log"}:
                archive.add(path, arcname=str(Path(root.name) / relative), recursive=False)
    temporary.replace(root / "review_bundle.tar.gz")
    print(f"Review bundle: {root / 'review_bundle.tar.gz'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["probes", "signal", "graph", "prepare_transfer", "transfer",
                                           "transfer_summary", "diagnostics", "all", "preflight"], required=True)
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    if config["dataset"] != "waterbirds" or config["student_model"] != "resnet18_large":
        raise ValueError("This version is explicitly Waterbirds / ResNet18_large")
    torch.set_num_threads(config["cpu_threads"])
    if args.stage == "preflight":
        locked_inputs(config)
        for _, _, root in source_runs(config):
            result, _ = _find_probe_artifacts(root, config["ssl_epoch"])
            if read_json(result.parent / "args.json")["model"] != config["student_model"]:
                raise ValueError(f"Unexpected student model: {result}")
        print("Preflight passed: frozen fingerprints and all 20 final feature/JSON pairs exist. No training launched.")
    elif args.stage in {"diagnostics", "all"}:
        for stage in ["probes", "signal", "graph"]:
            run_stage(config, stage)
        if args.stage == "all":
            prepare_transfer(config)
            for task in range(len(config["transfer_weights"]) * len(config["transfer_seeds"]) * 2):
                transfer_task(config, task)
            run_stage(config, "transfer_summary")
    elif args.stage == "prepare_transfer":
        prepare_transfer(config)
    elif args.stage == "transfer":
        if args.task_id is None:
            raise ValueError("Transfer requires --task-id")
        transfer_task(config, args.task_id)
    else:
        run_stage(config, args.stage)
    if args.stage in {"diagnostics", "all", "transfer_summary"}:
        package_review(config)


if __name__ == "__main__":
    main()
