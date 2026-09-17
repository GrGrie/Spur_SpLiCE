"""Post-hoc label-stratified illustrations; never modifies teacher graphs."""
from collections import Counter

RELATIONS = (
    ("same_y_same_a", "Same y, same a"),
    ("same_y_different_a", "Same y, different a"),
    ("different_y_same_a", "Different y, same a"),
    ("different_y_different_a", "Different y, different a"),
)


def relation_key(left, right):
    return ("same_y" if left[0] == right[0] else "different_y") + (
        "_same_a" if left[1] == right[1] else "_different_a")


def stratify(pairs):
    """Cover all eight relation/status cells, preferring unused source groups/images."""
    slots, source_counts, image_counts = [], Counter(), Counter()
    for retained in (True, False):
        for relation, title in RELATIONS:
            candidates = [p for p in pairs if p["retained"] == retained and p["stratum"] == relation]
            candidates.sort(key=lambda p: (source_counts[tuple(p["source_group"])],
                                            image_counts[p["left_id"]] + image_counts[p["right_id"]],
                                            -p["gain"], p["row"], p["column"]))
            pair = candidates[0] if candidates else None
            counts = Counter(str(tuple(p["source_group"])) for p in candidates)
            slots.append(dict(title=("Retained: " if retained else "Not retained: ") + title,
                              candidate_count=len(candidates), source_group_counts=dict(counts), pair=pair))
            if pair:
                source_counts[tuple(pair["source_group"])]+=1
                image_counts.update((pair["left_id"], pair["right_id"]))
    return slots


def discover_panels(cache, graph, metadata):
    """Recompute projected candidates from the ORIGINAL cache, not the old 20 pairs.

    Final retention/attribution always comes from the saved graph. Candidates
    retained under another concept are omitted from this concept's illustration.
    """
    from splice.cospro import (validate_splice_dataset_cache, CrpAuditConfig,
                               _AuditGeometry, _neighbor_geometry, _relation_geometry,
                               orthonormal_basis, topk_neighbors)
    cache = validate_splice_dataset_cache(cache)
    if cache["sample_ids"] != graph["sample_ids"]:
        raise ValueError("Cache and graph sample IDs/order differ; use the original graph cache")
    for key, value in graph.get("provenance", {}).items():
        if key in cache.get("provenance", {}) and cache["provenance"][key] != value:
            raise ValueError(f"Cache/graph provenance differs: {key}")
    config = CrpAuditConfig(**{k: v for k, v in graph["config"].items()
                              if k in CrpAuditConfig.__dataclass_fields__})
    centered = cache["centered_clip"]
    raw, _ = topk_neighbors(centered, config.projected_neighbors, config.similarity_chunk_size)
    audit = _AuditGeometry(centered, raw, cache["splice_codes"])
    labels = []
    for sample_id in graph["sample_ids"]:
        prefix, index = sample_id.split(":")
        if prefix != "waterbirds" or not 0 <= int(index) < len(metadata):
            raise ValueError(f"Invalid sample ID: {sample_id}")
        row = metadata[int(index)]
        if int(row["split"]) != 0:
            raise ValueError("Illustrations require training images only")
        labels.append((int(row["y"]), int(row["place"])))
    retained = {(i, j): (g, w) for i, (js, ws, gs) in enumerate(zip(
        graph["neighbor_indices"], graph["weights"], graph["group_ids"]))
        for j, w, g in zip(js, ws, gs) if w > 0}
    saved_gains = {(i, j): gain for i, (js, ws, gains) in enumerate(zip(
        graph["neighbor_indices"], graph["weights"], graph["intervention_gains"]))
        for j, w, gain in zip(js, ws, gains) if w > 0}
    groups = []
    for group in graph["groups"]:
        if not group.get("selected", False):
            continue
        gid, indices = group["group_id"], group["concept_indices"]
        if [cache["vocabulary"][k] for k in indices] != group["concepts"]:
            raise ValueError("Cache vocabulary does not match graph concept indices")
        basis = orthonormal_basis(cache["dictionary"][indices], config.orthogonal_tolerance)
        geometry = _neighbor_geometry(audit, basis, config,
                                      search_seed=config.seed + 1_000_003 * (gid + 1))
        geometry = _relation_geometry(audit, geometry, config, indices)
        activation = cache["splice_codes"][:, indices].sum(1)
        candidates = []
        seen_retained = set()
        neighbors = geometry["neighbours"].tolist()
        gains = geometry["gain"].tolist()
        similarities = geometry["projected_similarity"].tolist()
        residuals = geometry["residual_splice_similarity"].tolist()
        activations = activation.tolist()
        for i, js in enumerate(neighbors):
            for pos, j in enumerate(js):
                edge = retained.get((i, j))
                if edge and edge[0] != gid:
                    continue
                if edge:
                    seen_retained.add((i, j))
                gain, residual = gains[i][pos], residuals[i][pos]
                if edge and abs(gain - saved_gains[(i, j)]) > 1e-5:
                    raise ValueError(f"G{gid}: cached projection gain differs from saved graph for {i}->{j}")
                candidates.append(dict(row=i, column=j, left_id=graph["sample_ids"][i],
                    right_id=graph["sample_ids"][j], retained=bool(edge), gain=gain,
                    raw_similarity=similarities[i][pos]-gain, projected_similarity=similarities[i][pos],
                    activation_contrast=abs(activations[i]-activations[j]), residual_similarity=residual,
                    final_edge_weight=edge[1] if edge else 0.,
                    null_calibrated_confidence=group["null_excess_ratio"] * gain * (.5 + .5 * residual),
                    stratum=relation_key(labels[i], labels[j]), source_group=list(labels[i])))
        expected = {key for key, value in retained.items() if value[0] == gid}
        if seen_retained != expected:
            raise ValueError(f"G{gid}: reconstructed candidates miss saved edges; check original cache/search settings")
        slots = stratify(candidates)
        groups.append(dict(group_id=gid, concepts=group["concepts"], slots=slots,
                           pairs=[s["pair"] for s in slots if s["pair"]], candidate_count=len(candidates)))
        print(f"G{gid} {group['concepts']}: {len(candidates)} candidates, "
              f"{sum(s['pair'] is not None for s in slots)}/8 illustration cells", flush=True)
    if not groups:
        raise ValueError("Graph has no selected concept groups")
    return dict(selection="Post-hoc strata over all projected candidates of every selected concept; "
                "balance source groups and image reuse, then descending gain. Other-concept retained edges excluded.",
                groups=groups)
