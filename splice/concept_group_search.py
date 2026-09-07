"""Bounded, annotation-free composite-group proposals and selection metrics."""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F

from splice.crp import orthonormal_basis


def split_rows(sample_ids):
    order = sorted(range(len(sample_ids)), key=lambda i: hashlib.sha256(
        ("group-search-20260908:" + str(sample_ids[i])).encode()).hexdigest())
    middle = len(order) // 2
    return sorted(order[:middle]), sorted(order[middle:])


def subset(cache, rows):
    result = dict(cache)
    for key in ("splice_codes", "clip_embeddings", "centered_clip"):
        result[key] = cache[key][rows]
    result["sample_ids"] = [cache["sample_ids"][i] for i in rows]
    return result


def propose(cache, max_composites=24):
    """Pairs/triples/quads with complete-link text coherence; no coactivation gate.

    Six candidates in each threshold/size stratum avoid selecting only pairs.
    Frequency filtering is computed on discovery rows only. No named water list.
    """
    codes, dictionary = cache["splice_codes"], cache["dictionary"]
    frequency = (codes > 0).float().mean(0)
    active = torch.where((frequency >= .01) & (frequency <= .95))[0].tolist()
    directions = F.normalize(dictionary[active], dim=1)
    similarities = directions @ directions.T
    strata = [[] for _ in range(4)]
    for threshold_index, threshold in enumerate((.65, .75)):
        for i, anchor in enumerate(active):
            candidates = torch.where(similarities[i] >= threshold)[0].tolist()
            candidates.sort(key=lambda j: (-float(similarities[i, j]), active[j]))
            group = [i]
            for j in candidates:
                if j == i or any(float(similarities[j, k]) < threshold for k in group):
                    continue
                group.append(j)
                members = tuple(sorted(active[k] for k in group))
                union_support = float((codes[:, list(members)].sum(1) > 0).float().mean())
                coherence = min(float(similarities[a, b]) for a in group for b in group if a != b)
                strata[2 * threshold_index + (len(group) > 2)].append(
                    (coherence * math.sqrt(union_support), members))
                if len(group) == 4:
                    break
    chosen = []
    for stratum in strata:
        stratum.sort(key=lambda item: (-item[0], item[1]))
    # Round-robin the four strata. Deduplicate heavily overlapping vocab variants.
    while any(strata) and len(chosen) < max_composites:
        for stratum in strata:
            while stratum:
                _, candidate = stratum.pop(0)
                if candidate in chosen:
                    continue
                if any(len(set(candidate) & set(g)) / len(set(candidate) | set(g)) > .75 for g in chosen):
                    continue
                chosen.append(candidate)
                break
            if len(chosen) == max_composites:
                break
    return [list(g) for g in chosen]


def group_metrics(cache, members):
    codes = cache["splice_codes"][:, members]
    activation = codes.sum(1)
    mass = codes.sum(0)
    shares = mass / mass.sum().clamp_min(1e-12)
    basis = orthonormal_basis(cache["dictionary"][members])
    energy = (cache["centered_clip"] @ basis).square().sum(1)
    directions = F.normalize(cache["dictionary"][members], dim=1)
    text = directions @ directions.T
    coactivation = F.normalize(codes.T, dim=1) @ F.normalize(codes.T, dim=1).T
    mask = ~torch.eye(len(members), dtype=torch.bool)
    n = len(activation)
    hoyer = ((math.sqrt(n) - float(activation.sum() / activation.norm().clamp_min(1e-12)))
             / (math.sqrt(n) - 1)) if n > 1 and activation.any() else None
    return {
        "size": len(members), "rank": basis.shape[1],
        "support": float((activation > 0).float().mean()), "hoyer": hoyer,
        "effective_members": float(torch.exp(-(shares * shares.clamp_min(1e-12).log()).sum())),
        "max_member_mass_share": float(shares.max()),
        "active_members_given_active": float((codes > 0).sum(1)[activation > 0].float().mean()) if activation.any() else 0.,
        "text_min": float(text[mask].min()) if mask.any() else 1.,
        "coactivation_mean": float(coactivation[mask].mean()) if mask.any() else 1.,
        "removed_energy_mean": float(energy.mean()),
        "removed_energy_p95": float(torch.quantile(energy, .95)),
    }


def select_groups(audit_groups, metrics, policy, cap=12):
    """No labels, downstream metrics, or confirmation rows enter this ranking."""
    ranked = []
    for group, m in zip(audit_groups, metrics):
        if not group["selected"] or not .01 <= m["support"] <= .95 or m["removed_energy_p95"] > .5:
            continue
        score = group["null_excess_score"]
        if policy == "compact":
            if m["size"] > 1 and (m["effective_members"] < 1.5 or m["max_member_mass_share"] > .85):
                continue
            score *= math.sqrt(max(0., 1 - m["removed_energy_mean"])) / math.sqrt(m["rank"])
        elif policy != "semantic":
            raise ValueError(policy)
        ranked.append((score, group["group_id"], set(group["concept_indices"])))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    result, used = [], []
    for _, group_id, members in ranked:
        if any(len(members & old) / len(members | old) > .5 for old in used):
            continue
        result.append(group_id)
        used.append(members)
        if len(result) == cap:
            break
    return result
