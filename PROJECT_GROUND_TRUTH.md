# Spur_SpLiCE Project Ground Truth

**Status:** canonical project specification  
**Last updated:** 2026-08-18  
**Canonical proposal:** SpLiCE-CRP v2 — Concept-Projected Relational Pretraining  
**Implementation status:** frozen feature caching, audit, and teacher-graph construction implemented;
empirical validation and SSL relational training remain pending

> This file is the authoritative source of truth for the current project direction.
> If `README.md`, `SPUR_SPLICE_CHAT_SUMMARY.md`, `Spur_SpLiCE.tex`, old chat
> messages, scripts, or comments conflict with this document, follow this document.
> Do not silently reinterpret the method. Update this file first when a scientific
> decision changes.

## 1. One-paragraph project summary

The project asks whether sparse, language-aligned concepts from a frozen
OpenCLIP/SpLiCE model can improve the worst-group accuracy (WGA) of a separately
trained self-supervised ResNet without using per-image target labels, spurious
attributes, or group labels during concept discovery or SSL training. The current
proposal, SpLiCE-CRP v2, uses SpLiCE to propose semantic factor subspaces, removes
each candidate subspace from the **full centered CLIP embedding** by orthogonal
projection, and asks whether that intervention restores semantic neighbourhoods
independently supported by a frozen SSL verifier such as DINOv3. Stable
concept-conditioned relations define a soft counterfactual teacher graph. A
ResNet18-large is then trained with ordinary SimCLR plus a confidence-gated
relational distillation loss that matches this graph. The method does not use one
mined image as a permanent hard positive. Target and group annotations are used
only after SSL training for linear evaluation, WGA measurement, and explicitly
labelled diagnostic oracles.

## 2. Scientific objective

### Primary question

Can an interpretable frozen concept model discover and transfer nuisance-invariant
relations into an independent SSL encoder, improving WGA without per-image labels
or prior descriptions of the nuisance?

### Intended contribution

The intended contribution is not merely another edit of a frozen VLM. It is a new
interface between concept intervention and SSL:

1. SpLiCE supplies interpretable semantic factor candidates.
2. Full-space projection tests what the representation would look like without a
   factor.
3. Independent semantic consensus rejects likely class-destroying interventions.
4. The accepted counterfactual geometry supervises a separately trained ResNet.

In one sentence:

> Use concept removal to construct a label-free counterfactual relation graph,
> then transfer that graph into a standalone SSL encoder while preserving the
> ordinary SimCLR objective.

### What would count as success

The proposed method must:

- improve WGA over matched SimCLR;
- beat or clearly match the strongest same-setting fully SSL baseline, including
  Ghanooni et al.'s spectral regularization;
- outperform non-concept controls such as raw-CLIP relational distillation,
  DINO-only relational distillation, random subspaces, and shuffled SpLiCE codes;
- avoid a material collapse in average accuracy;
- be stable across paired seeds;
- use no target, spurious, or group annotations in discovery or SSL training.

A positive result against SimCLR alone is not sufficient evidence for the concept
mechanism.

## 3. Information setting and non-negotiable constraints

### Allowed during discovery and SSL training

- unlabeled training images;
- standard stochastic augmentations of those images;
- frozen OpenCLIP image embeddings;
- the frozen SpLiCE vocabulary, dictionary, and sparse codes;
- a frozen label-free visual verifier, recommended default: DINOv3;
- dataset indices needed to cache embeddings and construct a graph;
- statistics computed only from the preceding unlabeled representations;
- the trainable ResNet and its SimCLR objective.

### Forbidden during discovery and SSL training

- per-image target labels `y`;
- spurious attributes `a`;
- group identifiers `g=(y,a)`;
- a manually supplied description such as "water background", "gender", or
  "colour stripe";
- group-balanced sampling based on hidden metadata;
- selecting thresholds or checkpoints by validation WGA;
- silently using the test split for selection.

### Allowed only for evaluation and diagnostics

- target labels for downstream linear probing;
- group metadata for WGA and group-wise accuracy;
- post-hoc measurements of pair purity, opposite-spurious coverage, and oracle
  concept quality;
- privileged oracle methods, provided that every table labels them as oracles.

The canonical method is stricter than the already implemented SpLiCE-IU method:
SpLiCE-IU uses target labels, whereas SpLiCE-CRP v2 does not.

## 4. Models and their roles

### Trainable model

- **Default:** ResNet18-large trained from scratch with SimCLR.
- **Scale check:** ResNet50, only after the mechanism works on ResNet18-large.
- The downstream representation is the ResNet backbone feature, not the frozen
  CLIP or DINO representation.

### Frozen concept teacher

- **OpenCLIP:** ViT-B/32, matching the existing SpLiCE pipeline.
- **SpLiCE:** sparse non-negative decomposition over a language-aligned concept
  dictionary.
- Role: propose interpretable semantic factor groups and the subspaces to test.
- SpLiCE is not assumed to reconstruct the whole CLIP embedding exactly.

### Frozen semantic verifier

- **Recommended:** a small or base DINOv3 checkpoint, frozen and cached once.
- Role: provide an independent label-free similarity geometry used only to reject
  semantically unsafe concept-removal relations.
- DINOv3 is not treated as an oracle and may itself contain shortcuts.
- Required control: DINO-only relational distillation. If the proposed method does
  not beat that control, the incremental value of concepts is not established.

A lower-resource variant may replace DINOv3 with a warm-started EMA copy of the
SimCLR encoder plus multi-crop consistency. This is an ablation, not the canonical
low-risk configuration, because it is more susceptible to circular confirmation of
the ResNet's own early bias.

## 5. Why the previous proposals are not the current method

### Legacy routed augmentation

The implemented routing method reduces selected SpLiCE activations to a scalar and
chooses where to apply a stronger generic augmentation. It does not say which
semantic factor must change or what representation should replace it. It failed to
provide a stable WGA gain, and semantic routing underperformed shuffled routing in
the completed paired controls.

**Decision:** retain as a negative result and baseline; do not use as the main
method.

### Legacy correlation regularization

The implemented regularizer discourages linear dependence between ResNet features
and selected SpLiCE coordinates. It does not specify a counterfactual target and can
remove useful target information along with nuisance information.

**Decision:** retain as a baseline; do not use as the main method.

### SpLiCE-IU and residual-preserving class neutralization

SpLiCE-IU uses cross-fitted target probes and true target labels to rank concepts by
error repair minus damage. On the frozen Waterbirds CBM it selected `crow, heron,
birds` and improved WGA only from 39.85 to 40.60, while the privileged conditional-
group oracle selected background-like concepts and reached 48.87 WGA. This shows
that intervention utility is not the same as nuisance specificity.

The downstream class-median edit also requires target labels and preserves the
unexplained CLIP residual, which may still contain the nuisance.

**Decision:** keep the implementation and pending run as a scientifically useful
label-assisted negative/control track. It is superseded as the canonical proposal.

### SpLiCE-CRP v1: one hard mined positive

The first CRP proposal removed one active concept group, found one nearest neighbour
in the edited CLIP space, and treated that image as an additional positive. Three
problems make that version unsuitable as the paper's main method:

1. it did not define a defensible label-free rule for choosing which group is the
   nuisance;
2. subtracting only the sparse SpLiCE contribution can leave the nuisance in the
   unexplained CLIP residual;
3. a small number of minority images can become repeatedly sampled hubs, and a
   single false positive directly pulls different classes together.

**Decision:** keep hard-positive CRP only as an ablation demonstrating why the
relational formulation is needed.

## 6. Canonical architecture: SpLiCE-CRP v2

CRP v2 stands for **Concept-Projected Relational Pretraining**. The architecture
has two stages: a frozen audit/graph-construction stage and an SSL training stage.

### Stage A — cache frozen representations

For every training image `x_i`, cache:

- normalized OpenCLIP image embedding `z_i`;
- centered CLIP vector `u_i = z_i - mu`, using the same image mean as SpLiCE;
- sparse SpLiCE code `c_i`;
- normalized DINOv3 embedding `d_i`;
- optionally, the same representations for a small fixed set of deterministic
  crops used only to measure stability.

All frozen models run without gradients. Caches must be indexed by dataset sample
ID and record checkpoint/configuration identifiers. Content hashes may be added
when a cache builder can compute them automatically; hand-authored hashes are not
required because they create bookkeeping without verifying model contents.

**Implementation decision, 2026-08-18:** cache validation requires a schema version,
unique sample IDs, aligned tensor lengths, and finite values. Mandatory user-supplied
SHA-256 strings were rejected because the audit could not verify that those strings
actually described the loaded checkpoints. Consequence: cache builders may still add
automatic content digests under optional provenance, while cluster audit jobs no
longer reread a large cache solely to hash it. This does not change the label boundary
or any scientific selection criterion.

### Stage B — form distributed semantic concept groups

Do not treat a single word as a complete factor. Concepts such as `forest`,
`forests`, `woods`, `vegetation`, `bamboo`, and `rainforest` can encode one
distributed nuisance family.

Construct candidate groups using only:

- cosine similarity between centered text dictionary directions;
- coactivation similarity of sparse codes across unlabeled images;
- lexical deduplication for trivial singular/plural or near-identical strings;
- stability across bootstrap samples.

The initial implementation should build a graph over active concepts and cluster
that graph. Group size, clustering threshold, and active-concept frequency cutoffs
are audit hyperparameters and must be chosen from unlabeled stability/coverage,
not WGA.

For group `G`, stack its dictionary directions and compute an orthonormal basis:

```text
Q_G = orth(D_G^T)
```

Near-collinear directions must be removed during orthogonalization.

### Stage C — intervene on the full CLIP embedding

The canonical intervention is not sparse coefficient subtraction. It is a
projection of the full centered embedding:

```text
u_i^(-G) = normalize((I - Q_G Q_G^T) u_i)
```

This removes every component of `u_i` lying in the group subspace, including
components that SpLiCE's sparse reconstruction left in its residual.

Required intervention ablations are:

1. sparse coefficient subtraction;
2. full orthogonal projection — canonical;
3. sparse refitting while forbidding group `G`;
4. matched random subspace projection.

Projection can also remove useful information when core and nuisance directions
overlap. Therefore projection alone is never enough to accept a group.

### Stage D — construct concept-conditioned candidate relations

For anchor `i`, candidate `j`, and factor group `G`, compute the intervention gain:

```text
Delta(i,j,G) = cos(u_i^(-G), u_j^(-G)) - cos(u_i, u_j)
```

A relation is a candidate only when:

1. `Delta(i,j,G) > 0`: removing `G` caused the similarity increase;
2. the group activations differ by a sufficient unlabeled quantile;
3. `i` and `j` are reciprocal or otherwise stable neighbours in projected CLIP
   space;
4. their semantic compatibility is independently supported by DINOv3 similarity
   or DINOv3 neighbour agreement;
5. the relation persists under the fixed multi-crop/bootstrap audit;
6. neither endpoint is accepted solely because it is a high-degree hub.

Do not impose a high raw-CLIP similarity as the primary semantic constraint. On a
spurious dataset, raw CLIP similarity can be dominated by the very factor being
removed. A loose unrelated-image rejection floor is acceptable, but DINO agreement
and post-projection stability are the principal semantic guards.

### Stage E — score and select factor groups without labels

The method does not declare a group nuisance because its words sound like a
background. A group is useful only if removing it repeatedly reveals relations
supported by an independent semantic geometry.

A reference group score is:

```text
score(G) = robust_positive_gain(G)
           * coverage(G)
           * crop_bootstrap_stability(G)
           * semantic_agreement(G)
           / hubness_penalty(G)
```

Operational definitions:

- `robust_positive_gain`: trimmed mean or median of positive `Delta` values;
- `coverage`: fraction of anchors with at least one accepted relation for `G`;
- `crop_bootstrap_stability`: reproducibility of accepted relations and group rank;
- `semantic_agreement`: DINOv3 neighbour support;
- `hubness_penalty`: penalty based on maximum indegree, Gini coefficient, or
  effective donor count.

Selection must be calibrated against a label-free null distribution made from
matched random subspaces and shuffled SpLiCE codes. Prefer significance/stability
thresholds over an unconditional fixed top-K. Returning no selected group is a
valid outcome and must reduce the downstream method to the SimCLR baseline.

The exact score exponents and thresholds are not yet empirical facts. They must be
fixed by the frozen unlabeled audit and then held constant across datasets where
possible. Future agents must not invent target- or group-dependent tuning while
calling it label-free.

### Stage F — build a soft counterfactual teacher graph

Accepted relations are not converted into one permanent positive image. Instead,
construct a sparse weighted graph or row-stochastic teacher distribution:

```text
p_T(j | i) proportional to
    confidence(i,j,G) * exp(sim_projected(i,j,G) / tau_T)
```

Combine evidence from multiple accepted groups by normalized confidence. Limit the
top-k support per anchor, degree-normalize the graph, and use an indegree cap or a
Sinkhorn-style balancing step to prevent a few rare images from dominating.

Once a group has been validated by seed relations, its projection may be applied to
all sufficiently active images when constructing teacher similarities. This
propagates the factor-level evidence without requiring every majority example to
reuse one of the same few minority donors.

The graph artifact must store:

- neighbour indices and weights;
- responsible concept group(s);
- intervention gain;
- semantic-agreement confidence;
- crop/bootstrap stability;
- graph degree statistics;
- complete configuration and cache hashes.

### Stage G — train the ResNet with SimCLR plus relational distillation

Train the normal SimCLR model and preserve its NT-Xent loss:

```text
L = L_SimCLR + lambda(t) * L_rel
```

For student backbone features `h_i = normalize(f_theta(x_i))`, define a student
distribution over the teacher candidate set:

```text
p_S(j | i) proportional to exp(cos(h_i, h_j) / tau_S)
L_rel = sum_i q_i * KL(p_T(.|i) || p_S(.|i))
```

`q_i` is the frozen graph confidence. Relational matching is applied to backbone
similarities so that a separate MLP cannot absorb the entire concept signal without
changing the representation used by the downstream probe.

Implementation may use a cached top-k graph plus a graph-aware batch sampler or an
indexed student memory bank. It must not silently degenerate into one unweighted
hard positive per anchor.

Training safety mechanisms:

- SimCLR-only warm-up before enabling `L_rel`;
- gradual ramp of `lambda(t)`;
- zero loss for anchors without confident graph support;
- bounded per-anchor and per-node contribution;
- base SimCLR negatives/uniformity remain active;
- matched seeds and identical augmentation schedules for all controls.

## 7. Why this architecture addresses the three CRP v1 failures

### Failure 1: no definition of the active nuisance group

CRP v2 does not use names, target errors, or group labels to identify nuisance.
It asks whether removal of a factor restores stable relations supported by a
separate SSL geometry. A class factor is more likely to create cross-class
neighbours after deletion, which the independent semantic verifier should reject.
This is an inductive bias, not a mathematical guarantee.

### Failure 2: the unexplained residual can keep the nuisance

The canonical intervention projects the entire centered CLIP vector away from the
factor subspace. SpLiCE determines the interpretable subspace, but its incomplete
reconstruction no longer limits the strength of removal.

### Failure 3: contradictory thresholds and scarce donors

Raw CLIP similarity is demoted from a strict high threshold because it can encode
the nuisance. Semantic safety comes from independent agreement. Training uses a
soft, degree-controlled relation graph, not repeated attraction to one rare donor.

## 8. Rejected or demoted alternatives

### Crop-diffuseness as the sole nuisance score

It is plausible for Waterbirds backgrounds but not general: gender and hair colour
in CelebA occupy the same face region and can have similar crop stability.

**Use:** optional score component or Waterbirds diagnostic, not the main selector.

### Worst-cell or error-based group ranking

It gives a strong task-aligned signal but requires target labels and can become a
proxy for WGA optimization.

**Use:** privileged oracle only.

### Weight every concept group softly without selection

This avoids a discrete choice but mixes class and nuisance factors and can dilute
the useful intervention.

**Use:** ablation/control.

### Sparse subtraction while preserving the full residual

It is locally interpretable but may leave the nuisance in the residual and may not
change neighbour rankings.

**Use:** intervention ablation, not canonical removal.

### Reconstruct only from the remaining sparse concepts

It certainly removes the selected group but also discards all information not
captured by the dictionary, including potentially useful fine-grained class signal.

**Use:** aggressive upper/lower-bound ablation.

### One nearest neighbour as a hard positive

It is simple and related to NNCLR, but false positives and minority-donor hubness
can directly corrupt the representation.

**Use:** CRP v1 ablation only.

### DIAL+-style SAE editing inside the ResNet

It is close to existing VLM editing and requires deciding which layer and task
representation to decompose. It weakens the distinction between this project and
DIAL+.

**Use:** related work, not the planned method.

### Pseudo-environment clustering alone

Unsupervised clusters can primarily recover target classes rather than nuisance
environments, especially on Waterbirds.

**Use:** diagnostic baseline only.

### Spectral regularization as the proposed method

It is an important matched baseline and may be combined with CRP v2 after the main
mechanism is established. It is not required as part of the contribution.

**Use:** same-setting baseline; `CRP v2 + Spec` is optional after separate effects
are measured.

## 9. Frozen audit before any expensive SSL run

Do not launch a multi-seed 500/1000-epoch CRP v2 sweep before the frozen audit.

### Interventions to compare

1. raw CLIP;
2. sparse coefficient subtraction;
3. full orthogonal projection;
4. forbidden-group sparse refit;
5. matched random projection;
6. shuffled SpLiCE codes/groups.

### Label-free development metrics

- cosine change distribution;
- top-1/top-5 neighbour turnover;
- Jaccard@k before and after intervention;
- reciprocal-neighbour coverage;
- DINO semantic agreement;
- crop/bootstrap stability;
- graph indegree distribution, Gini coefficient, and effective donor count;
- selected-group score relative to the random/shuffled null.

High random-pair cosine correlation alone is not enough to declare the mechanism
inactive. If most anchors have `Jaccard@20 > 0.9` and unchanged top neighbours,
however, the intervention is operationally too weak.

### Post-hoc diagnostic metrics using hidden annotations

These metrics evaluate the mechanism but must not choose its hyperparameters:

- same-target precision of graph edges;
- opposite-spurious rate conditional on same target;
- edge coverage for every `(y,a)` group;
- frozen projected-CLIP average and worst-group accuracy;
- concept-family overlap with the conditional-group oracle.

### Recommended go/no-go conditions

Proceed to SSL only if:

- projection changes local rankings materially;
- same-target precision is not more than about 2 percentage points below the
  independent semantic-consensus baseline;
- same-target/opposite-spurious relations are substantially enriched relative to
  raw CLIP kNN, preferably at least 2x;
- coverage is not concentrated in only one target or group;
- hubness remains controlled after graph balancing;
- semantic SpLiCE groups outperform random subspaces and shuffled codes;
- selected factor families are stable across crops/bootstrap samples.

These numeric values are initial falsification thresholds, not reportable results.
They may be revised once, before observing downstream WGA, and the revision must be
recorded here.

## 10. Experiment matrix

### Stage 1 — frozen mechanism audit

Run the audit above on Waterbirds. Use `y` and `a` only in a separate post-hoc
evaluation function.

### Stage 2 — one-seed shortened pilot

Required configurations:

1. SimCLR;
2. SimCLR + raw-CLIP relational distillation;
3. SimCLR + DINO-only relational distillation;
4. SpLiCE-CRP v2;
5. SpLiCE-CRP v2 with shuffled concept codes;
6. SpLiCE-CRP v2 with matched random subspaces;
7. hard-positive CRP v1;
8. Ghanooni spectral regularization.

Optional only after the main comparison:

- SpLiCE-CRP v2 + spectral regularization;
- DINO replaced by EMA-ResNet consensus;
- no-DINO multi-crop-only verifier;
- coefficient subtraction or forbidden-group refit instead of projection.

### Stage 3 — paired multi-seed evaluation

Promote only methods that pass the frozen audit and one-seed controls. Use matched
initializations and report mean, standard deviation, and per-group accuracies.

Primary datasets:

- Waterbirds: background nuisance;
- CelebA: demographic/appearance nuisance;
- SpurCIFAR10: synthetic colour nuisance.

The same discovery code and unlabeled thresholds should be used across datasets as
far as computationally possible. Dataset-specific nuisance word lists are forbidden.

### Main reported baselines

Same-setting SSL baselines:

- SimCLR;
- SimSiam, if already supported or affordable;
- NNCLR;
- IFM;
- LA-SSL;
- SimCLR-LateTVG;
- Ghanooni et al. spectral regularization;
- raw-CLIP and DINO-only relational distillation.

External concept/VLM comparison table, clearly separated because the backbone and
evaluation protocol differ:

- zero-shot CLIP;
- DIAL and DIAL+;
- Bias Leaves a Gradient Tail;
- frozen SpLiCE-CBM;
- privileged SpLiCE oracle.

Upper bounds and non-comparable references must be labelled as such:

- oracle same-target/opposite-group pairs;
- Cross-Variant SSL;
- GroupDRO;
- DFR.

## 11. Interpretation rules

### Claims that are currently supported

- The repository trains a ResNet18-large SSL encoder; ViT-B/32 is the frozen
  OpenCLIP/SpLiCE extractor, not the trainable backbone.
- SpLiCE contains useful nuisance-related concepts: the privileged Waterbirds
  oracle improved frozen-CBM WGA by about 9 points.
- Existing automatic selectors have not found those nuisance concepts reliably.
- Routed augmentation and the existing regularizer did not provide a reliable WGA
  improvement.
- SpLiCE-IU is target-label-assisted and weakly positive only in the frozen CBM
  diagnostic; it is not a demonstrated SSL solution.

### Claims that are not yet supported

- SpLiCE-CRP v2 improves WGA.
- full projection is better than coefficient subtraction on the available data;
- DINO consensus reliably preserves target semantics;
- the proposed label-free group score identifies the true nuisance;
- relational distillation transfers the desired invariance into the ResNet.

Every one of these statements is a hypothesis until the corresponding control is
run. New agents must not present the proposal as an achieved result.

## 12. Main risks and falsifiers

### Fundamental identifiability

Without task labels, no method can universally know which visual factor is core and
which is nuisance. CRP v2 introduces an explicit inductive bias: factors whose
removal restores a relation supported by an independent SSL geometry are treated as
nuisance candidates. The claim is empirical, not universal.

### Shared bias between CLIP and DINO

Both pretrained models may encode the same background or demographic shortcut. The
DINO verifier therefore reduces risk but cannot prove semantic equivalence.

### Entangled concept subspaces

A concept group can mix target and nuisance directions. Full projection is stronger
than sparse subtraction and can therefore cause greater target damage. The semantic
guard, random-subspace controls, and average-accuracy reporting are essential.

### Graph sparsity or hubness

Waterbirds has only 240 minority training images out of 4,795. There are many
possible pairs but few unique counterfactual donors. The soft graph, degree control,
and factor-level propagation address this, but low coverage remains a possible
falsifier.

### Distillation without transfer

An auxiliary head can absorb teacher information without changing the backbone.
CRP v2 therefore matches relations on the backbone representation, while SimCLR
keeps its normal projection head.

### Teacher improvement does not imply student improvement

Even a better projected CLIP graph may be incompatible with the inductive bias or
capacity of ResNet18. This requires a matched one-seed pilot before scale-up.

## 13. Implementation roadmap

The proposed method is not yet implemented. Work should proceed in this order:

1. Add a frozen-feature audit script that consumes cached CLIP/SpLiCE features.
2. Add DINOv3 feature caching with dataset-index integrity checks.
3. Implement semantic concept grouping and store versioned group definitions.
4. Implement coefficient subtraction, full projection, forbidden-group refit, and
   random-subspace interventions behind one interface.
5. Implement neighbour-turnover and graph-quality diagnostics.
6. Implement label-free group scoring with shuffled/random null controls.
7. Export a versioned sparse teacher graph artifact.
8. Only after the audit passes, add relational distillation to the SimCLR loop.
9. Add unit tests for projection, cache alignment, graph normalization, label
   isolation, and deterministic graph construction.
10. Run the one-seed control matrix before any multi-seed submission.

Implementation changes should be surgical. Preserve the existing IU, routing, and
regularization modes as baselines instead of rewriting them in place.

## 14. Files and authority

Current important files:

- `PROJECT_GROUND_TRUTH.md` — authoritative scientific and architectural state;
- `README.md` — repository usage and entry points;
- `SPUR_SPLICE_CHAT_SUMMARY.md` — historical hand-off, currently describing the
  older SpLiCE-IU phase;
- `Spur_SpLiCE.tex` — current paper draft, also still centred on SpLiCE-IU;
- `spur_splice.py` — main implemented SSL entry point;
- `scripts/tools/discover_splice_spurious_concepts.py` — existing discovery code;
- `splice/ssl_regularization.py` — existing interventions and distillation code;
- `experiments/spurious_eval/` — datasets, models, losses, training, and metrics.

`splice/crp.py` now implements the label-free frozen audit and teacher-graph
artifact. The LaTeX paper and current training CLI still do not demonstrate CRP v2
training or a positive result.

## 15. Literature anchors

- SpLiCE: Bhalla et al., *Interpreting CLIP with Sparse Linear Concept Embeddings*,
  NeurIPS 2024: <https://openreview.net/forum?id=7UyBKTFrtd>
- Ghanooni et al., *Mitigating Spurious Features in Contrastive Learning with
  Spectral Regularization*, NeurIPS 2025.
- DIAL/DIAL+: Yalavarthi et al., *Label-Free Mitigation of Spurious Correlations in
  VLMs using Sparse Autoencoders*, ICLR 2026:
  <https://openreview.net/forum?id=NHOLsaHuFv>
- DINOv3, 2025: <https://arxiv.org/abs/2508.10104>
- PCA++, 2025: <https://arxiv.org/abs/2511.12278>
- NNCLR: Dwibedi et al., ICCV 2021:
  <https://openaccess.thecvf.com/content/ICCV2021/html/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.html>
- Waterbirds construction and group imbalance: Sagawa et al., ICLR 2020:
  <https://openreview.net/forum?id=ryxGuJrFvS>

## 16. Instructions for a new chat or agent

Before proposing, implementing, or describing the project:

1. Read this file completely.
2. State whether the requested work concerns legacy implemented methods or the
   proposed CRP v2 method.
3. Inspect the current code before claiming that a CRP v2 component exists.
4. Preserve the strict information boundary: no `y`, `a`, or group labels in CRP v2
   discovery/training.
5. Do not tune by WGA and call the method label-free.
6. Treat DINO-only, raw-CLIP, shuffled-code, and random-subspace controls as required,
   not optional decorations.
7. Do not claim positive CRP v2 results until real matched experiments exist.
8. If changing a canonical decision, update this file with the reason, rejected
   alternative, date, and consequences for experiments and paper claims.
