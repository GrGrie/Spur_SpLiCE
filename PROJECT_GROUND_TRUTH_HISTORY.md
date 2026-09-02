# Spur_SpLiCE — project ground truth (historical archive)

**Status:** canonical project specification

**Last updated:** 2026-09-02
**Implemented baseline:** SpLiCE-CRP v2 — Concept-Projected Relational Pretraining

**Implemented experimental extension:** SpLiCE-CQT v1 — Concept Quotient Transport

**Implementation status:** CRP and CQT graph construction plus shared SSL relational
training are implemented. Initial single-seed Waterbirds experiments exist, but
multi-seed stability and the causal value of the graph mechanism are not established.

> This file is the authoritative source of truth for the current project direction.
> If `README.md`, `SPUR_SPLICE_CHAT_SUMMARY.md`, `Spur_SpLiCE.tex`, old chat
> messages, scripts, or comments conflict with this document, follow this document.
> Do not silently reinterpret the method. Update this file first when a scientific
> decision changes.

## 1. One-paragraph project summary

The project asks whether sparse, language-aligned concepts from a frozen
OpenCLIP/SpLiCE model can improve the worst-group accuracy (WGA) of a separately
trained self-supervised ResNet without using per-image target labels, spurious
attributes, or group labels during concept discovery or SSL training. The existing
SpLiCE-CRP v2 method uses SpLiCE to propose semantic factor subspaces, removes
each candidate subspace from the **full centered CLIP embedding** by orthogonal
projection, and asks whether that intervention restores semantic neighbourhoods
independently supported by a frozen SSL verifier such as DINOv3. Stable
concept-conditioned relations define a soft counterfactual teacher graph. A
ResNet18-large is then trained with ordinary SimCLR plus a confidence-gated
relational distillation loss that matches this graph. The method does not use one
mined image as a permanent hard positive. Target and group annotations are used
only after SSL training for linear evaluation, WGA measurement, and explicitly
labelled diagnostic oracles. The experimental SpLiCE-CQT v1 mode reuses the same
cache and relational trainer, but searches for two-state concept factors, removes
only their rank-one state contrast, and builds relations by capacity-constrained
partial transport that preserves non-factor SpLiCE semantics.

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
3. Independent semantic checks reject likely class-destroying interventions.
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

The label-free CRP and CQT modes are stricter than the already implemented
SpLiCE-IU method: SpLiCE-IU uses target labels, whereas CRP and CQT do not.

## 4. Models and their roles

### Trainable model

- **Default:** ResNet18-large trained from scratch with SimCLR.
- **Scale check:** ResNet50-large from scratch, only after the mechanism works
  on ResNet18-large.
- **Pretraining check:** `resnet50_pretrained` initializes the same standard
  large-input ResNet50 architecture from torchvision ImageNet-1K V2 weights;
  the full encoder remains trainable during SimCLR plus CRP/CQT training.
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
- Role in CRP: pair-level semantic support and neighbour agreement.
- Role in CQT: independent local-geometry damage guard; DINO does not define CQT
  transport cost or the quotient direction.
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
- normalized DINOv3 embedding `d_i`.

All frozen models run without gradients. Caches must be indexed by dataset sample
ID and record checkpoint/configuration identifiers. CQT reuses this cache without
adding another cache schema. Hand-authored hashes are not required because they
create bookkeeping without verifying model contents.

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
- lexical deduplication for trivial singular/plural or near-identical strings.

The implementation unions active concepts satisfying those rules. Group size,
similarity thresholds, and active-concept frequency cutoffs are audit
hyperparameters and must be chosen from unlabeled coverage/null behavior, not WGA.

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

A relation is accepted by the current implementation only when:

1. `Delta(i,j,G) > 0`: removing `G` caused the similarity increase;
2. the group activations differ by a sufficient unlabeled quantile;
3. `i` and `j` are reciprocal neighbours in projected CLIP space;
4. `j` is in `i`'s raw-DINO neighbour set.

Do not impose a high raw-CLIP similarity as the primary semantic constraint. On a
spurious dataset, raw CLIP similarity can be dominated by the very factor being
removed. A loose unrelated-image rejection floor is acceptable, but DINO agreement
and post-projection stability are the principal semantic guards.

### Stage E — score and select factor groups without labels

The method does not declare a group nuisance because its words sound like a
background. A group is useful only if removing it repeatedly reveals relations
supported by an independent semantic geometry.

The implemented group score is:

```text
score(G) = robust_positive_gain(G)
           * coverage(G)
           * semantic_agreement(G)
           * activation_gain_alignment(G)
           / hubness_penalty(G)
```

Operational definitions:

- `robust_positive_gain`: trimmed mean or median of positive `Delta` values;
- `coverage`: fraction of anchors with at least one accepted relation for `G`;
- `semantic_agreement`: DINOv3 neighbour support;
- `activation_gain_alignment`: non-negative correlation between activation
  difference and positive intervention gain;
- `hubness_penalty`: penalty based on maximum indegree, Gini coefficient, or
  effective donor count.

Selection must be calibrated against a label-free null distribution made from
matched random subspaces and shuffled SpLiCE codes. Prefer significance/stability
thresholds over an unconditional fixed top-K. Returning no selected group is a
valid outcome and must reduce the downstream method to the SimCLR baseline.
An optional `max_selected_groups` cap is applied only after the null and coverage
gates, ranked by label-free null excess; zero disables the cap. The cap is an audit
and interpretability control, not permission to choose group count from downstream
WGA.

The exact score exponents and thresholds are not yet empirical facts. They must be
fixed by the frozen unlabeled audit and then held constant across datasets where
possible. Future agents must not invent target- or group-dependent tuning while
calling it label-free.

### Stage F — build a soft counterfactual teacher graph

Accepted relations are not converted into one permanent positive image. Instead,
construct a sparse weighted graph or row-stochastic teacher distribution:

```text
p_T(j | i) proportional to
    max_G confidence(i,j,G)
```

Projected similarity is audited/stored, while current graph weights are normalized
accepted-edge confidence rather than a second teacher-temperature softmax.

The current graph builder keeps the strongest evidence per directed pair, limits
top-k support per anchor, applies a hard global indegree cap, and row-normalizes
supported anchors. Sinkhorn balancing and post-selection propagation are not
implemented.

The graph artifact must store:

- neighbour indices and weights;
- responsible concept group(s);
- intervention gain;
- semantic-agreement confidence;
- graph degree statistics;
- complete configuration, sample IDs, and cache provenance.

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

The implementation uses a cached top-k graph plus a graph-aware batch sampler. It
does not use a student memory bank and does not degenerate into one unweighted hard
positive per anchor.

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

## 8. Experimental extension: SpLiCE-CQT v1

CQT stands for **Concept Quotient Transport**. It is an implemented experimental
graph-construction mode over the existing CRP cache and relational trainer. It does
not replace or modify CRP, and it is not yet supported by positive real-data
results.

### 8.1 Motivation and identifiability limit

CRP can select any visually strong concept group whose removal reorganizes image
neighbours. A coherent class concept can satisfy this criterion just like a
background concept. DINO support reduces obviously unsafe pairs, but cannot tell
which factor matters to an unknown downstream task and can share CLIP's bias.

This is partly fundamental: using task-agnostic unlabeled observations, “core” and
“spurious” are not universally identifiable. Exchanging their names can leave the
same observational distribution. CQT therefore does not claim to discover true
nuisance variables. It adds a narrower, testable inductive bias:

> Prefer two concept states that are mutually exclusive but retain overlapping
> non-factor semantics; erase only their state contrast and transport mass between
> compatible samples.

Target factors can also satisfy this assumption. Hidden-label pair purity and
average-accuracy preservation remain mandatory post-hoc falsifiers.

### 8.2 Cross-reproduced two-state proposals

CQT first reuses CRP's lexical/text/coactivation concept groups. To bound quadratic
pair search, v1 retains at most `max_candidate_groups`, ranked by balanced unlabeled
support, and at most `max_factors` final proposals.

For group pair `F=(G_A,G_B)` and cross-fit fold `s`, define exclusive pseudo-states:

```text
A_s = {i in fold s : activation(G_A)>0 and activation(G_B)=0}
B_s = {i in fold s : activation(G_B)>0 and activation(G_A)=0}
```

Coactive and inactive samples do not define this factor. A proposal must satisfy in
both deterministic random halves:

- at least `min_state_samples` per exclusive state;
- exclusivity above `min_exclusivity`;
- cosine agreement above `min_context_similarity` between mean semantic SpLiCE
  embeddings after all coordinates in `F` are removed.

The proposal must reproduce in both fold directions. For each direction, state
centroids are learned on one half and predictability/intervention evidence is
evaluated on the other. Version 1 intentionally supports exactly two states; an
adaptive number of states would add rank and model-selection degrees of freedom.

### 8.3 Rank-one concept quotient

Let each state direction be the normalized mean of its dictionary words:

```text
m_A = normalize(mean_{k in G_A} D_k)
m_B = normalize(mean_{k in G_B} D_k)
q_F = normalize(m_A - m_B)
```

For centered normalized CLIP vector `u_i`, CQT computes:

```text
r_i^F = normalize((I - q_F q_F^T) u_i)
```

This removes exactly the one-dimensional state contrast. Directions orthogonal to
`q_F`, including the shared state mean approximately proportional to `m_A+m_B`,
remain. CRP instead removes the full span of all selected group words. Rank is
fixed to one in CQT v1 and is never selected using downstream labels.

The implementation checks whether the operation is meaningful by cross-fitted
nearest-centroid pseudo-state classification. Raw pseudo-state balanced accuracy
must exceed `min_state_balanced_accuracy`, and quotient efficacy

```text
E_F = (BA_raw - BA_quotient) / max(BA_raw - 0.5, epsilon)
```

must exceed `min_quotient_efficacy` in both directions. This rejects readable word
pairs whose dictionary contrast has little effect on image geometry.

### 8.4 Explicit non-factor word geometry

CQT uses the human-readable SpLiCE decomposition directly in matching. After
excluding factor coordinates, define:

```text
w_i^(-F) = normalize(sum_{k not in F} c_ik D_k)
```

Then:

```text
cos(w_i^(-F), w_j^(-F))
  = normalized(c_i^(-F)^T D D^T c_j^(-F))
```

This is an implicit word-kernel similarity. The code computes sparse `c^(-F)D`
instead of materializing the full vocabulary kernel. Thus a pair is attractive only
when the quotient CLIP geometry improves and the other readable concepts remain
compatible.

### 8.5 Sparse capacity-constrained partial transport

For each factor/evaluation fold, CQT unions the top word-semantic candidates in
both `A -> B` and `B -> A` directions. On that fixed sparse bipartite support, cost
is:

```text
C_ij = 0.5 * rank_distance(cos(r_i^F, r_j^F))
       + 0.5 * (1 - cos(w_i^(-F), w_j^(-F))) / 2
```

`rank_distance` is normalized inside each anchor's candidate list. Equal weights
are fixed in v1. CQT solves the exact linear program on this sparse support:

```text
min_T sum_ij C_ij T_ij
subject to
    sum_j T_ij <= 1
    sum_i T_ij <= 1
    sum_ij T_ij  = M_F
    0 <= T_ij <= 1
```

`M_F` is a fixed fraction of the smaller state with a minimum pair count. A maximum
bipartite matching check rejects candidate supports unable to carry the requested
mass. `scipy.optimize.linprog` with HiGHS solves the sparse LP. The bipartite
cardinality-constrained polytope has integral extreme points up to numerical
tolerance, while the downstream graph also accepts fractional flow.

Partial mass avoids forcing poor pairs. Unit source/destination capacities prevent
factor-level donor hubs. The shared final graph builder still applies its top-k and
global indegree cap when several factors contribute edges.

### 8.6 DINO as a local-geometry damage guard

CQT deliberately does not require transported cross-state pairs to be raw-DINO
neighbours, and it does not quotient DINO. Either would recreate a hard raw-space
veto or risk deleting an unknown core factor in both teachers.

Within each pseudo-state, CQT records every point's sorted cosine similarities to
its raw-DINO local neighbours. Candidate damage is the mean absolute difference
between the two local similarity profiles. A transport plan passes only when its
mass-weighted damage is no worse than `dino_damage_quantile` of candidate-edge
damage. DINO therefore asks whether the map connects comparable local geometries;
it neither chooses the quotient nor enters transport cost.

This signature is a lightweight v1 proxy, not full Gromov-Wasserstein transport.
A stronger structural guard is a future ablation only if this proxy is falsified.

### 8.7 Null controls, gates, and factor confidence

Every factor is compared with:

- matched random rank-one CLIP contrasts, with transport re-solved on the same
  pseudo-states;
- shuffled pseudo-state assignments, preserving state sizes and re-solving
  candidates/transport.

A factor is selected only if every hard gate passes: cross-fold support/context,
raw pseudo-state predictability, quotient efficacy, positive intervention gain,
gain above the combined null quantile, non-factor word preservation, DINO local
safety, and minimum coverage. Selecting no factors is valid.

After these gates, factor graph confidence uses only:

```text
null_excess_gain(F) = max(0, gain(F) - null_threshold(F))
```

Proposal context, word similarity, and DINO are gates rather than multiplicative
“independent” score terms. This avoids double-counting quantities derived from the
same SpLiCE source.

Positive-mass transport pairs become symmetric graph relations. The artifact type
is `splice_cqt_v1_teacher_graph`; tensor `graph_version` remains 2 because the
row-stochastic trainer contract is unchanged. `cqt_relational` uses exactly the
same sampler, relational KL, SimCLR start/warm-up, optimizer, and evaluation path as
CRP. The scientific comparison therefore changes graph construction only.

### 8.8 Human-readable concept cards

Every CQT graph is one complete JSON artifact. Each factor card exposes:

- state-A and state-B words/indices;
- fold-wise exclusivity and non-factor context similarity;
- raw and quotient pseudo-state balanced accuracy and quotient efficacy;
- matched mass, transport gain, word similarity, coverage, and DINO damage;
- random-contrast and shuffled-state null scores;
- all hard-gate decisions and null-excess gain;
- top preserved non-factor words;
- representative sample-ID pairs with raw distance, quotient distance, word
  similarity, transport cost, and mass.

These words are part of the scientific output, not unused metadata. Sample IDs are
label-free; image/hidden-label rendering belongs in a separate post-hoc report.

### 8.9 Why keep SpLiCE rather than add another concept model

SpLiCE supplies three useful properties simultaneously: sparse non-negative
activations, direct language-readable words, and dictionary directions in the same
CLIP space used by the intervention. A new SAE would add a layer choice, training
procedure, naming procedure, and extra architecture without supplying the missing
task-identification signal. It would also move the contribution closer to DIAL+
while weakening simple word-level auditability.

CoBalT-style balancing is not interchangeable with this canonical setting when
its balancing uses target classes. DIAL+ likewise uses a supplied zero-shot class
set and prediction changes to decide what to edit. They are useful task-anchored
comparisons, but canonical CQT has no class names. Any future class-name variant
must be called `CQT-T` and reported separately.

### 8.10 CRP/CQT implementation comparison

| Component | SpLiCE-CRP v2 | SpLiCE-CQT v1 |
|---|---|---|
| Candidate | one word group | two mutually exclusive word-group states |
| CLIP edit | full group-span projection | rank-one state quotient |
| Relation builder | projected CLIP kNN | partial concept-preserving transport |
| Word use | group and activation difference | explicit non-factor word-kernel cost |
| DINO role | raw pair-neighbour hard support | local-geometry damage veto |
| Hub control | final indegree cap | transport capacities plus final cap |
| Nulls | random subspaces/shuffled activation | random contrasts/shuffled states |
| Student | SimCLR + relational KL | confidence-weighted graph positives + decaying relational KL |

CQT can miss continuous, one-sided, or multi-state nuisances. CRP remains runnable
and is a required baseline rather than dead code.

### 8.11 Running CRP and CQT

CRP and CQT have separate Slurm entry points with their method-specific parameters
grouped near the top:

```bash
sbatch scripts/train_crp.sbatch
sbatch scripts/train_cqt.sbatch
```

`train_crp.sbatch` defaults directly to `crp_relational`, while
`train_cqt.sbatch` fixes `cqt_relational` and supplies its configuration to the
shared W&B-enabled execution body. Both modes build/reuse
`outputs/crp/<dataset>_train_features.pt` and pass a row-stochastic graph to the
same training entry point. CRP graphs stay in `outputs/crp/<dataset>/`; complete
CQT graph and concept-card JSON artifacts are written to `outputs/cqt/<dataset>/`.
CQT-specific thresholds live in one commented configuration block and do not
borrow CRP audit variables. The full resolved graph config and artifact type are
added to the W&B run config after graph loading.

The sbatch script compares an existing artifact's stored config with the resolved
CRP/CQT config. A missing, unreadable, wrong-type, or configuration-mismatched graph
is rebuilt automatically while a valid frozen feature cache is reused. Therefore a
normal CQT run needs only `sbatch scripts/train_cqt.sbatch`; no separate graph command
or manual invalidation is required. `CRP_FORCE_REBUILD=true` remains available when
the frozen cache itself must also be reconstructed. Real full training keeps W&B
enabled as required by `AGENTS.md`; disabling it is for explicit smoke/local debug
only.

## 9. Rejected or demoted alternatives

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

## 10. Frozen audit before any expensive SSL run

Do not launch a multi-seed 500/1000-epoch CRP or CQT sweep before the frozen audit.

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
- graph indegree distribution, Gini coefficient, and effective donor count;
- selected-group score relative to the random/shuffled null.

For CQT additionally record:

- number and words of cross-reproduced state pairs;
- raw pseudo-state balanced accuracy and quotient efficacy;
- requested/feasible transport mass and factor coverage;
- non-factor word similarity and top preserved words;
- DINO local damage relative to its candidate quantile;
- random-contrast and shuffled-state transport nulls.

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
- selected factor families are reproducible across audit seeds.

Initial CQT-specific conditions, to be frozen before WGA is viewed, are:

- raw pseudo-state balanced accuracy at least 0.70;
- quotient efficacy at least 0.50;
- factor gain above a 0.95 combined null quantile;
- selection in at least two of three audit seeds;
- at least 10% supported anchors in the final graph;
- effective donor count at least 25% of supported anchors.

These numeric values are initial falsification thresholds, not reportable results.
They may be revised once, before observing downstream WGA, and the revision must be
recorded here.

## 11. Experiment matrix

### Stage 1 — frozen mechanism audit

Run the audit above on Waterbirds. Use `y` and `a` only in a separate post-hoc
evaluation function.

### Stage 2 — one-seed shortened pilot

Required configurations:

1. SimCLR;
2. SimCLR + raw-CLIP relational distillation;
3. SimCLR + DINO-only relational distillation;
4. SpLiCE-CRP v2;
5. SpLiCE-CQT v1;
6. SpLiCE-CRP v2 with shuffled concept codes;
7. SpLiCE-CRP v2 with matched random subspaces;
8. CQT with shuffled pseudo-states;
9. CQT with matched random contrasts;
10. hard-positive CRP v1;
11. Ghanooni spectral regularization.

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

## 12. Interpretation rules

### Claims that are currently supported

- The repository defaults to a from-scratch ResNet18-large SSL encoder and also
  supports from-scratch or ImageNet-pretrained ResNet50 students; ViT-B/32 is
  the frozen OpenCLIP/SpLiCE extractor, not the trainable backbone.
- SpLiCE contains useful nuisance-related concepts: the privileged Waterbirds
  oracle improved frozen-CBM WGA by about 9 points.
- Existing automatic selectors have not found those nuisance concepts reliably.
- Routed augmentation and the existing regularizer did not provide a reliable WGA
  improvement.
- SpLiCE-IU is target-label-assisted and weakly positive only in the frozen CBM
  diagnostic; it is not a demonstrated SSL solution.
- CRP cache/audit/graph construction and relational SSL training are implemented.
- CQT cross-reproduced proposal, rank-one quotient, sparse partial transport,
  DINO local guard, nulls, concept cards, and shared training mode are implemented
  and covered by synthetic tests.

### Claims that are not yet supported

- SpLiCE-CRP v2 improves WGA.
- SpLiCE-CQT improves CRP graph quality or WGA;
- CQT-selected two-state factors are true nuisances rather than target factors;
- full projection is better than coefficient subtraction on the available data;
- DINO consensus reliably preserves target semantics;
- the proposed label-free group score identifies the true nuisance;
- relational distillation transfers the desired invariance into the ResNet.

Every one of these statements is a hypothesis until the corresponding control is
run. New agents must not present the proposal as an achieved result.

## 13. Main risks and falsifiers

### Fundamental identifiability

Without task labels, no method can universally know which visual factor is core and
which is nuisance. CRP v2 introduces an explicit inductive bias: factors whose
removal restores a relation supported by an independent SSL geometry are treated as
nuisance candidates. The claim is empirical, not universal.

CQT narrows the bias to exchangeable concept states with shared non-factor
semantics. This is still not identification: target classes, poses, or viewpoints
can satisfy the same construction.

### Shared bias between CLIP and DINO

Both pretrained models may encode the same background or demographic shortcut. The
DINO verifier therefore reduces risk but cannot prove semantic equivalence.

### Entangled concept subspaces

A concept group can mix target and nuisance directions. Full projection is stronger
than sparse subtraction and can therefore cause greater target damage. The semantic
guard, random-subspace controls, and average-accuracy reporting are essential.

### Graph sparsity or hubness

Waterbirds has only 240 minority training images out of 4,795. There are many
possible pairs but few unique counterfactual donors. CRP's indegree cap and CQT's
transport capacities plus final cap reduce hubness, but low coverage remains a
possible falsifier.

### CQT misspecification

Continuous, one-sided, multi-state, or entangled nuisances violate CQT v1's
two-state rank-one model. Sparse word-neighbour support can also omit a valid pair.
Candidate count and higher-rank variants are audit ablations, not WGA-tuned fixes.

### Distillation without transfer

An auxiliary head can absorb teacher information without changing the backbone.
CRP v2 therefore matches relations on the backbone representation, while SimCLR
keeps its normal projection head.

### Teacher improvement does not imply student improvement

Even a better projected CLIP graph may be incompatible with the inductive bias or
capacity of ResNet18. This requires a matched one-seed pilot before scale-up.

## 14. Implementation status and next work

Implemented now:

1. a shared label-free CLIP/SpLiCE/DINO cache with dataset-order validation;
2. CRP concept grouping, full projection, audit nulls, soft graph, graph-aware
   sampler, relational KL, schedule, and empty-graph fallback;
3. CQT factor proposal, rank-one quotient, implicit word kernel, exact sparse
   partial-transport LP, DINO local guard, null controls, and concept-card JSON;
4. `crp_relational` and `cqt_relational` modes exposed through separate
   `scripts/train_crp.sbatch` and `scripts/train_cqt.sbatch` entry points;
5. W&B-enabled full-training launch with resolved graph artifact/config recorded;
6. unit tests for label isolation, projections, quotient, deterministic graphs,
   row stochasticity, batching, loss, and training-loop integration.

Next work is empirical rather than architectural:

1. run the frozen Waterbirds CRP/CQT audits and post-hoc graph falsifiers;
2. run the required raw-CLIP, DINO-only, random, and shuffled graph controls;
3. run one shortened paired pilot before any full multi-seed submission;
4. expose representative images from CQT sample IDs in a diagnostic-only report;
5. update the paper only after results establish which mode, if either, survives.

Implementation changes should be surgical. Preserve the existing IU, routing, and
regularization modes as baselines instead of rewriting them in place.

## 15. Files and authority

Current important files:

- `PROJECT_GROUND_TRUTH.md` — authoritative scientific and architectural state;
- `README.md` — repository usage and entry points;
- `SPUR_SPLICE_CHAT_SUMMARY.md` — historical hand-off, currently describing the
  older SpLiCE-IU phase;
- `Spur_SpLiCE.tex` — current paper draft, also still centred on SpLiCE-IU;
- `spur_splice.py` — main implemented SSL entry point;
- `scripts/tools/discover_splice_spurious_concepts.py` — existing discovery code;
- `splice/ssl_regularization.py` — existing interventions and distillation code;
- `splice/crp.py` — implemented CRP frozen audit and graph builder;
- `splice/cqt.py` — implemented CQT frozen audit, transport, and concept cards;
- `splice/crp_training.py` — shared CRP/CQT graph validation, sampler, and loss;
- `scripts/train_crp.sbatch` — CRP configuration and shared W&B-enabled runner;
- `scripts/train_cqt.sbatch` — CQT-specific configuration and Slurm entry point;
- `experiments/spurious_eval/` — datasets, models, losses, training, and metrics.

`spur_splice.py --splice_mode crp_relational` and `--splice_mode cqt_relational`
both implement confidence-weighted graph distillation on ResNet backbone features
with graph-aware sampling, a pure-SimCLR start period, a gradual loss ramp, and
empty-graph fallback. Initial single-seed experiments are promising but mixed;
matched multi-seed evidence and mechanism-isolating controls remain pending.

## 16. Literature anchors

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

## 17. Instructions for a new chat or agent

Before proposing, implementing, or describing the project:

1. At the beginning of every new chat, read this file completely before taking
   task-specific action. Do not rely only on a previous chat summary.
2. State whether the requested work concerns legacy methods, CRP v2, or CQT v1.
3. Inspect the current code before claiming that a CRP/CQT component exists.
4. Preserve the strict information boundary: no `y`, `a`, or group labels in CRP
   or CQT discovery/training.
5. Do not tune by WGA and call the method label-free.
6. Treat DINO-only, raw-CLIP, shuffled-code, and random-subspace controls as required,
   not optional decorations.
7. Do not claim positive CRP/CQT results until real matched experiments exist.
8. Keep CRP runnable and behaviorally separate from CQT.
9. Treat an empty selected graph as a valid falsification, not an error to bypass.
10. Surface CQT concept cards in analysis; interpretability is part of the method.
11. If changing a canonical decision, update this file with the reason, rejected
   alternative, date, and consequences for experiments and paper claims.
12. After every material method, architecture, loss, graph, experiment-protocol,
    reporting, or interpretation change, append a dated entry to Section 18 during
    the same task.
13. Do not silently edit old chronology entries. Add a correction or superseding
    entry so that the state of the project on any earlier reporting date remains
    reconstructable.
14. Always distinguish the current research state from the reporting snapshot.
    It is valid for a presentation to report the CRP-only state of 2026-08-30 while
    current research includes later CQT experiments, provided the date is explicit.

## 18. Dated research and implementation chronology

### How to maintain this chronology

This section records scientific and implementation milestones rather than every
mechanical line edit. Each entry should make it possible to reconstruct what method
existed on that date. Record:

- the track: legacy, CRP, CQT, shared trainer, baseline, or reporting;
- the state before the change;
- the change and its motivation;
- consequences for losses, graph semantics, experiments, and supported claims;
- the reporting snapshot affected by the change.

Current research and reported research are deliberately separate timelines. The
code may contain CQT while a dated presentation or manuscript describes only CRP.
Do not back-port later improvements into an earlier narrative without labelling
them as later work.

### 2026-08-07 — Frozen-CLIP sanity checks

**Track:** legacy diagnostics.

Waterbirds concept-ablation sanity checks were added to test whether frozen SpLiCE
concept interventions could change downstream behaviour at all. This was evidence
about available concept information, not yet a CRP training method.

### 2026-08-12 — Discovery rankings and diagnostics

**Track:** legacy concept discovery.

Concept-ranking variants and diagnostic outputs were expanded. The project was
still attempting to identify useful concepts directly; no relational CRP student
objective existed yet.

### 2026-08-18 — Intervention utility and the CRP v2 frozen stage

**Track:** legacy IU, CRP graph construction, diagnostic baselines.

The label-assisted intervention-utility track was implemented, together with the
first CRP v2 feature cache, frozen audit, soft graph construction, post-hoc graph
evaluation, raw-CLIP/DINO-style baseline graphs, and a privileged relational oracle.
CRP edge evidence already had pair confidence, but the standalone ResNet relational
trainer had not yet been implemented. Therefore there was no historical unweighted
CRP training loss before this date; CRP was still a frozen graph/audit mechanism.

### 2026-08-24 — First implemented CRP relational trainer

**Track:** CRP and shared trainer.

CRP relational SSL training was introduced with the graph-aware batch sampler,
backbone-level student similarities, empty-support handling, and validation of the
teacher graph. From its first implementation, the relational loss was confidence
weighted:

```text
L_CRP = mean over supported anchors i of
        q_i * KL(p_T(.|i) || p_S(.|i))

L_total = L_SimCLR + lambda(t) * L_CRP
```

The original anchor confidence was relative to the strongest retained edge in the
same graph:

```text
q_i_old = max_j edge_confidence(i,j)
          / max_k max_j edge_confidence(k,j)
```

The schedule had a SimCLR-only start and a linear ramp, after which `lambda(t)`
stayed constant. Graph-linked donors were still ordinary SimCLR negatives. This is
the earliest implemented CRP loss and is the correct starting point for historical
method descriptions.

### 2026-08-25 — Unified training entry point

**Track:** experiment infrastructure.

Older Waterbirds launch scripts were consolidated into `scripts/train.sbatch`.
Run naming was clarified on the same day. This changed experiment operation, not
the CRP mathematical objective.

### 2026-08-27 — Automatic frozen graph preparation

**Track:** CRP experiment infrastructure.

The CRP frozen-audit configuration was expanded and graph preparation became part
of the normal training launch. A missing or incompatible graph could be rebuilt
before SSL training. This reduced manual experiment drift without changing the
meaning of the relational loss.

### 2026-08-28 — Cluster resource and checkpoint workflow

**Track:** experiment infrastructure.

Memory allocation, output paths, feature-cache validation, and checkpoint cleanup
options were revised. These were operational changes and should not be presented as
method innovations.

### 2026-08-30 — CRP reporting snapshot and CQT integration

**Track:** CRP reporting; CQT research implementation.

The CRP manuscript snapshot described the confidence-weighted relational loss with
`q_i`, graph-aware sampling, a start at epoch 10, and a ten-epoch ramp to a constant
relational weight. Its strongest recorded single-seed CRP configuration reported
53.49% average accuracy and 48.95% WGA against approximately 48.94% average accuracy
and 40.66% WGA for the matched SimCLR baseline. These numbers establish a promising
pilot, not seed stability.

The same date also introduced the CQT implementation and shared CRP/CQT graph
validation. CQT was an experimental research extension, not part of the CRP-only
reporting snapshot. The manuscript saved on this date already contains `q_i`; the
later September work did not introduce confidence weighting itself.

### 2026-08-31 — CQT operations, empty-graph fallback, and model scale check

**Track:** CQT, shared trainer, and baselines.

An empty CRP/CQT graph was made an explicit SimCLR fallback instead of a failed
training job. CQT graph configuration was added to artifact paths and run labels,
and a dedicated CQT investigation workflow was added. Support for an
ImageNet-initialized ResNet50 student was added as an exploratory scale/pretraining
check. CoBalT reproduction code was also imported as a comparison track.

The ResNet50 experiment must not be cited as evidence for CQT when its selected
graph is empty: in that case the run is operationally a SimCLR run from a different
initialization and architecture.

### 2026-09-01 — Absolute confidence, null-margin calibration, graph positives,
and concept reports

**Track:** CRP, CQT, shared trainer, interpretability, and audit tooling.

Before this change, every non-empty graph normalized its strongest anchor to
confidence one. That discarded the absolute strength of graph evidence. Anchor
confidence now remains on its absolute bounded scale:

```text
q_i_current = clamp(max_j calibrated_edge_confidence(i,j), 0, 1)
```

For CRP, selected group evidence is additionally scaled by how far its group score
exceeds its label-free null threshold. A group that barely passes the threshold now
contributes less than a group with a large null margin. The multiplication by `q_i`
inside the KL mean did not change; the construction and calibration of `q_i` did.

Graph-linked samples were changed from simultaneous ordinary SimCLR negatives into
weighted additional contrastive positives. This addresses the earlier conflict in
which relational KL pulled a donor toward an anchor while NT-Xent pushed the same
pair apart. The total objective is therefore now more precisely described as:

```text
L_total(t) = L_graph-positive-SimCLR
             + lambda(t) * mean_i[q_i * KL(p_T(.|i) || p_S(.|i))]
```

An optional late linear decay of `lambda(t)` was added so graph supervision can
shape the representation early and return control to SimCLR later. Graph records
were moved to complete JSON artifacts. Post-hoc Waterbirds oracle graph audit and
CRP concept-usage reports were added. A concept report distinguishes concepts whose
directions were projected during teacher construction from concepts proven absent
from the final ResNet; the latter claim is not justified by graph metadata alone.

CRP precision defaults were also tightened: a higher activation-difference gate,
a larger minimum intervention gain, fewer neighbours per anchor, and more null
trials. These defaults are new experimental candidates, not established optimal
values.

### 2026-09-01 — Separate CRP and CQT cluster entry points

**Track:** experiment infrastructure and reporting ergonomics.

Before this change, `scripts/train.sbatch` contained CRP, CQT, shared-training, and
legacy settings in one long configuration block. Switching methods required finding
and editing `MODE` while unrelated method parameters remained interleaved.

The old entry point was renamed to `scripts/train_crp.sbatch` and reorganized so
dataset, model, main CRP loss controls, and CRP graph-audit parameters are grouped
at the top. `scripts/train_cqt.sbatch` was added with dataset, model, main shared
loss controls, and all CQT proposal/quotient/transport parameters at the top. The
CQT file supplies those values to the same execution body, preserving graph
preparation, evaluation, checkpoint handling, and mandatory W&B tracking across
both methods. Direct submission commands are now method-specific, so changing one
method's common experimental knobs does not require navigating the other method's
configuration.

Both Slurm entry points now request eight hours instead of sixteen. This is a
cluster scheduling/resource request only; it does not change epoch count, optimizer,
loss, graph construction, or the scientific interpretation of a run.

### 2026-09-01 — First results with graph positives, decay, and absolute confidence

**Track:** CRP/CQT experimental interpretation and next-run selection.

The matched seed-0 SimCLR reference finishes at 48.94% average accuracy and 40.66%
WGA when measured by the average of the final ten linear-probe results.

The revised CQT configuration (`q=0.04`, DINO damage quantile `0.60`, relational
weight `0.01`, student temperature `0.25`, graph positives enabled, decay from
epoch 200 to 350) finishes at 54.74% average accuracy and 52.27% WGA. Its graph is
very sparse: 72 supported anchors and 79 edges, or 1.50% anchor coverage. The
weighted relational loss is only about `3e-7` before decay and zero afterward.
Consequently this run establishes a promising pipeline result but does not show
that relational KL caused the gain. Graph-aware batches, sparse extra positives,
and seed variation remain competing explanations.

The revised CRP precision configuration with the same training controls finishes
at 48.32% average accuracy and 42.81% WGA. Its graph covers 4,095 anchors with
12,755 edges, or 85.40% coverage. Its weighted relational loss is about `3.1e-4`
at weight `0.01`, versus approximately `0.059--0.061` throughout the earlier best
CRP run. The earlier best CRP configuration, before absolute confidence, graph
positives, and decay, finished at 53.49% average accuracy and 48.95% WGA.

**Interpretation:** the CQT result must first be replicated without changing its
hyperparameters. CRP's old weight sweep is no longer on the same effective scale:
the confidence recalibration made `0.01--0.05` nearly remove the KL contribution.
The next CRP experiments should disable graph positives for the dense graph,
disable late decay, retain temperature `0.25`, and sweep larger weights chosen by
the observed loss ratio rather than by downstream WGA. Initial safe candidates are
`0.5` and `1.0`; `2.0` approximately targets the old weighted-loss magnitude and is
a higher-risk follow-up. Selection still requires matched seeds and must not be
presented as label-free hyperparameter selection if WGA is used to choose it.

### 2026-09-01 — Concrete concept-ablation report and KL-only training ablation

**Track:** CRP and CQT diagnostics; shared trainer; experiment protocol.

Before this change, graph artifacts recorded group-level intervention statistics
but there was no single human-readable file showing the actual Waterbirds images,
their hidden diagnostic annotations, and pairwise cosine similarity before and
after every selected intervention. The shared trainer also fixed the SimCLR term at
unit weight, so it could not isolate whether relational KL can train a useful
backbone without contrastive supervision.

A diagnostic-only Slurm entry point now builds or reuses the normal frozen cache
and teacher graph, chooses two explicitly labelled Waterbirds pairs, and writes one
self-contained HTML. One pair holds the waterbird target fixed while changing the
background; the other holds the land background fixed while changing the target.
For every selected CRP group it applies the canonical full-subspace projection; in
CQT mode it applies the selected factor's canonical rank-one quotient. The report
shows group words, graph evidence, cosine before/after, and `delta = after - before`.
Hidden annotations affect only post-hoc pair selection and rendering, never graph
discovery or training. Example selection maximizes absolute displayed cosine change
within each required metadata constraint and is explicitly labelled illustrative,
so it cannot support aggregate or causal claims by itself.

The trainer now exposes an explicit non-negative SimCLR loss weight, defaulting to
one so canonical CRP/CQT behaviour is unchanged. A separate W&B-enabled KL-only
Slurm entry point sets that weight to zero, activates relational KL from epoch one,
disables its late decay and graph-positive SimCLR modification, and retains the
standard downstream linear classifier with average and worst-group accuracy. A
non-empty teacher graph is required because an empty graph would leave this
ablation without any training objective. This is a shared-trainer ablation for both
CRP and CQT; it does not replace the canonical combined objective, change graph
semantics, or update the CRP-only 2026-08-30 reporting snapshot. Its result can test
whether KL alone is sufficient in a particular graph/model setting, but it cannot
by itself establish that SimCLR or the ResNet architecture is generally unnecessary.

### 2026-09-01 — Optional DINO guard and label-free CoBalT balance check

**Track:** CRP and CQT graph construction; baseline/ablation protocol; experiment
infrastructure.

Before this change, every CRP/CQT frozen cache and graph audit required DINO
features. DINO neighbour support was mandatory for CRP relations and DINO local
damage was a mandatory CQT factor gate. Concept frequency and coactivation used
the empirical training distribution directly, and the independent CoBalT
reproduction was not connected to CRP/CQT group construction.

Both audit configurations and their Slurm entry points now expose independent
`use_dino` and `cobalt` booleans. With `use_dino=false`, cache construction does
not import or run DINO. CRP retains reciprocal projected-neighbour support and
delta-based intervention confidence but removes DINO support/similarity from its
gate; CQT omits the DINO local-damage calculation and gate. The canonical default
remains `use_dino=true`. No-DINO caches and teacher graphs have distinct paths,
and run names, graph configuration, and W&B metadata record the ablation.

With `cobalt=true`, a separately trained, fixed CoBalT Stage-1 artifact is aligned
to the frozen cache by source sample index. Its discovered memberships produce
mean-one sample weights proportional to summed inverse concept frequency. These
weights affect only SpLiCE concept-frequency filtering and coactivation used to
form candidate groups. No target, spurious-attribute, or group annotation crosses
the graph-discovery boundary. A dedicated W&B-enabled Slurm job performs the
label-free CoBalT discovery and membership extraction. The supervised
class-balancing/classifier stage from CoBalT is deliberately not inserted into
CRP/CQT.

These switches were added to test whether DINO contributes enough validation to
justify the extra frozen model and whether an independently discovered concept
partition stabilizes group formation under imbalance. Results with either switch
must be identified as ablations. In particular, `use_dino=false` has a weaker
semantic-preservation check, while `cobalt=true` is a CoBalT-inspired label-free
grouping check rather than a reproduction of the complete supervised CoBalT
training protocol. Neither changes the shared SSL trainer or the current
canonical CRP/CQT defaults.

### 2026-09-01 — Seed-0 tuning protocol, bounded concept view, and relational diagnostics

**Track:** CRP/CQT frozen audit, shared trainer diagnostics, experiment protocol,
and reporting.

Before this change, the cluster entry points hard-coded most frozen graph
parameters, so controlled Slurm arrays could not override them. CRP allowed every
null-passing group into the graph and `min_group_size=1` produced a diagnostic
artifact with 139 audited groups, 119 selected groups, and mostly singleton words.
The HTML could display all of them and always contained both diagnostic pairs.
Training logged only the already scheduled relational loss, which hid whether a
small value came from sparse batch support, low anchor confidence, or an already
small unweighted KL.

CRP now supports an optional post-gate, label-free `max_selected_groups` cap ranked
by null-excess score. CRP and CQT graph settings are environment-overridable and
variant-tagged paths prevent graph sweeps from overwriting one another. Separate
Slurm arrays were added for frozen CRP grouping and CQT graph audits and for seed-0
full-training sweeps. The training arrays explicitly separate graph-aware sampling,
graph positives, relational-weight scale, and student temperature. Full SSL jobs
retain mandatory W&B tracking; frozen preparation jobs remain diagnostics.

The concept-ablation HTML now supports one requested pair or both pairs and a
display-only intervention limit. Its default Slurm view shows the
waterbird-on-land/landbird-on-land pair and the twelve interventions most used by
retained teacher edges, with an explicit statement that display truncation does not
alter the graph. The relational trainer now logs scheduled weight, supported-anchor
fraction, mean supported-anchor confidence, unweighted KL, and confidence-weighted
KL. These additions do not change the KL formula.

Consequences: graph grouping must be chosen from words, size distribution,
coverage, null margin, and hubness before downstream WGA is inspected. The new
seed-0 arrays are mechanism-finding pilots, not evidence of seed stability. Existing
CRP/CQT results and the 2026-08-30 CRP-only reporting snapshot are unchanged; no new
performance claim follows from this infrastructure and diagnostic update.

### 2026-09-02 — Seed-0 CRP loss-scale sweep, first four runs

**Track:** CRP experimental interpretation.

Before these runs, absolute anchor confidence had reduced the effective relational
term enough that the earlier `0.01--0.05` weight sweep was no longer informative;
weights `0.5`, `1.0`, and then `2.0` were proposed from the observed loss scale.
The first completed runs on the bounded `g2_t070_c020_k12` graph disabled graph
positives and late decay and used student temperature `0.25`. At seed 0, weight
`1.0` finished at 52.45% final-ten average accuracy and 47.17% final-ten WGA,
while weight `2.0` finished at 54.15% and 50.54%, respectively. The shared graph
covered 3,060 anchors (63.82%) with 6,268 edges; mean supported-anchor confidence
was about 0.00232 and the confidence-weighted KL was about 0.012 before application
of the scheduled weight. Thus doubling the weight materially increased the active
relational contribution and improved both reported downstream metrics in this
single-seed comparison.

Two otherwise matched weight-`1.0` temperature checks (`0.10` and `0.50`) were
still running at roughly 50 of 500 epochs when inspected. Their early probe values
are not final-ten estimates and do not yet support a temperature ranking. None of
these four runs establishes seed stability, comparison with required non-concept
controls, or a causal concept-removal effect. The weight-`2.0` result is a stronger
seed-0 CRP pilot than the earlier recorded CRP pilot, but it remains tuning evidence
and must not be promoted to a label-free general claim based on WGA selection.

### 2026-09-02 — Paper rewritten for the current CRP architecture

**Track:** CRP reporting.

Before this change, `Spur_SpLiCE.tex` mixed an earlier CRP reporting snapshot with
newer implementation details and an older seed-0 sweep. It did not present absolute
null-margin confidence, optional graph-positive SimCLR, late relational decay,
empty-graph fallback, or the newer relational diagnostics as one coherent system.

The paper was rewritten as a current-state CRP-only description. It now follows the
implemented pipeline from frozen cache, concept grouping, full centered-CLIP
projection, reciprocal and DINO-gated relations, random/shuffled null calibration,
absolute anchor confidence and sparse graph construction through graph-aware
sampling, backbone relational KL, optional weighted graph positives, and the
start/ramp/decay schedule. Its empirical section reports the completed bounded-graph
seed-0 weight-`1.0` and weight-`2.0` runs and explicitly separates current
entry-point defaults from the mechanism-isolation configuration used by those runs.
This reporting update changes no method, loss, graph, or experiment protocol and
does not retroactively alter the dated 2026-08-30 reporting snapshot.

### 2026-09-02 — Configuration-safe visual CRP graph audit

**Track:** CRP frozen audit and reporting diagnostics.

Before this change, CRP graph separation depended on a manually supplied variant
name. Reusing a name with different frozen-audit parameters could rebuild and
overwrite the previous graph, while every seed-level concept-ablation report used
one shared diagnostics filename. The report selected the image pair with maximum
absolute displayed intervention change and applied every shown group to that pair,
even when the pair was not a retained teacher edge. It therefore illustrated a
best-looking intervention but did not provide a representative or complete view of
the graph used for training.

CRP graph paths now include a human-readable hierarchy containing every resolved
graph-construction parameter, the DINO and CoBalT switches, and the seed. An
optional variant name is only an additional readable parent label; distinct
resolved configurations still receive distinct paths. Each HTML audit is stored
beside its exact teacher graph. The grouping sweep now emits both the numeric audit
summary and this HTML automatically.

The CRP HTML now always displays four post-hoc Waterbirds relation types: same
target/opposite background, opposite target/same background, same target/same
background, and opposite target/opposite background. Each example is selected by
the pair whose raw centered-CLIP cosine is nearest the median of all eligible pairs
of that type; intervention results do not affect selection. The report adds full
resolved configuration, graph degree/confidence statistics, hidden-label edge
diagnostics, richer word-level group evidence, and representative actual retained
edges selected around median edge confidence. Hidden annotations remain reporting
only and do not affect graph discovery or selection.

Consequences: frozen graph sweeps can be compared without silent artifact loss,
and the report distinguishes generic intervention behavior from relations that
actually supervise the student. These changes affect CRP audit storage and
reporting only; concept selection, graph mathematics, SSL losses, and existing
experimental results are unchanged.

### 2026-09-02 — Open Images V7 vocabulary source and cleanup

**Track:** shared SpLiCE vocabulary and experiment infrastructure.

Before this change, the repository exposed only the bundled LAION vocabulary,
whose 37,445 entries contain many malformed fragments, social tags, names, and
other low-value strings. The Open Images V7 download page provides class-label
metadata separately from the image pixels, so the project can obtain a more
object-oriented candidate vocabulary without downloading the dataset images.

The loader now supports `openimages_v7`. It downloads the official class-description
CSV on demand, keeps the display-name column, normalizes Unicode and whitespace,
deduplicates case-insensitively, writes only `openimages_v7.txt`, and removes the
temporary source CSV. A standalone downloader and a local parser test were added;
the generated repository vocabulary contains 20,931 non-empty unique labels.
LAION remains available for reproducibility, and Open Images is selectable rather
than silently replacing historical defaults. The current CRP/CQT Slurm launchers
also accept `SPLICE_VOCAB` and `SPLICE_VOCAB_SIZE` environment overrides so a new
vocabulary can be selected without editing the runner.

Consequences: a cache built with `openimages_v7` is not comparable with a cache
built from LAION and must be rebuilt before CRP or CQT graph construction. The
dictionary change does not alter CRP/CQT graph mathematics, the shared SSL loss,
or any existing result. New Open Images runs require a fresh frozen audit and the
same label-free controls; no quality or WGA claim follows from the vocabulary
swap alone. The 2026-08-30 CRP reporting snapshot is unchanged.

### 2026-09-02 — English graph reports and expanded Open Images audit sweep

**Track:** CRP frozen-audit experiment infrastructure and reporting only.

Before this change, the concept-ablation HTML renderer mixed Russian and English,
and the graph-preparation array covered six configurations. The renderer now emits
English text throughout, including pair descriptions, table headings, warnings,
and methodological notes; its test also rejects Cyrillic output. The preparation
array now covers twelve label-free configurations, adding group sizes 1--4,
lower text/coactivation thresholds, and selected-group caps of 8--16. The Slurm
launcher accepts an externally built Open Images cache and preserves the cache
rebuild override as an environment setting.

The reason is to make reports shareable with English-speaking reviewers and to
probe whether the Open Images vocabulary yields usable multi-word teacher groups
across a broader operating range. Consequences: one cache-build command should be
run before the array, then the array can reuse the frozen cache without concurrent
rebuilds. The sweep uses an explicit Open Images graph-variant prefix so graph
artifacts cannot be confused with historical LAION paths. The new configurations
and English reporting do not change CRP graph mathematics, the SSL loss, or
historical results. They create a new Open Images audit surface; graph quality and
downstream WGA remain unestablished until the resulting JSON/HTML artifacts and
matched training runs are reviewed.

### 2026-09-02 — Dedicated Open Images cache job and legacy cache-script removal

**Track:** CRP frozen-cache experiment infrastructure.

Before this change, Open Images cache preparation was documented as a direct
terminal Python command, while the repository also contained a legacy
`SpLiCE_CRP_v2_cache_features.sbatch` with LAION-only defaults and a twelve-hour
limit. The legacy job was removed and replaced by `cache_openimages_crp.sbatch`,
which requests two hours on one V100, activates the cluster environment itself,
uses the official Open Images V7 vocabulary, enables DINO by default, and writes
to a distinct `waterbirds_train_features_oi_v7.pt` path. `train_crp.sbatch` now
accepts an explicit `CRP_CACHE_PATH`, allowing the graph sweep and later training
to consume this cache without overwriting the historical LAION cache.

The reason is operational: this workflow must be launchable entirely through
Slurm, without requiring Python commands in an interactive cluster shell. The
separate output path and the `oi_v7` graph-variant prefix prevent accidental
mixing of vocabulary-specific frozen artifacts. CRP graph mathematics, SSL loss,
and historical results are unchanged; this affects preparation infrastructure
only and does not establish graph quality or downstream WGA.

### Reporting snapshots currently available

- **CRP-only report dated 2026-08-30:** use the CRP formulation and results recorded
  in the 2026-08-30 entry. Do not include later graph-positive, decay, absolute
  confidence, CQT, or concept-report changes as if they existed in that snapshot.
- **Current research state dated 2026-09-01:** CRP and CQT share the revised trainer;
  CQT is experimentally ahead of the CRP-only narrative, while neither method has
  yet established multi-seed stability or a causal concept-removal claim.
