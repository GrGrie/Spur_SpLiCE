# Spur_SpLiCE — project ground truth

**Purpose:** compact entry point for a new agent. Read this section before changing,
running, or interpreting the project. The full dated chronology is in
`Project History.md`; current research state and historical reporting snapshots
are deliberately separate.

**Last updated:** 2026-09-03

**Current tracks:** SpLiCE-CRP v3 (active internal no-DINO/CoBalT protocol),
SpLiCE-CRP v2 (historical/compatibility) and SpLiCE-CQT v1 (experimental)
**Current evidence:** implementation and initial Waterbirds single-seed pilots exist.
Multi-seed stability, required controls, and a causal concept-removal claim are not
established.

## 1. Project in one minute

Spur_SpLiCE asks whether a frozen, language-readable concept model can help a
separately trained self-supervised image encoder become less dependent on spurious
visual correlations. The intended contribution is a label-free counterfactual
teacher graph: intervene on frozen CLIP concepts, find relations that become more
semantically coherent, and distil those relations into a standalone SimCLR ResNet.

The target outcome is improved worst-group accuracy (WGA) on Waterbirds, CelebA, and
SpurCIFAR10 without target, spurious-attribute, or group labels during discovery or
SSL training. Frozen OpenCLIP, SpLiCE, and DINO are teachers/auditors only; the
evaluated representation is the trainable ResNet backbone.

A successful claim requires matched SimCLR and same-setting controls, preserved
average accuracy, paired multi-seed replication, and evidence against non-concept
explanations. A gain over SimCLR alone is insufficient.

## 2. Information boundary

During concept discovery and SSL training use only:

- unlabeled training images and augmentations;
- frozen OpenCLIP ViT-B/32 embeddings;
- frozen SpLiCE dictionary, sparse codes, and vocabulary;
- frozen DINOv3 geometry (canonical default; optional ablation);
- sample IDs and statistics computed from these unlabeled representations;
- the trainable ResNet and SimCLR objective.

Never use during discovery or SSL training:

- target labels y, spurious attributes a, or group IDs g=(y,a);
- manually supplied nuisance descriptions or dataset-specific nuisance word lists;
- hidden-metadata balancing;
- WGA/test data for selecting concepts, graphs, thresholds, or checkpoints.

Labels and group metadata are allowed only for linear evaluation, WGA/group-wise
accuracy, post-hoc graph diagnostics, and explicitly labelled oracle methods. A
cache or graph containing annotation keys is invalid.

## 3. Shared architecture

~~~text
unlabeled images
   ├─ OpenCLIP ViT-B/32 → centered CLIP vectors
   ├─ SpLiCE → sparse codes + readable dictionary
   └─ DINOv3 → independent semantic geometry
                    │
             CRP or CQT frozen audit
                    │
              sparse teacher graph
                    │
        ResNet18-large + SimCLR + graph KL
                    │
              backbone → linear probe/WGA
~~~

The frozen cache contains normalized CLIP embeddings, centered CLIP vectors using
the SpLiCE image mean, sparse SpLiCE codes, dictionary/vocabulary, and by default
normalized DINO embeddings. The complete Open Images V7 label vocabulary is the
default dictionary; LAION remains an explicit historical/ablation option. The cache
is indexed by sample ID, validates aligned lengths
and finite values, and records model/configuration provenance. CQT reuses it.

The default student is a from-scratch large-input ResNet18. ResNet50 from scratch
and ImageNet-initialized ResNet50 are exploratory scale checks. The shared trainer
uses backbone similarities and a row-stochastic graph:

~~~text
L_total(t) = L_graph-positive-SimCLR
             + lambda(t) * mean_i[q_i * KL(p_T(.|i) || p_S(.|i))]
~~~

p_T is the graph distribution, p_S uses normalized student backbone features, and
q_i is frozen absolute anchor confidence. The trainer supports SimCLR warm-up,
relational ramp, optional decay, graph-aware sampling, bounded graph support, and
optional graph-linked SimCLR positives. Unsupported anchors contribute zero.
There is no student memory bank or permanent hard positive. An empty graph falls
back to SimCLR, except in the explicitly non-empty KL-only ablation.

## 4. CRP v2 — historical canonical method

CRP means Concept-Projected Relational Pretraining. It is the canonical method and
the required baseline for CQT.

1. **Group concepts:** combine distributed semantic concepts using text-direction
   similarity, unlabeled sparse-code coactivation, and lexical deduplication.
2. **Project all of centered CLIP:** for group G with orthonormal dictionary basis
   Q_G, use u_i^(-G) = normalize((I - Q_G Q_G^T)u_i). This is canonical;
   sparse subtraction, forbidden-group sparse refitting, and random subspaces are
   intervention ablations.
3. **Find relations:** keep positive intervention-gain pairs with an unlabeled
   activation difference, reciprocal projected-CLIP neighbourhoods, and, when
   enabled, raw-DINO neighbour support.
4. **Select groups without labels:** combine robust positive gain, coverage,
   independent semantic agreement, activation/gain alignment, and hubness; compare
   against matched random-subspace and shuffled-code nulls. Null/stability gates
   precede the optional max_selected_groups cap. Selecting no group is valid.
5. **Build a graph:** retain strongest evidence per directed pair, limit top-k,
   cap global indegree, and row-normalize. Store edges, group evidence, gains,
   confidence, degree statistics, complete config, sample IDs, and provenance.

CRP's inductive bias is that removing a factor can restore relations supported by
independent semantic geometry. It may select a class factor as well as a nuisance;
this is not a general identification guarantee.

## 4.1 CRPv3 — active internal no-DINO/CoBalT protocol

CRPv3 keeps the CRP v2 projection and null-test structure, but the active protocol
for the CoBalT/no-DINO track adds four label-free controls:

1. The final graph uses fixed `top_k=3` and absolute `max_indegree=10`; the old
   relative indegree rule remains only as a compatibility fallback.
2. With `use_dino=false`, an optional residual SpLiCE gate compares the remaining
   sparse concepts after excluding the projected group. It is enabled by default
   and controlled by `use_residual_splice_gate`.
3. Accepted group relations are checked on deterministic cross-fold subsets and
   require configured edge persistence. This is an out-of-sample relation check;
   it is not a downstream label-based selector.
4. CoBalT concept artifacts may carry spatial slot-separation confidence. When
   available, confidence downweights uncertain memberships in the concept-balance
   weights; older artifacts fall back explicitly to neutral confidence.

CRPv3 emits `splice_crp_v3_teacher_graph` and remains separate from CQT. This is
an internal research protocol, not evidence that the new gates improve WGA or
identify a true nuisance factor. The exact graph configuration and fold results
must be reported with each audit.

## 5. CQT v1 — experimental extension

CQT means Concept Quotient Transport. It reuses the cache, graph interface, and
shared trainer but changes graph construction only; it does not replace CRP.

Hypothesis: two mutually exclusive concept states may differ in one nuisance-like
direction while sharing non-factor semantics. Remove only that state contrast and
transport mass between compatible samples. Target classes, poses, or viewpoints
can satisfy the same assumption, so CQT does not identify true nuisances.

1. Reuse CRP groups and propose two cross-reproduced exclusive states A and B.
2. Form q = normalize(m_A - m_B) from the two state dictionary means.
3. Apply the rank-one quotient r_i^F = normalize((I - qq^T)u_i).
4. Match quotient geometry with the similarity of remaining readable SpLiCE
   semantics, excluding factor coordinates.
5. Solve sparse capacity-constrained partial transport with unit source/destination
   capacities. Partial mass avoids poor pairs and donor hubs.
6. With DINO enabled, reject transport that damages within-state local DINO
   geometry beyond the configured candidate quantile.
7. Gate on cross-fold support/context, pseudo-state predictability, quotient
   efficacy, positive gain above random-contrast and shuffled-state nulls, word
   preservation, local safety, and coverage. Emit concept cards with words, metrics,
   gates, nulls, and sample-ID pairs.

CQT is a two-state rank-one model and can miss continuous, one-sided, multi-state,
or entangled nuisances. Its artifact type is splice_cqt_v1_teacher_graph.

| Component | CRP v2 | CQT v1 |
|---|---|---|
| Candidate | one distributed group | two exclusive group states |
| Intervention | full group-span projection | rank-one state quotient |
| Graph | reciprocal projected-CLIP neighbours | partial concept-preserving transport |
| DINO | pair-neighbour support | local-geometry damage guard |
| Hub control | final indegree cap | transport capacities plus final cap |
| Null | random subspaces/shuffled codes | random contrasts/shuffled states |

## 6. Status, claims, and results

Implemented: label-isolated cache; CRP and CQT audits/graphs; shared graph sampler,
backbone KL, schedules, graph positives, empty-graph fallback; separate Slurm
entry points; W&B-enabled full training; concept cards and tests.

Supported now:

- SpLiCE contains nuisance-related information, but automatic label-free selection
  has not reliably recovered the privileged Waterbirds concepts.
- Legacy routed augmentation and correlation regularization are baselines/negative
  tracks.
- CRP/CQT construction and relational training are implemented.
- Existing single-seed runs are pipeline evidence only.

Not supported: that CRP/CQT improves WGA; that CQT finds true nuisances; that full
projection wins; that DINO preserves task semantics; or that graph KL transfers
the desired invariance. These require matched controls and multiple seeds.

Recorded pilots, all non-final:

- SimCLR seed 0: 48.94% average accuracy / 40.66% WGA.
- Earlier CRP pilot: 53.49% / 48.95%.
- CQT pilot: 54.74% / 52.27%, but only 72 supported anchors and nearly inactive KL.
- Dense CRP pilot: 48.32% / 42.81%.
- Bounded CRP loss-scale check: weight 1.0 → 52.45% / 47.17%; weight 2.0 →
  54.15% / 50.54%; single seed and tuning evidence only.

The CQT gain may be due to graph-aware batches, sparse extra positives, or seed
variation; the CRP weight result is tuning evidence. Required controls remain
pending.

The 2026-08-30 CRP-only report is a historical snapshot and must not be described
with later CQT, confidence, graph-positive, or decay changes. The current state
contains both CRP and CQT.

## 7. Experiment protocol

Before expensive SSL, audit raw CLIP, sparse subtraction, full projection,
forbidden-group refitting, random projection, and shuffled concepts. Inspect
neighbour turnover/Jaccard, reciprocal coverage, DINO agreement, hubness, null
excess, and for CQT state accuracy/efficacy, transport mass, word preservation,
and local DINO damage. Hidden-label diagnostics are falsifiers, not selectors.

Required pilot controls: SimCLR; raw-CLIP relational distillation; DINO-only
relational distillation; CRP; CQT; shuffled concepts/states; matched random
subspaces/contrasts; hard-positive CRP v1; and Ghanooni spectral regularization.
Only then promote to paired seeds on Waterbirds, CelebA, and SpurCIFAR10 with
matched initialization, augmentations, and evaluation. Report mean, standard
deviation, average accuracy, WGA, and per-group accuracy.

Every full SSL script must enable W&B and log project/entity, run name, group, tags,
resolved configuration, runtime versions, per-epoch SSL metrics, and evaluation
metrics. Diagnostic cache/graph/audit/report jobs do not need W&B.

## 8. Repository map and commands

| Path | Role |
|---|---|
| spur_splice.py | main SSL entry point, modes, W&B, evaluation |
| splice/crp.py | CRP cache validation, grouping, audit, graph |
| splice/cqt.py | CQT proposals, quotient, transport, concept cards |
| splice/crp_training.py | shared graph validation, sampler, KL, schedule |
| splice/ssl_regularization.py | legacy interventions and baselines |
| experiments/spurious_eval/ | datasets, ResNets, SSL loop, probes, metrics |
| scripts/train_crp.sbatch | CRP Slurm runner |
| scripts/train_ssl_graph_shortlist.sbatch | matched 100-epoch SimCLR/CRP graph screen |
| scripts/train_cqt.sbatch | CQT Slurm runner |
| scripts/prepare_crp_group_sweep.sbatch | CRP frozen graph sweep |
| scripts/prepare_crp_semantic_family_sweep.sbatch | seed-0 semantic-family graph sweep |
| scripts/prepare_crp_splice_only_sweep.sbatch | no-DINO/no-CoBalT SpLiCE-only graph controls |
| scripts/prepare_cqt_graph_sweep.sbatch | CQT frozen graph sweep |
| scripts/*.conf | editable Slurm configurations for training and preparation |
| scripts/load_config.sh | shared config loader |
| scripts/tools/*.py | executable cache/audit/report modules called by Slurm jobs |
| scripts/concept_ablation_examples.sbatch | post-hoc diagnostic HTML |
| scripts/tools/cache_crp_features.py | frozen cache construction |
| scripts/cache_openimages_crp.sbatch | Open Images cache job |
| CoBalT/scripts/prepare_concepts.sbatch | label-free CoBalT concept artifact |
| data/vocab/ | selectable vocabularies |
| README.md | usage/setup |
| Spur_SpLiCE.tex | paper draft; may lag current state |

~~~bash
sbatch scripts/train_crp.sbatch
sbatch scripts/train_cqt.sbatch
~~~

CRP artifacts are under outputs/crp/<dataset>/ and CQT artifacts/concept cards under
outputs/cqt/<dataset>/. Reuse a graph only with its exact matching cache and
resolved audit configuration. Open Images and LAION caches are not interchangeable.

## 9. Legacy tracks

Keep these runnable as comparisons, not as the canonical method: routed augmentation;
correlation regularization; label-assisted SpLiCE-IU/class neutralization; CRP v1
hard positives; crop-diffuseness; worst-cell/error ranking; pseudo-environment
clustering; sparse subtraction; remaining-concepts-only reconstruction; SAE/DIAL
editing; and combined CRP-plus-spectral variants before separate effects are known.

## 10. Rules for the next agent

1. Read this file fully and inspect relevant code before making claims.
2. Identify the track: legacy, CRP, CQT, shared trainer, baseline, or reporting.
3. Preserve the label boundary; never use annotations to choose concepts, graphs,
   thresholds, checkpoints, or SSL batches.
4. Treat raw-CLIP, DINO-only, shuffled, random, and spectral controls as required.
5. Treat an empty graph as a valid falsification.
6. Keep CRP and CQT behaviorally separate; expose CQT concept cards.
7. Do not call a single-seed gain stable or causal.
8. Keep W&B enabled for real full SSL runs; disable only for explicit smoke/debug.
9. After a material method, loss, graph, protocol, reporting, or interpretation
   change, append a dated chronology entry below with prior state, change, reason,
   consequences, track, and affected reporting snapshot. Do not rewrite history.

## 11. Immediate next work

1. Finish Waterbirds CRP/CQT frozen audits and post-hoc graph falsifiers.
2. Run raw-CLIP, DINO-only, random, shuffled, and spectral controls.
3. Run one shortened paired pilot before multi-seed submission.
4. Inspect CQT concept cards and representative sample-ID reports.
5. Update the paper only after controls determine which mode, if either, survives.

## 12. Project history

The complete dated research and implementation chronology is in
[Project History.md](Project%20History.md). Do not read it by default. Open it
when reconstructing an earlier method, experiment, reporting snapshot, or
implementation decision.

The history file is authoritative for past states; this file is authoritative for
the current state. Never back-port later decisions into an earlier snapshot without
labelling them as later work. When a material method, loss, graph, protocol,
reporting, or interpretation change is made, append the new dated entry to
Project History.md in the same task.

## 13. 2026-09-02 — CRPv3 implementation chronology

- **State before:** CRP v2 used CoBalT only to reweight SpLiCE concept frequency
  and coactivation grouping. No-DINO relation scoring disabled the independent
  semantic gate, and the final indegree cap was derived from candidate density.
  CoBalT concept artifacts stored hard memberships without assignment confidence.
- **Change:** introduced the CRPv3 no-DINO/CoBalT protocol with fixed graph
  density (`top_k=3`, `max_indegree=10`), an optional default-on residual SpLiCE
  semantic gate, deterministic cross-fold relation validation, and spatial
  slot-separation confidence propagated from concept extraction into CoBalT
  balancing weights. Added v3 graph validation/reporting while retaining v2
  compatibility and leaving CQT behavior on its existing format.
- **Reason:** make graph density comparable across settings, restore a
  DINO-free semantic consistency check, reject unstable relations, and prevent
  uncertain CoBalT assignments from dominating concept grouping.
- **Consequences:** CRPv3 graphs may be sparser and have lower coverage than the
  previous dense no-DINO graphs; selected groups now need cross-fold support and
  graph training receives confidence-calibrated edges. Existing CoBalT artifacts
  without confidence remain runnable but do not receive the new confidence signal
  until concept extraction is rerun. These changes require fresh frozen audits,
  matched graph-density controls, and paired SSL seeds before any robustness claim.
- **Track:** CRP and shared graph/reporting infrastructure; CoBalT concept
  extraction is an upstream CRP input. CQT and historical reporting snapshots
  are not superseded by this entry.

## 14. 2026-09-03 — Current manuscript architecture update

- **State before:** the paper described an earlier frozen-audit architecture with
  an auxiliary visual-geometry branch and treated CoBalT only as related work.
- **Change:** architectural passages now present CoBalT discovery, confidence-aware
  concept balancing, residual SpLiCE agreement, cross-fold persistence and fixed
  graph-density limits as the sole CRP method. Existing results are scoped to the
  relational trainer.
- **Reason:** align the manuscript with the active CoBalT-balanced CRP protocol.
- **Consequences:** the paper's method and parameter table match current research.
  Fresh end-to-end audits and paired runs remain required for claims about the
  complete pipeline. This is a reporting-only change; code, losses and existing
  experiment artifacts are unchanged.
- **Track:** CRP reporting; current manuscript snapshot. CQT and historical
  reporting snapshots are unchanged.

## 15. 2026-09-03 — Semantic-family CRP audit variants

- **State before:** the Open Images CRP grouping sweep required positive
  coactivation for every non-lexical grouping edge and used selected-group caps in
  nearly all lower-threshold variants. On Waterbirds this produced mostly
  two-concept near-synonym groups for individual bird species, so the capped audit
  could spend most of its factor budget on repeated bird semantics.
- **Change:** appended three frozen-audit variants with text thresholds 0.75, 0.70,
  and 0.65, `coactivation_threshold=0`, `min_group_size=1`, and
  `max_selected_groups=0`. Because cached SpLiCE codes are non-negative, zero
  coactivation makes the existing grouping graph semantic-only; connected
  components remain data-dependent and have no maximum concept count. Existing
  coactivation-gated variants remain unchanged as controls.
- **Reason:** test whether mutually exclusive but semantically related concepts can
  form broad families, reducing repeated species-level factors and exposing
  background families to the same label-free audit without curated nuisance words.
- **Consequences:** the sweep now has fifteen tasks. Semantic-only components may
  become large through transitive links, so their words, basis ranks, coverage,
  null margins, and graph density must be inspected before SSL. A background group
  is not guaranteed and must still pass the unchanged null and cross-fold gates.
  No training result or robustness claim changes until fresh audits and matched
  controls are run.
- **Track:** CRP frozen-audit experiment protocol only. CQT, the shared trainer,
  baselines, and earlier reporting snapshots are unchanged.

## 16. 2026-09-03 — Open Images V7 as the repository-wide vocabulary default

- **State before:** the dedicated Open Images cache launcher selected the complete
  Open Images V7 vocabulary explicitly, but generic Python entry points, legacy
  SpLiCE helpers, the shared cluster configuration, and the environment check still
  defaulted to a 10,000-entry LAION vocabulary.
- **Change:** centralized Python defaults on `openimages_v7` with the full vocabulary
  (`vocab_size=-1`) and applied the same pair to every generic CLI, SpLiCE helper,
  shared cluster configuration, and environment check. LAION remains supported only
  when explicitly selected.
- **Reason:** make the richer named object vocabulary used by the active CRP audits
  the consistent zero-override behavior and prevent entry-point-dependent dictionary
  changes.
- **Consequences:** commands that omit vocabulary flags now build Open Images-based
  caches and concept artifacts. Historical LAION caches remain valid only with their
  recorded configuration and must not be mixed with new default artifacts. Fresh
  graph audits are required after the dictionary change; no existing result or
  robustness claim is updated.
- **Track:** shared SpLiCE cache/discovery infrastructure affecting CRP, CQT, and
  legacy SpLiCE baselines when they omit explicit overrides. The shared SSL trainer
  and historical reporting snapshots are unchanged.

## 17. 2026-09-03 — Comparable semantic audits and matched SSL graph screen

- **State before:** the general grouping array inherited its seed from the Slurm
  task index, so different grouping variants also changed the null-test seed. The
  three semantic-family settings existed only at the end of the fifteen-task
  array. There was no small launcher that compared shortlisted CRPv3 graphs with a
  true SimCLR run under an otherwise matched shortened protocol.
- **Change:** fixed the grouping sweep seed at zero and added a dedicated
  three-task CoBalT/no-DINO semantic-family preparation array with explicit Open
  Images, residual-gate, cross-fold, and graph-density settings. Added a separate
  100-epoch W&B-tracked SSL screen containing pure SimCLR plus the label-free
  shortlist `g3_t065_c015_k8` and `g2_t070_c020_k12`, both with relational weight
  2.0, temperature 0.25, graph positives disabled, warmup, and no late decay. The
  shared launcher now accepts `MODE=none` so the control does not load a teacher
  graph or use graph-aware batches.
- **Reason:** isolate grouping parameters from audit randomness, run the requested
  semantic-only variants without repeating completed graphs, and obtain an early
  matched SSL signal for one higher-null-margin/variable-group graph and one denser
  coverage-oriented graph.
- **Consequences:** newly prepared semantic variants are directly comparable at
  seed zero; older array artifacts with other seeds remain valid diagnostics but
  are not exact controlled variant comparisons. The 100-epoch jobs are screening
  evidence only and cannot be compared directly with historical 500-epoch values
  or support robustness claims. Promotion still requires inspection of loss scale,
  supported-anchor fraction and downstream metrics, followed by full paired seeds
  and the required non-concept controls. No graph mathematics or SSL loss changed.
- **Track:** CRP frozen-audit and experiment protocol; shared training launcher and
  SimCLR baseline plumbing. CQT and historical reporting snapshots are unchanged.

## 18. 2026-09-03 — SpLiCE-only teacher-graph ablation before CoBalT

- **State before:** the active no-DINO CRPv3 graph runs used CoBalT-derived
  confidence-weighted sample balancing, so completed SSL screens could not isolate
  whether the frozen SpLiCE concepts and their interventions were useful without
  CoBalT. The existing shortlist contained SimCLR and CoBalT-balanced graphs only.
- **Change:** added a two-task seed-0 graph preparation array with `USE_DINO=false`
  and `COBALT=false` for the matched `g3_t065_c015_k8` and
  `g2_t070_c020_k12` configurations. It reuses the same frozen Open Images SpLiCE
  cache but computes frequency and coactivation with uniform sample mass. Expanded
  the 100-epoch SSL screen to five matched tasks: SimCLR, both SpLiCE-only graphs,
  and both CoBalT-balanced counterparts.
- **Reason:** measure the incremental contribution of SpLiCE/CRP before adding
  CoBalT, and then measure the contribution of CoBalT while holding seed, student,
  graph hyperparameters, loss scale, augmentations, and evaluation fixed.
- **Consequences:** the experiment now supports two explicit comparisons:
  SimCLR versus SpLiCE-only CRP, and SpLiCE-only CRP versus CoBalT-balanced CRP.
  The first graph stage remains a frozen, label-free diagnostic; the shortened SSL
  stage remains screening evidence and cannot support a robustness claim without
  full paired seeds and non-concept controls. No graph formula, loss, vocabulary,
  CQT behavior, or historical result changed.
- **Track:** CRP frozen-audit protocol, SpLiCE-only baseline, CoBalT ablation, and
  shared SSL experiment protocol. Reporting snapshots are unchanged.
