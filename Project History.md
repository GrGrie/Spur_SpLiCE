# Spur_SpLiCE — project history

This file contains the dated research and implementation chronology moved out of
PROJECT_GROUND_TRUTH.md. Read it when reconstructing an earlier method, experiment,
reporting snapshot, or implementation decision. The ground truth keeps only the
current state and a link to this history.

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

### 2026-09-02 — Compact ground-truth entry point

**Track:** reporting/documentation only.
Before this change, the file mixed the current specification, implementation map,
experiment protocol, and long historical explanations, making project orientation
slow. Sections 1–17 were condensed into a short entry point that puts the project
goal, CRP/CQT distinction, shared architecture, information boundary, status,
experiment protocol, repository map, and next-agent rules first. The historical
chronology in Section 18 was left intact.

The reason is navigability for new agents and collaborators. No code, loss, graph
semantics, experiment result, or scientific claim changed. The 2026-08-30 CRP-only
reporting snapshot and all other historical snapshots remain unchanged.

### 2026-09-02 — Cluster-only launcher configuration and CoBalT default

**Track:** experiment infrastructure; shared CRP/CQT trainer; CoBalT concept check.

Before this change, the main Slurm launchers kept most training, graph, cache, and
sweep values in shell blocks, while several workflows still required long
`sbatch --export` commands. CoBalT discovery also had no local launcher config,
and its preparation entry point defaulted to the paper ResNet-50 pretrained
backbone. Local PowerShell launchers and the older standalone CQT investigation
launcher were not part of the cluster workflow but remained in `scripts/`.

The change moves editable values into adjacent `.conf` files in `scripts/`, adds a
shared shell config loader, and moves CoBalT discovery settings to
`scripts/prepare_concepts.conf`. The default CoBalT model is now the project's
`resnet18_large` convention, mapped to CoBalT's internal `resnet18` backbone with
the large-input stem and no pretrained initialization. CoBalT discovery accepts
this as an explicit non-paper backbone run rather than mislabelling it as a smoke
test. Sweep matrices are also stored in their own configs. The six PowerShell
launchers and obsolete `cqt_investigate.sbatch` were removed; current CRP/CQT
training, graph preparation, cache, CoBalT concept preparation, and required
diagnostic entry points remain.

The reason is a cluster-only workflow with one place to edit experiment numbers
and short, reproducible `sbatch scripts/<entry>.sbatch` commands. CRP/CQT graph
mathematics, SSL loss, label boundary, and existing results are unchanged. The
CoBalT default is a protocol change for future concept artifacts: artifacts made
with the new default must not be compared as identical to the old pretrained
ResNet-50 discovery without recording the model configuration. This affects
preparation infrastructure and the CoBalT concept-check track; it does not alter
the CRP/CQT scientific claims or the historical reporting snapshots.

### 2026-09-03 — Manuscript aligned with the CoBalT-balanced CRP protocol

**Track:** CRP reporting only; current manuscript snapshot.

**State before:** `Spur_SpLiCE.tex` presented the earlier frozen-audit architecture
with an auxiliary visual-geometry branch, older graph-density settings and CoBalT
only as related work. Its reported seed-0 pilots predated the complete current
pipeline.

**Change:** the architectural and method passages now present label-free CoBalT
discovery as the upstream balancing stage. They describe spatial concept
memberships, assignment-confidence weighting, weighted SpLiCE frequency and
coactivation, residual SpLiCE pair agreement, deterministic cross-fold edge
persistence, three outgoing edges, absolute indegree ten and the existing shared
relational trainer. The manuscript uses the public name CRP throughout and presents
this configuration as the sole method. Pilot results are explicitly scoped to the
relational trainer.

**Reason:** align the paper's method narrative with the active CoBalT-balanced CRP
research track and prevent earlier architectural details from being attributed to
the current method.

**Consequences:** the mathematical graph description and default-parameter table
now match the active protocol. Existing pilot values remain historical
mechanism-finding evidence and provide no end-to-end evaluation of the full
pipeline. Fresh frozen audits, matched controls and paired SSL seeds remain
required for robustness claims. No implementation, loss or experiment artifact was
changed.

### 2026-09-03 — Semantic-family CRP audit variants

**Track:** CRP frozen-audit experiment protocol; current development only.

Before this change, the Open Images CRP grouping sweep required positive
coactivation for every non-lexical grouping edge and capped selected groups in
nearly all lower-threshold variants. The observed Waterbirds audit consequently
contained mostly two-word near-synonym groups for separate bird species, allowing
repeated bird semantics to occupy the capped selection.

Three semantic-family variants were appended with text thresholds 0.75, 0.70, and
0.65, zero coactivation threshold, minimum group size one, and no selected-group
cap. Since SpLiCE codes are non-negative, zero coactivation uses the existing
grouping implementation as a semantic-only connected-component graph. Component
size remains data-dependent with no upper bound. All prior coactivation-gated
variants remain available as matched controls.

The purpose is to let mutually exclusive but semantically related names form broad
families and thereby give background families an opportunity to enter the same
label-free audit, without dataset-specific nuisance words or annotations. The sweep
now has fifteen tasks. Semantic components can percolate through transitive links,
so group contents, basis ranks, coverage, null margins, and graph density must be
audited before SSL. Background discovery is not guaranteed, and no training result,
method claim, CQT behavior, shared-trainer behavior, or historical reporting
snapshot changes until new audits and controls are completed.

### 2026-09-03 — Open Images V7 as the repository-wide vocabulary default

**Track:** shared SpLiCE cache/discovery infrastructure; CRP, CQT, and legacy
SpLiCE baselines when no explicit vocabulary override is supplied.

Before this change, the dedicated Open Images cache launcher explicitly selected
the full Open Images V7 vocabulary, while generic Python entry points, legacy helper
functions, the shared cluster configuration, and the environment check still
defaulted to a 10,000-entry LAION vocabulary. Consequently, two otherwise similar
commands could silently construct incompatible concept dictionaries.

Python defaults are now centralized on `openimages_v7` with `vocab_size=-1`, and
the same vocabulary/size pair is used by every generic CLI, helper fallback, shared
cluster configuration, and environment check. LAION support is retained for
explicit historical controls and ablations. The OpenCLIP pretrained checkpoint is
unchanged because it is the encoder configuration rather than the concept
vocabulary.

Commands without vocabulary flags now create Open Images-based caches and concept
artifacts. Historical LAION artifacts remain valid only with their recorded exact
configuration and cannot be mixed with the new defaults. Graphs must be rebuilt
after a dictionary change. This updates future cache/discovery behavior but changes
no shared SSL loss, existing result, supported robustness claim, or historical
reporting snapshot.

### 2026-09-03 — Comparable semantic audits and matched SSL graph screen

**Track:** CRP frozen-audit and experiment protocol; shared training launcher and
SimCLR baseline plumbing.

Before this change, the general grouping array inherited its seed from the Slurm
task index, coupling each grouping setting to a different null-test seed. The three
semantic-family settings also required submitting the complete fifteen-task array,
and no small launcher compared promising CRPv3 graphs with a true matched SimCLR
control.

The grouping sweep now fixes seed zero. A dedicated three-task preparation array
runs only the CoBalT/no-DINO semantic-family settings with explicit Open Images,
residual-gate, cross-fold, and fixed-density values. A separate W&B-enabled
100-epoch SSL array runs pure SimCLR and two graphs selected from label-free frozen
audit evidence: `g3_t065_c015_k8` as the variable-group, higher-null-margin
candidate and `g2_t070_c020_k12` as the denser coverage comparator. Both CRP tasks
use relational weight 2.0, temperature 0.25, graph positives off, warmup, and no
late decay. The shared training launcher accepts `MODE=none` so the SimCLR task
loads neither a graph nor the graph-aware sampler.

This separates configuration effects from null-test randomness and permits an
inexpensive matched mechanism screen while the new semantic audits run. Existing
nonzero-seed artifacts remain useful diagnostics but are not exact controlled
variant comparisons. The shortened runs cannot be compared directly with prior
500-epoch results and do not establish robustness; a winner must still pass loss
and supported-anchor checks before full paired seeds and required controls. Graph
construction, the relational loss, CQT, and historical reporting snapshots are
unchanged.

### 2026-09-03 — SpLiCE-only teacher-graph ablation before CoBalT

**Track:** CRP frozen-audit protocol, SpLiCE-only baseline, CoBalT ablation, and
shared SSL experiment protocol.

Before this change, the active no-DINO CRPv3 graphs applied confidence-weighted
sample balancing derived from fixed CoBalT memberships. Although SpLiCE supplied
the concept dictionary, sparse codes and intervention geometry, there was no
matched current-protocol graph or SSL run isolating those components before CoBalT.

A two-task seed-0 graph array now builds `g3_t065_c015_k8` and
`g2_t070_c020_k12` with both DINO and CoBalT disabled. These jobs reuse the same
frozen Open Images SpLiCE cache as the CoBalT runs; the only grouping-input change
is that concept frequency and coactivation use uniform sample mass instead of
CoBalT-derived weights. The shortened SSL array now contains five matched tasks:
pure SimCLR, both SpLiCE-only CRP graphs, and both CoBalT-balanced counterparts.

This establishes a direct decomposition of the mechanism: SimCLR versus
SpLiCE-only CRP estimates the value or harm of the frozen SpLiCE teacher graph,
while SpLiCE-only versus CoBalT-balanced CRP estimates the incremental balancing
contribution. Seed, graph hyperparameters, student configuration, relational-loss
scale, augmentations, and evaluation remain fixed. These 100-epoch runs are
screening evidence only; full paired seeds and non-concept controls remain required.
No graph formula, loss, vocabulary, CQT behavior, existing result, or historical
reporting snapshot changed.

### 2026-09-03 — One-command Windows SpLiCE-only mechanism screen

**Track:** local experiment infrastructure, SpLiCE-only CRP ablation, SimCLR
baseline, and reporting plumbing.

Before this change, the no-DINO/no-CoBalT graph controls and matched SSL screen were
available only through cluster-oriented launchers. Running them from a fresh
Windows clone required manually acquiring Waterbirds, constructing the vocabulary
and cache, building a graph, and assembling separate training commands.

An idempotent PowerShell entry point now uses the existing `grgrie-train` Conda
environment, verifies a working CUDA tensor operation, obtains and materializes
Waterbirds and the Open Images vocabulary, constructs a frozen SpLiCE cache and one
`g2_t070_c020_k12` teacher graph with DINO and CoBalT disabled, renders the graph
audit, and trains matched SimCLR and SpLiCE-only CRP students sequentially. The
default quick budget is 50 SSL epochs and a single 30-epoch final validation probe.
It preserves logs, final checkpoints, a CSV result table, and offline W&B records.
The local audit uses 16 null trials to limit runtime.

This provides a reproducible home-GPU mechanism screen but is not directly
comparable with the canonical 32-null-trial audit or 500-epoch SSL protocol. Exact
completion time remains dependent on downloads, drivers, and the local environment.
A promising result must be confirmed under the full paired-seed protocol. The
graph formula, relational loss, CoBalT, CQT, existing results, and historical
reporting snapshots are unchanged.

### 2026-09-03 — Matched Windows CoBalT contribution arm

**Track:** local CRPv3 experiment protocol, CoBalT ablation, SpLiCE-only baseline,
and SimCLR control.

Before this change, the local screen ended after matched SimCLR and SpLiCE-only
CRP training. It could not evaluate current CoBalT-balanced graph construction.
Applying a CoBalT switch while retaining the exact same teacher graph would not
change SSL because fixed CoBalT memberships participate only in graph construction,
not in the student loss or optimizer.

The Windows entry point now accepts `-IncludeCobalt`. It reuses a completed
Waterbirds download, frozen SpLiCE cache, SimCLR run, and SpLiCE-only CRP run. It
then trains the current seed-0 CoBalT discovery configuration, extracts fixed
memberships and confidence, rebuilds the same `g2_t070_c020_k12` graph setting with
CoBalT weighting enabled, renders a second audit, and trains a third student with
the same quick-screen SSL parameters. Results for all three arms are written to one
CSV. Log names include the SSL/probe budgets so changed budgets do not silently
reuse older results.

This isolates CoBalT at its actual intervention point while holding the cache,
non-CoBalT graph settings, seed, student, optimizer, augmentations, relational
schedule, and evaluation fixed. The two teacher graphs are expected to differ;
that difference is the treatment being measured. The result remains a shortened
single-seed screen and cannot establish robustness without full paired runs. No
graph formula, SSL loss, CQT behavior, existing result, or historical snapshot
changed.

### 2026-09-03 — CRPv4 spatial balancing in the SpLiCE vocabulary

**Track:** CRP method and frozen-audit infrastructure; label-free spatial concept
discovery. The shared student trainer is format-compatible but mathematically
unchanged. CQT and historical reporting snapshots are unchanged.

Before this change, CRPv3's optional CoBalT input came from a separate visual
ResNet and an anonymous vector-quantized dictionary. It influenced SpLiCE grouping
through dataset-level sample weights, so the spatial branch and SpLiCE described
images with different concept systems. Spatial slot confidence existed, but there
was no direct per-image balance factor over named SpLiCE concepts.

CRPv4 adds a separate spatial path over the same frozen OpenCLIP ViT-B/32 used by
SpLiCE. It exposes native patch tokens, optionally learns Slot Attention with
label-free two-view attention and projected-semantic consistency, applies CLIP's
own frozen normalization/projection only after slot aggregation, and compares
patches or slots directly with the frozen SpLiCE dictionary. The bounded ablation
matrix contains vanilla and SCLIP patch features, each with and without slots;
the SCLIP path reuses the frozen final-layer projections with query-query plus
key-key correlative attention and adds no learned language adapter or new concept
dictionary.

The reason for the change is to remove duplicate teacher-side concept systems:
SpLiCE should continue to answer which named concepts explain an image, while the
spatial branch should describe where and how consistently those same concepts are
represented. This also makes spatial balancing independently ablatable without
changing the student or rewriting the original decomposition.

Extraction writes sparse image-specific evidence and spatial confidence aligned
exactly to the frozen cache's sample IDs and vocabulary. Graph construction creates
a separate balanced code tensor from the original SpLiCE weights and the spatial
factors, interpolates uncertain samples toward neutral balance, and preserves each
sample's original total sparse-code mass. These balanced codes affect grouping,
activation differences, nulls, and the no-DINO residual gate; the cached weights,
CLIP projection intervention, teacher-graph loss, and student remain unchanged.
CRPv4 has a distinct graph artifact/version. Combining this path with legacy
CoBalT sample balancing is rejected so their effects remain identifiable.

The implementation establishes a runnable hypothesis, not evidence that slots,
SCLIP, or spatial balancing improve WGA or identify nuisance concepts. All four
spatial variants require fresh frozen audits, direct comparison with unbalanced
CRPv3 and frequency-only controls, inspection of coverage/null margins, and paired
SSL seeds before any robustness or causal claim. Earlier CRPv2/v3 results and the
current manuscript snapshot are not retroactively reinterpreted.

### 2026-09-04 — Sequential Windows CRPv4 four-way screen

**Track:** CRPv4 experiment protocol and local launcher infrastructure. The CRP
graph construction and shared student loss are unchanged.

Before this change, the four CRPv4 spatial variants had a cluster preparation
array, but there was no one-command Windows path that prepared each spatial
artifact, built its matching CRPv4 graph, and trained all four students under the
same local 100-epoch protocol. The earlier Windows launcher covered SimCLR,
SpLiCE-only CRPv3, and legacy CoBalT-balanced CRPv3 instead.

A resumable PowerShell launcher now runs vanilla patchwise, vanilla slots, SCLIP
patchwise, and SCLIP slots sequentially at seed zero. It reuses a validated frozen
Open Images SpLiCE cache, trains only the two slot branches, extracts aligned
image-specific spatial evidence, constructs a separately validated CRPv4 graph for
each variant, and performs W&B-tracked 100-epoch relational SSL followed by one
final validation probe. All graph, student, optimizer, augmentation, relational
schedule, and evaluation settings are matched across the four arms. Partial and
final result tables record graph support alongside the final probe metrics; an
empty graph remains a valid falsification and uses the documented SimCLR fallback.

This enables the requested overnight single-GPU architecture screen while keeping
the four spatial treatments identifiable. Its default 16-trial null audit is a
local runtime compromise and is not directly equivalent to the 32-trial cluster
audit. The single seed and shortened 100-epoch schedule are screening evidence
only; they do not establish stability, a causal concept interpretation, or an
improvement over matched non-concept controls. No current or historical result is
reinterpreted.

### 2026-09-04 — Isolated diversity-constrained CRPv4 experiment

**Track:** optional CRPv4 frozen-audit experiment and local experiment launcher.
The canonical CRP implementation, four-way CRPv4 screen, graph format, and shared
student trainer are unchanged.

Before this change, the active Waterbirds grouping configuration required at least
two concepts joined by both text similarity and sparse-code coactivation. The
resulting candidate set consisted only of small bird-name families even though
several scene and environment concepts had nonzero unlabeled SpLiCE support. The
final group cap ranked null-passing candidates by quality but did not constrain
semantic or edge redundancy.

An opt-in module now implements a separate diversity-constrained CRPv4 audit. It
allows singleton factors, lowers the candidate-frequency floor, partitions group
centroids into deterministic unlabeled semantic clusters, and budgets expensive
audits across those clusters using activation entropy plus agreement with the
existing spatial artifact. It then applies the unchanged projection, relation,
null, residual-agreement, and cross-fold gates. Only groups passing those gates are
eligible for a cluster-capped MMR selection that penalizes text-centroid similarity
and teacher-edge overlap. The resulting artifact remains a standard CRPv4 teacher
graph accepted by the unchanged trainer and records the full preselection and
selection trace.

A separate resumable PowerShell launcher reuses one completed spatial artifact
from the four-way screen, writes all new graph, audit, training, W&B, and result
files under an independent output root, and trains a matched 100-epoch seed-zero
student. This isolates rollback to the optional module, launcher, tests, and this
history entry; no running or canonical code path imports the experiment.

The mechanism encourages semantic breadth but cannot guarantee a particular named
scene concept: a candidate is still rejected if it fails the unchanged label-free
quality gates. Spatial multiplication also cannot introduce a concept whose
original sparse SpLiCE weight is identically zero. The experiment is therefore a
test of diverse selection among supported concepts, not an oracle scene-concept
injection, and remains single-seed screening evidence.

### 2026-09-04 — Periodic CRPv4 probes and no-DINO diverse-cache correction

**Track:** CRPv4 experiment protocol and optional diversity launcher plumbing.
The canonical audit, diversity objective, graph format, and student losses are
unchanged.

The first local diversity launch exposed an interface edge case: the optional
module validated a no-DINO feature cache twice, and the canonical first pass
represented the absent optional embedding as `None`; the second pass then tried
to normalize that value as a tensor. The diverse path now performs one CLI-level
validation and defensively removes only that absent optional field from a shallow
copy when its audit function receives an already validated cache. The correction
is confined to the opt-in module and is covered by a regression fixture; the
canonical CRP implementation remains untouched.

Both Windows CRPv4 launchers now request a validation linear probe every 25 SSL
epochs, including epoch 100, rather than only after the final SSL epoch. Probe
frequency is an explicit launcher parameter, and log names, checkpoint roots,
W&B names, and tags encode it so completed final-only runs are preserved and are
not silently reused as periodic runs. Each probe still trains for 30 epochs by
default. This changes evaluation cadence and runtime, not SSL optimization.

### 2026-09-04 — Resumable three-arm local diverse screen

**Track:** optional diversity experiment protocol and Windows launcher. The
diversity selector, frozen audit, teacher graph, and shared student losses are
unchanged.

The first diverse launcher invocation accepted one spatial variant and therefore
stopped after the requested default `vanilla_slots` student. It did not provide a
single-command matched control screen. The default invocation now orchestrates
three sequential arms: the completed diverse vanilla-slot graph/student, a
diverse vanilla-patchwise graph/student, and a pure SimCLR baseline with the same
student, augmentations, optimizer, seed, 100-epoch schedule, and periodic linear
probe protocol. Explicit `-Variant` selection remains available for isolated
runs.

Each arm has an independent output, checkpoint, log, and W&B directory. Completed
arms are validated and reused, while missing arms continue in order, and a root
result table is assembled after all three finish. The baseline explicitly uses
no SpLiCE or teacher graph and records zero graph statistics. This launcher change
enables the intended matched comparison without altering the already completed
vanilla-slot evidence or treating an earlier baseline from another protocol as
interchangeable.

### 2026-09-04 — Seed-parameterized Windows CRPv3 replication

**Track:** local CRPv3 experiment protocol and reporting. CRP graph mathematics,
legacy CoBalT discovery, student losses, and the seed-zero results are unchanged.

The original three-arm Windows screen fixed seed zero throughout and reused a
single output namespace. It therefore could not run a clean second stochastic
replicate, and its result parser matched an intermediate linear-probe status line
rather than the final probe summary. The launcher now accepts an experiment seed,
uses that seed consistently for the CRP audit, CoBalT discovery, and all three SSL
students, and gives nonzero seeds an independent default output directory. The
frozen SpLiCE feature cache is reused across seeds because it is fixed input data,
while both teacher graphs and the learned CoBalT artifact are regenerated inside
the seed-specific directory.

The 100-epoch protocol retains linear probes every 25 SSL epochs. Completion now
requires both the final SSL epoch and its completed final probe, log names encode
the seed and probe cadence, W&B identity fields encode the seed, and the result
table reads the final probe summary. A local side-by-side HTML index was also added
for presenting the existing seed-zero SpLiCE-only and CoBalT-balanced graph audits;
this is reporting-only and does not modify either graph artifact.
