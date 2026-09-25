# Concept-guided late-layer views: a design study

2026-09-25. A working note on bringing SpLiCE concepts into student training itself, in the style of
LateTVG. It covers the mechanism LateTVG relies on, one constraint every late-layer intervention
must satisfy, four candidate methods and the experiment that separates them.

## 1. What LateTVG does

LateTVG (Hamidieh et al., ICLR 2024, arXiv 2406.18562) keeps the image augmentations of SimCLR or
SimSiam and adds a transformation in weight space. View 1 passes through the full encoder `f`. View 2
passes through `f~`, whose last `L` convolutions lose the fraction `a` of smallest-magnitude weights.
The mask is recomputed at every step, and `f` and `f~` share weights. The loss aligns `f(x1)` with
`f~(x2)`.

The argument has two parts. Late layers hold the high-level features, spurious and core alike. Pruning
removes features stored in low-magnitude weights first, and in supervised networks those are
disproportionately the features that serve minority examples (Hooker et al., 2019). Asking the
representation to survive pruning therefore raises the loss on examples whose features live in the
weak part of the network. The transformation knows nothing about *which* features it removes; the
minority effect is a statistical side effect of magnitude.

Reported Waterbirds WGA with a from-scratch ResNet-18: SimCLR 43.8 to 55.4, SimSiam 48.3 to 56.3.
These are the best points of a grid over `L` and `a`, which the paper does not say it selected on
validation.

Our implementation lives in `cospro/training/late_pruning.py` and runs from
`experiments/manifests/waterbirds_latetvg.yaml`. Two choices follow the paper where it is explicit and
fix a value where it is silent: the magnitude threshold is pooled over all targeted convolutions
(the paper's `Top_a(θ)`), and the pruned pass keeps its own batch-norm statistics (the paper keeps
`f~` as a separate model). The paper's Algorithm 1 returns `f~`; we evaluate the full encoder `f`,
like every other arm. With CoSpRo on top, the relational KL sees the mean of the full view-1 and
pruned view-2 embeddings.

## 2. The placement constraint

Any view built by transforming the network has to change the representation the probe measures.
Write the encoder as `v = F(h)`, where `h` is an intermediate feature map, and the SSL loss as a
comparison of `g(v1)` with `g(v2~)` through the projection head `g`.

* **Transformation after the evaluated representation.** Suppose view 2 becomes `v2~ = (I - P) v2`
  for a fixed projector `P`. The loss is minimized when `g` ignores `span(P)`: the head learns
  `g(v) = g((I - P) v)` and the encoder never has to change. The linear probe then reads the removed
  directions from `v` at full strength. This rules out every variant that edits the pooled
  512-dimensional feature, including a concept bottleneck whose context coordinates are simply
  zeroed on one view.
* **Transformation before the evaluated representation.** With `v2~ = F(T h2)` the difference
  `F(h) - F(T h)` passes through the remaining nonlinear layers. The head cannot cancel it with one
  fixed linear map, so the invariance lands in `F` and therefore in `v`. The minimizer makes `F`
  insensitive to what `T` removes.

LateTVG satisfies the constraint because it prunes inside `layer4`. Every method below does the
same: the intervention sits in `layer3` or `layer4`, upstream of global pooling.

A second consequence concerns what the network can do in response. A fixed `T` lets the network
relocate the removed information to channels `T` does not touch. A transformation re-estimated from
the current network, as LateTVG's per-step mask is, turns the relocation into a moving target. This
is the same logic as iterative nullspace projection (Ravfogel et al., 2020): each round removes what
is currently linearly present, and the fixed point is a representation where the removed information
is absent or unused.

## 3. Candidate methods

All four use artifacts the pipeline already produces: the SpLiCE dataset cache (sparse concept codes
`s_i` per training sample, indexed by the stable sample indices the relational method already
receives) and the concept-type scores (`cospro/pipeline/concept_type.py`), which rank concepts from
object-like to context-like through the text encoder alone. Split the concept coordinates into
context concepts `s_ctx` and object concepts `s_obj` by that score. No step uses labels or groups.

### M1. Concept-targeted late pruning (recommended first)

LateTVG with a different mask criterion. Instead of pruning the smallest weights, drop the `layer4`
channels whose activations carry the context concepts.

1. Pool each channel of an intermediate map (output of `layer4.0` or `layer3`) over space:
   `a_i ∈ R^C` for sample `i`.
2. Keep exponential moving averages of the covariances between `a`, `s_ctx` and `s_obj` over
   training batches.
3. Score channel `c` by its partial covariance with the context concepts given the object concepts:
   `Σ_{a s_ctx | s_obj} = Σ_{a s_ctx} - Σ_{a s_obj} Σ_{s_obj s_obj}^{-1} Σ_{s_obj s_ctx}`, and take the
   row norm per channel, normalised by the channel's residual standard deviation.
4. For view 2, zero a fraction `k` of channels, sampled with probability increasing in the score, in
   the same functional forward the LateTVG code already has.

The partial covariance is where the concept decomposition earns its place. In the Waterbirds
training set, context and object agree for 95% of images, so a plain correlation with "water" also
selects channels that encode "waterbird". Conditioning on the object concepts leaves only the
variance in which context and object disagree, which the minority images supply. Magnitude pruning
has no access to this distinction.

Expected failure modes: the conditional covariance is estimated from few disagreeing images and
will be noisy early in training (warm up with LateTVG's magnitude mask, then switch); SpLiCE codes
describe the full image while the view is a crop, which adds noise to the statistics but leaves them
unbiased over the population.

This method admits a sharp causal test. Target the object channels instead (swap `s_ctx` and `s_obj`
in step 3). If concept targeting works as intended, that arm must lose WGA relative to random channel
dropping, and the context arm must gain.

### M2. Concept-subspace erasure inside the network

M1 restricts removal to whole channels. M2 removes a direction. From the same moving covariances,
compute the LEACE projector (Belrose et al., 2023) that makes the context scores, residualised on the
object scores, linearly unpredictable from the pooled feature `a`. Apply `h~ = (I - P) h` at every
spatial position of the intermediate map for view 2 and continue the forward pass through the
remaining blocks.

LEACE gives a guarantee M1 lacks: after erasure, no linear predictor on the pooled feature recovers
the residual context scores better than a constant, while the feature changes minimally in the
covariance norm. The SSL invariance then asks `F` to produce the same representation with or without
that subspace. M1 is the special case of M2 with a diagonal projector, so M1 is the ablation of M2's
extra precision.

This is also the student-side twin of CoSpRo's teacher. The teacher removes a concept subspace from
CLIP to decide which images are related; M2 removes the matching information from the student's own
features to build a view. Both use the same concept groups and the same gate.

### M3. A concept bottleneck in place of the last stage

Replace global pooling with `v = [c(x); r(x)]`, where `c` predicts the SpLiCE codes (trained on the
cached codes) and `r` is an unconstrained residual. At evaluation, zero the context coordinates of
`c`. This is the literal reading of "replace the last layers with concepts", and it has three
problems:

* The placement constraint: zeroing coordinates of `v` for one view is absorbed by the head.
* Leakage: the residual `r` carries the context information whenever that lowers the SSL loss, a
  known property of concept bottlenecks (Mahinpei et al., 2021).
* Capacity: a ResNet-18 trained from scratch on 4,795 images predicts SpLiCE codes poorly. Our
  direct-transfer control already measures the regression part: `splice_reconstruction` gained
  2.8 and 6.2 WGA points over matched SimCLR on seeds 3 and 1, and the bottleneck would inherit its
  ceiling.

Transformer layers belong in this family. A cross-attention layer whose queries are the CLIP text
embeddings of the vocabulary needs student features aligned with CLIP space, which a from-scratch
student lacks. It becomes viable only with a pretrained initialisation, and then the study changes
setting. M3 is useful as an interpretability analysis of a trained student, and weak as a training
method.

### M4. Spatial concept masking of the feature map

Waterbirds and MetaShift place the spurious attribute in a different region of the image from the
object. The CLIP teacher can localise it offline: MaskCLIP-style dense features (the value path of the
last attention layer) give per-patch similarities to the context concepts, a 7x7 context map per
image for ViT-B/32. During training, map the crop and flip of view 2 onto that grid, resample it to
the `layer3` resolution and suppress feature positions with high context score before `layer4`.

This is the content-style picture of von Kügelgen et al. (2021) made explicit: view 2 keeps the
content and loses the style, so the invariant representation keeps content. Suppressing features
inside the network avoids the pixel-level shortcut in which the model learns that grey regions mean
"background". Cost: the SSL loader must return crop parameters, the dense CLIP maps are coarse and
noisy, and the method helps only where the spurious attribute is spatially separable (background yes,
gender in CelebA no). Among the four it is the most specific to Waterbirds and the most expensive.

## 4. Recommended plan

1. Run `waterbirds_latetvg.yaml` first. It fixes the LateTVG reference under our protocol and tells
   whether late-layer views help this student at all.
2. Implement M1 as a second mask criterion inside `LatePruning`, with channel-structured masks. Arms,
   all at the same number of dropped channels: magnitude (LateTVG), random channels, context-targeted
   and object-targeted. The ordering object < random < context on WGA is the evidence that the
   concepts, and not the act of pruning, produce the gain.
3. If context-targeted beats random, add M2 as the direction-level refinement and CoSpRo + M2 as the
   combined method: the concept projection used across images (graph) and within the network
   (view).
4. Keep M4 as a follow-up for datasets with spatially separable spurious attributes. Use M3 only as
   post-hoc analysis.

## References

* Hamidieh et al., Views Can Be Deceiving: Improved SSL Through Feature Space Augmentation, ICLR 2024.
  https://arxiv.org/abs/2406.18562
* Hooker et al., What Do Compressed Deep Neural Networks Forget?, 2019. https://arxiv.org/abs/1911.05248
* Ravfogel et al., Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection,
  ACL 2020. https://arxiv.org/abs/2004.07667
* Belrose et al., LEACE: Perfect Linear Concept Erasure in Closed Form, NeurIPS 2023.
  https://arxiv.org/abs/2306.03819
* Mahinpei et al., Promises and Pitfalls of Black-Box Concept Learning Models, 2021.
  https://arxiv.org/abs/2106.13314
* Zhou et al., Extract Free Dense Labels from CLIP (MaskCLIP), ECCV 2022. https://arxiv.org/abs/2112.01071
* von Kügelgen et al., Self-Supervised Learning with Data Augmentations Provably Isolates Content
  from Style, NeurIPS 2021. https://arxiv.org/abs/2106.04619
