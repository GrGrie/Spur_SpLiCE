# ECS kill test and CSA stage 1 on MetaShift, 2026-11-24

Study `factors_metashift`, seeds 1 to 4, validation WGA (mean of the last four probes). Full tables:
`outputs/reports/factors_metashift/concept_curves.md` (`python -m cospro.cli.summarize_concept_curves --study factors_metashift`).

## ECS: K1 fail, K2 fail, so ECS ends

- **K1.** Detection AUROC 0.50 on every seed. The onset score follows the region area (Pearson 0.99 per seed), so
  the seeds agree at Spearman 0.99 because they share the regions. Area normalization keeps AUROC at 0.50. Only
  four of twelve units carry a hidden-label type, so the AUROC rests on four pairs. Unit 0 collects 71 of 130
  factors, appears in every image and covers 61 percent of it.
- **K2.** `curves` gains +5.0 points over `simclr` on 2 of 4 seeds, and `curves_oracle` gains +2.9. One seed carries
  both means: `simclr` seed 3 scored 31.6, while every other arm scored 40 to 55 on that seed. Without seed 3 the
  changes lie within about 3 points of zero.
- **Reliance check.** Every swapping arm lowers the measured reliance on its swapped units on 4 of 4 seeds.
  `curves_bottom` and `curves_unit_1` keep unit 3 near 0.135; swapping unit 3 brings it to 0.05.

Under `PROJECT_ECS.md`, Section 3, both stages fail and the oracle fails too, so the swap itself is the weak
point and ECS ends.

## CSA stage 1: outcome weak, every unit neutral

Nine audited units, each swapped alone (`curves_unit_<k>`), 36 runs.

- **Every verdict is neutral.** The effects against `simclr` run from −0.6 to +5.2 points with sd 8 to 12, and all
  of them inherit the low `simclr` seed 3.
- **The identity of the swapped unit explains nothing.** A unit-by-seed F test gives F = 0.76, p = 0.64. The
  run-to-run sd is 4.1 points, and pairing by seed removes no variance: the 450 epochs after the shared early
  window decorrelate the runs.
- **Against the mean of the unit arms on the same seed,** unit 3 (lawn / surfers / pasture, attribute) sits
  +1.8 on 4 of 4 seeds and unit 11 (beaches) sits −4.1 on 3 of 4 seeds. Both stay below the noise rule.
- **H3 holds.** The swapped unit's own share falls on 9 of 9 units, by about a factor 2.5 for unit 3, so the
  reliance curve follows the intervention.
- **H2 is uninformative here.** The Spearman correlation is 0.60, but it relates the onset score to effects that
  sit inside the noise.

A single-unit effect of 2 points would need about 30 seeds at this noise level. The training-time swap of one
unit is too weak a lever on MetaShift, so CSA moves to the attribution at inference of `PROJECT_CSA.md`,
Section 6, which pairs clean and occluded images inside one encoder.

## Next

1. `csa_audit` on seeds 1 to 8: audit-only SimCLR runs that keep their final encoder.
2. The attribution at inference on those eight encoders.
3. Thesis finding to write up: the swap moves the representation (reliance falls 2.5 times) and leaves the
   probe's WGA unchanged.
