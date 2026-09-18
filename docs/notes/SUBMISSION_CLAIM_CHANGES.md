# Claim calibration

Exact replacements in `CoSpRo.tex`; numerical results and probe protocol are unchanged.

## Replacement 1

Before:

> Our results suggest that language-aligned concept representations provide a useful mechanism for studying and modifying spurious relations in self-supervised learning.

After:

> These results provide positive empirical evidence that concept-guided relational pretraining can improve worst-group behavior in this spurious-correlation setting.

## Replacement 2

Before:

> CoSpRo improves mean WGA by 3.82 percentage points relative to SimCLR and 2.16 points relative to matched raw-CLIP supervision (Table~\ref{tab:main}).

After:

> CoSpRo achieves higher mean WGA by 3.82 percentage points relative to SimCLR and 2.16 points relative to matched raw-CLIP supervision in the four-seed Waterbirds evaluation (Table~\ref{tab:main}).

## Replacement 3

Before:

> The sampler-only comparisons show that the relational objective, rather than graph-aware batching alone, accounts for most of the final average gain.

After:

> The sampler-only comparisons support a contribution from the relational objective beyond graph-aware batching. These results provide positive empirical evidence of improved robustness under the evaluated protocol; they do not establish statistical superiority across training seeds or settings.

## Replacement 4

Before:

> The shuffled target is substantially weaker, supporting the presence of useful structure in the SpLiCE reconstruction rather than a generic auxiliary-loss effect.

After:

> The lower mean result for the shuffled target supports the presence of useful structure in the SpLiCE reconstruction beyond a generic auxiliary-loss effect in this two-seed comparison.

## Replacement 5

Before:

> CoSpRo improves mean WGA by 4.42 points over the matched semantic graph.

After:

> CoSpRo achieves a 4.42-point higher mean WGA than the matched semantic graph in this two-seed validation comparison, providing preliminary evidence for intervention-derived relations.

## Replacement 6

Before:

> The direct-transfer comparison does not establish that its negative result extends to every transfer objective or temperature.

After:

> The direct-transfer comparison provides evidence of useful reconstruction transfer under its evaluated objective and temperature; it does not establish the relative superiority of relational and direct transfer.

## Replacement 7

Before:

> The four-seed Waterbirds comparison improves average and worst-group accuracy over SimCLR and matched raw-CLIP supervision. The semantic-graph and corrected direct-transfer controls further support the usefulness of concept-derived structure.

After:

> The four-seed Waterbirds comparison yields higher mean average and worst-group accuracy than SimCLR and matched raw-CLIP supervision, providing positive empirical evidence of robustness benefits under this protocol. The semantic-graph and corrected direct-transfer controls offer additional evidence for the usefulness of concept-derived structure, while broader superiority remains to be established.
