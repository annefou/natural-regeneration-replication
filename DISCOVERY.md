# Discovery and plan (handoff)

Written on 2026-10-03, before the replication started, to hand over everything found
during discovery to the session that runs the replication. Read this first, together
with the paper PDF in `paper/`.

## Why this paper

- **Paper:** Williams, B. A. et al. (2024). *Global potential for natural regeneration in
  deforested tropical regions.* Nature 636, 131–137.
  [doi:10.1038/s41586-024-08106-4](https://doi.org/10.1038/s41586-024-08106-4)
  (open access; PMC11618091). Preprint: [doi:10.21203/rs.3.rs-3235955/v1](https://doi.org/10.21203/rs.3.rs-3235955/v1).
- **Author Correction** (Nature, 6 Dec 2024,
  [doi:10.1038/s41586-024-08481-y](https://doi.org/10.1038/s41586-024-08481-y)): in Fig. 2,
  Brazil's area with natural-regeneration potential was corrected from **55.1 Mha to
  43.71 Mha**. Use the corrected figures.
- **Not replicated yet:** Replication Radar `replication_status` returned `open`
  (no verification on the nanopub network) on 2026-10-03.
- **Publicly evaluated, with doubts:** The Unjournal published two expert evaluations
  (3 June 2025, CC-BY), registered in Crossref as a review of the paper:
  - Summary and metrics: [doi:10.21428/d28e8e57.5411b150](https://doi.org/10.21428/d28e8e57.5411b150)
    (<https://unjournal.pubpub.org/pub/evalsumnaturalregeneration>)
  - Evaluation 1: [doi:10.21428/d28e8e57.5411b150/6a67da7e](https://doi.org/10.21428/d28e8e57.5411b150/6a67da7e)
    (<https://unjournal.pubpub.org/pub/e1naturalregeneration>)
  - Evaluation 2: [doi:10.21428/d28e8e57.5411b150/640bb340](https://doi.org/10.21428/d28e8e57.5411b150/640bb340)
    (<https://unjournal.pubpub.org/pub/e2naturalregeneration>)
  - Full texts are mirrored in the MIT-licensed repo
    [unjournal/unjournaldata](https://github.com/unjournal/unjournaldata)
    (`unjournal_evaluations/e1naturalregeneration.md`, `e2naturalregeneration.md`,
    `evalsumnaturalregeneration.md`; ratings in `data/rsx_evalr_rating.csv`).

## The claims

From the abstract (verify the exact wording against the PDF before drafting any quote):

1. An area of **215 million hectares** has potential for natural forest regeneration,
   representing **23.4 Gt C** (range 21.1–25.7 Gt) of above-ground carbon
   sequestration over 30 years.
2. **Five countries** (Brazil, Indonesia, China, Mexico, Colombia) account for **52%**
   of this potential.
3. Methods claim: a random-forest model at **30 m** predicts natural regeneration with
   **87.9%** validation accuracy (87.8% out-of-bag); accuracy varies spatially and is
   lowest in parts of Southeast Asia.

## What the evaluators said

| | Evaluator 1 | Evaluator 2 |
|---|---|---|
| Overall | 50 [40–60] | 50 [30–80] |
| Belief in the main claim | 20 [15–25] | 40 [20–60] |
| Methods | 20 [15–25] | 30 [10–50] |
| Open science | 95 [90–100] | 30 [15–45] |
| Real-world relevance | 90 [80–100] | 60 [50–70] |

**Evaluator 1, main critique (testable):** three predictors are recorded *during* the
outcome period (natural regrowth 2000–2012, persisting to 2016), not before it: **net
primary productivity, burned area and road density**; ESA CCI land cover is taken from
the year 2000. The model therefore "predicts the present rather than the future", and
the reported accuracy "likely overstates the true predictive performance". Suggested
fix: restrict predictors to those **observed before 2000 or time-invariant** (ESA CCI
land cover exists from 1992; climate; soils). Second critique: the prediction is not
"pure biophysical potential", because socioeconomic factors are embedded; 215 Mha is
likely a lower bound of biophysical potential. Finds the method "tractable and
replicable with readily available data sources".

**Evaluator 2:** uncorrected omission errors in the input regrowth map and possible
conflation of regrowth types; a rough recalculation suggests about **80–180 Mha**.
Suggested check: compare other algorithms (gradient boosting, neural networks) for at
least some regions.

## Data (all open)

| Role | Source | Access |
|---|---|---|
| Authors' outputs (for comparison) | Zenodo [10.5281/zenodo.7428804](https://doi.org/10.5281/zenodo.7428804), CC-BY-4.0, 77 files | `pnv_pct_30m_tile_*` (continuous %, 10° tiles, ~9.3 GB total), `pnv_bin_30m.zip` (binary >0.5, ~2.3 GB), `prop_pnv_v1_1km.tif` (display only) |
| Training labels: natural regrowth vs plantations 2000–2012 | Fagan, Kim et al. (2022), *The expansion of tree plantations across tropical biomes*, Nature Sustainability 5, 681–688, [doi:10.1038/s41893-022-00904-w](https://doi.org/10.1038/s41893-022-00904-w) | Maps: Global Forest Watch, "Pantropical tree plantation expansion 2000–2012" (<https://data.globalforestwatch.org/content/pantropical-tree-plantation-expansion-2000-2012/about>); input-preparation code: <https://github.com/dohyung-kim/plantation> |
| Predictors | As cited in the paper (Methods and Supplementary Table 2): 19 bioclimatic variables reduced to 5 PCs; elevation and slope (~30 m); net primary productivity (MOD17A3, 2000–2015); burned area (monthly, 2001–2017); distance to water; population density; GDP; HDI; road density; distance to urban areas; protected areas; ESA CCI land cover (2000) | Open datasets; several are on Google Earth Engine |
| Authors' code | "Available on request" from Hawthorne Beyer | Not open. We write our own implementation (independent code). Optionally email to ask. |

The spatial predictions in the paper come from a model with **biophysical variables
only**; the socioeconomic variables were used in model comparison. Training: ten
random-forest models, each on 500,000 balanced records drawn from a pool of six
million. Check the Methods and Supplementary Information 1 for details before
designing the replication.

## Scope decided

**Region: Colombia** (one of the five countries holding 52% of the potential;
manageable size; several biomes). Not the whole tropics.

## Plan: three steps, each with its own verdict

1. **Reproduction (cheap).** Sum the authors' published 30 m raster over Colombia and
   compare with the country figure in the paper (Fig. 2 / Supplementary tables).
   Question: do the published numbers follow from the published map?
2. **Independent replication.** Our own random-forest pipeline, with the open Fagan
   et al. labels and the same biophysical predictors, over Colombia. Compare predicted
   area, accuracy and spatial agreement with the authors' raster.
3. **Robustness check proposed by the evaluators.** Refit without the predictors
   measured during the outcome period (NPP, burned area, road density; land cover
   from before 2000 instead of 2000), keeping only pre-2000 or time-invariant
   predictors. If accuracy and predicted area drop clearly, that supports Evaluator 1.
   Optional: one alternative algorithm (gradient boosting), as Evaluator 2 suggests.

Report each step honestly, including what could not be reproduced and why. Any of
*confirms / qualifies / refutes* is a valid outcome.

## Constraints

- **Do not publish nanopublications yet.** Draft the FORRT chain, but publishing
  waits for Anne's decision.
- When the chain is published, cite both the paper and The Unjournal's evaluations
  (the evaluation summary DOI above).
- Use the corrected Brazil figure if Brazil is mentioned anywhere.
- Verify every quote against the PDF (`docs/verify-before-drafting.md`); no quotes from
  memory or from this file.
- Independent data and code wherever possible; say clearly where we reuse the authors'
  outputs (step 1 only).

## To do before starting

- [ ] Put the paper PDF in `paper/` as `williams-2024.pdf` (open access from Nature or
  Europe PMC; automated download was blocked from the server).
- [ ] Optionally add the Author Correction PDF and the two Unjournal evaluations.
- [ ] Then start the replication workflow (`/replication-study`), Phase 1 paper analysis.

## Context (not for publication)

The Unjournal evaluates papers by reading them; it does not rerun them. Their
evaluators' "suggested robustness checks" are effectively a replication protocol, which
is why this paper was chosen. Anne is in contact with The Unjournal's co-director
(IOSP workshop, Leiden, October 2026); the replication is independent of that workshop.
