# Natural regeneration in Colombia: a replication of Williams et al. (2024)

> **Reference paper:** Williams, B. A. et al. (2024). *Global potential for natural regeneration in
> deforested tropical regions.* Nature 636, 131–137.
> [doi:10.1038/s41586-024-08106-4](https://doi.org/10.1038/s41586-024-08106-4)
> (Author Correction: [doi:10.1038/s41586-024-08481-y](https://doi.org/10.1038/s41586-024-08481-y)).
>
> **Public evaluation:** The Unjournal, two evaluations and a summary
> ([doi:10.21428/d28e8e57.5411b150](https://doi.org/10.21428/d28e8e57.5411b150)). Their suggested
> robustness checks shaped this replication's protocol.

## The claim tested

The paper maps the potential for natural forest regeneration at 30 m with a random-forest model and
reports, on page 133:

> "The validation accuracy estimate of 87.9% was based on an independent set of 4.87 million random
> points (equally stratified with respect to the two levels of the dependent variable; further details
> are provided in Supplementary Information 1 and Extended Data Fig. 4)."

We tested this accuracy claim in **Colombia**, one of the five countries holding 52 % of the estimated
potential. The pantropical totals (215 Mha, 23.4 Gt C) were not tested.

## Verdict: partially supported

```{figure} figures/main_result.png
:name: fig-main
**A.** The accuracy claim under increasingly strict validation. **B.** Colombia's area with
regeneration potential. **C.** Forward test: regrowth predicted from 2000–2012 against regrowth
observed in 2012–2024 (MapBiomas). Accuracies on balanced classes (0.5 = chance); areas with exact
WGS84 pixel areas.
```

- **Reproduced as reported.** Rebuilt with our own code, the paper's ten biophysical predictors and
  the same kind of labels (Fagan et al. 2022), the model reaches **0.887** on independent random points
  (paper: 0.879): 0.853 of regrowth points and 0.920 of non-regrowth points are classified correctly.
- **Lower for the map's actual use**, predicting places and periods not seen in training:
  0.854–0.869 with spatially blocked validation in Colombia, 0.812–0.834 across the Neotropics, **0.767**
  for a model trained outside Colombia, and **0.717** when predicting 2012–2024 regrowth from 2000–2012
  (independent MapBiomas labels).
- **Area figures are fragile.** Summing probability × area, as the paper does, over-predicted the
  regrowth that actually followed by about seven times (7.37 vs 1.00 Mha); calibrating to the observed
  regrowth prevalence predicted 0.98 Mha. The published Supplementary Table 4 areas match a nominal
  0.09 ha pixel count, not the stated Mollweide areas.
- **The training ratio sets the area, not the ranking.** Retraining with 50 %, 20 %, 5 % and 1.6 %
  (the real share) of regrowth points shrinks the uncalibrated area from 3.99 to 0.24 Mha, while the
  ranking skill (AUC 0.92–0.95) barely moves; after the prior-shift correction all four agree
  (0.19–0.24 Mha). This confirms, in Colombia, the point made by Cloud (2026, draft) for Brazil.
- **Drawing non-regrowth near regrowth ("paired" sampling) helps a little locally, not for the
  future.** It raises correct rejection of non-regrowth within 3 km of regrowth from 0.842 to 0.904,
  costs a little on random points (0.887 → 0.879), and leaves calibrated areas unchanged. It does not
  improve the forward test (0.710 vs 0.717). In a Colombia-only model, climate carries only 13–15 % of
  the predictive signal (distance to forest 57 %), so the "climate classifier" problem Cloud finds in
  the paper's pantropical sample is largely absent here.
- **No sign of leakage from the predictor years.** One evaluator suggested using only predictors observed
  before 2000. Training with 1992 or 1999 land cover instead of 2000 gives the same accuracy (0.887), as
  expected: the regrowth labels start after 2000, so 2000 land cover describes the starting state, not the
  outcome. Dropping the predictors that do overlap the outcome period (NPP, burned area, road density)
  also changes little (0.897 with them, 0.893 without).
- **Improvement found:** adding land-use *history* (MapBiomas, 1985–1999: years since the pixel was last
  forest, years under human use) raises accuracy from 0.778 to 0.834 (0.750 → 0.813 with blocked
  validation). This is extra information, not a leakage fix; it supports the evaluators' point that
  land use, not only biophysics, shapes where regrowth happens.

The full conclusion, evidence and limitations are in the Outcome draft
([`nanopubs/drafts/05_outcome.md`](nanopubs/drafts/05_outcome.md)); every implementation choice the
paper leaves open is listed in [`docs/deviations.md`](docs/deviations.md).

## What was done

| Step | Question | Notebook |
|---|---|---|
| 1. Reproduction | Do the published Colombia (and Costa Rica) areas follow from the authors' published map? | `03a` |
| 2. Replication | Does an independent implementation reach the reported accuracy? | `01b`, `02`, `02b`, `03` |
| 3. Robustness | Land cover from 1992 or 1999 instead of 2000, spatially blocked validation, variable selection with and without predictors measured during the outcome period, gradient boosting, CHELSA climate | `03c` |
| Diagnostics | Training on the Neotropics, transfer to Colombia, transferability curve with HEALPix blocks | `02c`, `03d` |
| Area of Applicability | Are low-accuracy predictions extrapolations (Meyer & Pebesma 2021)? | `03e` |
| Independent labels | MapBiomas Colombia: label comparison, forward test, land-use history, cross-scoring | `01c`, `02d`, `03f` |
| Sampling design | Training regrowth share (50 % to 1.6 %); non-regrowth drawn near regrowth ("paired", Cloud 2026) vs at random; forward test | `03h` |

## Data

All inputs are open and downloaded by the pipeline: the authors' outputs (Zenodo
[10.5281/zenodo.7428804](https://doi.org/10.5281/zenodo.7428804), CC BY 4.0), Fagan et al. (2022)
regrowth labels (Global Forest Watch, CC BY-NC 4.0), MapBiomas Colombia Collection 3 (CC BY 4.0),
Hansen Global Forest Change v1.13, ESA CCI Land Cover v2.0.7, SoilGrids 2017, WorldClim 2.1 and
CHELSA 2.1, RESOLVE Ecoregions 2017, GADM 4.1, GlobFire, MOD17A3 and GRIP4. Sources, versions and
checksums are recorded by the download notebooks; access notes are in
[`docs/data-access-probe.md`](docs/data-access-probe.md).

## Derived dataset

The aggregated maps (HEALPix NESTED on WGS84, depths 15 to 8, GRID4EARTH Zarr) and the sample tables with every
predictor, label and prediction are archived on Zenodo:
[doi:10.5281/zenodo.23138775](https://doi.org/10.5281/zenodo.23138775) (CC BY-NC 4.0).

## Running it

The full pipeline downloads tens of GB, needs a NASA Earthdata login (`~/.netrc`) and takes several
hours on 12 cores; the burned-area download alone can take hours when the PANGAEA server is busy.

```bash
git clone https://github.com/annefou/natural-regeneration-replication.git
cd natural-regeneration-replication
pixi install
pixi run snakemake --cores 12 --config smoke=1   # small end-to-end test, a few minutes
pixi run snakemake --cores 12                    # full run
```

The pages of this book are the notebooks as executed in the full run.

## Citation

- This replication: [`CITATION.cff`](CITATION.cff), DOI [10.5281/zenodo.23137988](https://doi.org/10.5281/zenodo.23137988).
- The original paper: [10.1038/s41586-024-08106-4](https://doi.org/10.1038/s41586-024-08106-4).
- The Unjournal evaluation: [10.21428/d28e8e57.5411b150](https://doi.org/10.21428/d28e8e57.5411b150).

The FORRT nanopublication chain (quote, claim, study design, outcome, citation) is published on Science Live
and listed in [`nanopubs/PUBLISHED.md`](nanopubs/PUBLISHED.md); the outcome is
[RAZF-V9BsAmgy5a0KezH74Fd78oZy2CS6PLsizcXMTNWg](https://w3id.org/sciencelive/np/RAZF-V9BsAmgy5a0KezH74Fd78oZy2CS6PLsizcXMTNWg).
