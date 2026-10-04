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
  (paper: 0.879).
- **Lower for the map's actual use**, predicting places and periods not seen in training:
  0.854–0.869 with spatially blocked validation in Colombia, 0.812–0.834 across the Neotropics, **0.767**
  for a model trained outside Colombia, and **0.717** when predicting 2012–2024 regrowth from 2000–2012
  (independent MapBiomas labels).
- **Area figures are fragile.** Summing probability × area, as the paper does, over-predicted the
  regrowth that actually followed by about seven times (7.37 vs 1.00 Mha); calibrating to the observed
  regrowth prevalence predicted 0.98 Mha. The published Supplementary Table 4 areas match a nominal
  0.09 ha pixel count, not the stated Mollweide areas.
- **Improvement found:** adding land-use history from 1985–1999 raises accuracy from 0.778 to 0.834
  (0.750 → 0.813 with blocked validation), as one of the evaluators suggested.

The full conclusion, evidence and limitations are in the Outcome draft
([`nanopubs/drafts/05_outcome.md`](nanopubs/drafts/05_outcome.md)); every implementation choice the
paper leaves open is listed in [`docs/deviations.md`](docs/deviations.md).

## What was done

| Step | Question | Notebook |
|---|---|---|
| 1. Reproduction | Do the published Colombia (and Costa Rica) areas follow from the authors' published map? | `03a` |
| 2. Replication | Does an independent implementation reach the reported accuracy? | `01b`, `02`, `02b`, `03` |
| 3. Robustness | Pre-2000 land cover, spatially blocked validation, variable selection with and without predictors measured during the outcome period, gradient boosting, CHELSA climate | `03c` |
| Diagnostics | Training on the Neotropics, transfer to Colombia, transferability curve with HEALPix blocks | `02c`, `03d` |
| Area of Applicability | Are low-accuracy predictions extrapolations (Meyer & Pebesma 2021)? | `03e` |
| Independent labels | MapBiomas Colombia: label comparison, forward test, land-use history, cross-scoring | `01c`, `02d`, `03f` |

## Data

All inputs are open and downloaded by the pipeline: the authors' outputs (Zenodo
[10.5281/zenodo.7428804](https://doi.org/10.5281/zenodo.7428804), CC BY 4.0), Fagan et al. (2022)
regrowth labels (Global Forest Watch, CC BY-NC 4.0), MapBiomas Colombia Collection 3 (CC BY 4.0),
Hansen Global Forest Change v1.13, ESA CCI Land Cover v2.0.7, SoilGrids 2017, WorldClim 2.1 and
CHELSA 2.1, RESOLVE Ecoregions 2017, GADM 4.1, GlobFire, MOD17A3 and GRIP4. Sources, versions and
checksums are recorded by the download notebooks; access notes are in
[`docs/data-access-probe.md`](docs/data-access-probe.md).

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

The FORRT nanopublication chain (claim, study design, outcome) will be listed in
[`nanopubs/PUBLISHED.md`](nanopubs/PUBLISHED.md) once published.
