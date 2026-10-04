# 05 — FORRT Replication Outcome

> Run the pre-flight checklist in `docs/forrt-form-fields.md` § Pre-flight checklist before drafting.
>
> **Verify the actual numerical results first** by reading `results/` and `notebooks/03_analysis.py`. Don't quote numbers from memory. See `docs/verify-before-drafting.md`.

## Field-by-field draft

<!-- field: outcome -->
### Short URI suffix for outcome ID (text input, required)

Slug. Use kebab-case.

```
williams2024-rf-accuracy-colombia-outcome
```

<!-- field: label -->
### Plain-text label for the outcome (text input, required)

Descriptive title.

```
Replication outcome: 87.9 % validation accuracy of the natural-regeneration random forest (Williams et al. 2024), tested in Colombia
```

<!-- field: study -->
### Choose study (search/select, required)

URI of the Replication Study published in step 04. Pull from `nanopubs/PUBLISHED.md`.

```

```

<!-- field: repo -->
### Repository URL (text input, required)

Use the Zenodo **version DOI** URL for the release the results came from — not a
bare branch URL, and not the concept DOI.

> **Why not the bare repo URL.** `https://github.com/ORG/REPO` names a *moving
> branch*. This Outcome asserts "this code produced this number", in a signed,
> immutable record. A branch URL means that assertion points at whatever `main`
> happens to be years from now — code that may never have produced the number
> above. A concept DOI has the same flaw: it resolves to the latest version.
> The version DOI pins the exact release. `docs/chain-decision-tree.md` § Anchor
> ranks the options: SWHID > Zenodo DOI > repo URL > Wayback.
>
> Both DOIs and the SWHID are in `CITATION.cff` under `identifiers:`, recorded
> automatically at release by `.github/workflows/release-identifiers.yml`. Take
> the one described as *"Version DOI"*.

```
https://doi.org/10.5281/zenodo.23139192
```

<!-- field: date -->
### Choose completion date (text input, required)

```
2026-10-04
```

<!-- field: validationStatus -->
### Choose validation status (dropdown, required)


This dropdown maps to the CiTO intention in step 06: Validated → `confirms`, PartiallySupported → `qualifies`, Contradicted → `disputes`.

- [ ] contradicted
- [ ] inconclusive
- [ ] not tested
- [x] partially supported
- [ ] validated

<!-- field: confidenceLevel -->
### Choose confidence level (dropdown, required)

- [ ] high - Strong evidence, mostly agrees with original
- [ ] low - Limited evidence, significant disagreement
- [x] moderate - Adequate evidence, partial agreement
- [ ] very high - Extensive evidence, high agreement with original
- [ ] very low - Minimal evidence, major disagreement

<!-- field: conclusion -->
### Describe the overall conclusion about the original claim (textarea, required)

Substantive interpretation. Headline comparison: replication's number vs the paper's number, sign + significance.

```
Partially supported (Colombia). Rebuilt independently with the paper's ten biophysical predictors and the Fagan et al. (2022) regrowth labels, the random forest reproduces the reported accuracy under the paper's own validation design: 0.887 on independent balanced random points (paper: 0.879). The accuracy does not carry over to the map's intended use, predicting regeneration in places and periods not seen in training. Spatially blocked cross-validation gives 0.854–0.869 within Colombia and 0.812–0.834 across the Neotropics; a model trained outside Colombia reaches 0.767 there; and a forward test with independent MapBiomas labels (trained on 2000–2012 regrowth, tested on 2012–2024) reaches 0.717 balanced accuracy. The 87.9 % is therefore a sound measure of agreement with the training labels under random validation, but it overstates out-of-sample predictive accuracy by up to 16 percentage points. The protocol follows the robustness checks suggested in The Unjournal's evaluation of the paper (doi:10.21428/d28e8e57.5411b150).
```

<!-- field: evidence -->
### Describe the evidence that supports your conclusion (textarea, required)

Numerical results, test statistics, model coefficients. Read directly from `results/`.

```
All values Colombia; accuracy on balanced classes (0.5 = chance); areas with exact WGS84 pixel areas.
Reproduction (authors' Zenodo rasters, 10.5281/zenodo.7428804): Supp. Table 4 binary area 13.70 Mha is matched only by a nominal 0.09 ha pixel count (13.699 Mha; exact area 11.642 Mha), as in Costa Rica (1.219 vs 1.22 Mha; exact 1.026). Supp. Table 3 expected area 11.19 Mha is not reproduced (10.514 Mha exact; 10.584 Mha Mollweide).
Replication, Fagan labels: validation accuracy 0.887, out-of-bag 0.884; random 5-fold CV 0.887; HEALPix-blocked CV (WGS84, ~100/50/25 km) 0.854/0.864/0.869.
Robustness: land cover 1992 or 1999 instead of 2000: 0.887/0.887. Variable selection with or without NPP, burned area and road density: 0.897/0.893. Gradient boosting: 0.877.
Transfer: Neotropical sample, random CV 0.892; blocked ~100/200/400 km 0.834/0.823/0.812. Trained without Colombia, tested on Colombia: 0.767.
Independent labels (MapBiomas Colombia Collection 3): random 0.778, blocked ~100 km 0.750; forward 2012–2024: 0.717 (AUC 0.80). Adding land-use history 1985–1999: 0.834 (blocked 0.813).
Area (every 30 m pixel of the prediction domain): authors' map 10.28 Mha; our model 3.93 Mha as probability × area, 0.18 Mha calibrated to the regrowth prevalence. Forward test: probability × area predicted 7.37 Mha of 2012–2024 regrowth against 1.00 Mha observed; calibrated with the 2000–2012 prevalence, 0.98 Mha.
Authors' map: scores > 0.5 on 52.6 % of Colombian non-regrowth validation points (Fagan labels) and 52.0 % (MapBiomas labels); our model 5.8 %. The spatial pattern agrees (HEALPix depth 8, r = 0.86).
Data and code: derived dataset doi:10.5281/zenodo.23138775 (HEALPix maps and sample tables); software doi:10.5281/zenodo.23139192.
```

<!-- field: limitations -->
### Describe what limits the conclusions of the study (textarea, optional)

Honest caveats. If the result is partial or contradicted, say so plainly. Don't overclaim.

```
Colombia only: the pantropical figures (215 Mha, 23.4 Gt C) and the five-country share were not tested.
Independent implementation: the authors' code is not public, so details the paper leaves open were chosen by us (tree-cover threshold 30 %, random-forest settings set to R defaults, 1 km-radius forest density, newer data versions such as Hansen GFC v1.13 instead of v1.6); all are listed in docs/deviations.md. Why the authors' map scores about 2.6 times higher than any of our models remains unexplained.
Labels: Fagan and MapBiomas regrowth disagree strongly (1.25 Mha vs 0.25 Mha of 2000–2012 regrowth; only 16 % of Fagan regrowth area is MapBiomas regrowth). Both are satellite classifications; no independent reference sample was available (MapBiomas' interpreted validation points are not public), so absolute regrowth areas and on-the-ground accuracy remain uncertain.
Predictors only used in the paper's model comparison could not all be obtained (WDPA August 2020, distance to water); they are not in the final model. Our model's areas use every 30 m pixel; the 1-in-100 systematic sample used in step 2 over-estimated them by 0.7–1.5 % (cause not identified), which changes no conclusion. The calibrated areas assume the label prevalence equals the true prevalence.
```

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 05.
