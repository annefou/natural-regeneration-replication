# 04 — FORRT Replication Study

> Run the pre-flight checklist in `docs/forrt-form-fields.md` § Pre-flight checklist before drafting.
>
> **Verify code first:** read the actual reproduction script in `notebooks/03_analysis.py` before writing the methodology field. See `docs/verify-before-drafting.md`.

## Field-by-field draft

<!-- field: study -->
### Short URI suffix for study ID (text input, required)

Slug. Use kebab-case.

```
williams2024-natural-regeneration-colombia
```

<!-- field: label -->
### Label/name of replication study (text input, required)

Human-readable title.

```
Reproduction and independent replication in Colombia of the 87.9 % validation accuracy of the natural-regeneration random forest (Williams et al. 2024)
```

<!-- field: type -->
### Choose the study type (dropdown, required)

- [ ] Replication Study - replication with different methodology or conditions
- [x] Reproduction/Replication Study - study that is both, reproduction and replication
- [ ] Reproduction Study - direct reproduction: same methodology, same tools

<!-- field: claim -->
### Choose FORRT claim (search/select, required)

URI of the Claim published in step 03. Pull from `nanopubs/PUBLISHED.md`.

```

```

<!-- field: scope -->
### Describe what part of the claim is reproduced/replicated. (textarea, required)

The **scope** of the claim being tested. Which aspect, what's in/out of scope. NOT methodology. NOT results. See `docs/pico-study-outcome-levels.md`.

```
The claim tested is the paper's reported validation accuracy (87.9 %) of its random-forest model of where natural regeneration occurs, and whether that accuracy describes the model where it is used: in places and periods not seen in training. The test is restricted to Colombia, one of the five countries holding 52 % of the estimated potential.
In scope: (1) whether the authors' published map reproduces Colombia's area figures in Supplementary Tables 3 and 4; (2) whether an independent implementation reaches the reported accuracy under the paper's validation design; (3) the accuracy under spatially blocked, cross-country and forward-in-time validation, with the paper's training labels and with independent labels; (4) the robustness checks suggested by The Unjournal's evaluators (doi:10.21428/d28e8e57.5411b150).
Out of scope: the pantropical totals (215 Mha, 23.4 Gt C), the carbon estimates and the five-country share.
```

<!-- field: methodology -->
### Describe how the claim is reproduced/replicated. (textarea, required)

The **method** in plain prose. Read `notebooks/03_analysis.py` and any config files first. NOT exact numerical results.

```
Reproduction: the authors' 30 m continuous and binary maps (doi:10.5281/zenodo.7428804) were summed over Colombia (GADM 4.1) with exact WGS84 pixel areas, Mollweide areas and a nominal 0.09 ha per pixel, and compared with Supplementary Tables 3 and 4; Costa Rica was checked the same way.
Replication: new Python code. A random forest (scikit-learn set to R randomForest defaults, 500 trees) with the paper's ten biophysical predictors rebuilt from open data: forest density in a 1 km disk and distance to forest from Hansen Global Forest Change v1.13 (tree cover at least 30 %), SoilGrids 2017 organic carbon density and pH (0-30 cm), WorldClim 2.1 bioclimatic principal components 1-4, ESA CCI land cover and RESOLVE 2017 biome; natural-regrowth labels from Fagan et al. (2022). Balanced training sample (39,522 points), independent balanced validation points, and prediction with 2018 forest and 2015 land cover on every 30 m pixel, aggregated conservatively to HEALPix on WGS84.
Validation beyond the paper: random 5-fold cross-validation; spatially blocked 5-fold cross-validation with HEALPix cells of about 25 to 400 km (repeated with latitude/longitude squares); a model trained on a Neotropical sample excluding Colombia; independent labels from MapBiomas Colombia Collection 3, with a forward test (trained on 2000-2012 regrowth, tested on 2012-2024); area of applicability (Meyer and Pebesma 2021).
Robustness checks: land cover from 1992 or 1999; variable selection with and without NPP, burned area and road density; gradient boosting; CHELSA climate; land-use history 1985-1999. Expected areas are also calibrated to the observed regrowth prevalence.
Code: doi:10.5281/zenodo.23139192. Derived data: doi:10.5281/zenodo.23138775.
```

<!-- field: deviation -->
### Describe any deviations from original methodology. (textarea, optional)

What's different from the original method. Verify against the actual code, don't guess.

```
The authors' code is not public, so details the paper leaves open were chosen by us (all listed in docs/deviations.md): tree-cover threshold 30 %; forest 2018 = tree cover 2000 minus loss 2001-2018 from Hansen v1.13 (the authors used v1.6); forest density in a 1 km-radius disk; distance to forest capped at 25 km; R randomForest defaults reproduced in scikit-learn; our own mapping of ESA CCI classes to 11 classes; the climate principal components re-derived, because their loadings are not published.
The study covers Colombia only, with sample sizes scaled to Colombia's share of the regrowth patches. Predictors used only in the paper's model comparison were partly unavailable (WDPA August 2020, distance to water); they are not in the final model.
Additions not in the paper: spatially blocked, cross-country and forward-in-time validation, independent MapBiomas labels, calibration to the regrowth prevalence, land-use history, and full-resolution HEALPix aggregation.
```

<!-- field: keyword -->
### Search keywords (Wikidata) (search/select, optional)

Provide labels (not QIDs) — the Wikidata search picks up labels.

- _Label 1: ___
- _Label 2: ___

```
natural regeneration (Q11442890)
secondary forest (Q2140056)
random forest (Q245748)
remote sensing (Q199687)
land cover (Q3001793)
Colombia (Q739)
HEALPix (Q5629401)
MapBiomas (Q115768950)
```

<!-- field: discipline -->
### Search discipline (Wikidata) (search/select, optional)

Provide labels.

- _Discipline label: ___

```
forest ecology (Q2249329)
```

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 04.
