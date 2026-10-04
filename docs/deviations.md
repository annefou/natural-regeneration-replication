# Deviations, implementation choices and extensions

Each entry states what the paper did (or did not specify), what this replication
does, and why. "Gap n" refers to the numbered list of unspecified details in
`nanopubs/drafts/00_paper_summary.md`. The code that implements each choice is
named so the entry can be checked against it.

## Scope

| # | Paper | This replication | Code |
|---|---|---|---|
| S1 | Pantropical (±25°, tropical/subtropical moist, dry, coniferous forest biomes) | **Colombia only** (GADM 4.1 level 0) ∩ RESOLVE 2017 biomes 1–3. Costa Rica is used only for a step-1 cross-check. | `02b`, `03a` |
| S2 | Authors' code "available on request" | Independent code written from the Methods and SI; the authors' code was not requested or used. Step 1 reuses the authors' published **outputs** (Zenodo 10.5281/zenodo.7428804). | all |

## Step 1 — reproduction from the published map

| # | Paper | This replication | Code |
|---|---|---|---|
| R1 | Area "calculated in Mollweide projection", country sums with GADM (2022) | Primary: **exact WGS84 ellipsoidal area** of each 0.00025° pixel. Sensitivity: sphere of radius *a* (identical to PROJ's WGS84 Mollweide) and nominal 0.09 ha per pixel. GADM 4.1, pixel-centre rule. | `03a` |
| R2 | Continuous product stored as integer % (gap 14) | Integer storage and sub-1 % NoData reported as bounds, not corrected. | `03a` |
| R3 | "Available for restoration" mask undefined (gap 13) | Candidate masks reported (continuous-valid, binary-valid, and binary-valid within our study domain). | `03a`, `03` |

## Step 2 — independent random-forest replication

| # | Paper (gap) | This replication | Why | Code |
|---|---|---|---|---|
| M1 | R `randomForest`; hyperparameters not reported (gap 1) | `sklearn.ensemble.RandomForestClassifier` with the R defaults for classification: `n_estimators=500`, `max_features=floor(sqrt(p))` (3 for 10 variables), `min_samples_leaf=1`, bootstrap n with replacement, Gini. Probability = `predict_proba` (equals R vote fraction for pure leaves). Seed 20261003. | Closest open equivalent of the default R model | `03`, `03c` |
| M2 | Categorical land cover and biome as R factors | Integer-coded (sklearn random forests have no native categorical splits) | Library limitation; affects which splits are reachable, not the information available | `03` |
| M3 | Sample sizes: 6 M pool, 1 M final model, 10 × 500 k for selection, 4.87 M validation (gap 9) | Scaled by Colombia's share of Fagan regrowth patches (188,921 / 4.78 M = 3.95 %): final model 39,523 records, selection fits 19,762, pool + validation ≈ 429 k (split 6 : 4.87), balanced by class. Validation points never used in training. | "Proportional to the paper" | `02b`, `03` |
| M4 | Class-1 points sampled in regrowth polygons with `st_sample`; class 0 "ellipsoid-aware" in the domain | Bernoulli sampling of **30 m pixels**: class 1 = pixel centre inside a Fagan `regrowth` polygon; class 0 = pixel in the class-0 domain; inclusion probability ∝ exact pixel area. Points are pixel centres. | Covariates are per pixel; equivalent to point sampling followed by pixel lookup | `02b` |
| M5 | Fagan label version not stated (gap 18) | GFW S3 GeoPackage `pantropical_tree_plantation_expansion_2000_2012.gpkg` (ETag recorded in `data/raw/sources_predictors.json`); **all confidence ranks** of `regrowth` are class 1; `plantation` and `open` polygons are excluded from class 0. | The paper excludes "forestry" polygons; `open` (low-confidence) polygons are also tree gain and not clearly class 0 | `01b`, `02b` |
| M6 | "Tree cover in 2000" threshold not reported (gap 2) | Forest = Hansen GFC v1.13 `treecover2000` ≥ **30 %** | Common GFW default | `02b` |
| M7 | 2018 tree cover construction not reported (gap 2). Peer review file, Response 35: the authors used "the 2018 layer" from the Hansen GFC **v1.6** download page; v1.6 has no tree-cover-2018 layer, only treecover2000/gain/lossyear | Forest 2018 = forest 2000 AND NOT `lossyear` 2001–2018, from GFC **v1.13** (gain not added). Changed 2026-10-04 from an earlier "(forest 2000 OR gain) AND NOT loss" | Usual GFW convention; GFW advises against combining gain and loss. Version v1.13 vs v1.6: loss 2011+ was reprocessed in later releases | `02b` |
| M8 | Forest density: "1 km² area" / "1 km radius" / "1-km circular buffer" (gap 3) | Mean forest fraction in a **1 km-radius** disk on the 30 m grid (ellipsoidal metres per degree at the tile centre) | Two of the three descriptions say radius | `02b` |
| M9 | Distance to forest: method not reported | Euclidean distance on the 30 m grid (ellipsoidal metres per degree at tile centre), computed in 1° tiles with a 0.25° margin and **truncated at 25 km** | Bounded memory; regrowth is almost always within a few km of forest | `02b` |
| M10 | Land cover: ESA CCI simplified 31 → 11 classes, mapping not listed; training year implied 2000 (gap 4) | Our 11-class mapping (`data/clean/landcover_classes.csv`); 2000 for training, 2015 for prediction | — | `02` |
| M11 | Prediction: "2015 ESA CCI combined with 2018 tree cover"; NoData for 2018 forest, open water, urban, rock/bare (gap 11) | Prediction domain = study domain − forest 2018 − ESA 2015 classes 190, 200–202, 210. Sparse vegetation (150–153) is predicted (excluded only from class-0 training). "Combined" is read as: land-cover value from ESA 2015, forest mask from 2018 tree cover. | Literal reading of the Methods | `02b` |
| M12 | Resampling of 250 m / 300 m / 1 km layers to 30 m not reported (gap 12) | **Nearest neighbour** at the 30 m pixel centre for all coarse layers (categorical and continuous) | No information is invented; same as nearest resampling | `02b` |
| M13 | Soil: "SoilGrids250m", OCDENS / PHIHOX, top 30 cm depth-weighted (gap 8) | SoilGrids **v2017** (2017-03-10 release) OCDENS and PHIHOX, depths 0/5/15/30 cm, trapezoidal mean over 0–30 cm | Matches the paper's variable names (decision by Anne) | `02` |
| M14 | Bioclim PCA: `prcomp` on 1 M random land points; loadings, centring and scaling not published (Supp. Table 1 gives only SDs) | Re-derived: 1,000,000 points uniform by area on the sphere, kept where WorldClim is valid (global land incl. Antarctica); both `prcomp` variants fitted; the one whose component SDs match Supp. Table 1 is used (`data/clean/bioclim_pca_summary.csv`). **Result: centre-only (R default, `scale.=FALSE`)** — cumulative variance 0.791 / 0.940 / 0.974 / 0.991 / 0.996 vs paper 0.780 / 0.948 / 0.976 / 0.990 / 0.994; SDs 893 / 387 / 184 / 131 / 76 vs paper 780 / 362 / 147 / 103 / 60 (~15–25 % higher, consistent with a different land sample, e.g. Antarctica or the sampling scheme). Centre + scale gives SDs 3.1 / 2.1 / 1.4 / 1.1 / 0.8 and is clearly **not** what the paper used, so the brief's "centred + scaled" was not followed. | Loadings unavailable; the data decide the variant | `02` |
| M15 | Biome predictor (Dinerstein 2017) | RESOLVE Ecoregions 2017 `BIOME_NUM`, rasterised at 30 m (pixel centre) | — | `02`, `02b` |
| M16 | Areas in Mollweide | Exact WGS84 pixel area (primary); Mollweide-sphere and 0.09 ha as sensitivity columns. Prediction areas are estimated from a **1-in-100 systematic sample** of 30 m pixels (every 10th row and column), not from a full 30 m prediction. Domain areas and the authors' map within our domain are exact 30 m sums. | Full 30 m RF prediction over ~1 G pixels was not necessary for area totals; systematic sampling error is reported | `02b`, `03` |
| M17 | No Colombia accuracy in the paper (gap 19) | Compared with the global (87.9 % / 87.8 %) and Neotropics (86.5 %) figures | — | `03` |

## Extension (not a deviation): probability calibration

The paper sums uncalibrated RF vote fractions trained on 50/50 samples ("expected
area"). Those scores are not probabilities at the true regrowth prevalence. We
additionally report (`03_analysis.py`):

- prevalence π = exact area of Fagan regrowth pixels / exact area of the sampling
  domain (class-0 ∪ class-1 masks), from the 30 m tile sums of `02b`;
- prior-shift correction p′ = π·p / (π·p + (1 − π)(1 − p)) applied to the scores,
  with expected area Σ p′ × area and area with p′ > 0.5;
- cross-check: isotonic calibration fitted on a random half of the validation set
  with case weights that reweight the balanced set to prevalence π (equivalent in
  expectation to drawing at prevalence π); never used in training;
- Brier score and reliability diagram on the other (evaluation) half, weighted
  the same way.

## Step 3 — robustness

| # | Choice | Code |
|---|---|---|
| B1 | (a) Land cover from ESA CCI **1992** (earliest year) and **1999** replaces 2000 in training; samples unchanged (class-0 domain still defined with ESA 2000) | `03c` |
| B2 | (b) 5-fold CV on the training pool: stratified random folds vs `GroupKFold` grouped by HEALPix NESTED cell (healpix-geo, `ellipsoid="WGS84"`) at depths 6, 7, 8; each fold trains on a 39,523-record draw. Distance-to-training analysis: 3-D chord distance on the WGS84 ellipsoid, 0.5 km bins to 8 km | `03c` |
| B3 | (c) Selection: 10 fits on 19,762 balanced records; importance = permutation mean decrease in accuracy on a separate 10,000-record balanced hold-out from the pool (R uses OOB permutation); rank sum; forward addition evaluated on the validation set; selected size = smallest k within 0.1 percentage points of the best. Candidates: final 10 + PC5, slope (SRTMGL1), cropland density (ESA 2000, 5 km radius), distance to urban (ESA 2000); "with" adds NPP (MOD17A3 C5.5 mean 2000–2015, NTSG), burned area (GlobFire, fraction of 204 months burned 2001–2017 at 15″), road density (GRIP4 5′). | `03c` |
| B3a | Burned area source: GlobFire monthly `FinalArea` polygons (final perimeter of each fire; `ActiveArea` = daily spread, not used), rasterised at 15″. If any GlobFire month cannot be fetched after 8 attempts, the whole series switches to MODIS MCD64A1 v061 (`Burn Date` > 0, 6 sinusoidal tiles, nearest to 15″) — the product GlobFire is built from. The source actually used is recorded in `data/raw/burned_source.json` and in `burned.nc` attributes. | `01b`, `02` |
| B3b | Road density: GRIP4 `grip4_total_dens_m_km2.asc` (5′); the zip also contains `grip4_area_land_km2.asc`, which an early version read by mistake (caught in the smoke test, fixed before the full run). | `02` |
| B4 | Not included: 10 of the 12 SoilGrids properties, distance to water (Kummu 2011 raster not distributed), protected areas (WDPA Aug 2020 not publicly archived), socioeconomic variables | — |
| B5 | (d) `HistGradientBoostingClassifier`, sklearn defaults, native categorical splits for land cover and biome. Optional rule (`optional_variants`), run after (a)–(c). | `03c` (`VARIANTS=de`) |
| B6 | (e) **SENSITIVITY check, not a fix:** bioclim PC1–PC4 from CHELSA v2.1 1981–2010 (`https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/CHELSA_bio{1..19}_1981-2010_V.2.1.tif`, files dated 2024-09-13) instead of WorldClim v2.1 1970–2000. Same PCA procedure and the same 1,000,000 sample points (WorldClim pixel centres mapped to the nearest CHELSA pixel; 998,736 valid — CHELSA ends at 84°N), same variant (centre only). The CHELSA files are strip-organised GeoTIFFs (not COGs), so the global read streams every row once over `/vsicurl/` (~5.8 GB transfer, nothing stored). Values converted with the files' scale/offset. CHELSA's period overlaps the 2000–2012 outcome period, so it is not a temporally cleaner predictor. Random and HEALPix-blocked CV are reported for it. Optional rule. | `02`, `03c` (`VARIANTS=de`) |

## Testing

Every rule was smoke-tested end to end (`snakemake --config smoke=1`: 1° tiles in
-75…-73°E, 4…6°N; 3,000 points per class; 100 trees; outputs in
`data/clean_smoke/`, `results/smoke/`, `figures/smoke/`) before the full run. The
smoke test also exercised the MCD64A1 fallback (2 months, tile h10v08) and the
Earthdata `~/.netrc` login from a detached (`setsid`) process.
