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

## Diagnostic 3 — transferability (training scope)

**Question.** Our step-2 model (Colombia only) gives an uncalibrated expected
area of 4.0 Mha against 10.3 Mha for the authors' pantropical map in the same
domain. On Colombian non-regrowth validation points, the authors' map scores
p > 0.5 on 52.6 % (mean 0.52) and ours on 6–8 % (mean 0.17–0.19). Leading
hypothesis: training scope. Diagnostic 3 trains **the same model specification**
on a Neotropical sample and tests transfer to Colombia (`02c`, `03d`). This is a
diagnostic extension, not part of the paper's method.

| # | Choice | Why | Code |
|---|---|---|---|
| D1 | **Region:** RESOLVE 2017 biomes 1–3, latitude −25…25°, longitude −120…−30° (the Americas). No realm filter; the two Nearctic pine-oak ecoregions of northern Mexico (biome 3) inside ±25° are included. | Neotropical part of the paper's domain, read literally | `02c` §1 |
| D2 | **Sub-regions:** explicit ecoregion → sub-region table in `02c` (Mesoamerica, Caribbean, Andes incl. inter-Andean dry valleys, Amazon incl. Guiana Shield, Atlantic forest incl. Brazilian Atlantic dry forests, Cerrado/Chaco edge = Chiquitano, Mato Grosso dry forests, Caatinga, Maranhão Babaçu). A seventh stratum, **"Chocó-Pacific and northern lowlands"** (Chocó-Darién, western Ecuador, Tumbes-Piura, Magdalena-Urabá, Catatumbo, Sinú, Maracaibo, Lara-Falcón, Apure-Villavicencio, Orinoco Delta), holds the ecoregions that fit none of the six named ones. Caatinga is biome 2 in RESOLVE 2017, so it is in the domain. Cerrado and Chaco proper are savanna biomes, outside the domain. | Every domain pixel must belong to a stratum; the names are taken from RESOLVE, not invented | `02c` §1 |
| D3 | **Cells:** HEALPix depth 5, NESTED, `healpix_geo` with `ellipsoid="WGS84"` (41,509 km² each, ~200 km). They are described on a 0.05° lattice by domain area, dominant sub-region and dominant biome. A cell is **eligible** if ≥ 25 % of its area is domain: 313 of the 457 cells that touch the domain. Stratum = dominant sub-region × dominant biome (14 strata with eligible cells). **48 cells** are drawn at random (seed 20261003): one per stratum, the remaining 34 allocated proportional to stratum domain area (largest-remainder, capped by eligible cells). Result: Amazon/1: 26; Cerrado-Chaco edge/2: 5; Atlantic forest/1: 3; Andes/1: 3; Mesoamerica/1: 2; one each for Cerrado-Chaco/1, Mesoamerica/2, Mesoamerica/3, Chocó-Pacific/1, Chocó-Pacific/2, Atlantic forest/2, Andes/2, Caribbean/1, Caribbean/2. Four cells touch Colombia (7486, 7613, 7650, 7657). Cell list with counts: `data/clean/neotropics_cells.parquet`; map: `figures/diag3_training_cells.png`. | Proportional allocation keeps the sample close to self-weighting by area, like the paper's area-uniform class-0 sampling. The minimum of one cell per stratum guarantees that every sub-region and biome is represented. A square-root allocation, which would weight small strata more, was not used. | `02c` §2 |
| D4 | **Labels and pixels** as in `02b` (30 m Hansen lattice, pixel-centre rule). Class 1 = domain ∩ Fagan `regrowth`. Class 0 = domain − forest 2000 − ESA CCI 2000 {150–153, 190, 200–202, 210} − all Fagan polygons. Fagan polygons are read remotely (`/vsicurl/`, every layer, per cell bbox + 0.1°). The ESA CCI 2000 exclusion is applied **at the sampled points**, not on the 30 m mask. This is equivalent, because the Bernoulli draw is independent per pixel, so dropping excluded pixels afterwards is a thinning of the same draw. | Avoids a wall-to-wall ESA mosaic; same definition | `02c` §5, §7 |
| D5 | **Sampling:** Bernoulli per 30 m pixel, inclusion ∝ exact WGS84 pixel area, with **per-cell rates**. Class 1: 3 × cap / regrowth pixels in the cell (regrowth area = geodesic area of the Fagan regrowth polygons whose representative point is in the cell). Class 0: 30 × cap / domain pixels in the cell. Then **cap = 1,000 points per class and cell** (a 1.3 × cap pre-cap before the remote soil reads), drop points with a missing predictor, and balance the classes by random reduction of the larger class. Split 75 % training pool / 25 % validation. Target: ≤ 48,000 per class. Cells with little regrowth contribute fewer class-1 points, so the realised total is reported in `neotropics_meta.json`. | Cap so that regrowth hotspots (Mesoamerica, Atlantic forest) do not dominate. Training draws of 39,523 (as step 2) leave headroom for the "excluding Colombia" pool. | `02c` §5, §11 |
| D6 | **Predictors** (2000 only; the Neotropical sample is used only for training and CV). Forest = tree cover 2000 ≥ 30 %. Forest density (1 km disk) and distance to forest (EDT, capped at 25 km) are computed on 1° tiles with a 0.25° margin, **the same code as `02b`**. Forest 2018 (02b, 2026-10-04: forest 2000 AND NOT loss 2001–2018, no gain) is not needed for the Neotropical sample. All 2018 inputs in `03d` come from the Colombian `samples.parquet` / `pred_grid.parquet` of `02b`. Soil: SoilGrids v2017 OCDENS/PHIHOX sl1–sl4, trapezoidal 0–30 cm, read remotely at the points. Bioclim: WorldClim v2.1 at the points with the **existing** PCA loadings (`data/clean/bioclim_pca_loadings.csv`, not refitted), PC1–PC4. Land cover: ESA CCI 2000 at the points, the same 31 → 11 table (`data/clean/landcover_classes.csv`). Biome: RESOLVE raster at 30 m. All coarse layers use nearest neighbour. | Same definitions as step 2 | `02c` §5–9 |
| D7 | **Check:** points inside the Colombian window are compared with the `02` layers (`data/clean/*.nc`). In the smoke test, 379 of 379 points had identical pc1–pc4, ocdens, phihox and lc_2000. The full-run check is in `neotropics_meta.json` (`consistency_check`). | Proves that the remote point reads reproduce the Colombian predictors | `02c` §10 |
| D8 | **Disk:** no wall-to-wall Hansen, SoilGrids or WorldClim. Hansen tree-cover 2000 tiles (10°) are downloaded per group (md5 from `x-goog-hash`) into `data/raw/hansen_tmp_neotropics/` and deleted once no remaining group needs them. The download aborts if free space would drop below 10 GB. The Colombian tiles in `data/raw/hansen/` are reused and kept. The 30 m lossyear and gain layers are not downloaded (not needed). | Disk constraint (~22 GB free) | `02c` §4, §6 |
| D9 | **Test 1** (`03d`). Three models with the step-2 specification (`make_rf`: 500 trees, `max_features=3`, `min_samples_leaf=1`, bootstrap, seed 20261003) and 39,523 balanced records: (i) Colombia, the exact step-2 draw; (ii) the Neotropical pool without points in GADM Colombia; (iii) the full Neotropical pool. Neotropical points on the same 30 m pixel as a Colombian validation point are removed. Reported: accuracy on the Colombian validation set (2000 inputs); mean p and share p > 0.5 on Colombian non-regrowth validation points (2000 and 2018 inputs); uncalibrated expected area and p > 0.5 area on the Colombian 1-in-100 grid (2018 inputs, exact WGS84 areas, as in `03`); Pearson r with the authors' map per pixel and at HEALPix depth 8 (cells > 50 % in the domain, as in `03`). The authors' row uses `auth_pct / 100` at the same pixels: NoData counts as 0 in areas and is excluded from point metrics, and its accuracy is computed on the validation points where the map has a value. | — | `03d` |
| D10 | **Test 2** (`03d`). 5-fold CV on the Neotropical training pool (incl. Colombia): stratified random folds vs `GroupKFold` by HEALPix cell (WGS84) at depths 6, 5, 4 (~100, ~200, ~400 km). Each fold trains on min(39,523, training-fold size) records. Reported: mean ± sd accuracy, balanced accuracy (blocked folds are not class-balanced), and mean p / share p > 0.5 on held-out non-regrowth points. Depth-5 groups coincide with the sampled cells. | Same scheme as step 3(b), at coarser block sizes | `03d` |

- **Diagnostic 3, unclassified Fagan patches:** 276 Neotropical gain polygons have a null `pred3class` (unclassified in Fagan et al. 2022, e.g. missing Sentinel-1 data). They are excluded from both classes. Colombia has none, so step 2/3 are unaffected.
