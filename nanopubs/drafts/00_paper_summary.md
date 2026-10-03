# Paper summary

> This is a working scratchpad for the paper-analysis phase. The output of this file feeds the Quote / AIDA / Claim drafts. It is not itself a nanopub.

**Reference paper:** Global potential for natural regeneration in deforested tropical regions

**DOI:** 10.1038/s41586-024-08106-4 (Nature 636, 131–137, 5 December 2024; received 4 August 2023, accepted 24 September 2024, published online 30 October 2024; open access, CC BY-NC-ND 4.0)

**Author Correction:** 10.1038/s41586-024-08481-y (Nature 637, E9, published online 6 December 2024). In Fig. 2, Brazil's area with potential for natural regeneration was corrected from 55.1 Mha to 43.71 Mha. The corrected figure is used everywhere below. Note: Supplementary Table 3 as downloaded already lists Brazil = 43.71 Mha; Supplementary Table 4 (binary >50 % sensitivity) lists Brazil = 55.12 Mha, which suggests the original Fig. 2 error took the Brazil value from the binary table.

**Authors (19, as printed in the PDF):** Brooke A. Williams, Hawthorne L. Beyer, Matthew E. Fagan, Robin L. Chazdon, Marina Schmoeller, Starry Sprenkle-Hyppolite, Bronson W. Griscom, James E. M. Watson, Anazélia M. Tedesco, Mariano Gonzalez-Roglich, Gabriel A. Daldegan, Blaise Bodin, Danielle Celentano, Sarah Jane Wilson, Jonathan R. Rhodes, Nikola S. Alexandre, Do-Hyung Kim, Diego Bastos & Renato Crouzeilles. Williams and Beyer contributed equally. Corresponding author: Brooke A. Williams. Code contact: Hawthorne L. Beyer (code "available on request").

**Year:** 2024

**Sources read (all in `paper/`):** `williams-2024.pdf` (12 pages, source of truth); `williams-2024-author-correction.pdf`; `williams-2024_PMC11618091.xml` (Europe PMC JATS, text aid only); `supplementary/` MOESM1 (SI 1, extended methods, Tables A1.1–A1.5), MOESM2 (Reporting Summary, 3 pages), MOESM3 (Supp. Table 1, bioclim PCA), MOESM4 (Supp. Table 2, data sources), MOESM5 (Supp. Table 3, per-country continuous potential), MOESM6 (Supp. Table 4, per-country binary >50 %), MOESM7 (Peer Review File).

**PDF vs JATS XML discrepancies:** none found in the candidate sentences (abstract, results, Methods study-region sentence are identical after whitespace normalisation). The XML additionally contains a one-sentence editorial summary ("An estimated area of 215 million hectares has the potential for natural forest regeneration across tropical forested countries and biomes, representing an above-ground carbon sequestration potential of 23.4 Gt C.") that is **not** printed in the PDF; do not quote it.

## Headline claim

Primary (abstract, PDF p. 131):

> We estimate that an area of 215 million hectares—an area greater than the entire country of Mexico—has potential for natural forest regeneration, representing an above-ground carbon sequestration potential of 23.4 Gt C (range, 21.1–25.7 Gt) over 30 years.

255 characters. Copied into `01_quote.md`. Verification: the `forrt-research` MCP `verify_quote` tool was **not available** in this session, so the quote is formally **unverified** by that tool. It was checked mechanically instead: it matches the `pdftotext` extraction of the PDF after whitespace normalisation only (equivalent to the `normalized` tier; no character changed), and matches the JATS XML. Re-run `verify_quote(pdf_path="paper/williams-2024.pdf", quotation=...)` before publishing.

Alternative candidates (same verification status, all `normalized`):

- B, 52 % five countries (abstract, p. 131, 211 characters):
  > Five countries (Brazil, Indonesia, China, Mexico and Colombia) account for 52% of this estimated potential, showcasing the need for targeting restoration initiatives that leverage natural regeneration potential.
- C, 87.9 % accuracy (results, p. 133, 268 characters):
  > The validation accuracy estimate of 87.9% was based on an independent set of 4.87 million random points (equally stratified with respect to the two levels of the dependent variable; further details are provided in Supplementary Information 1 and Extended Data Fig. 4).

Why A is primary: it is the paper's core empirical estimate, it is in the abstract, and step 1 (raster sum) and step 2 (independent pipeline) both produce an area number that maps onto it (via the Colombia share). B is the natural anchor if the chain is scoped to Colombia's share; C is the anchor for step 3 (accuracy under temporally clean predictors). Each could be its own atomic AIDA.

## Methodology summary

**Data sources**

- **Dependent variable / training labels:** Fagan et al. (2022), *The expansion of tree plantations across tropical biomes*, Nat. Sustain. 5, 681–688 (ref. 22). Natural regrowth patches ≥0.45 ha with vegetation >5 m tall, gained 2000–2012 and persisting to 2016 ("hereafter, 2000–2016"), derived from the 30 m Global Forest Watch (Hansen et al. 2013) tree-cover time series and separated from plantations by a machine-learning classifier. Pantropical total: 31.6 ± 11.9 Mha of natural regrowth in 4.78 million patches (mean patch 1.2 ha); the model used "a sample of 5.4 Mha of natural regrowth detected previously". Fagan accuracy (humid biome, SI Table A1.1): overall three-class 90.6 % (±0.7); natural-regrowth class 88.9 % (±1.2); regrowth producer's accuracy 78.8 (±5.6) by count but 18.7 (±5.4) area-based; user's accuracy 85.1 ± 5.6.
- **Predictors** (Methods + Supp. Table 2; full timing table below): ESA CCI land cover (300 m, annual 1992–2015, simplified 31 → 11 classes) plus derived cropland density (5 km circular buffer) and distance to urban area; GFW/Hansen tree cover (30 m) → forest density (1 km buffer) and distance to forest; WDPA protected areas (Aug 2020 version + 768 Chinese PAs from June 2017); biome (Dinerstein et al. 2017); SoilGrids250m (12 properties, depth-weighted mean of top 30 cm); slope from SRTM (1 arc-second ≈ 30 m); NPP (MOD17A3, annual 2000–2015, 1 km²); WorldClim v2.1 1970–2000 (19 bioclim at 30 arc-seconds ≈ 1 km → 5 PCs, 99.4 % variance, PCA via R `prcomp` on 1 million random land points; Supp. Table 1); GHS-POP population (R2019A, epochs 1975/1990/2000/2015); GDP and HDI gridded (Kummu et al. 2020, 1990–2015); road density (GRIP, Meijer et al. 2018, m per km²); burned area (Artés et al. 2019, monthly average 2001–2017); distance to water (Kummu et al. 2011). GADM (2022) for country summaries. Carbon: Cook-Patton et al. (2020) 1 km accumulation rates.
- **Study region / sampling domain:** tropical and subtropical dry broadleaf, moist broadleaf and coniferous forest biomes (Dinerstein et al. 2017) within ±25° latitude. Non-regeneration (0) domain = those biomes excluding (1) ESA CCI **2000** classes water (210), bare (200), urban (190), sparse vegetation (150); (2) areas already tree cover in GFW 2000; (3) natural-regeneration polygons; (4) forestry/plantation polygons (Fagan et al.).

**Model**

- R `randomForest` (Liaw & Wiener 2002), binary classification (regeneration 1 / no regeneration 0). Pool: 6 million random points, stratified equally by class, no other spatial stratification; regen points sampled inside polygons with `sf::st_sample`; non-regen points sampled ellipsoid-aware within ±25° and filtered by the domain rules; after removing records with any NoData covariate: **5.8 million** records (SI 1).
- Variable selection: ten RF models, each on 500,000 balanced records, "with all biophysical covariates"; variables ranked by mean decrease in accuracy, overall rank = sum of rank positions across the ten models; then models of increasing size adding one variable at a time until accuracy stops improving (Extended Data Fig. 1: ten variables). Ten 500k models: accuracy mean 0.886, range 0.885–0.887 (SI 1).
- **Final spatial model: ten biophysical variables**, fit on **1 million** records: forest density, distance to forest, soil organic carbon (density; Ext. Data Fig. 3 label "OCDENS"), soil pH ("PHIHOX"), bioclim PC1–PC4, land use/cover, biome (SI 1; Extended Data Fig. 2). **NPP, burned area, road density, slope, cropland density, distance to urban, distance to water, protected areas, population, GDP, HDI are not in the final spatial model.**
- Model comparison: socioeconomic + biophysical vs biophysical only, accuracy 0.886 vs 0.880; spatial predictions use biophysical only.
- Prediction: covariates of the final model, with three updated to contemporary conditions — forest density and distance to forest from GFW **2018** tree cover (30 m), land use/cover from **2015** ESA CCI combined with that tree cover. 2018 forest, open water, urban, rock/bare set to NoData. Output: probability 0–1 per 30 m cell, labelled "present (2015)" and "near future (2030)" under the assumption that 2000–2016 conditions persist.
- Threshold: none for the headline area (continuous). Binary products and sensitivity analysis use >0.5.
- Area: 30 m × 30 m cell area × probability ("weighted-area" = expected area), computed in Mollweide projection; country sums with GADM. Sensitivity: cells >50 % counted whole → 263 Mha.
- Carbon: Cook-Patton et al. (2020) t C ha⁻¹ yr⁻¹ at 1 km, ×30 (years), resampled to 30 m, ×0.09 ha per cell, × probability, summed.
- Validation: independent balanced random points; accuracy also examined in 16 × 0.5 km distance-to-training intervals (>95 % at 0–0.5 km, 81.4 % at 2.0–2.5 km; bootstrapped balanced accuracy never below 75 %).

**Sample sizes:** 6 M pool → 5.8 M usable records; 10 × 500,000 for variable selection; 1,000,000 for the final model; validation 4.87 M points (main text, Extended Data Fig. 4) — but SI Tables A1.2–A1.5 report n = 1,000,000 reference points; 62,493 random 30 m cells for the forest-edge statistic; 3,533,732 training points with potential >0.5 (Fig. 3).

**Headline numerical results**

| Quantity | Value (as printed) | Location |
|---|---|---|
| Area with potential (continuous, expected) | 215 Mha (CI 214.78–215.22) | abstract; results p. 132 |
| — Neotropics / Indomalayan / Afrotropics | 98 Mha (97.80–98.20) / 90 Mha (89.82–90.18) / 25.5 Mha (25.47–25.53) | p. 132 |
| Area, binary >50 % sensitivity | 263 Mha | Methods; Supp. Table 4 (sum 263.21) |
| Above-ground carbon over 30 yr | 23.4 Gt C (range 21.1–25.7) | abstract; p. 134 |
| — Neotropics / Indomalayan / Afrotropics | 11.1 Gt (10.0–12.2) / 5.42 Gt (4.87–5.96) / 3.1 Gt (2.83–3.37) | p. 134 (note: these sum to 19.6, not 23.4 Gt; not explained) |
| Five countries' share | 52 % (Brazil 20.3, Indonesia 13.6, China 7.2, Mexico 5.6, Colombia 5.2 %) | abstract; p. 132; Discussion |
| Five countries' share of study region land | 24.3 % | p. 132 |
| Validation accuracy | 87.9 % (4.87 M balanced points) | p. 133 |
| Out-of-bag accuracy | 87.8 % | p. 133 |
| Biome validation accuracy | moist 87.9 %, dry 87.9 %, coniferous 87.8 % | p. 133 |
| Global confusion matrix (SI A1.2) | OA 87.8 ± 0.064 %; area-based OA 84.8 ± 0.001; REG user's acc. 92, producer's acc. 85.2 (count) / 38 (area-based); REG map area 215 Mha vs **corrected area 512 Mha** | SI 1 |
| Neotropics (SI A1.3) | OA 86.5 ± 0.107; area-based 82.5; REG map 98.78 Mha, corrected 276 Mha | SI 1 |
| Afrotropics (SI A1.4) | OA 92.2 ± 0.0948; area-based 93.3; REG map 25.5 Mha, corrected 43 Mha | SI 1 |
| Indomalaysia (SI A1.5) | OA 85.1 ± 0.127; area-based 72.9; REG map 90.86 Mha, corrected 275 Mha | SI 1 |
| Model comparison | socio+bio 0.886 vs bio-only 0.880 | p. 133; SI 1 |
| Forest-edge statistic | 98.1 % of cells with potential >0.5 within 300 m of forest edge | p. 133 |

Regional accuracy in the main text: "Spatial variation in accuracy was considerable, with the lowest accuracies occurring in portions of Southeast Asia" (no per-country accuracy reported; nothing for Colombia).

### Colombia-specific numbers

| Source | Available for restoration (M. ha) | PNR (M. ha) | Proportion available for restoration with PNR |
|---|---|---|---|
| Supp. Table 3 (continuous, expected area), row "Colombia \| Neotropics" | 93.78 | 11.19 | 0.12 |
| Supp. Table 4 (binary >50 %), row "Colombia \| Neotropics" | 93.78 | 13.70 | 0.15 |

- Share of global potential: 5.2 % (text, p. 132 and Discussion). Check: 11.19 / 215.14 (sum of Supp. Table 3) = 5.20 %.
- Fig. 2 (p. 133): Colombia appears in the Neotropics sector; the green-bar labels printed there include 11.19 (and 11.99 for Mexico), consistent with Supp. Table 3. The yellow "available" bar for Colombia carries no printed number.
- Fig. 1b: map inset of Colombia (example of potential and 2018 tree cover); no number.
- Colombia carbon potential: **not reported**. Colombia accuracy: **not reported**.
- Country boundary: GADM (2022). The "available for restoration" denominator is the domain area within the study biomes; its exact definition at prediction time is not fully specified (see gaps).

### Predictor timing relative to the outcome period (labels: gain 2000–2012, persisting to 2016)

| Predictor | Source, resolution | Time window used (as reported) | Relative to outcome | In final 10-variable spatial model? |
|---|---|---|---|---|
| Forest density (mean forest within 1 km buffer) | GFW/Hansen tree cover, 30 m | 2000 for training; 2018 for prediction | baseline at start (training); after (prediction) | yes |
| Distance to forest | GFW/Hansen, 30 m | 2000 training; 2018 prediction | baseline / after | yes |
| Land use/cover (11 classes) | ESA CCI, 300 m (annual 1992–2015 available) | implied 2000 for training ("updated … rather than conditions in 2000"); 2015 ESA CCI + 2018 tree cover for prediction. Peer-review response says "2015 ESA CCI dataset" — ambiguous | start of period (training) | yes |
| Cropland density (5 km buffer) | ESA CCI, 300 m | year not reported | unknown | no |
| Distance to urban | ESA CCI, 300 m | year not reported | unknown | no |
| Soil OC density, soil pH (+10 other soil props evaluated) | SoilGrids250m (Hengl 2017), 250 m | not dated (static product) | time-invariant (treated as such) | SOC, pH: yes |
| Bioclim PC1–PC5 | WorldClim v2.1, 30 arc-s (~1 km) | 1970–2000 | before | PC1–4: yes |
| Biome | Dinerstein et al. 2017 | static | time-invariant | yes |
| Slope | SRTM 1 arc-s (~30 m) | acquisition date not reported | time-invariant | no |
| Net primary productivity | MOD17A3, 1 km² | mean annual 2000–2015 | **during** | no |
| Burned area | Artés et al. 2019; resolution not reported | monthly average 2001–2017 | **during** | no |
| Road density | GRIP (Meijer 2018), m per km² | reference year not reported | **during/after** (dataset published 2018) | no |
| Protected areas | WDPA Aug 2020 + China June 2017 | 2020 snapshot | **after** | no |
| Distance to water | Kummu et al. 2011 | not reported | probably static | no |
| Population density | GHS-POP R2019A | epochs 1975/1990/2000/2015; epoch used not reported | unknown | no (socioeconomic) |
| GDP, HDI | Kummu et al. 2020 | 1990–2015; year(s) used not reported | unknown | no (socioeconomic) |

**Consequence for step 3:** Evaluator 1's "concurrent predictor" critique (NPP, burned area, road density) applies to the variable-selection and model-comparison stages (the 0.886 / 0.880 accuracies) but those three variables are **not in the final ten-variable model** that produced the 215 Mha map and the 87.9 % / 87.8 % accuracies (SI 1; Extended Data Fig. 2). Dropping them from the final model is therefore a no-op. The temporally relevant levers in the final model are (a) land cover year (2000 vs a pre-2000 year such as 1992–1999), (b) the 2000 → 2018 tree-cover / 2000 → 2015 land-cover swap between training and prediction, and (c) spatial autocorrelation in validation. Step 3 should be reframed accordingly; also re-run the variable selection with and without NPP / burned area to see whether ranking or accuracy changes. WDPA 2020 (post-period) is a further temporally misaligned candidate not mentioned by the evaluators.

## Replication design choice

Which of the three FORRT Study Types fits this replication?

- [ ] **Reproduction Study** — direct reproduction: same methodology, same tools.
- [ ] **Replication Study** — replication with different methodology or conditions.
- [x] **Reproduction/Replication Study** — both.

Step 1 is a Reproduction: sum the authors' published 30 m continuous raster (`pnv_pct_30m` tiles, integer percentages; Zenodo 10.5281/zenodo.7428804) over Colombia (GADM boundary, Mollweide cell areas) and compare with Supp. Table 3 (11.19 Mha), and the binary product (`pnv_bin_30m`) with Supp. Table 4 (13.70 Mha). This tests whether the published numbers follow from the published map; it reuses the authors' output, not their code (which is only "available on request"). Steps 2–3 are a Replication: an independent random-forest pipeline over Colombia written from the Methods, using the open Fagan et al. labels and the same ten biophysical predictors, compared on accuracy, predicted area and spatial agreement; then a robustness variant with temporally clean predictors (pre-2000 land cover; NPP / burned area / road density excluded at the selection stage) as Evaluator 1 proposed. Regional subsetting, independent code and unspecified details (below) mean steps 2–3 are a Replication with different conditions, not a reproduction.

## Details needed for the independent implementation that the paper does not specify

1. **RF hyperparameters**: `ntree`, `mtry`, `nodesize`, class weights, and whether `predict` used vote fraction — not reported.
2. **Forest definition from GFW tree cover**: canopy-cover % threshold for "tree cover in 2000/2018", for forest density and distance to forest, and how 2000–2018 loss/gain was combined to build the 2018 layer — not reported.
3. **Forest density window**: described inconsistently as "a 1 km2 area" (p. 132), "within a 1 km radius" (p. 133), "1-km circular buffer" (Methods, Fig. 3). Radius vs area differ ~3×.
4. **Land cover year used in training**: implied 2000 (Methods) vs "2015 ESA CCI" (peer-review response). Also how the 2015 ESA CCI was "combined" with 2018 tree cover for prediction — not specified.
5. **Year/epoch of cropland density, distance to urban, population, GDP, HDI, road density, distance to water**; burned-area resolution and aggregation — not reported.
6. **Which variables count as "biophysical"** vs "socioeconomic" (e.g. cropland density, distance to urban, protected areas, road density) — no explicit list.
7. **Full variable-importance ranking** and the accuracy curve values (only figures; Extended Data Figs. 1–2), so the selection cannot be checked numerically.
8. **Soil variable naming**: Methods list "organic carbon (g per kg)" but the model uses "OCDENS" (density); depth weighting described but source version of SoilGrids not stated.
9. **Validation set size**: 4.87 M points (main text, Ext. Data Fig. 4) vs n = 1,000,000 in SI Tables A1.2–A1.5; whether validation points were excluded from the 5.8 M pool used for training draws.
10. **"87.9 % based on autocorrelation effects"** — how this number differs from the 87.8 % OA in Table A1.2 is not explained; spatial-block or distance-based validation scheme not formalised.
11. **Prediction-time NoData mask**: "open water, urban areas and rock or bare ground" — sparse vegetation (code 150) is excluded from the training domain but not listed at prediction; ESA year and resampling of 300 m to 30 m (nearest? which grid alignment?) not reported.
12. **Resampling/alignment** of all coarse predictors (250 m, 300 m, 1 km) to the 30 m grid — method not reported.
13. **"Available for restoration" area** in Supp. Tables 3–4: exact mask and year not defined.
14. **Integer-percent storage** of the published raster: rounding effect on area sums not discussed (relevant for step 1).
15. **Carbon range 21.1–25.7 Gt**: how the range was derived (Cook-Patton uncertainty layer?) — not reported; regional carbon values (11.1 + 5.42 + 3.1 = 19.6 Gt) do not sum to 23.4 Gt.
16. **CI on area (214.78–215.22 Mha)**: method not reported.
17. **Random seeds, software versions** (R, randomForest, sf) — not reported; code on request only.
18. **Fagan et al. label version**: which release of the "Pantropical tree plantation expansion 2000–2012" map, and how the "forestry activity" exclusion polygons were defined — cited, not specified.
19. **Colombia-level accuracy** and carbon: not reported, so step 2 has no paper-side Colombia accuracy to compare against (use SI A1.3 Neotropics: OA 86.5 %).

## Notes for downstream drafts

- Quote A uses an em dash (—) and an en dash in "21.1–25.7"; keep both characters.
- "215 million hectares" is the **expected** (probability-weighted) area, not a thresholded area; the binary >0.5 estimate is 263 Mha. A Colombia replication should compare like with like (11.19 Mha continuous; 13.70 Mha binary).
- The SI confusion matrix itself gives a corrected REG area of 512 Mha globally vs a mapped 215 Mha (area-based producer's accuracy 38 %), which bears on Evaluator 2's omission-error critique and on any AIDA that treats 215 Mha as an estimate of the true area.
- "Five countries … 52 %" is arithmetic on the same map; for Colombia the comparable atomic claim is "Colombia accounts for 5.2 %" / 11.19 Mha.
- Paper frames prediction as 2015 / 2030 potential under 2000–2016 conditions; the AIDA should not call it a forecast without that assumption.
- Geographic scope: spatial paper — see `09_geo_coverage.md` (records the original pantropical study area, not our Colombia subset).
