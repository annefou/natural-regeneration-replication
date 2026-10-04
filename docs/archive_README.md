# Natural-regeneration potential in Colombia on HEALPix (WGS84)

Derived dataset of the replication *natural-regeneration-replication* (software:
[doi:10.5281/zenodo.23137989](https://doi.org/10.5281/zenodo.23137989)), which tests the 87.9 %
accuracy claim of Williams et al. (2024), *Global potential for natural regeneration in deforested
tropical regions*, Nature 636, 131–137 ([doi:10.1038/s41586-024-08106-4](https://doi.org/10.1038/s41586-024-08106-4)),
in Colombia. The replication protocol follows the robustness checks suggested in The Unjournal's
evaluation of the paper ([doi:10.21428/d28e8e57.5411b150](https://doi.org/10.21428/d28e8e57.5411b150)).

## Contents

| Path | What |
|---|---|
| `natural_regeneration_colombia_healpix.zarr.zip` | Zarr v3, GRID4EARTH layout, HEALPix NESTED on WGS84 (one uncompressed zip; open it directly or unzip to a `.zarr` folder) |
| `tables/` | Parquet tables: sample points and prediction grid with every predictor, label and prediction |
| `sources.json` | Input datasets (DOIs or URLs, versions, checksums) as recorded by the download notebooks |
| `checksums.csv` | SHA-256 of every file in this record |

### Zarr: three layer families

All variables are **areas in m² summed over the cell** (`cell_methods: "area: sum"`), so any coarser
level is the exact sum of its four children (`parent = cell >> 2`). Each level group carries the
`dggs` convention attributes and a CF grid-mapping variable `crs`; each family states its
`support`.

| Group | Levels | Support | Variables |
|---|---|---|---|
| `measurements/full_resolution` | 15 (~200 m) → 8 (~25 km) | every 30 m pixel, binned by pixel centre, value × exact WGS84 pixel area | country, study-domain and prediction-domain area; our model's expected area (uncalibrated and prevalence-calibrated) and area with p > 0.5; the authors' published map (expected, valid and binary area); Fagan et al. (2022) regrowth |
| `measurements/sample_based` | 12 (~1.6 km) → 8 | 1-in-100 systematic sample of pixels (each point = 10 × 10 pixels) | `n_points`; MapBiomas Colombia regrowth and labelled area for 2000–2012 and 2012–2024 |
| `measurements/area_of_applicability` | 8 | the same sample | area outside the area of applicability (Meyer & Pebesma 2021) of three models, shares and mean dissimilarity |

Totals over Colombia (Mha): prediction domain 18.79; our expected area 3.93 (uncalibrated) and 0.18
(calibrated); the authors' map 10.51 (whole country); Fagan regrowth in the study domain 0.25.

### Reading it

```python
import xarray as xr, xdggs, zarr
store = zarr.storage.ZipStore("natural_regeneration_colombia_healpix.zarr.zip", mode="r")
dt = xr.open_datatree(store, engine="zarr", consolidated=True)
ds = dt["measurements/full_resolution/12"].ds.drop_vars("crs").dggs.decode()
print(ds.dggs.grid_info)          # HEALPix level 12, nested, WGS84
share = ds.expected_area_uncalibrated / ds.prediction_domain_area
```

## Methods in brief

- **Our model:** random forest (scikit-learn, R `randomForest` defaults), trained in Colombia on
  Fagan et al. (2022) natural-regrowth labels with the paper's ten biophysical predictors (forest
  density and distance to forest from Hansen GFC v1.13 at ≥ 30 % tree cover; SoilGrids 2017 organic
  carbon density and pH; WorldClim 2.1 bioclimatic PC1–4; ESA CCI land cover; RESOLVE biome), and
  predicted with 2018 forest and 2015 land cover, as in the paper. Calibrated values apply a prior
  shift to the Fagan regrowth prevalence (1.64 %).
- **MapBiomas labels:** Collection 3 annual land cover, with MapBiomas Colombia's secondary-vegetation
  rule (anthropic two years, then natural three years) and forest in every year of the window.
- **Known limitation:** the step-2 analysis used the 1-in-100 sample for areas, which over-estimated
  them by 0.7–1.5 %; the full-resolution layers here are the reference.
- Full documentation, code and the list of implementation choices: the software record above and
  `docs/deviations.md` in it.

## Licence and attribution

**CC BY-NC 4.0.** Layers derive from inputs whose licences restrict commercial use: Fagan et al.
(2022) regrowth labels (CC BY-NC 4.0) and the GADM 4.1 country boundary (non-commercial use). Other
inputs: Williams et al. (2024) map, Zenodo [10.5281/zenodo.7428804](https://doi.org/10.5281/zenodo.7428804)
(CC BY 4.0); MapBiomas Colombia Collection 3 (CC BY 4.0); Hansen et al. (2013) Global Forest Change
(CC BY 4.0); ESA CCI Land Cover; SoilGrids 2017; WorldClim 2.1; RESOLVE Ecoregions 2017. Please cite
the original datasets as well as this record.

Creator: Anne Fouilloux, LifeWatch ERIC, ORCID [0000-0002-1784-2920](https://orcid.org/0000-0002-1784-2920).
