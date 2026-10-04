# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05b — Dataset archive: GRID4EARTH Zarr (HEALPix, WGS84) + Parquet tables
#
# Builds `archive/` for a Zenodo dataset deposit, separate from the software release.
#
# **Zarr** (`natural_regeneration_colombia_healpix.zarr`, Zarr v3), HEALPix NESTED on
# WGS84, GRID4EARTH layout: one multiscale group per layer family, numeric level
# groups, `dggs` convention attributes and a CF grid-mapping variable `crs`, both
# from `healpix_connector.dggs_zarr`. Every variable is an area in m² summed over the
# cell (`cell_methods: "area: sum"`), so coarser levels are exact parent sums
# (`cell >> 2`). Each family declares its support:
#
# | Family | Levels | Support |
# |---|---|---|
# | `full_resolution` | 15 → 8 | every 30 m pixel, from `05a` (conservative, exact WGS84 pixel areas) |
# | `sample_based` | 12 → 8 | the 1-in-100 systematic sample of pixels (each point stands for 10 × 10 pixels); MapBiomas labels |
# | `area_of_applicability` | 8 | the sample, from `03e` (already aggregated) |
#
# **Parquet tables**: the sample points with every predictor and label, the
# prediction grid with predictors, step-2 predictions and MapBiomas labels, so the
# models can be refitted without downloading the ~21 GB of inputs.
#
# Inputs are not redistributed; `sources.json` lists them with DOIs or URLs and
# checksums.

# %%
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from healpix_connector.dggs_zarr import cf_grid_mapping_attrs, dggs_attrs
from healpix_resample import ConservativeResampler

torch.set_num_threads(8)

# %%
SMOKE = os.environ.get("SMOKE", "0") == "1"
RAW = Path("../data/raw")
CLEAN = Path("../data/clean_smoke" if SMOKE else "../data/clean")
DERIVED = Path("../data/derived_smoke" if SMOKE else "../data/derived")
RESULTS = Path("../results/smoke" if SMOKE else "../results")
OUT = Path("../archive_smoke" if SMOKE else "../archive")
ZARR = OUT / "natural_regeneration_colombia_healpix.zarr"
FULL_DEPTH, SAMPLE_DEPTH, MIN_DEPTH = 15, 12, 8
GRID_W = 100  # each systematic grid point stands for 10 x 10 pixels
VERSION = "1.0.0"
SOFTWARE_DOI = "10.5281/zenodo.23137989"  # v1.0.0 of the replication software

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "tables").mkdir(parents=True)

FULL_META = {
    "country_area": "Area of Colombia (GADM 4.1 level 0, pixel centres inside)",
    "study_domain_area": "Area of the study domain: Colombia within RESOLVE 2017 biomes 1-3 (tropical/subtropical forests)",
    "prediction_domain_area": "Area of the prediction domain: study domain minus forest 2018 and ESA CCI 2015 urban/bare/water, with complete predictors",
    "expected_area_uncalibrated": "Expected area of natural regeneration, sum of p x pixel area, p = random-forest vote fraction (step-2 model, Fagan et al. 2022 labels, 2018 inputs)",
    "expected_area_calibrated": "Expected area of natural regeneration with p calibrated to the regrowth prevalence (prior shift)",
    "area_p_gt05": "Area of pixels with uncalibrated p > 0.5",
    "authors_expected_area": "Williams et al. (2024) published map (Zenodo 10.5281/zenodo.7428804): sum of potential (%)/100 x pixel area",
    "authors_valid_area": "Area where the published continuous map has a value (0-100 %)",
    "authors_binary_area": "Area where the published binary map equals 1 (potential > 0.5)",
    "fagan_regrowth_area": "Natural regrowth 2000-2012 persisting to 2016 (Fagan et al. 2022) inside the study domain",
}
SAMPLE_META = {
    "n_points": ("1", "Number of systematic sample points (1 per 10 x 10 pixels) in the study domain"),
    "sampled_domain_area": ("m2", "Study-domain area represented by the sample points"),
    "mapbiomas_regrowth_2000_2012_area": ("m2", "MapBiomas Colombia C3 regrowth: anthropic in 2000, regeneration event by 2012, forest every year 2012-2016"),
    "mapbiomas_labelled_2000_2012_area": ("m2", "Area labelled regrowth or non-regrowth for the 2000-2012 window (others excluded)"),
    "mapbiomas_regrowth_2012_2024_area": ("m2", "MapBiomas regrowth: anthropic in 2012, regeneration event by 2020, forest every year 2020-2024"),
    "mapbiomas_labelled_2012_2024_area": ("m2", "Area labelled regrowth or non-regrowth for the 2012-2024 window"),
}


# %% [markdown]
# ## Helpers: coarsening and writing one level group

# %%
def coarsen(df: pd.DataFrame, from_depth: int, to_depth: int) -> pd.DataFrame:
    parent = df.index.to_numpy() >> np.uint64(2 * (from_depth - to_depth)) if df.index.dtype == np.uint64 else \
        df.index.to_numpy() >> (2 * (from_depth - to_depth))
    return df.groupby(parent).sum()


def level_dataset(df: pd.DataFrame, depth: int, var_attrs: dict, support: str) -> xr.Dataset:
    ds = xr.Dataset(coords={"cell_ids": ("cells", df.index.to_numpy().astype("int64"))})
    ds["cell_ids"].attrs.update(long_name=f"HEALPix NESTED cell index at refinement level {depth} (WGS84)")
    ds["crs"] = xr.DataArray(np.int8(0), attrs=cf_grid_mapping_attrs(depth))
    for v, (units, long_name) in var_attrs.items():
        ds[v] = ("cells", df[v].to_numpy().astype("float32" if units == "m2" else "int32"))
        ds[v].attrs.update(units=units, long_name=long_name, grid_mapping="crs",
                           cell_methods="area: sum" if units == "m2" else "area: sum (count)")
    ds.attrs.update(dggs_attrs(depth))
    ds.attrs.update(support=support, Conventions="CF-1.11")
    return ds


def write_family(name: str, levels: dict[int, pd.DataFrame], var_attrs: dict, support: str) -> None:
    path = f"measurements/{name}"
    for depth, df in levels.items():
        level_dataset(df, depth, var_attrs, support).to_zarr(ZARR, group=f"{path}/{depth}", mode="w", zarr_format=3,
                                                             consolidated=False)
    g = zarr.open_group(ZARR, path=path, mode="a", zarr_format=3)
    conv = dggs_attrs(max(levels))["zarr_conventions"]
    g.attrs.update({
        "zarr_conventions": [{"uuid": "d35379db-88df-4056-af3a-620245f8e347", "name": "multiscales",
                              "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
                              "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
                              "description": "Multiscale layout of zarr datasets"}] + conv,
        "multiscales": {"layout": [{"asset": str(d), "dggs": dggs_attrs(d)["dggs"]} for d in sorted(levels, reverse=True)]},
        "support": support,
    })
    print(f"{name}: levels {sorted(levels)}; cells at finest level {len(levels[max(levels)]):,}")


# %% [markdown]
# ## Family 1: full resolution (05a), depth 15 → 8

# %%
full = pd.read_parquet(DERIVED / f"fullres_healpix_d{FULL_DEPTH}.parquet").set_index("cell_ids")
full_levels = {FULL_DEPTH: full}
for d in range(FULL_DEPTH - 1, MIN_DEPTH - 1, -1):
    full_levels[d] = coarsen(full_levels[d + 1], d + 1, d)
for d in full_levels:  # parent sums must conserve every total exactly (float64)
    assert np.allclose(full_levels[d].sum().to_numpy(), full.sum().to_numpy(), rtol=1e-9), d  # float64 summation order: ~1e-12
write_family("full_resolution", full_levels, {k: ("m2", v) for k, v in FULL_META.items()},
             "every 30 m pixel; value x exact WGS84 pixel area binned by pixel centre "
             "(healpix_resample.ConservativeResampler), coarser levels are parent sums")

# %% [markdown]
# ## Family 2: sample-based layers (MapBiomas labels), depth 12 → 8

# %%
grid = pd.read_parquet(CLEAN / "pred_grid.parquet", columns=["lon", "lat", "grow", "gcol", "area_ell"])
mb = pd.read_parquet(CLEAN / "mapbiomas_grid.parquet", columns=["grow", "gcol", "mb_label_e1", "mb_label_fw2"])
gm = grid.merge(mb, on=["grow", "gcol"], how="inner")
a = gm.area_ell.to_numpy() * GRID_W
vals = np.stack([np.ones(len(gm)) / a, np.ones(len(gm)),
                 (gm.mb_label_e1 == 1).to_numpy(), (gm.mb_label_e1 >= 0).to_numpy(),
                 (gm.mb_label_fw2 == 1).to_numpy(), (gm.mb_label_fw2 >= 0).to_numpy()]).astype("float64")
r = ConservativeResampler(gm.lon.to_numpy(), gm.lat.to_numpy(), level=SAMPLE_DEPTH, area=a, ellipsoid="WGS84",
                          verbose=False).resample(vals)
smp = pd.DataFrame(np.asarray(r.cell_data).reshape(len(SAMPLE_META), -1).T, columns=list(SAMPLE_META),
                   index=np.asarray(r.cell_ids).astype("int64"))
smp["n_points"] = smp.n_points.round().astype("int64")
sample_levels = {SAMPLE_DEPTH: smp}
for d in range(SAMPLE_DEPTH - 1, MIN_DEPTH - 1, -1):
    sample_levels[d] = coarsen(sample_levels[d + 1], d + 1, d)
write_family("sample_based", sample_levels, SAMPLE_META,
             "1-in-100 systematic sample of 30 m pixels (each point = 10 x 10 pixels); area = point pixel area x 100; "
             "reliable only where n_points is large")

# %% [markdown]
# ## Family 3: area of applicability (03e), depth 8 only

# %%
aoa = xr.open_dataset(RESULTS / "aoa_healpix_d8.nc")
cell_dim = [d for d in aoa.dims][0]
cell_var = "cell_ids" if "cell_ids" in aoa.variables else ("cell" if "cell" in aoa.variables else cell_dim)
aoa_df = aoa.to_dataframe().reset_index().set_index(cell_var)
aoa_vars = {v: (str(aoa[v].attrs.get("units", "m2")), str(aoa[v].attrs.get("long_name", v)))
            for v in aoa.data_vars if aoa[v].dims == (cell_dim,) and np.issubdtype(aoa[v].dtype, np.number)}
aoa_df = aoa_df[[v for v in aoa_vars]]
aoa_vars = {v: (("m2" if u in ("m2", "m^2") else "1"), ln) for v, (u, ln) in aoa_vars.items()}
for v, (u, ln) in list(aoa_vars.items()):
    if u == "1":  # shares are not additive: keep as plain values, not area sums
        aoa_df[v] = aoa_df[v].astype("float32")
ds_aoa = xr.Dataset(coords={"cell_ids": ("cells", aoa_df.index.to_numpy().astype("int64"))})
ds_aoa["crs"] = xr.DataArray(np.int8(0), attrs=cf_grid_mapping_attrs(MIN_DEPTH))
for v, (u, ln) in aoa_vars.items():
    ds_aoa[v] = ("cells", aoa_df[v].to_numpy().astype("float32"))
    ds_aoa[v].attrs.update(units=u, long_name=ln, grid_mapping="crs",
                           cell_methods="area: sum" if u == "m2" else "area: mean")
ds_aoa.attrs.update(dggs_attrs(MIN_DEPTH))
ds_aoa.attrs.update(support="1-in-100 systematic sample, aggregated in 03e", Conventions="CF-1.11")
ds_aoa.to_zarr(ZARR, group=f"measurements/area_of_applicability/{MIN_DEPTH}", mode="w", zarr_format=3,
               consolidated=False)
print("area_of_applicability:", list(aoa_vars))

# %% [markdown]
# ## Root metadata

# %%
root = zarr.open_group(ZARR, mode="a", zarr_format=3)
root.attrs.update({
    "title": "Natural-regeneration potential in Colombia on HEALPix (WGS84): Williams et al. (2024) map, an independent replication model, and regrowth labels",
    "summary": ("Multiscale HEALPix NESTED (WGS84) aggregation of (1) the published natural-regeneration potential map of "
                "Williams et al. (2024), (2) an independent random-forest replication of that model for Colombia, "
                "(3) Fagan et al. (2022) and MapBiomas Colombia regrowth. All values are areas (m2) summed per cell."),
    "Conventions": "CF-1.11",
    "version": VERSION,
    "license": "CC-BY-NC-4.0",
    "creator_name": "Anne Fouilloux", "creator_orcid": "https://orcid.org/0000-0002-1784-2920",
    "institution": "LifeWatch ERIC",
    "software": f"https://doi.org/{SOFTWARE_DOI}",
    "references": "https://doi.org/10.1038/s41586-024-08106-4; https://doi.org/10.5281/zenodo.7428804; "
                  "https://doi.org/10.1038/s41893-022-00904-w; https://colombia.mapbiomas.org; "
                  "https://doi.org/10.21428/d28e8e57.5411b150",
    "geospatial_bounds_crs": "HEALPix NESTED on WGS84 (healpix-geo, authalic latitude)",
})
zarr.consolidate_metadata(ZARR, zarr_format=3)

# %% [markdown]
# ## Check: read back with xarray

# %%
dt = xr.open_datatree(ZARR, engine="zarr", consolidated=True)
d15 = dt[f"measurements/full_resolution/{FULL_DEPTH}"].ds
d8 = dt[f"measurements/full_resolution/{MIN_DEPTH}"].ds
check = pd.DataFrame({"depth_15_Mha": (d15[list(FULL_META)].sum() / 1e10).to_pandas(),
                      "depth_8_Mha": (d8[list(FULL_META)].sum() / 1e10).to_pandas()})
print(check.round(4).to_string())
print(dt[f"measurements/full_resolution/{FULL_DEPTH}"].ds.attrs["dggs"])

# %% [markdown]
# ## Parquet tables

# %%
TABLES = {
    "samples_colombia.parquet": (CLEAN / "samples.parquet", "Colombian training/validation sample points with every predictor (2000, 2015/2018, 1992/1999), Fagan label, authors' map values"),
    "samples_neotropics.parquet": (CLEAN / "neotropics_samples.parquet", "Neotropical sample (diagnostic 3), HEALPix depth-5 stratified"),
    "prediction_grid_features.parquet": (CLEAN / "pred_grid.parquet", "1-in-100 systematic grid over the study domain with every predictor and label"),
    "prediction_grid_step2.parquet": (DERIVED / "step2_predictions_grid.parquet", "Step-2 model predictions on the grid (uncalibrated, prior-shift, isotonic) and the authors' map"),
    "mapbiomas_labels_grid.parquet": (CLEAN / "mapbiomas_grid.parquet", "MapBiomas Colombia C3 regrowth labels (all windows), 2012 predictors and land-use history at the grid points"),
}
for out, (src, _) in TABLES.items():
    shutil.copy(src, OUT / "tables" / out)

sources = []
for f in sorted(RAW.glob("sources*.json")):
    if "smoke" not in f.name:
        sources.append({"file": f.name, "content": json.load(open(f))})
json.dump(sources, open(OUT / "sources.json", "w"), indent=1)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


shutil.copy("../docs/archive_README.md", OUT / "README.md")  # dataset description (versioned in docs/)
shutil.copy("../docs/archive_zenodo.json", OUT / "zenodo_metadata.json")  # deposit metadata for the upload

files = [p for p in sorted(OUT.rglob("*")) if p.is_file()]
pd.DataFrame({"path": [str(p.relative_to(OUT)) for p in files], "bytes": [p.stat().st_size for p in files],
              "sha256": [sha256(p) for p in files]}).to_csv(OUT / "checksums.csv", index=False)
total = sum(p.stat().st_size for p in files)
print(f"{len(files):,} files, {total / 1e9:.2f} GB")
json.dump({"tables": {k: v[1] for k, v in TABLES.items()}, "zarr": ZARR.name, "bytes": total},
          open(RESULTS / "archive_manifest.json", "w"), indent=2)
