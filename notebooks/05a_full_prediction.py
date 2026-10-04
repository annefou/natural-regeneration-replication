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
# # 05a — Full-resolution prediction over Colombia, aggregated to HEALPix depth 15
#
# Step 2 predicted on a 1-in-100 systematic sample of 30 m pixels. This notebook
# predicts the same model on **every** 30 m pixel of the prediction domain and
# aggregates, per 1° tile, to HEALPix NESTED **depth 15** (WGS84, ~200 m cells) with
# `healpix_resample.ConservativeResampler`: each pixel is binned into the cell that
# contains its centre and contributes value × exact WGS84 pixel area, so every total
# is conserved (checked against our own healpix-geo sums: identical).
#
# - **Same model as step 2.** Refit with the same training draw, seed and settings
#   as `03_analysis` (the predictions at the step-2 grid pixels are compared below).
# - **Same predictors as step 2.** The per-tile code is copied from `02b`: forest
#   2018 = tree cover 2000 ≥ 30 % minus loss 2001–2018; forest density in a 1 km
#   disk; distance to forest capped at 25 km; coarse layers by nearest neighbour.
# - **Layers** (all sums of value × pixel area, m², per depth-15 cell): country,
#   study domain and prediction domain areas; our expected area uncalibrated and
#   prevalence-calibrated, and the area with p > 0.5; the authors' map (expected
#   area, valid area, binary area); Fagan et al. regrowth inside the study domain.
#
# **Why depth 15.** Each depth-15 cell is an exact sum of ~50 pixels (pixel centres
# lie in exactly one cell, so binning at depth 15 equals binning at depth 17 and
# summing children). Depth 17 (~49 m) would be ~460 M cells over Colombia (~18 GB for
# ten layers), too large to archive; depth 15 is ~29 M cells.
#
# Run time: about 1–2 h on 12 cores (random-forest prediction dominates).

# %%
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import torch
import xarray as xr
from healpix_resample import ConservativeResampler
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from scipy import ndimage, signal
from sklearn.ensemble import RandomForestClassifier

torch.set_num_threads(1)  # one tile per worker process

# %%
RAW = Path("../data/raw")
SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_BBOX = (-75.0, 4.0, -73.0, 6.0)
CLEAN = Path("../data/clean_smoke" if SMOKE else "../data/clean")
DERIVED = Path("../data/derived_smoke" if SMOKE else "../data/derived")
RESULTS = Path("../results/smoke" if SMOKE else "../results")
for d in (DERIVED, RESULTS):
    d.mkdir(parents=True, exist_ok=True)
RES = 0.00025
TILE_PX, MARGIN_PX = 4000, 1000
TREE_THRESHOLD = 30
DENSITY_RADIUS_M = 1000.0
DIST_CAP_M = 25_000.0
GRID_STRIDE = 10
HP_DEPTH = 15
SEED = 20261003
N_TREES = 100 if SMOKE else 500
N_WORKERS = int(os.environ.get("N_WORKERS", min(8, os.cpu_count() or 1)))  # ~5 GB peak per tile
CHUNK = 1_000_000
FINAL_VARS = ["forest_density_2000", "dist_forest_2000", "ocdens", "phihox", "pc1", "pc2", "pc3", "pc4", "lc_2000", "biome"]
PRED_MAP = {"forest_density_2000": "forest_density_2018", "dist_forest_2000": "dist_forest_2018", "lc_2000": "lc_2015"}
COL_SHARE = 188_921 / 4_780_000
N_TRAIN = 2_000 if SMOKE else int(round(1_000_000 * COL_SHARE))
ESA_EXCL_PRED = [190, 200, 201, 202, 210]
LAYERS = ["country_area", "study_domain_area", "prediction_domain_area", "expected_area_uncalibrated",
          "expected_area_calibrated", "area_p_gt05", "authors_expected_area", "authors_valid_area",
          "authors_binary_area", "fagan_regrowth_area"]

WGS84_A, WGS84_F = 6378137.0, 1 / 298.257223563
E2 = WGS84_F * (2 - WGS84_F)
EC = np.sqrt(E2)


def _q(phi: np.ndarray) -> np.ndarray:
    s = np.sin(phi)
    return (1 - E2) * (s / (1 - E2 * s * s) - np.log((1 - EC * s) / (1 + EC * s)) / (2 * EC))


def area_ellipsoid(lat_top: np.ndarray) -> np.ndarray:
    return WGS84_A**2 / 2 * np.radians(RES) * (_q(np.radians(lat_top)) - _q(np.radians(lat_top - RES)))


def metres_per_degree(lat: float) -> tuple[float, float]:
    phi = np.radians(lat)
    s2 = np.sin(phi) ** 2
    m = WGS84_A * (1 - E2) / (1 - E2 * s2) ** 1.5
    n = WGS84_A / np.sqrt(1 - E2 * s2)
    return float(np.radians(1) * m), float(np.radians(1) * n * np.cos(phi))


# %% [markdown]
# ## Model: the step-2 refit (same draw, seed and settings as `03_analysis`)

# %%
samples = pd.read_parquet(CLEAN / "samples.parquet")
sums = pd.read_csv(CLEAN / "tile_sums.csv").drop(columns="tile").sum()
PI = float(sums["class1_area_ell"] / (sums["class1_area_ell"] + sums["class0_area_ell"]))
pool = samples[samples.set == "train_pool"]
train = pd.concat([g.sample(n=min(N_TRAIN // 2, len(g)), random_state=SEED) for _, g in pool.groupby("label")])
rf = RandomForestClassifier(n_estimators=N_TREES, max_features=3, min_samples_leaf=1, bootstrap=True,
                            oob_score=True, n_jobs=-1, random_state=SEED).fit(train[FINAL_VARS].to_numpy(),
                                                                                train.label.to_numpy())
rf.n_jobs = 1  # prediction runs inside forked workers
print(f"training records {len(train):,}; OOB accuracy {rf.oob_score_:.4f}; prevalence pi = {PI:.5f}")


def prior_shift(p: np.ndarray, pi: float = PI, train_prev: float = 0.5) -> np.ndarray:
    num = p * pi / train_prev
    return num / (num + (1 - p) * (1 - pi) / (1 - train_prev))


# %% [markdown]
# ## Vector inputs, coarse layers and the authors' tiles (as in `02b`)

# %%
col = gpd.read_file(RAW / "gadm" / "gadm41_COL.gpkg", layer="ADM_ADM_0")
COL_GEOM = shapely.make_valid(col.geometry.iloc[0])
biomes = gpd.read_parquet(CLEAN / "biomes_colombia.parquet")
study = biomes[biomes.BIOME_NUM.isin([1, 2, 3])]
DOMAIN_GEOM = shapely.intersection(COL_GEOM, shapely.union_all(study.geometry.values))
if SMOKE:
    DOMAIN_GEOM = shapely.clip_by_rect(DOMAIN_GEOM, *SMOKE_BBOX)
    COL_GEOM = shapely.clip_by_rect(COL_GEOM, *SMOKE_BBOX)
fagan = gpd.read_parquet(RAW / "fagan" / "fagan2022_colombia.parquet")
fagan["code"] = fagan.pred3class.map({"regrowth": 1, "plantation": 2, "open": 3}).fillna(4).astype("uint8")
fagan_tree = shapely.STRtree(fagan.geometry.values)

tiles = []
w, s, e, n = COL_GEOM.bounds
for lat0 in range(int(np.floor(s)), int(np.ceil(n))):
    for lon0 in range(int(np.floor(w)), int(np.ceil(e))):
        if shapely.box(lon0, lat0, lon0 + 1, lat0 + 1).intersects(COL_GEOM):
            tiles.append((lon0, lat0))
MAX_TILES = int(os.environ.get("MAX_TILES", "0"))  # testing only: limit the number of tiles
if MAX_TILES:
    tiles = sorted(tiles, key=lambda t: (abs(t[0] + 74.5), abs(t[1] - 4.5)))[:MAX_TILES]
print(len(tiles), "tiles of 1° intersecting the country")

COARSE = {}
for f, vars_ in [("bioclim_pca.nc", ["pc1", "pc2", "pc3", "pc4"]), ("soil_0_30cm.nc", ["ocdens", "phihox"]),
                 ("landcover.nc", ["lc_2015", "esa_2015"])]:
    ds = xr.open_dataset(CLEAN / f).load()
    for v in vars_:
        da = ds[v]
        lat, lon = da.lat.values, da.lon.values
        COARSE[v] = dict(arr=da.values, lat0=lat[0], dlat=lat[1] - lat[0], lon0=lon[0], dlon=lon[1] - lon[0])


def coarse_grid(name: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    c = COARSE[name]
    r = np.clip(np.rint((lat - c["lat0"]) / c["dlat"]).astype(int), 0, c["arr"].shape[0] - 1)
    k = np.clip(np.rint((lon - c["lon0"]) / c["dlon"]).astype(int), 0, c["arr"].shape[1] - 1)
    return c["arr"][r[:, None], k[None, :]]


ZEN = RAW / "zenodo_7428804"
AUTH_PCT, AUTH_BIN = [], []
for f in ZEN.glob("pnv_pct_30m_tile_*.tif"):
    with rasterio.open(f) as r:
        AUTH_PCT.append((str(f), round(r.bounds.left), round(r.bounds.top), r.width, r.height))
for f in (ZEN / "pnv_bin_30m").glob("*.tif"):
    with rasterio.open(f) as r:
        AUTH_BIN.append((str(f), round(r.bounds.left), round(r.bounds.top), r.width, r.height))


def read_authors(index: list, lon0: int, lat0: int, fill: int) -> np.ndarray:
    for path, left, top, wd, ht in index:
        if left <= lon0 and lon0 + 1 <= left + wd * RES + 1e-9 and top - ht * RES - 1e-9 <= lat0 and lat0 + 1 <= top:
            col0, row0 = int(round((lon0 - left) / RES)), int(round((top - (lat0 + 1)) / RES))
            with rasterio.open(path) as r:
                return r.read(1, window=Window(col0, row0, TILE_PX, TILE_PX))
    return np.full((TILE_PX, TILE_PX), fill, dtype="uint8")


def disk_kernel(radius_m: float, dy: float, dx: float) -> np.ndarray:
    ry, rx = int(np.ceil(radius_m / dy)), int(np.ceil(radius_m / dx))
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    k = ((yy * dy) ** 2 + (xx * dx) ** 2 <= radius_m**2).astype("float32")
    return k / k.sum()


def focal(forest: np.ndarray, kernel: np.ndarray, dy: float, dx: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    kr, kc = kernel.shape[0] // 2, kernel.shape[1] // 2
    sub = forest[m - kr:m + TILE_PX + kr, m - kc:m + TILE_PX + kc].astype("float32")
    dens = signal.fftconvolve(sub, kernel, mode="valid")
    if forest.any():
        dist = ndimage.distance_transform_edt(~forest, sampling=(dy, dx))[m:m + TILE_PX, m:m + TILE_PX]
    else:
        dist = np.full((TILE_PX, TILE_PX), np.inf)
    return np.clip(dens, 0, 1).astype("float32"), np.minimum(dist, DIST_CAP_M).astype("float32")


# %% [markdown]
# ## Per-tile worker: predictors → prediction → conservative HEALPix sums (depth 15)

# %%
def process_tile(tile: tuple[int, int]) -> dict:
    lon0, lat0 = tile
    m = MARGIN_PX
    left, top = lon0 - m * RES, lat0 + 1 + m * RES
    size = TILE_PX + 2 * m
    arrs = {}
    for layer in ["treecover2000", "lossyear"]:
        with rasterio.open(CLEAN / f"hansen_{layer}.vrt") as src:
            w_ = from_bounds(left, top - size * RES, left + size * RES, top, src.transform).round_offsets().round_lengths()
            arrs[layer] = src.read(1, window=Window(w_.col_off, w_.row_off, size, size), boundless=True, fill_value=0)
    forest18 = (arrs["treecover2000"] >= TREE_THRESHOLD) & ~((arrs["lossyear"] >= 1) & (arrs["lossyear"] <= 18))
    del arrs
    my, mx = metres_per_degree(lat0 + 0.5)
    dy, dx = RES * my, RES * mx
    dens18, dist18 = focal(forest18, disk_kernel(DENSITY_RADIUS_M, dy, dx), dy, dx, m)
    f18 = forest18[m:m + TILE_PX, m:m + TILE_PX]
    del forest18

    core_tr = from_origin(lon0, lat0 + 1, RES, RES)
    shape = (TILE_PX, TILE_PX)
    tbox = shapely.box(lon0, lat0, lon0 + 1, lat0 + 1)
    country = rasterize([(shapely.clip_by_rect(COL_GEOM, *tbox.bounds), 1)], out_shape=shape, transform=core_tr,
                        fill=0, dtype="uint8").astype(bool)
    if not country.any():
        return {"tile": tile, "cells": np.zeros(0, "int64"), "sums": np.zeros((len(LAYERS), 0)), "check": None}
    bshapes = [(shapely.clip_by_rect(g, *tbox.bounds), int(b)) for g, b in zip(biomes.geometry, biomes.BIOME_NUM)]
    bshapes = [(g, b) for g, b in bshapes if not g.is_empty]
    biome = rasterize(bshapes, out_shape=shape, transform=core_tr, fill=0, dtype="uint8") if bshapes else np.zeros(shape, "uint8")
    idx = fagan_tree.query(tbox)
    fsub = fagan.iloc[idx].sort_values("code", ascending=False)
    fag = (rasterize(((g, int(c)) for g, c in zip(fsub.geometry, fsub.code)), out_shape=shape, transform=core_tr,
                     fill=0, dtype="uint8") if len(fsub) else np.zeros(shape, "uint8"))

    lat_c = lat0 + 1 - (np.arange(TILE_PX) + 0.5) * RES
    lon_c = lon0 + (np.arange(TILE_PX) + 0.5) * RES
    esa15 = coarse_grid("esa_2015", lon_c, lat_c)
    dom = country & np.isin(biome, [1, 2, 3])
    pred = dom & ~f18 & ~np.isin(esa15, ESA_EXCL_PRED)

    feats = {"forest_density_2018": dens18, "dist_forest_2018": dist18, "biome": biome}
    for v in ["ocdens", "phihox", "pc1", "pc2", "pc3", "pc4", "lc_2015"]:
        feats[v] = coarse_grid(v, lon_c, lat_c)
    r_, c_ = np.nonzero(pred)
    X = np.column_stack([feats[PRED_MAP.get(v, v)][r_, c_].astype("float32") for v in FINAL_VARS])
    ok = ~np.isnan(X).any(axis=1)
    p = np.zeros(len(X), dtype="float32")
    good = np.flatnonzero(ok)
    for i in range(0, len(good), CHUNK):
        j = good[i:i + CHUNK]
        p[j] = rf.predict_proba(X[j])[:, 1]
    del X, feats
    p_img = np.zeros(shape, dtype="float32")
    ok_img = np.zeros(shape, dtype=bool)
    p_img[r_, c_], ok_img[r_, c_] = p, ok
    pred_ok = pred & ok_img

    pct = read_authors(AUTH_PCT, lon0, lat0, 255)
    bn = read_authors(AUTH_BIN, lon0, lat0, 3)
    pct_ok = pct <= 100

    rr, cc = np.nonzero(country)
    a = area_ellipsoid(lat_c + RES / 2)[rr]
    vals = np.stack([
        np.ones(len(rr)), dom[rr, cc], pred_ok[rr, cc],
        np.where(pred_ok, p_img, 0)[rr, cc], np.where(pred_ok, prior_shift(p_img), 0)[rr, cc],
        (pred_ok & (p_img > 0.5))[rr, cc],
        np.where(pct_ok, pct, 0)[rr, cc] / 100.0, pct_ok[rr, cc], (bn == 1)[rr, cc],
        (dom & (fag == 1))[rr, cc],
    ]).astype("float64")
    res = ConservativeResampler(lon_c[cc], lat_c[rr], level=HP_DEPTH, area=a, ellipsoid="WGS84",
                                verbose=False, num_threads=1).resample(vals)
    cells = np.asarray(res.cell_ids, dtype="int64")
    out_sums = np.asarray(res.cell_data, dtype="float64").reshape(len(LAYERS), -1)

    grow0 = int(round((90 - (lat0 + 1)) / RES))
    gcol0 = int(round((lon0 + 180) / RES))
    on_grid = (((grow0 + np.arange(TILE_PX)) % GRID_STRIDE == GRID_STRIDE // 2)[:, None]
               & ((gcol0 + np.arange(TILE_PX)) % GRID_STRIDE == GRID_STRIDE // 2)[None, :] & pred_ok)
    gr, gc = np.nonzero(on_grid)
    check = pd.DataFrame({"grow": grow0 + gr, "gcol": gcol0 + gc, "p_full": p_img[gr, gc]})
    return {"tile": tile, "cells": cells, "sums": out_sums, "check": check}


# %% [markdown]
# ## Run all tiles (parallel, fork)

# %%
t0 = time.time()
cell_parts, sum_parts, checks = [], [], []
with mp.get_context("fork").Pool(N_WORKERS, maxtasksperchild=2) as pool_:
    for i, res in enumerate(pool_.imap_unordered(process_tile, tiles)):
        cell_parts.append(res["cells"])
        sum_parts.append(res["sums"])
        if res["check"] is not None:
            checks.append(res["check"])
        print(f"{i + 1}/{len(tiles)} tiles, {time.time() - t0:.0f} s", flush=True)

cells = np.concatenate(cell_parts)
S = np.concatenate(sum_parts, axis=1)
dhp = pd.DataFrame(S.T, columns=LAYERS).assign(cell_ids=cells).groupby("cell_ids", sort=True).sum()  # tiles share edge cells
dhp.reset_index().to_parquet(DERIVED / f"fullres_healpix_d{HP_DEPTH}.parquet")
print(f"{len(dhp):,} depth-{HP_DEPTH} cells; {time.time() - t0:.0f} s")

# %% [markdown]
# ## Checks
#
# 1. At the step-2 grid pixels the full-resolution prediction must equal step 2's.
# 2. The totals are compared with the step-1 and step-2 estimates.

# %%
chk = pd.concat(checks, ignore_index=True)
step2 = pd.read_parquet(DERIVED / "step2_predictions_grid.parquet", columns=["grow", "gcol", "p"])
j = chk.merge(step2, on=["grow", "gcol"], how="inner")
max_diff = float(np.abs(j.p_full - j.p).max()) if len(j) else float("nan")
print(f"grid pixels compared: {len(j):,} of {len(step2):,} step-2 points; max |p_full - p_step2| = {max_diff:.2e}")

tot = dhp.sum() / 1e10
s2 = pd.read_csv(RESULTS / "step2_replication_colombia.csv").set_index("metric").value
summary = pd.DataFrame([
    ("prediction_domain_area_mha", tot.prediction_domain_area, s2.get("prediction_domain_area_mha"), "02b exact 30 m sum"),
    ("expected_area_uncalibrated_mha", tot.expected_area_uncalibrated, s2.get("expected_area_uncalibrated_mha"), "step 2: 1-in-100 sample"),
    ("expected_area_calibrated_mha", tot.expected_area_calibrated, s2.get("expected_area_prior_shift_mha"), "step 2: 1-in-100 sample"),
    ("area_p_gt05_mha", tot.area_p_gt05, s2.get("area_p_gt_0.5_uncalibrated_mha"), "step 2: 1-in-100 sample"),
    ("authors_expected_in_country_mha", tot.authors_expected_area, 10.5143, "step 1, exact (whole country)"),
    ("study_domain_area_mha", tot.study_domain_area, s2.get("study_domain_area_mha"), "02b exact 30 m sum"),
    ("country_area_mha", tot.country_area, 113.7241, "step 1, exact"),
    ("fagan_regrowth_in_domain_mha", tot.fagan_regrowth_area, None, ""),
    ("grid_check_max_abs_diff_p", max_diff, 0.0, f"{len(j)} grid pixels"),
], columns=["metric", "full_resolution", "reference", "reference_source"])
summary.to_csv(RESULTS / "fullres_summary.csv", index=False)
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
json.dump({"depth": HP_DEPTH, "n_cells": int(len(dhp)), "n_tiles": len(tiles), "pi": PI, "oob": float(rf.oob_score_),
           "resampler": "healpix_resample.ConservativeResampler (centre binning, value x exact WGS84 pixel area)",
           "ellipsoid": "WGS84", "layers": LAYERS}, open(DERIVED / f"fullres_healpix_d{HP_DEPTH}.json", "w"), indent=2)
