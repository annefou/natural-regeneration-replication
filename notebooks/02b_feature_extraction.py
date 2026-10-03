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
# # 02b — 30 m domain masks, sampling and predictor extraction (Colombia)
#
# Works on the native 30 m Hansen lattice (0.00025°, identical to the authors'
# map), in 1° × 1° tiles with a 0.25° margin, in parallel. For every tile that
# intersects Colombia ∩ study biomes:
#
# 1. **Forest** = Hansen tree cover 2000 ≥ `TREE_THRESHOLD` (%). Forest 2018 =
#    (forest 2000 OR gain 2000–2012) AND NOT loss 2001–2018.
# 2. **Focal predictors** (2000 for training, 2018 for prediction): forest density =
#    forest fraction in a 1 km-radius disk; distance to forest (m, Euclidean on
#    the ellipsoid-scaled grid, truncated at 25 km).
# 3. **Masks** (pixel-centre rule):
#    - study domain = Colombia (GADM 4.1) ∩ RESOLVE biomes 1, 2, 3;
#    - class 1 = study domain ∩ Fagan `regrowth` polygons;
#    - class 0 = study domain − forest 2000 − ESA CCI 2000 classes 150–153, 190,
#      200–202, 210 − all Fagan polygons (regrowth, plantation, open);
#    - prediction domain = study domain − forest 2018 − ESA CCI 2015 classes 190,
#      200–202, 210 (open water, urban, bare as in the paper).
# 4. **Sampling.** Class 0 and class 1 candidates by Bernoulli sampling with
#    inclusion probability ∝ exact pixel area (ellipsoid-aware). A systematic
#    1-in-100 grid (every 10th row and column of the global lattice) gives the
#    prediction sample.
# 5. **Coarse predictors** sampled at the pixel centre (nearest neighbour) from
#    `data/clean/`; slope (Horn) from SRTMGL1 computed per tile.
# 6. **Exact sums** over all 30 m pixels of each tile: domain areas, and the
#    authors' map restricted to the study domain.
#
# **Smoke mode** (`SMOKE=1`): only the 1° tiles inside the smoke sub-window
# (-75…-73°E, 4…6°N), 3,000 points per class; outputs in `data/clean_smoke/`.

# %%
import json
import multiprocessing as mp
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from scipy import ndimage, signal

# %%
RAW = Path("../data/raw")
SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_BBOX = (-75.0, 4.0, -73.0, 6.0)
CLEAN = Path("../data/clean_smoke" if SMOKE else "../data/clean")
RES = 0.00025
TILE_DEG, MARGIN_PX = 1, 1000
TILE_PX = int(round(TILE_DEG / RES))
TREE_THRESHOLD = 30
DENSITY_RADIUS_M = 1000.0
DIST_CAP_M = 25_000.0
GRID_STRIDE = 10
SEED = 20261003
N_WORKERS = int(os.environ.get("N_WORKERS", min(12, os.cpu_count() or 1)))

# Sample sizes proportional to the paper: Colombia holds 188,921 of the 4.78 M
# Fagan regrowth patches (3.95 %). Paper: pool 6 M + validation 4.87 M.
COL_SHARE = 188_921 / 4_780_000
TARGET_PER_CLASS = 3_000 if SMOKE else int(round((6_000_000 + 4_870_000) / 2 * COL_SHARE))
print(f"Colombia share {COL_SHARE:.4f}; target {TARGET_PER_CLASS:,} points per class (pool + validation)")

WGS84_A, WGS84_F = 6378137.0, 1 / 298.257223563
E2 = WGS84_F * (2 - WGS84_F)
EC = np.sqrt(E2)


def _q(phi: np.ndarray) -> np.ndarray:
    s = np.sin(phi)
    return (1 - E2) * (s / (1 - E2 * s * s) - np.log((1 - EC * s) / (1 + EC * s)) / (2 * EC))


def area_ellipsoid(lat_top: np.ndarray) -> np.ndarray:
    return WGS84_A**2 / 2 * np.radians(RES) * (_q(np.radians(lat_top)) - _q(np.radians(lat_top - RES)))


def area_sphere_a(lat_top: np.ndarray) -> np.ndarray:
    return WGS84_A**2 * np.radians(RES) * (np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_top - RES)))


def metres_per_degree(lat: float) -> tuple[float, float]:
    phi = np.radians(lat)
    s2 = np.sin(phi) ** 2
    m = WGS84_A * (1 - E2) / (1 - E2 * s2) ** 1.5
    n = WGS84_A / np.sqrt(1 - E2 * s2)
    return float(np.radians(1) * m), float(np.radians(1) * n * np.cos(phi))


# %% [markdown]
# ## Vector inputs and tile list

# %%
col = gpd.read_file(RAW / "gadm" / "gadm41_COL.gpkg", layer="ADM_ADM_0")
COL_GEOM = shapely.make_valid(col.geometry.iloc[0])
biomes = gpd.read_parquet(CLEAN / "biomes_colombia.parquet")
study = biomes[biomes.BIOME_NUM.isin([1, 2, 3])]
DOMAIN_GEOM = shapely.intersection(COL_GEOM, shapely.union_all(study.geometry.values))
if SMOKE:
    DOMAIN_GEOM = shapely.clip_by_rect(DOMAIN_GEOM, *SMOKE_BBOX)
print("study biomes present:", study.BIOME_NAME.tolist())

fagan = gpd.read_parquet(RAW / "fagan" / "fagan2022_colombia.parquet")
FAGAN_CODE = {"regrowth": 1, "plantation": 2, "open": 3}
fagan["code"] = fagan.pred3class.map(FAGAN_CODE).astype("uint8")
fagan_tree = shapely.STRtree(fagan.geometry.values)

regrowth_dom = fagan[fagan.code == 1].geometry.intersection(DOMAIN_GEOM)
from pyproj import Geod  # noqa: E402

regrowth_area = sum(abs(Geod(ellps="WGS84").geometry_area_perimeter(g)[0]) for g in regrowth_dom if not g.is_empty)
domain_area = abs(Geod(ellps="WGS84").geometry_area_perimeter(DOMAIN_GEOM)[0])
PIX = 765.0  # typical 30 m pixel area (m²) in Colombia, for setting rates only
RATE1 = min(1.0, 1.5 * TARGET_PER_CLASS / (regrowth_area / PIX))
RATE0 = min(1.0, 4.0 * TARGET_PER_CLASS / (domain_area / PIX))
print(f"regrowth area in domain {regrowth_area / 1e4:,.0f} ha; domain {domain_area / 1e10:.2f} Mha; "
      f"rates class1 {RATE1:.4f}, class0 {RATE0:.5f}")

tiles = []
w, s, e, n = DOMAIN_GEOM.bounds
for lat0 in range(int(np.floor(s)), int(np.ceil(n))):
    for lon0 in range(int(np.floor(w)), int(np.ceil(e))):
        if shapely.box(lon0, lat0, lon0 + 1, lat0 + 1).intersects(DOMAIN_GEOM):
            tiles.append((lon0, lat0))
print(len(tiles), "tiles of 1°")

# %% [markdown]
# ## Coarse layers (loaded once; shared with forked workers)

# %%
COARSE = {}


def _add(ds: xr.Dataset, var: str, name: str | None = None) -> None:
    da = ds[var]
    lat, lon = da.lat.values, da.lon.values
    COARSE[name or var] = dict(arr=da.values, lat0=lat[0], dlat=lat[1] - lat[0], lon0=lon[0], dlon=lon[1] - lon[0])


for f, vars_ in [("bioclim_pca.nc", ["pc1", "pc2", "pc3", "pc4", "pc5"]),
                 ("chelsa_pca.nc", ["cpc1", "cpc2", "cpc3", "cpc4", "cpc5"]), ("soil_0_30cm.nc", ["ocdens", "phihox"]),
                 ("landcover.nc", ["lc_1992", "lc_1999", "lc_2000", "lc_2015", "esa_2000", "esa_2015",
                                   "cropland_density_2000", "cropland_density_2015", "dist_urban_2000", "dist_urban_2015"]),
                 ("npp.nc", ["npp"]), ("roads.nc", ["road_density"]), ("burned.nc", ["burned_frac"])]:
    ds = xr.open_dataset(CLEAN / f).load()
    for v in vars_:
        _add(ds, v)
print("coarse layers:", list(COARSE))


def sample_coarse(name: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    c = COARSE[name]
    r = np.rint((lat - c["lat0"]) / c["dlat"]).astype(int)
    k = np.rint((lon - c["lon0"]) / c["dlon"]).astype(int)
    ok = (r >= 0) & (r < c["arr"].shape[0]) & (k >= 0) & (k < c["arr"].shape[1])
    out = np.full(lon.shape, np.nan, dtype="float32")
    out[ok] = c["arr"][r[ok], k[ok]]
    return out


def coarse_grid(name: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Nearest-neighbour values of a coarse layer on a full tile (outer product of lat rows, lon cols)."""
    c = COARSE[name]
    r = np.clip(np.rint((lat - c["lat0"]) / c["dlat"]).astype(int), 0, c["arr"].shape[0] - 1)
    k = np.clip(np.rint((lon - c["lon0"]) / c["dlon"]).astype(int), 0, c["arr"].shape[1] - 1)
    return c["arr"][r[:, None], k[None, :]]


# %% [markdown]
# ## Authors' map tiles (for comparison at the same pixels)

# %%
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


# %% [markdown]
# ## Per-tile worker

# %%
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


def slope_tile(lon0: int, lat0: int, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    with rasterio.open(CLEAN / "srtm.vrt") as src:
        win = from_bounds(lon0, lat0, lon0 + 1, lat0 + 1, src.transform).round_offsets().round_lengths()
        win = Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2)
        z = src.read(1, window=win, boundless=True, fill_value=-32768).astype("float32")
        tr = src.window_transform(win)
    z[z == -32768] = np.nan
    my, mx = metres_per_degree(lat0 + 0.5)
    dy, dx = abs(tr.e) * my, tr.a * mx
    zp = np.pad(z, 1, mode="edge")
    gx = ((zp[:-2, 2:] + 2 * zp[1:-1, 2:] + zp[2:, 2:]) - (zp[:-2, :-2] + 2 * zp[1:-1, :-2] + zp[2:, :-2])) / (8 * dx)
    gy = ((zp[2:, :-2] + 2 * zp[2:, 1:-1] + zp[2:, 2:]) - (zp[:-2, :-2] + 2 * zp[:-2, 1:-1] + zp[:-2, 2:])) / (8 * dy)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))[1:-1, 1:-1]
    r = np.clip(((tr.f + tr.e) - lat) / -tr.e, 0, slope.shape[0] - 1).astype(int)
    c = np.clip((lon - (tr.c + tr.a)) / tr.a, 0, slope.shape[1] - 1).astype(int)
    return slope[r, c]


ESA_EXCL_TRAIN = [150, 151, 152, 153, 190, 200, 201, 202, 210]
ESA_EXCL_PRED = [190, 200, 201, 202, 210]
COARSE_POINT_VARS = ["pc1", "pc2", "pc3", "pc4", "pc5", "cpc1", "cpc2", "cpc3", "cpc4", "cpc5", "ocdens", "phihox", "lc_1992", "lc_1999", "lc_2000", "lc_2015",
                     "cropland_density_2000", "cropland_density_2015", "dist_urban_2000", "dist_urban_2015",
                     "npp", "road_density", "burned_frac"]


def process_tile(tile: tuple[int, int]) -> dict:
    lon0, lat0 = tile
    rng = np.random.default_rng([SEED, lon0 + 1000, lat0 + 1000])
    m = MARGIN_PX
    left, top = lon0 - m * RES, lat0 + 1 + m * RES
    size = TILE_PX + 2 * m
    win_tr = from_origin(left, top, RES, RES)
    arrs = {}
    for layer in ["treecover2000", "gain", "lossyear"]:
        with rasterio.open(CLEAN / f"hansen_{layer}.vrt") as src:
            w = from_bounds(left, top - size * RES, left + size * RES, top, src.transform).round_offsets().round_lengths()
            arrs[layer] = src.read(1, window=Window(w.col_off, w.row_off, size, size), boundless=True, fill_value=0)
    forest00 = arrs["treecover2000"] >= TREE_THRESHOLD
    lost = (arrs["lossyear"] >= 1) & (arrs["lossyear"] <= 18)
    forest18 = (forest00 | (arrs["gain"] == 1)) & ~lost
    my, mx = metres_per_degree(lat0 + 0.5)
    dy, dx = RES * my, RES * mx
    kern = disk_kernel(DENSITY_RADIUS_M, dy, dx)
    dens00, dist00 = focal(forest00, kern, dy, dx, m)
    dens18, dist18 = focal(forest18, kern, dy, dx, m)
    core = (slice(m, m + TILE_PX), slice(m, m + TILE_PX))
    f00, f18 = forest00[core], forest18[core]
    del arrs, forest00, forest18, lost

    core_tr = from_origin(lon0, lat0 + 1, RES, RES)
    shape = (TILE_PX, TILE_PX)
    tbox = shapely.box(lon0, lat0, lon0 + 1, lat0 + 1)
    country = rasterize([(shapely.clip_by_rect(COL_GEOM, *tbox.bounds), 1)], out_shape=shape, transform=core_tr,
                        fill=0, dtype="uint8").astype(bool)
    bshapes = [(shapely.clip_by_rect(g, *tbox.bounds), int(b)) for g, b in zip(biomes.geometry, biomes.BIOME_NUM)]
    bshapes = [(g, b) for g, b in bshapes if not g.is_empty]
    biome = rasterize(bshapes, out_shape=shape, transform=core_tr, fill=0, dtype="uint8") if bshapes else np.zeros(shape, "uint8")
    idx = fagan_tree.query(tbox)
    fsub = fagan.iloc[idx].sort_values("code", ascending=False)  # regrowth (1) burned in last = wins overlaps
    fag = (rasterize(((g, int(c)) for g, c in zip(fsub.geometry, fsub.code)), out_shape=shape, transform=core_tr,
                     fill=0, dtype="uint8") if len(fsub) else np.zeros(shape, "uint8"))

    lat_c = lat0 + 1 - (np.arange(TILE_PX) + 0.5) * RES
    lon_c = lon0 + (np.arange(TILE_PX) + 0.5) * RES
    esa00 = coarse_grid("esa_2000", lon_c, lat_c)
    esa15 = coarse_grid("esa_2015", lon_c, lat_c)
    dom = country & np.isin(biome, [1, 2, 3])
    cls1 = dom & (fag == 1)
    cls0 = dom & ~f00 & ~np.isin(esa00, ESA_EXCL_TRAIN) & (fag == 0)
    pred = dom & ~f18 & ~np.isin(esa15, ESA_EXCL_PRED)

    a_ell = area_ellipsoid(lat_c + RES / 2)[:, None]
    a_sph = area_sphere_a(lat_c + RES / 2)[:, None]
    pct = read_authors(AUTH_PCT, lon0, lat0, 255)
    bn = read_authors(AUTH_BIN, lon0, lat0, 3)
    pct_ok = pct <= 100
    sums = {"tile": f"{lon0}_{lat0}"}
    for mname, mask in [("country", country), ("domain", dom), ("class1", cls1), ("class0", cls0), ("pred", pred)]:
        sums[f"{mname}_area_ell"] = float((mask * a_ell).sum())
        sums[f"{mname}_area_sph"] = float((mask * a_sph).sum())
        sums[f"{mname}_n"] = int(mask.sum())
    for mname, mask in [("domain", dom), ("pred", pred)]:
        for aname, a in [("ell", a_ell), ("sph", a_sph), ("nom", 900.0)]:
            sums[f"auth_expected_{mname}_{aname}"] = float((mask * pct_ok * np.where(pct_ok, pct, 0) / 100 * a).sum())
            sums[f"auth_bin1_{mname}_{aname}"] = float((mask * (bn == 1) * a).sum())
            sums[f"auth_binvalid_{mname}_{aname}"] = float((mask * (bn <= 1) * a).sum())
            sums[f"auth_pctvalid_{mname}_{aname}"] = float((mask * pct_ok * a).sum())

    # Bernoulli sampling (inclusion ∝ pixel area) and systematic grid
    w_area = a_ell / a_ell.max()
    u = rng.random(shape)
    pick1 = cls1 & (u < RATE1 * w_area)
    pick0 = cls0 & (u < RATE0 * w_area)
    grow0 = int(round((90 - (lat0 + 1)) / RES))
    gcol0 = int(round((lon0 + 180) / RES))
    rr = (grow0 + np.arange(TILE_PX)) % GRID_STRIDE == GRID_STRIDE // 2
    cc = (gcol0 + np.arange(TILE_PX)) % GRID_STRIDE == GRID_STRIDE // 2
    grid = dom & rr[:, None] & cc[None, :]

    out = {"sums": sums}
    for name, sel in [("pool", pick1 | pick0), ("grid", grid)]:
        r, c = np.nonzero(sel)
        lon, lat = lon_c[c], lat_c[r]
        df = pd.DataFrame({
            "lon": lon, "lat": lat, "grow": grow0 + r, "gcol": gcol0 + c,
            "area_ell": a_ell[r, 0], "label": np.where(cls1[r, c], 1, np.where(cls0[r, c], 0, -1)).astype("int8"),
            "is_pred_domain": pred[r, c], "forest2000": f00[r, c], "forest2018": f18[r, c],
            "fagan_code": fag[r, c], "biome": biome[r, c],
            "forest_density_2000": dens00[r, c], "forest_density_2018": dens18[r, c],
            "dist_forest_2000": dist00[r, c], "dist_forest_2018": dist18[r, c],
            "auth_pct": pct[r, c], "auth_bin": bn[r, c],
        })
        for v in COARSE_POINT_VARS:
            df[v] = sample_coarse(v, lon, lat)
        df["slope"] = slope_tile(lon0, lat0, lon, lat) if len(df) else np.array([], dtype="float32")
        out[name] = df
    return out


# %% [markdown]
# ## Run all tiles (parallel, fork)

# %%
pool_parts, grid_parts, sums = [], [], []
ctx = mp.get_context("fork")
with ctx.Pool(N_WORKERS, maxtasksperchild=4) as p:
    for i, res in enumerate(p.imap_unordered(process_tile, tiles)):
        pool_parts.append(res["pool"])
        grid_parts.append(res["grid"])
        sums.append(res["sums"])
        if i % 10 == 0:
            print(f"{i + 1}/{len(tiles)} tiles", flush=True)
pool = pd.concat(pool_parts, ignore_index=True)
grid = pd.concat(grid_parts, ignore_index=True)
sums = pd.DataFrame(sums).sort_values("tile")
print(len(pool), "pool candidates;", len(grid), "grid points")

# %% [markdown]
# ## Balanced draw: training pool and independent validation set
#
# From the candidates, draw `TARGET_PER_CLASS` points per class (or as many as
# exist), drop points with any missing final-model predictor, then split each
# class into a training pool (6/10.87 of the points, as 6 M vs 4.87 M in the
# paper) and an independent validation set. The two sets never share a pixel.

# %%
FINAL_VARS = ["forest_density_2000", "dist_forest_2000", "ocdens", "phihox", "pc1", "pc2", "pc3", "pc4", "lc_2000", "biome"]
rng = np.random.default_rng(SEED)
cand = pool[pool.label >= 0].dropna(subset=FINAL_VARS)
parts = []
for lab in [0, 1]:
    c = cand[cand.label == lab]
    n = min(TARGET_PER_CLASS, len(c))
    parts.append(c.iloc[rng.choice(len(c), n, replace=False)])
n_bal = min(len(x) for x in parts)
parts = [x.iloc[:n_bal] for x in parts]
samples = pd.concat(parts, ignore_index=True)
samples["set"] = "train_pool"
for lab in [0, 1]:
    ix = samples.index[samples.label == lab]
    val = rng.choice(ix, int(round(len(ix) * 4.87 / 10.87)), replace=False)
    samples.loc[val, "set"] = "validation"
print(samples.groupby(["set", "label"]).size())
samples.to_parquet(CLEAN / "samples.parquet")
grid.to_parquet(CLEAN / "pred_grid.parquet")
sums.to_csv(CLEAN / "tile_sums.csv", index=False)
json.dump(dict(TREE_THRESHOLD=TREE_THRESHOLD, DENSITY_RADIUS_M=DENSITY_RADIUS_M, DIST_CAP_M=DIST_CAP_M,
               GRID_STRIDE=GRID_STRIDE, TARGET_PER_CLASS=TARGET_PER_CLASS, RATE0=RATE0, RATE1=RATE1,
               n_candidates=int(len(cand)), n_per_class=int(n_bal), n_tiles=len(tiles)),
          open(CLEAN / "features_meta.json", "w"), indent=2)

# %%
tot = sums.drop(columns="tile").sum()
print("Exact areas (Mha, ellipsoid / Mollweide-sphere):")
for k in ["country", "domain", "class1", "class0", "pred"]:
    print(f"  {k:8s} {tot[f'{k}_area_ell'] / 1e10:9.3f} {tot[f'{k}_area_sph'] / 1e10:9.3f}")
print("Authors' map within the study domain (Mha):")
for k in ["expected", "bin1", "binvalid", "pctvalid"]:
    print(f"  {k:9s}", "  ".join(f"{a}={tot[f'auth_{k}_domain_{a}'] / 1e10:.3f}" for a in ["ell", "sph", "nom"]))
