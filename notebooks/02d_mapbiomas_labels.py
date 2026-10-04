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
# # 02d — Independent regrowth labels from MapBiomas Colombia Collection 3
#
# **Source rule.** MapBiomas Colombia, *Apéndice – Módulo de pérdida de vegetación
# y vegetación secundaria en Colección 3*, v1.0, October 2025 (Huertas, Rojas,
# Medina). The live URL returns 404; the Wayback snapshot of 2026-05-14 is in
# `data/raw/mapbiomas_docs/`. MapBiomas publishes the resulting
# secondary-vegetation maps only through its platform, not in the public bucket,
# so the rule is re-implemented here at points:
#
# 1. **Groups** (module Table 1). Anthropic = pasture, agriculture, forest
#    plantation, palm oil, mosaic of agriculture and pasture, urban
#    infrastructure, mining, aquaculture. Natural = forest, mangrove, flooded
#    forest, flooded natural non-forest formation, herbaceous formation, other
#    natural non-forest formation. Not included = hypersaline tidal flat, other
#    non-vegetated areas, river/lake/ocean, glacier, not observed. The C3 codes
#    are mapped below. Codes that Table 1 does not name are assigned as
#    documented in `docs/deviations.md`.
# 2. **Trajectories** (§2.2), from 1987, year by year:
#    - *Loss* in year t: Natural in t−1 and t−2 (in the maps of the previous
#      steps), and Anthropic in t and t+1 (in the input). This is primary loss,
#      or secondary loss if the pixel was secondary vegetation.
#    - *Regeneration* in year t: Anthropic in t−1 and t−2 (previous steps), and
#      Natural in t, t+1 and t+2 (input). The pixel becomes secondary vegetation.
#    - Any other change of group is noise and is reverted to the previous year's
#      group (§2.2.4).
#    - A pixel classified as "Other" in any year of the input series is class
#      Other (Table 2).
# 3. **Regrowth labels** (our analogue of Fagan et al.'s 2000–2012 natural regrowth;
#    this step is ours, not MapBiomas'). For a window (S, O, P):
#    - **class 1:** Anthropic in year S; a MapBiomas regeneration event in a year
#      t ∈ [S+1, O]; secondary vegetation without any loss from t through P; and
#      the raw class is forest (codes 3, 5, 6, 49) in every year O…P.
#    - **class 0:** never Other; non-forest and not urban / mining / aquaculture /
#      solar in year S; never forest in S…P; no regeneration event in S+1…P.
#    - every other pixel is −1 (excluded).
#
#    Windows (S, O, P): E1 (2000, 2012, 2016), with persistence sensitivity
#    (2000, 2012, 2014) and (2000, 2012, 2018); forward test period 1
#    (2000, 2008, 2012) and period 2 (2012, 2020, 2024): equal 8-year onset windows,
#    4-year forest persistence as in E1, and no overlap between the periods.
#
# **Points.** The 1-in-100 systematic grid of `02b` (12.2 M points over the study
# domain) and the step-2 sample points. MapBiomas uses a 0.000269494585° grid that
# is not aligned with the 0.00025° Hansen lattice, so each point takes the value of
# the MapBiomas pixel that contains it. Nothing is resampled.
#
# **Forward-test predictors for 2012** at the grid points:
# - forest 2012 = Hansen tree cover 2000 ≥ 30 % and no loss in 2001–2012;
# - forest density (1 km disk) and distance to forest (capped at 25 km), computed
#   with the same code and tiles as `02b`;
# - land cover = ESA CCI 2012 in the same 11 classes.
#
# 2000 values are recomputed in the same pass as a check against `02b`.
#
# **History predictors (extension, E3)** from MapBiomas 1985–1999: years since the
# pixel was last forest before 2000 (16 = not forest in 1985–1999), and the number
# of Anthropic years in 1985–1999.

# %%
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from scipy import ndimage, signal

# %%
SMOKE = os.environ.get("SMOKE", "0") == "1"
RAW = Path("../data/raw")
CLEAN = Path("../data/clean_smoke" if SMOKE else "../data/clean")
MB_DIR = RAW / "mapbiomas_c3"
MB_URL = ("https://storage.googleapis.com/mapbiomas-public/initiatives/colombia/collection_3/coverage/"
          "colombia_coverage_{y}.tif")
YEARS = np.arange(1985, 2025)
N_WORKERS = int(os.environ.get("N_WORKERS", min(12, os.cpu_count() or 1)))
RES = 0.00025
TILE_PX = 4000
MARGIN_PX = 1000
TREE_THRESHOLD = 30
DENSITY_RADIUS_M = 1000.0
DIST_CAP_M = 25_000.0
WGS84_A, WGS84_F = 6378137.0, 1 / 298.257223563
E2 = WGS84_F * (2 - WGS84_F)

FOREST = [1, 3, 5, 6, 49]
NATURAL = FOREST + [10, 11, 12, 13, 50, 81, 82]
ANTHROPIC = [9, 14, 15, 18, 19, 20, 21, 24, 30, 31, 35, 36, 39, 40, 41, 46, 47, 48, 62, 74, 75]
OTHER = [0, 22, 23, 25, 26, 27, 29, 32, 33, 34, 68]
NOT_CLASS0_AT_START = [24, 30, 31, 75]  # urban, mining, aquaculture, solar (analogue of the ESA exclusions)
GROUP = np.zeros(256, dtype="uint8")  # 0 = Other / not included
GROUP[ANTHROPIC] = 1
GROUP[NATURAL] = 2
IS_FOREST = np.zeros(256, dtype=bool)
IS_FOREST[FOREST] = True
ANTH, PRIM, SEC, LOSS_P, REGEN, LOSS_S, OTH = 1, 2, 3, 4, 5, 6, 7
# e1_p2 / e1_p6: persistence sensitivity of E1 (forest for 2 / 6 years after 2012).
# fw1 / fw2: forward test, same 8-year onset window and 4-year persistence as e1;
# period-2 predictors are measured in 2012, when period-1 outcomes are complete.
WINDOWS = {"e1": (2000, 2012, 2016), "e1_p2": (2000, 2012, 2014), "e1_p6": (2000, 2012, 2018),
           "fw1": (2000, 2008, 2012), "fw2": (2012, 2020, 2024)}
FW_YEAR = 2012

grid = pd.read_parquet(CLEAN / "pred_grid.parquet", columns=["lon", "lat", "grow", "gcol", "area_ell",
                                                              "forest_density_2000", "dist_forest_2000", "lc_2000"])
samples = pd.read_parquet(CLEAN / "samples.parquet", columns=["lon", "lat", "grow", "gcol", "set", "label"])
pts = pd.concat([grid[["lon", "lat"]], samples[["lon", "lat"]]], ignore_index=True)
print(f"{len(grid):,} grid points, {len(samples):,} sample points")

# %% [markdown]
# ## 1. MapBiomas codes at the points, 1985–2024

# %%
def mb_path(y: int) -> str:
    p = MB_DIR / f"colombia_coverage_{y}.tif"
    return str(p) if p.exists() else f"/vsicurl/{MB_URL.format(y=y)}"


with rasterio.open(mb_path(2000)) as r:
    MB_T, MB_H, MB_W = r.transform, r.height, r.width
ROW = np.floor((MB_T.f - pts.lat.to_numpy()) / -MB_T.e).astype(np.int64)
COL = np.floor((pts.lon.to_numpy() - MB_T.c) / MB_T.a).astype(np.int64)
INSIDE = (ROW >= 0) & (ROW < MB_H) & (COL >= 0) & (COL < MB_W)
ORDER = np.argsort(ROW, kind="stable")
STRIP = 512


def read_year(y: int) -> np.ndarray:
    out = np.zeros(len(ROW), dtype="uint8")
    o = ORDER[INSIDE[ORDER]]
    strips = ROW[o] // STRIP
    bounds = np.flatnonzero(np.diff(strips)) + 1
    with rasterio.open(mb_path(y)) as src:
        for idx in np.split(o, bounds):
            if not len(idx):
                continue
            r0 = int(ROW[idx].min())
            r1, c0, c1 = int(ROW[idx].max()) + 1, int(COL[idx].min()), int(COL[idx].max()) + 1
            a = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            out[idx] = a[ROW[idx] - r0, COL[idx] - c0]
    return out


t0 = time.time()
with mp.get_context("fork").Pool(min(N_WORKERS, len(YEARS))) as p:
    CODES = np.column_stack(p.map(read_year, YEARS.tolist()))
print(f"read {len(YEARS)} years at {len(ROW):,} points in {time.time() - t0:.0f} s; outside raster {(~INSIDE).sum()}")
vals, cnt = np.unique(CODES, return_counts=True)
print("code frequencies:", dict(zip(vals.tolist(), cnt.tolist())))
unknown = sorted(set(vals.tolist()) - set(FOREST + NATURAL + ANTHROPIC + OTHER))
print("codes not in the mapping (treated as Other):", unknown)

# %% [markdown]
# ## 2. Trajectory analysis (module §2.2, Table 2)

# %%
def trajectories(codes: np.ndarray) -> np.ndarray:
    g = GROUP[codes]
    n, ny = g.shape
    out = np.zeros((n, ny), dtype="uint8")
    eff = g.copy()  # group after reverting noise (the "maps of the previous steps")
    status = np.where(g[:, 0] == 1, ANTH, PRIM).astype("uint8")
    for k in (0, 1):
        status = np.where(g[:, k] == 1, ANTH, np.where(g[:, k] == 2, PRIM, status))
        out[:, k] = status
    for k in range(2, ny):
        prev, prev2, cur = eff[:, k - 1], eff[:, k - 2], g[:, k]
        nxt1 = g[:, k + 1] if k + 1 < ny else np.zeros(n, "uint8")
        nxt2 = g[:, k + 2] if k + 2 < ny else np.zeros(n, "uint8")
        change = cur != prev
        loss = change & (prev == 2) & (prev2 == 2) & (cur == 1) & (nxt1 == 1)
        regen = change & (prev == 1) & (prev2 == 1) & (cur == 2) & (nxt1 == 2) & (nxt2 == 2)
        eff[:, k] = np.where(change & ~loss & ~regen, prev, cur)
        ev = np.zeros(n, "uint8")
        ev[loss] = np.where(status[loss] == SEC, LOSS_S, LOSS_P)
        ev[regen] = REGEN
        status = np.where(loss, ANTH, np.where(regen, SEC, status))
        out[:, k] = np.where(ev > 0, ev, status)
    out[(g == 0).any(axis=1)] = OTH
    return out


t0 = time.time()
TRAJ = trajectories(CODES)
print(f"trajectories in {time.time() - t0:.0f} s")
yi = {int(y): i for i, y in enumerate(YEARS)}
for y in (2000, 2012, 2016, 2024):
    v, c = np.unique(TRAJ[:, yi[y]], return_counts=True)
    print(y, dict(zip(v.tolist(), (c / len(TRAJ)).round(4).tolist())))


def window_labels(S: int, O: int, P: int) -> tuple[np.ndarray, np.ndarray]:
    s, o, p = yi[S], yi[O], yi[P]
    tr = TRAJ[:, s:p + 1]
    is_regen = tr == REGEN
    last_regen = np.where(is_regen.any(axis=1), tr.shape[1] - 1 - np.argmax(is_regen[:, ::-1], axis=1), -1)
    yrs = np.arange(tr.shape[1])
    after = yrs[None, :] >= last_regen[:, None]
    secondary_since = np.all(~after | np.isin(tr, [SEC, REGEN]), axis=1)
    onset_ok = (last_regen >= 1) & (last_regen <= o - s)
    forest_end = IS_FOREST[CODES[:, o:p + 1]].all(axis=1)
    anth_start = np.isin(TRAJ[:, s], [ANTH, LOSS_P, LOSS_S])
    cls1 = anth_start & onset_ok & secondary_since & forest_end & (TRAJ[:, s] != OTH)
    never_forest = ~IS_FOREST[CODES[:, s:p + 1]].any(axis=1)
    cls0 = ((TRAJ[:, s] != OTH) & ~IS_FOREST[CODES[:, s]] & ~np.isin(CODES[:, s], NOT_CLASS0_AT_START) & never_forest
            & ~is_regen[:, 1:].any(axis=1))
    lab = np.full(len(tr), -1, dtype="int8")
    lab[cls0] = 0
    lab[cls1] = 1
    onset = np.where(cls1, S + last_regen, 0).astype("int16")
    return lab, onset


LAB = {}
for name, (S, O, P) in WINDOWS.items():
    LAB[name], LAB[f"onset_{name}"] = window_labels(S, O, P)
    v, c = np.unique(LAB[name][:len(grid)], return_counts=True)
    print(f"{name} {S}-{O}-{P}: grid label counts", dict(zip(v.tolist(), c.tolist())))
regen_any_e1 = (TRAJ[:, yi[2001]:yi[2012] + 1] == REGEN).any(axis=1)

# %% [markdown]
# ## 3. History predictors from 1985–1999 (extension)

# %%
hist = CODES[:, yi[1985]:yi[1999] + 1]
was_forest = IS_FOREST[hist]
last_forest = np.where(was_forest.any(axis=1), 1999 - np.argmax(was_forest[:, ::-1], axis=1), 0)
YSF = np.where(last_forest > 0, 2000 - last_forest, 16).astype("float32")
NANTH = (GROUP[hist] == 1).sum(axis=1).astype("float32")

# %% [markdown]
# ## 4. Forest density / distance to forest and land cover for 2012 at the grid points

# %%
def metres_per_degree(lat: float) -> tuple[float, float]:
    phi = np.radians(lat)
    s2 = np.sin(phi) ** 2
    return (float(np.radians(1) * WGS84_A * (1 - E2) / (1 - E2 * s2) ** 1.5),
            float(np.radians(1) * WGS84_A / np.sqrt(1 - E2 * s2) * np.cos(phi)))


def disk_kernel(radius_m: float, dy: float, dx: float) -> np.ndarray:
    ry, rx = int(np.ceil(radius_m / dy)), int(np.ceil(radius_m / dx))
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    k = ((yy * dy) ** 2 + (xx * dx) ** 2 <= radius_m**2).astype("float32")
    return k / k.sum()


def focal(forest: np.ndarray, kernel: np.ndarray, dy: float, dx: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Identical to 02b.focal."""
    kr, kc = kernel.shape[0] // 2, kernel.shape[1] // 2
    sub = forest[m - kr:m + TILE_PX + kr, m - kc:m + TILE_PX + kc].astype("float32")
    dens = signal.fftconvolve(sub, kernel, mode="valid")
    if forest.any():
        dist = ndimage.distance_transform_edt(~forest, sampling=(dy, dx))[m:m + TILE_PX, m:m + TILE_PX]
    else:
        dist = np.full((TILE_PX, TILE_PX), np.inf)
    return np.clip(dens, 0, 1).astype("float32"), np.minimum(dist, DIST_CAP_M).astype("float32")


lon0s, lat0s = np.floor(grid.lon.to_numpy()).astype(int), np.floor(grid.lat.to_numpy()).astype(int)
tile_key = pd.Series(list(zip(lon0s, lat0s)))
TILES = tile_key.groupby(tile_key).indices
GROW, GCOL = grid.grow.to_numpy(), grid.gcol.to_numpy()


def process_tile(tile: tuple[int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lon0, lat0 = tile
    idx = TILES[tile]
    m = MARGIN_PX
    left, top = lon0 - m * RES, lat0 + 1 + m * RES
    size = TILE_PX + 2 * m
    arrs = {}
    for layer in ["treecover2000", "lossyear"]:
        with rasterio.open(CLEAN / f"hansen_{layer}.vrt") as src:
            w = from_bounds(left, top - size * RES, left + size * RES, top, src.transform).round_offsets().round_lengths()
            arrs[layer] = src.read(1, window=Window(w.col_off, w.row_off, size, size), boundless=True, fill_value=0)
    f00 = arrs["treecover2000"] >= TREE_THRESHOLD
    f12 = f00 & ~((arrs["lossyear"] >= 1) & (arrs["lossyear"] <= FW_YEAR - 2000))
    del arrs
    my, mx = metres_per_degree(lat0 + 0.5)
    dy, dx = RES * my, RES * mx
    kern = disk_kernel(DENSITY_RADIUS_M, dy, dx)
    r = GROW[idx] - int(round((90 - (lat0 + 1)) / RES))
    c = GCOL[idx] - int(round((lon0 + 180) / RES))
    out = {}
    for yr, f in (("2000", f00), ("2012", f12)):
        dens, dist = focal(f, kern, dy, dx, m)
        out[f"forest_density_{yr}"], out[f"dist_forest_{yr}"] = dens[r, c], dist[r, c]
        out[f"forest{yr}"] = f[m + r, m + c]
    return idx, out


F = {k: np.full(len(grid), np.nan, dtype="float32") for k in
     ["forest_density_2000", "dist_forest_2000", "forest_density_2012", "dist_forest_2012", "forest2000", "forest2012"]}
t0 = time.time()
with mp.get_context("fork").Pool(N_WORKERS, maxtasksperchild=4) as p:
    for i, (idx, out) in enumerate(p.imap_unordered(process_tile, list(TILES))):
        for k, v in out.items():
            F[k][idx] = v
        if i % 20 == 0:
            print(f"{i + 1}/{len(TILES)} tiles, {time.time() - t0:.0f} s", flush=True)
chk = {v: float(np.nanmax(np.abs(F[v] - grid[v].to_numpy()))) for v in ["forest_density_2000", "dist_forest_2000"]}
print("check against 02b (max abs difference, 2000 inputs):", chk)

lc_classes = pd.read_csv(CLEAN / "landcover_classes.csv")
LUT = np.zeros(256, dtype="float32")
for k, codes in zip(lc_classes["class"], lc_classes.esa_cci_codes):
    LUT[[int(x) for x in str(codes).split()]] = k


def esa_at(year: int) -> np.ndarray:
    with rasterio.open(RAW / "esacci_lc" / f"esacci_lc_{year}_colombia.tif") as src:
        a = src.read(1)
        t = src.transform
    rr = np.clip(np.floor((t.f - grid.lat.to_numpy()) / -t.e).astype(int), 0, a.shape[0] - 1)
    cc = np.clip(np.floor((grid.lon.to_numpy() - t.c) / t.a).astype(int), 0, a.shape[1] - 1)
    return a[rr, cc]


esa12 = esa_at(FW_YEAR)
lc_2012 = LUT[esa12]
lc_check = float((LUT[esa_at(2000)] == grid.lc_2000.to_numpy()).mean())
print(f"ESA 2000 lookup reproduces grid lc_2000 at {lc_check:.4%} of grid points")

# %% [markdown]
# ## 5. Write

# %%
ng = len(grid)
gout = pd.DataFrame({"grow": grid.grow.to_numpy(), "gcol": grid.gcol.to_numpy(),
                     "mb_code_2000": CODES[:ng, yi[2000]], "mb_code_2012": CODES[:ng, yi[2012]],
                     "mb_traj_2000": TRAJ[:ng, yi[2000]], "mb_traj_2012": TRAJ[:ng, yi[2012]],
                     "mb_regen_any_2001_2012": regen_any_e1[:ng],
                     "mb_years_since_forest": YSF[:ng], "mb_years_anthropic_8599": NANTH[:ng],
                     "esa_2012": esa12, "lc_2012": lc_2012,
                     "forest2012": F["forest2012"].astype(bool),
                     "forest_density_2012": F["forest_density_2012"], "dist_forest_2012": F["dist_forest_2012"]})
for k in WINDOWS:
    gout[f"mb_label_{k}"] = LAB[k][:ng]
    gout[f"mb_onset_{k}"] = LAB[f"onset_{k}"][:ng]
gout.to_parquet(CLEAN / "mapbiomas_grid.parquet")
sout = pd.DataFrame({"grow": samples.grow.to_numpy(), "gcol": samples.gcol.to_numpy(), "set": samples.set.to_numpy(),
                     "label": samples.label.to_numpy(), "mb_code_2000": CODES[ng:, yi[2000]],
                     "mb_traj_2000": TRAJ[ng:, yi[2000]], "mb_regen_any_2001_2012": regen_any_e1[ng:],
                     "mb_years_since_forest": YSF[ng:], "mb_years_anthropic_8599": NANTH[ng:]})
for k in WINDOWS:
    sout[f"mb_label_{k}"] = LAB[k][ng:]
    sout[f"mb_onset_{k}"] = LAB[f"onset_{k}"][ng:]
sout.to_parquet(CLEAN / "mapbiomas_samples.parquet")
pd.DataFrame(CODES, columns=[f"mb_{y}" for y in YEARS]).assign(
    is_grid=np.r_[np.ones(ng, bool), np.zeros(len(samples), bool)]).to_parquet(CLEAN / "mapbiomas_codes.parquet")
meta = dict(windows=WINDOWS, n_grid=ng, n_samples=len(samples), unknown_codes=unknown, check_02b_max_abs_diff=chk,
            check_lc_2000_match=lc_check, n_tiles=len(TILES),
            grid_label_counts={k: {int(a): int(b) for a, b in zip(*np.unique(LAB[k][:ng], return_counts=True))}
                               for k in WINDOWS})
json.dump(meta, open(CLEAN / "mapbiomas_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
