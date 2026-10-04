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
# # 03g — Does the blocked cross-validation depend on HEALPix or on WGS84?
#
# Step 3(b) groups the training pool into HEALPix cells (WGS84) and holds whole cells out
# together. A reader could ask whether the blocked accuracies (0.854–0.869) are an artefact
# of that grid. Here the **same procedure** (pool, `GroupKFold(5)`, per-fold training
# subsample, seeds) is repeated with:
#
# - HEALPix NESTED on WGS84 (as in step 3(b); must reproduce its numbers),
# - HEALPix NESTED on the sphere,
# - latitude/longitude squares with the same area as the HEALPix cells of each depth,
# - the same squares shifted by half a block (to show how much an arbitrary placement of
#   block boundaries moves the result).
#
# Only the grouping changes; the model, data and folds' sizes are otherwise identical.

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from healpix_geo.nested import lonlat_to_healpix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

# %%
SMOKE = os.environ.get("SMOKE", "0") == "1"
CLEAN = Path("../data/clean_smoke" if SMOKE else "../data/clean")
RESULTS = Path("../results/smoke" if SMOKE else "../results")
N_TREES = 100 if SMOKE else 500
SEED = 20261003
COL_SHARE = 188_921 / 4_780_000
N_TRAIN = 2_000 if SMOKE else int(round(1_000_000 * COL_SHARE))
FINAL_VARS = ["forest_density_2000", "dist_forest_2000", "ocdens", "phihox", "pc1", "pc2", "pc3", "pc4", "lc_2000", "biome"]
DEPTHS = [6, 7, 8]
R_AUTHALIC = 6371007.181  # m; HEALPix cell area = 4 pi R^2 / (12 * 4^depth)
KM_PER_DEG = 111.32

samples = pd.read_parquet(CLEAN / "samples.parquet")
pool = samples[samples.set == "train_pool"].reset_index(drop=True)
pv = pool.dropna(subset=FINAL_VARS)
Xp, yp = pv[FINAL_VARS].to_numpy(), pv.label.to_numpy()
lon, lat = pv.lon.to_numpy(), pv.lat.to_numpy()
print(f"pool records {len(pv):,}")


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=N_TREES, max_features=int(np.floor(np.sqrt(len(FINAL_VARS)))),
                                  min_samples_leaf=1, oob_score=True, n_jobs=-1, random_state=SEED)


def cell_side_deg(depth: int) -> float:
    area = 4 * np.pi * R_AUTHALIC**2 / (12 * 4**depth)
    return float(np.sqrt(area) / 1000 / KM_PER_DEG)


def squares(side: float, shift: float = 0.0) -> np.ndarray:
    i = np.floor((lon + 180 + shift) / side).astype("int64")
    j = np.floor((lat + 90 + shift) / side).astype("int64")
    return i * 100_000 + j


# %%
schemes: dict[str, tuple[str, int | None, np.ndarray | None]] = {"random": ("random", None, None)}
for d in DEPTHS:
    side = cell_side_deg(d)
    schemes[f"healpix_wgs84_d{d}"] = ("HEALPix, WGS84", d, lonlat_to_healpix(lon, lat, d, ellipsoid="WGS84"))
    schemes[f"healpix_sphere_d{d}"] = ("HEALPix, sphere", d, lonlat_to_healpix(lon, lat, d, ellipsoid="sphere"))
    schemes[f"latlon_d{d}"] = ("lat/lon squares", d, squares(side))
    schemes[f"latlon_shifted_d{d}"] = ("lat/lon squares, shifted half a block", d, squares(side, side / 2))

rows = []
for name, (family, depth, groups) in schemes.items():
    splits = (StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xp, yp) if groups is None
              else GroupKFold(5).split(Xp, yp, groups=groups))
    accs = []
    for k, (tri, tei) in enumerate(splits):
        sub = np.random.default_rng(SEED + k).choice(tri, min(N_TRAIN, len(tri)), replace=False)
        m = make_rf().fit(Xp[sub], yp[sub])
        accs.append(float(((m.predict_proba(Xp[tei])[:, 1] > 0.5) == yp[tei]).mean()))
    rows.append(dict(scheme=name, family=family, depth=depth,
                     block_km=None if depth is None else round(cell_side_deg(depth) * KM_PER_DEG, 1),
                     n_blocks=None if groups is None else int(len(np.unique(groups))),
                     mean_accuracy=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)),
                     folds=" ".join(f"{a:.4f}" for a in accs)))
    print(rows[-1])
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "step3_blocking_sensitivity.csv", index=False)

# %% [markdown]
# ## Same block size, different grids

# %%
tab = res[res.depth.notna()].pivot_table(index=["depth", "block_km"], columns="family", values="mean_accuracy")
tab["max - min"] = tab.max(axis=1) - tab.min(axis=1)
print(f"random CV: {res.loc[res.scheme == 'random', 'mean_accuracy'].iloc[0]:.4f}")
print(tab.round(4).to_string())
