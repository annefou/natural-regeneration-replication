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
# # 03c — Step 3: robustness checks (Colombia)
#
# Every variant reuses the step-2 samples (same training draw, same validation
# set, same prediction grid) and the step-2 random-forest settings. Each is
# compared with the step-2 baseline.
#
# - **(a) Land cover before the outcome period.** Train with ESA CCI 1992 or 1999
#   instead of 2000 (Evaluator 1: use predictors observed before 2000). Prediction
#   still uses 2015 land cover, as in the paper.
# - **(b) Spatial validation.** 5-fold cross-validation on the training pool,
#   random folds vs folds grouped by HEALPix cell (WGS84) at depths 6, 7, 8
#   (~110, ~55, ~27 km). Accuracy of the step-2 model on the validation set by
#   distance to the nearest training point (0.5 km bins, as in the paper).
# - **(c) Variable selection as in the paper**, with and without the three
#   predictors recorded during the outcome period (NPP 2000–2015, burned area
#   2001–2017, road density from GRIP4): rank by permutation mean decrease in
#   accuracy over 10 balanced fits, then add variables one at a time.
# - **(d) Gradient boosting** (`HistGradientBoostingClassifier`, native
#   categorical splits) on the step-2 predictors.
# - **(e) Climate-source SENSITIVITY check (not a fix):** bioclim PC1–PC4 from
#   CHELSA v2.1 (1981–2010) instead of WorldClim v2.1 (1970–2000), same PCA
#   procedure and sample points (`02_data_clean`). CHELSA's period overlaps the
#   2000–2012 outcome period.
#
# Areas use exact WGS84 pixel areas on the 1-in-100 grid; calibrated areas use the
# step-2 prior-shift correction with the same prevalence π.

# %%
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from healpix_geo.nested import lonlat_to_healpix
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

plt.style.use("seaborn-v0_8-whitegrid")

# %%
SMOKE = os.environ.get("SMOKE", "0") == "1"  # tiny test configuration (snakemake --config smoke=1)
VARIANTS = os.environ.get("VARIANTS", "abc")  # "abc" = main robustness rule; "de" = optional variants rule
CLEAN = Path("../data/clean_smoke" if SMOKE else "../data/clean")
RESULTS = Path("../results/smoke" if SMOKE else "../results")
FIGURES = Path("../figures/smoke" if SMOKE else "../figures")
DERIVED = Path("../data/derived_smoke" if SMOKE else "../data/derived")
N_TREES = 100 if SMOKE else 500
TAG = "robustness" if VARIANTS == "abc" else "optional_variants"
print("variants:", VARIANTS, "| smoke:", SMOKE)
SEED = 20261003
MHA = 1e10
GRID_W = 100
COL_SHARE = 188_921 / 4_780_000
N_TRAIN = 2_000 if SMOKE else int(round(1_000_000 * COL_SHARE))
N_SEL = 1_000 if SMOKE else int(round(500_000 * COL_SHARE))
N_HOLDOUT = 1_000 if SMOKE else 10_000

FINAL_VARS = ["forest_density_2000", "dist_forest_2000", "ocdens", "phihox", "pc1", "pc2", "pc3", "pc4", "lc_2000", "biome"]
PRED_MAP = {"forest_density_2000": "forest_density_2018", "dist_forest_2000": "dist_forest_2018", "lc_2000": "lc_2015",
            "lc_1992": "lc_2015", "lc_1999": "lc_2015", "cropland_density_2000": "cropland_density_2015",
            "dist_urban_2000": "dist_urban_2015"}

samples = pd.read_parquet(CLEAN / "samples.parquet")
grid = pd.read_parquet(CLEAN / "pred_grid.parquet")
grid = grid[grid.is_pred_domain].reset_index(drop=True)
PI = json.load(open(DERIVED / "step2_meta.json"))["pi"]
pool = samples[samples.set == "train_pool"].reset_index(drop=True)
val = samples[samples.set == "validation"].reset_index(drop=True)
train = pd.concat([g.sample(n=min(N_TRAIN // 2, len(g)), random_state=SEED) for _, g in pool.groupby("label")])
print(len(pool), "pool;", len(train), "train;", len(val), "validation; pi =", PI)


def make_rf(p: int, seed: int = SEED) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=N_TREES, max_features=max(1, int(np.floor(np.sqrt(p)))), min_samples_leaf=1,
                                  oob_score=True, n_jobs=-1, random_state=seed)


def prior_shift(p: np.ndarray) -> np.ndarray:
    num = p * PI / 0.5
    return num / (num + (1 - p) * (1 - PI) / 0.5)


def predict_grid(model, vars_: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = grid[[PRED_MAP.get(v, v) for v in vars_]].to_numpy()
    ok = ~np.isnan(X).any(axis=1)
    p = np.full(len(grid), np.nan)
    p[ok] = model.predict_proba(X[ok])[:, 1]
    return p, ok


def areas(p: np.ndarray, ok: np.ndarray) -> dict[str, float]:
    a = grid.area_ell.to_numpy()[ok] * GRID_W
    pp = p[ok]
    pc = prior_shift(pp)
    return {"expected_uncal_mha": (pp * a).sum() / MHA, "expected_prior_mha": (pc * a).sum() / MHA,
            "gt05_uncal_mha": ((pp > 0.5) * a).sum() / MHA, "gt05_prior_mha": ((pc > 0.5) * a).sum() / MHA,
            "grid_coverage": float(ok.mean())}


rows = []


def fit_eval(name: str, vars_: list[str], model=None, note: str = "") -> dict:
    tr = train.dropna(subset=vars_)
    va = val.dropna(subset=vars_)
    model = model if model is not None else make_rf(len(vars_))
    model.fit(tr[vars_].to_numpy(), tr.label.to_numpy())
    acc = float(((model.predict_proba(va[vars_].to_numpy())[:, 1] > 0.5) == va.label.to_numpy()).mean())
    p, ok = predict_grid(model, vars_)
    ap, ab = grid.auth_pct.to_numpy(), grid.auth_bin.to_numpy()
    sp, sb = ok & (ap <= 100), ok & (ab <= 1)
    row = dict(variant=name, n_vars=len(vars_), variables=" ".join(vars_), n_train=len(tr), n_val=len(va),
               oob_accuracy=getattr(model, "oob_score_", np.nan), validation_accuracy=acc, **areas(p, ok),
               pearson_vs_authors_pct=float(stats.pearsonr(p[sp], ap[sp] / 100)[0]),
               kappa_vs_authors_bin=float(cohen_kappa_score(p[sb] > 0.5, ab[sb] == 1)), note=note)
    rows.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "variables"})
    return row


# %% [markdown]
# ## Baseline (step 2) and (a) pre-2000 land cover
#
# The baseline row is refitted in both rules so each output table is self-contained.

# %%
fit_eval("baseline_step2", FINAL_VARS, note="final 10-variable model, LC 2000")
if VARIANTS == "abc":
    for yr in [1992, 1999]:
        fit_eval(f"a_landcover_{yr}", [f"lc_{yr}" if v == "lc_2000" else v for v in FINAL_VARS],
                 note=f"ESA CCI {yr} instead of 2000 for training")

# %% [markdown]
# ## (d) Gradient boosting and (e) CHELSA climate sensitivity (optional rule, `VARIANTS=de`)

# %%
CHELSA_VARS = [{"pc1": "cpc1", "pc2": "cpc2", "pc3": "cpc3", "pc4": "cpc4"}.get(v, v) for v in FINAL_VARS]
if VARIANTS == "de":
    cat_mask = [v in ("lc_2000", "biome") for v in FINAL_VARS]
    fit_eval("d_hist_gradient_boosting", FINAL_VARS,
             model=HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=SEED),
             note="sklearn defaults (max_iter=100, lr=0.1, early stopping auto); categorical LC and biome")
    fit_eval("e_chelsa_bioclim_SENSITIVITY", CHELSA_VARS,
             note="SENSITIVITY check, not a fix: CHELSA 1981-2010 overlaps the outcome period")

# %% [markdown]
# ## (b) Random vs spatially blocked cross-validation (HEALPix, WGS84)
#
# Main rule: step-2 variable set. Optional rule: CHELSA variant (e).

# %%
cv_rows = []
cv_sets = [("baseline_step2", FINAL_VARS)] if VARIANTS == "abc" else [("e_chelsa_bioclim_SENSITIVITY", CHELSA_VARS)]
for vset_name, vset in cv_sets:
    pv = pool.dropna(subset=vset)
    Xp, yp = pv[vset].to_numpy(), pv.label.to_numpy()
    schemes = {"random": StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xp, yp)}
    for depth in [6, 7, 8]:
        cells = lonlat_to_healpix(pv.lon.to_numpy(), pv.lat.to_numpy(), depth, ellipsoid="WGS84")
        schemes[f"healpix_d{depth}"] = GroupKFold(5).split(Xp, yp, groups=cells)
    for name, splits in schemes.items():
        accs = []
        for k, (tri, tei) in enumerate(splits):
            sub = np.random.default_rng(SEED + k).choice(tri, min(N_TRAIN, len(tri)), replace=False)
            m = make_rf(len(vset)).fit(Xp[sub], yp[sub])
            accs.append(float(((m.predict_proba(Xp[tei])[:, 1] > 0.5) == yp[tei]).mean()))
        cv_rows.append(dict(variables=vset_name, scheme=name, mean_accuracy=np.mean(accs), sd=np.std(accs, ddof=1),
                            folds=" ".join(f"{a:.4f}" for a in accs)))
        print(cv_rows[-1])
cv = pd.DataFrame(cv_rows)
cv.to_csv(RESULTS / ("step3_spatial_cv.csv" if VARIANTS == "abc" else "step3_spatial_cv_chelsa.csv"), index=False)

# %% [markdown]
# Accuracy of the step-2 model on the validation set by distance to the nearest
# training point (great-circle distance via 3-D chord on the WGS84 ellipsoid).

# %%
def ecef(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lo, la = np.radians(lon), np.radians(lat)
    nn = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    return np.column_stack([nn * np.cos(la) * np.cos(lo), nn * np.cos(la) * np.sin(lo), nn * (1 - e2) * np.sin(la)])


def balanced_draw(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return pd.concat([g.sample(n=min(n // 2, len(g)), random_state=seed) for _, g in df.groupby("label")])


if VARIANTS == "abc":
    base = make_rf(len(FINAL_VARS)).fit(train[FINAL_VARS].to_numpy(), train.label.to_numpy())
    d, _ = cKDTree(ecef(train.lon.to_numpy(), train.lat.to_numpy())).query(ecef(val.lon.to_numpy(), val.lat.to_numpy()))
    correct = (base.predict_proba(val[FINAL_VARS].to_numpy())[:, 1] > 0.5) == val.label.to_numpy()
    edges = np.arange(0, 8.5, 0.5)
    b = np.digitize(d / 1000, edges) - 1
    dist_rows = []
    for i in range(len(edges)):
        sel = b == i
        lab = f"{edges[i]:.1f}-{edges[i] + 0.5:.1f} km" if i < len(edges) - 1 else f">{edges[-1]:.1f} km"
        if sel.any():
            dist_rows.append(dict(bin=lab, n=int(sel.sum()), accuracy=float(correct[sel].mean()),
                                  share_regrowth=float(val.label.to_numpy()[sel].mean())))
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(RESULTS / "step3_accuracy_by_distance.csv", index=False)
    print(dist.to_string(index=False))

# %% [markdown]
# ## (c) Variable selection as in the paper, with and without NPP, burned area, roads
#
# A predictor that is entirely missing (e.g. burned area if neither GlobFire nor
# MCD64A1 could be fetched) is dropped from the candidate list and reported.

# %%
if VARIANTS == "abc":
    CANDIDATES_ALL = FINAL_VARS + ["pc5", "slope", "cropland_density_2000", "dist_urban_2000",
                                   "npp", "burned_frac", "road_density"]
    missing = [v for v in CANDIDATES_ALL if pool[v].notna().mean() < 0.5]
    print("candidates dropped for missing data:", missing)
    CANDIDATES_ALL = [v for v in CANDIDATES_ALL if v not in missing]
    DURING = [v for v in ["npp", "burned_frac", "road_density"] if v in CANDIDATES_ALL]
    pool_c = pool.dropna(subset=CANDIDATES_ALL).reset_index(drop=True)
    val_c = val.dropna(subset=CANDIDATES_ALL).reset_index(drop=True)
    print(len(pool_c), "pool records with all candidates;", len(val_c), "validation")

    def select(cands: list[str], tag: str) -> tuple[list[str], pd.DataFrame]:
        ranks = []
        for k in range(10):
            tr = balanced_draw(pool_c, N_SEL, SEED + k)
            rest = pool_c.drop(tr.index)
            ho = balanced_draw(rest, N_HOLDOUT, SEED + 100 + k)
            m = make_rf(len(cands), SEED + k).fit(tr[cands].to_numpy(), tr.label.to_numpy())
            pi_ = permutation_importance(m, ho[cands].to_numpy(), ho.label.to_numpy(), scoring="accuracy",
                                         n_repeats=3, random_state=SEED + k, n_jobs=-1)
            ranks.append(pd.Series(pi_.importances_mean, index=cands).rank(ascending=False))
        order = pd.concat(ranks, axis=1).sum(axis=1).sort_values().index.tolist()
        tr = balanced_draw(pool_c, N_SEL, SEED + 999)
        curve = []
        for k in range(1, len(order) + 1):
            vs = order[:k]
            m = make_rf(k).fit(tr[vs].to_numpy(), tr.label.to_numpy())
            acc = float(((m.predict_proba(val_c[vs].to_numpy())[:, 1] > 0.5) == val_c.label.to_numpy()).mean())
            curve.append(dict(set=tag, k=k, added=order[k - 1], oob_accuracy=m.oob_score_, validation_accuracy=acc))
        curve = pd.DataFrame(curve)
        best = curve.validation_accuracy.max()
        k_sel = int(curve.loc[curve.validation_accuracy >= best - 0.001, "k"].min())
        print(tag, "ranking:", order, "| selected k =", k_sel)
        return order[:k_sel], curve

    sel_all, curve_all = select(CANDIDATES_ALL, "with_npp_burned_roads")
    sel_wo, curve_wo = select([v for v in CANDIDATES_ALL if v not in DURING], "without_npp_burned_roads")
    curves = pd.concat([curve_all, curve_wo])
    curves.to_csv(RESULTS / "step3_variable_selection_curves.csv", index=False)
    fit_eval("c_selected_with_during_vars", sel_all, note=f"selection incl. {DURING}; dropped {missing}")
    fit_eval("c_selected_without_during_vars", sel_wo, note="selection excl. NPP, burned area, roads")
    fit_eval("c_all_candidates", CANDIDATES_ALL, note=f"all {len(CANDIDATES_ALL)} candidates")
    fit_eval("c_all_candidates_without_during_vars", [v for v in CANDIDATES_ALL if v not in DURING],
             note=f"{len(CANDIDATES_ALL) - len(DURING)} candidates")

# %%
out = pd.DataFrame(rows)
out.to_csv(RESULTS / ("step3_robustness_colombia.csv" if VARIANTS == "abc" else "step3_optional_variants_colombia.csv"),
           index=False)
with pd.option_context("display.width", 250):
    print(out.drop(columns="variables").to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# %%
if VARIANTS == "abc":
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for tag, c in curves.groupby("set"):
        axes[0].plot(c.k, c.validation_accuracy, "o-", label=tag)
    axes[0].set_xlabel("number of variables (forward order)")
    axes[0].set_ylabel("validation accuracy")
    axes[0].legend(fontsize=8)
    axes[0].set_title("(c) Variable selection curves")
    axes[1].bar(dist.bin, dist.accuracy)
    axes[1].set_ylim(0.5, 1)
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set_title("(b) Validation accuracy by distance to nearest training point")
    fig.tight_layout()
    fig.savefig(FIGURES / "step3_robustness.png", dpi=150, bbox_inches="tight")
    plt.show()
