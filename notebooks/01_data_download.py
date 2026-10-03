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
# # 01 — Data download (step 1: reproduction from the authors' published map)
#
# Fetches every input for step 1 (Colombia reproduction of the natural-regeneration
# potential area) into `data/raw/`. No credentials are needed.
#
# | Input | Source | Licence |
# |---|---|---|
# | Continuous potential, 30 m, 10° tiles (`pnv_pct_30m_tile_*.tif`, integer %) | Williams et al. (2022), Zenodo [10.5281/zenodo.7428804](https://doi.org/10.5281/zenodo.7428804) | CC-BY-4.0 |
# | Binary potential > 0.5, 30 m (`pnv_bin_30m.zip`) | same record | CC-BY-4.0 |
# | Country boundaries, level 0 (`STEP1_COUNTRIES`: Colombia, plus Costa Rica as a second-country check) | GADM 4.1 (`gadm41_<ISO3>.gpkg`), <https://gadm.org> | GADM licence (free for academic, non-commercial use; no redistribution) |
#
# Only tiles that intersect the GADM Colombia polygon are downloaded. Tile
# selection is computed from the polygon, not hard-coded. Every file is checked
# against the Zenodo md5 (authors' files) or a pinned md5 (GADM). Downloads are
# idempotent: a file that already exists with the right checksum is not fetched
# again.

# %%
import hashlib
import json
import shutil
import zipfile
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from shapely.geometry import box

# %%
RAW_DIR = Path("../data/raw")
ZEN_DIR = RAW_DIR / "zenodo_7428804"
BIN_DIR = ZEN_DIR / "pnv_bin_30m"
GADM_DIR = RAW_DIR / "gadm"
for d in (ZEN_DIR, BIN_DIR, GADM_DIR):
    d.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD = "7428804"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
STEP1_COUNTRIES = ["COL", "CRI"]
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso3}.gpkg"
# GADM publishes no checksum; pinned from the first download (2026-10-03).
GADM_MD5 = {
    "COL": "d1ed49e54c2429fd9f5577bccd4a6851",  # Last-Modified 2022-07-18, 64,114,688 bytes
    "CRI": "083df869e4f97f7b48260a976087941f",  # Last-Modified 2022-07-18, 12,754,944 bytes
}


# %%
def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, out: Path, md5: str) -> Path:
    """Download url to out unless it already exists with the expected md5."""
    if out.exists() and md5sum(out) == md5:
        print(f"ok (cached)  {out.name}")
        return out
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f, length=1 << 22)
    got = md5sum(tmp)
    if got != md5:
        tmp.unlink()
        raise RuntimeError(f"md5 mismatch for {out.name}: {got} != {md5}")
    tmp.rename(out)
    print(f"downloaded   {out.name}")
    return out


# %% [markdown]
# ## Colombia boundary (GADM 4.1, level 0)
#
# The paper used GADM (2022) for country sums; GADM 4.1 (July 2022) is that release.

# %%
gadm_paths = {iso: download(GADM_URL.format(iso3=iso), GADM_DIR / f"gadm41_{iso}.gpkg", GADM_MD5[iso])
              for iso in STEP1_COUNTRIES}
countries = pd.concat([gpd.read_file(p, layer="ADM_ADM_0") for p in gadm_paths.values()], ignore_index=True)
print(countries[["GID_0", "COUNTRY"]].to_string())
col_geom = countries.geometry.union_all()  # union of all step-1 countries, used for tile selection

# %% [markdown]
# ## Authors' published map (Zenodo 10.5281/zenodo.7428804)
#
# File list and md5 checksums come from the Zenodo REST API. Continuous tiles are
# named `pnv_pct_30m_tile_<xmin>_<xmax>_<ymin>_<ymax>.tif`; a tile is kept when
# that rectangle intersects the Colombia polygon (islands included).

# %%
record = requests.get(ZENODO_API, timeout=60).json()
files = {f["key"]: f for f in record["files"]}
print(record["metadata"]["title"], "|", record["metadata"]["license"]["id"], "|", len(files), "files")


def tile_box(key: str):
    xmin, xmax, ymin, ymax = (float(v) for v in key.removesuffix(".tif").split("_")[-4:])
    return box(xmin, ymin, xmax, ymax)


pct_keys = sorted(
    k for k in files if k.startswith("pnv_pct_30m_tile_") and tile_box(k).intersects(col_geom)
)
print("continuous tiles intersecting Colombia:", pct_keys)

# %%
for key in pct_keys + ["pnv_bin_30m.zip"]:
    f = files[key]
    download(f["links"]["self"], ZEN_DIR / key, f["checksum"].removeprefix("md5:"))

# %% [markdown]
# The binary product is a zip of 84 numbered tiles. Extract only the members whose
# raster bounds (read from the header inside the zip) intersect Colombia.

# %%
zip_path = ZEN_DIR / "pnv_bin_30m.zip"
bin_members = []
with zipfile.ZipFile(zip_path) as zf:
    for name in zf.namelist():
        if not name.endswith(".tif"):
            continue
        with rasterio.open(f"/vsizip/{zip_path.resolve()}/{name}") as r:
            if not box(*r.bounds).intersects(col_geom):
                continue
        out = BIN_DIR / Path(name).name
        info = zf.getinfo(name)
        if not (out.exists() and out.stat().st_size == info.file_size):
            with zf.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        bin_members.append(out.name)
print("binary tiles intersecting Colombia:", sorted(bin_members))

# %% [markdown]
# ## Source log

# %%
SOURCES = [
    {
        "name": "Williams et al. natural regeneration potential, continuous 30 m tiles",
        "doi": "10.5281/zenodo.7428804",
        "url": ZENODO_API,
        "license": record["metadata"]["license"]["id"],
        "accessed_on": date.today().isoformat(),
        "files": [
            {"key": k, "md5": files[k]["checksum"].removeprefix("md5:"), "size": files[k]["size"]}
            for k in pct_keys
        ],
    },
    {
        "name": "Williams et al. natural regeneration potential, binary (>0.5) 30 m",
        "doi": "10.5281/zenodo.7428804",
        "url": files["pnv_bin_30m.zip"]["links"]["self"],
        "license": record["metadata"]["license"]["id"],
        "accessed_on": date.today().isoformat(),
        "md5": files["pnv_bin_30m.zip"]["checksum"].removeprefix("md5:"),
        "extracted_members": sorted(bin_members),
    },
] + [
    {
        "name": f"GADM 4.1 {iso} (level 0-2)",
        "doi": None,
        "url": GADM_URL.format(iso3=iso),
        "license": "GADM licence: free for academic and other non-commercial use; redistribution not allowed",
        "accessed_on": date.today().isoformat(),
        "md5": GADM_MD5[iso],
        "sha256": sha256sum(path),
    }
    for iso, path in gadm_paths.items()
]
with open(RAW_DIR / "sources.json", "w") as f:
    json.dump({"sources": SOURCES}, f, indent=2)
print(f"Logged {len(SOURCES)} source(s) to {RAW_DIR / 'sources.json'}")
