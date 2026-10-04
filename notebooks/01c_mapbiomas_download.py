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
# # 01c — Data download: MapBiomas Colombia Collection 3 and ESA CCI 2012
#
# Inputs of the MapBiomas label experiments (`02d`, `03f`).
#
# - **MapBiomas Colombia Collection 3**, annual land cover 1985–2024 (CC BY 4.0;
#   cite "Proyecto MapBiomas Colombia – Colección 3 de la serie anual de mapas de
#   cobertura y uso del suelo de Colombia"). These are public Cloud-Optimised GeoTIFFs in the
#   `mapbiomas-public` Google Cloud Storage bucket (no login): one 63,923 × 68,191
#   Byte raster per year on a 0.000269494585° grid, NoData 0. All 40 years are
#   downloaded (about 4.5 GB) and checked against the md5 that GCS publishes, because the label
#   rule needs the whole time series at 12 M points spread over the country.
#   That reads every block of every file anyway, so it is cheaper to read them
#   locally once. The download stops if free disk space would fall below 10 GB.
# - **ESA CCI Land Cover v2.0.7, 2012**: the Colombia window, as in `01b`. It is the
#   land-cover predictor of the forward test.
# - **Method documents**: the MapBiomas Colombia C3 general appendix and the C3
#   vegetation-loss / secondary-vegetation module appendix. The module appendix's
#   live URL returns 404; the 2026-05-14 Wayback Machine snapshot is used.
#
# **Smoke mode** (`SMOKE=1`): nothing is downloaded from MapBiomas, because `02d`
# then reads the smoke window remotely. The source log is
# `data/raw/sources_mapbiomas_smoke.json`.

# %%
import base64
import hashlib
import json
import os
import shutil
import time
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
import shapely
from rasterio.windows import from_bounds

# %%
RAW = Path("../data/raw")
MB_DIR = RAW / "mapbiomas_c3"
DOC_DIR = RAW / "mapbiomas_docs"
TODAY = date.today().isoformat()
SMOKE = os.environ.get("SMOKE", "0") == "1"
MIN_FREE_GB = 10.0
YEARS = list(range(1985, 2025))
MB_URL = ("https://storage.googleapis.com/mapbiomas-public/initiatives/colombia/collection_3/coverage/"
          "colombia_coverage_{y}.tif")
ESA = "https://dap.ceda.ac.uk/neodc/esacci/land_cover/data/land_cover_maps/v2.0.7/ESACCI-LC-L4-LCCS-Map-300m-P1Y-{y}-v2.0.7.tif"
DOCS = {
    "Apendice-1-Colombia-Coleccion-3.0.pdf":
        "https://colombia.mapbiomas.org/wp-content/uploads/sites/9/2026/10/Apendice-1-Colombia-Coleccion-3.0.pdf",
    "Apendice_modulo_perdida_de_vegetacion.pdf":
        "https://web.archive.org/web/20260514084706id_/https://colombia.mapbiomas.org/wp-content/uploads/sites/3/"
        "2025/10/Apendice_modulo_perdida_de_vegetacion.pdf",
    "Codigo-de-la-Leyenda-coleccion-3.pdf":
        "https://colombia.mapbiomas.org/wp-content/uploads/sites/9/2026/09/Codigo-de-la-Leyenda-coleccion-3.pdf",
}

col = gpd.read_file(RAW / "gadm" / "gadm41_COL.gpkg", layer="ADM_ADM_0")
W, S, E, N = shapely.make_valid(col.geometry.iloc[0]).bounds
BBOX = (np.floor(W - 0.5), np.floor(S - 0.5), np.ceil(E + 0.5), np.ceil(N + 0.5))


def file_hash(path: Path, algo: str = "md5", chunk: int = 1 << 22) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def gcs_meta(url: str) -> tuple[str | None, int]:
    head = requests.head(url, timeout=60)
    head.raise_for_status()
    md5 = None
    for part in head.headers.get("x-goog-hash", "").split(","):
        k, _, v = part.strip().partition("=")
        if k == "md5":
            md5 = base64.b64decode(v + "=" * (-len(v) % 4)).hex()
    return md5, int(head.headers.get("content-length", 0))


def free_gb() -> float:
    return shutil.disk_usage(RAW).free / 1e9


def download(url: str, out: Path, md5: str | None = None, retries: int = 6) -> Path:
    """Idempotent download, verified against md5 when known (as in 01b)."""
    if out.exists() and (md5 is None or file_hash(out) == md5):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r.raw, f, length=1 << 22)
            break
        except (requests.RequestException, OSError) as err:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for {out.name}: {err}")
            time.sleep(10 * (attempt + 1))
    if md5 is not None and (got := file_hash(tmp)) != md5:
        tmp.unlink()
        raise RuntimeError(f"md5 mismatch for {out.name}: {got} != {md5}")
    tmp.rename(out)
    return out


def window_to_gtiff(src_url: str, out: Path, bbox: tuple[float, float, float, float] = BBOX) -> Path:
    """As in 01b: read a lon/lat window of a remote raster and save it as a tiled GeoTIFF."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_url) as src:
        win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        prof = src.profile | dict(driver="GTiff", width=data.shape[1], height=data.shape[0],
                                  transform=src.window_transform(win), tiled=True, blockxsize=512, blockysize=512,
                                  compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
    tmp = out.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(data, 1)
    tmp.rename(out)
    return out


SOURCES: list[dict] = []

# %% [markdown]
# ## 1. MapBiomas Colombia Collection 3 annual coverage, 1985–2024

# %%
mb_files = []
for y in YEARS:
    url = MB_URL.format(y=y)
    md5, size = gcs_meta(url)
    out = MB_DIR / f"colombia_coverage_{y}.tif"
    if not SMOKE:
        if not out.exists() and free_gb() - size / 1e9 < MIN_FREE_GB:
            raise RuntimeError(f"free disk {free_gb():.1f} GB: downloading {out.name} would leave < {MIN_FREE_GB} GB")
        download(url, out, md5)
    mb_files.append(dict(year=y, url=url, md5=md5, bytes=size, local=str(out) if out.exists() else None))
print(f"{sum(f['local'] is not None for f in mb_files)} of {len(YEARS)} MapBiomas years local; free disk {free_gb():.1f} GB")
with rasterio.open(f"/vsicurl/{MB_URL.format(y=2000)}") as r:
    mb_grid = dict(width=r.width, height=r.height, transform=list(r.transform)[:6], nodata=r.nodata, crs=r.crs.to_string())
print(mb_grid)
SOURCES.append(dict(name="MapBiomas Colombia Collection 3, annual land cover 1985-2024",
                    citation="Proyecto MapBiomas Colombia - Coleccion 3 de la serie anual de mapas de cobertura y uso "
                             "del suelo de Colombia", license="CC-BY-4.0", accessed_on=TODAY, grid=mb_grid,
                    files=mb_files))

# %% [markdown]
# ## 2. ESA CCI Land Cover 2012 (Colombia window, as in `01b`)

# %%
esa12 = window_to_gtiff("/vsicurl/" + ESA.format(y=2012), RAW / "esacci_lc" / "esacci_lc_2012_colombia.tif")
SOURCES.append(dict(name="ESA CCI Land Cover v2.0.7, 2012 (Colombia window)", doi="10.5285/4761751d7c844e228ec2f5fe11b2e3b0",
                    url=ESA.format(y=2012), license="ESA CCI data policy (free, attribution)", accessed_on=TODAY,
                    window_wsen=BBOX, file=esa12.name, sha256=file_hash(esa12, "sha256")))

# %% [markdown]
# ## 3. Method documents

# %%
docs = []
for name, url in DOCS.items():
    p = download(url, DOC_DIR / name)
    docs.append(dict(file=name, url=url, sha256=file_hash(p, "sha256")))
SOURCES.append(dict(name="MapBiomas Colombia Collection 3 method documents", accessed_on=TODAY, files=docs,
                    note="vegetation-loss module appendix: live URL 404 on 2026-10-04, Wayback snapshot 2026-05-14"))
out = RAW / ("sources_mapbiomas_smoke.json" if SMOKE else "sources_mapbiomas.json")
json.dump(SOURCES, open(out, "w"), indent=2)
print("wrote", out)
