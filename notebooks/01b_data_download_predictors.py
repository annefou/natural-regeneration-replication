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
# # 01b — Data download for steps 2–3 (labels and predictors, Colombia)
#
# Fetches the inputs of the independent random-forest replication into
# `data/raw/`. Where a provider serves a global file that supports HTTP range
# reads, only a Colombia window (bounding box of GADM 4.1 COL + 0.5° margin) is
# read and saved; otherwise the file is downloaded whole. Every step is idempotent.
#
# | Input | Source | Access | Licence |
# |---|---|---|---|
# | Natural-regrowth / plantation polygons 2000–2012 (labels) | Fagan et al. 2022, GFW S3 GeoPackage | remote spatial-index read (Colombia only) | CC BY-NC 4.0 |
# | Tree cover 2000, gain 2000–2012, loss year 2001–2024 | Hansen GFC-2025-v1.13 (Google Storage) | tile download, md5 from `x-goog-hash` | CC BY 4.0 |
# | Land cover 1992, 1999, 2000, 2015 (v2.0.7) | ESA CCI LC, CEDA archive | remote window | ESA CCI data policy |
# | Soil organic carbon density, pH (0, 5, 15, 30 cm) | SoilGrids250m v2017 (ISRIC) | remote window | CC BY 4.0 |
# | Bioclim bio1–bio19 (1970–2000, 30 s) | WorldClim v2.1 | full zip (global PCA needs global sample) | WorldClim terms |
# | Biomes | RESOLVE Ecoregions 2017 (Dinerstein et al.) | zip, md5 from `x-goog-hash` | CC BY 4.0 |
# | NPP mean 2000–2015 (MOD17A3, C5.5, 30 s) | NTSG, Univ. Montana | file | NASA / NTSG open |
# | Burned area, monthly 2001–2017 | GlobFire / GWIS (Artés et al. 2019), PANGAEA | monthly zips, Colombia subset kept | CC BY-SA 4.0 |
# | Road density | GRIP4 (Meijer et al. 2018), PBL | zip | open with attribution |
# | Elevation (for slope) | SRTMGL1 v003, NASA LP DAAC | Earthdata login (netrc) | NASA open |
#
# **Smoke mode** (`SMOKE=1`, set by `snakemake --config smoke=1`): same code on a
# tiny configuration. Only 2 GlobFire months, 2 MCD64A1 months (to test the
# fallback), and SRTM tiles of the smoke sub-window are fetched; the source log is
# written to `sources_predictors_smoke.json`. Shared inputs are cached in `data/raw/`.
#
# **Burned area fallback.** If any GlobFire month cannot be fetched after 8
# attempts (the PANGAEA file host returned HTTP 503 at times on 2026-10-03), the
# whole burned-area series switches to MODIS MCD64A1 v061 (Earthdata), the
# product GlobFire is derived from. The source used is written to
# `data/raw/burned_source.json`.
#
# **Run time: this step can take very long.** In the run rendered here
# (2026-10-03) it took 2 h 27 min. The PANGAEA host answered "503 Service
# Unavailable" so often that 161 of the 204 GlobFire monthly files needed retries
# (433 "retry …" lines in the output below; the worst file needed 7 of its 8
# attempts). All 204 months were finally downloaded from GlobFire, with no failed
# month and no fallback (`burned_source.json`: `"source": "globfire"`). The retry
# lines are therefore noise, not errors. Each retry waits 10, 20, … 70 s, and a
# stalled connection can hold an attempt for up to 10 min, so on a bad day this
# step can run for many hours, or end on the MCD64A1 fallback. Burned area is used
# only by robustness check 3(c), where it was not selected into the final model.
#
# **Credentials.** SRTM (and the MCD64A1 fallback) need one: NASA Earthdata, read by `earthaccess` from
# `~/.netrc` (`machine urs.earthdata.nasa.gov login … password …`). In CI, write
# that file from a secret (e.g. `EARTHDATA_NETRC_BASE64`). Nothing else needs a login.

# %%
import base64
import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from osgeo import gdal, ogr
from rasterio.windows import from_bounds
from shapely.geometry import box

gdal.UseExceptions()
ogr.UseExceptions()
gdal.SetConfigOption("GDAL_HTTP_MULTIRANGE", "YES")
gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "5")
gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "5")
gdal.SetConfigOption("CPL_VSIL_CURL_CHUNK_SIZE", "4194304")
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

# %%
RAW = Path("../data/raw")
RAW.mkdir(parents=True, exist_ok=True)
TODAY = date.today().isoformat()
SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_BBOX = (-75.0, 4.0, -73.0, 6.0)

col = gpd.read_file(RAW / "gadm" / "gadm41_COL.gpkg", layer="ADM_ADM_0")
COL_GEOM = shapely.make_valid(col.geometry.iloc[0])
W, S, E, N = COL_GEOM.bounds
MARGIN = 0.5
BBOX = (np.floor(W - MARGIN), np.floor(S - MARGIN), np.ceil(E + MARGIN), np.ceil(N + MARGIN))
print("Colombia window (W, S, E, N):", BBOX)

SOURCES: list[dict] = []


# %%
def file_hash(path: Path, algo: str = "md5", chunk: int = 1 << 22) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def gcs_md5(url: str) -> str | None:
    """md5 published by Google Cloud Storage in the x-goog-hash header (hex)."""
    head = requests.head(url, timeout=60)
    head.raise_for_status()
    for part in head.headers.get("x-goog-hash", "").split(","):
        k, _, v = part.strip().partition("=")
        if k == "md5":
            return base64.b64decode(v + "=" * (-len(v) % 4)).hex()
    return None


def download(url: str, out: Path, md5: str | None = None, retries: int = 6, auth_session=None) -> Path:
    """Idempotent download; verifies md5 when known, otherwise records sha256."""
    if out.exists() and (md5 is None or file_hash(out) == md5):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    sess = auth_session or requests
    for attempt in range(retries):
        try:
            with sess.get(url, stream=True, timeout=600) as r:
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
    """Read a lon/lat window of a remote raster and save it as a tiled, compressed GeoTIFF."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_url) as src:
        win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        prof = src.profile | dict(
            driver="GTiff", width=data.shape[1], height=data.shape[0],
            transform=src.window_transform(win), tiled=True, blockxsize=512, blockysize=512,
            compress="deflate", predictor=2, BIGTIFF="IF_SAFER",
        )
    tmp = out.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(data, 1)
    tmp.rename(out)
    return out


# %% [markdown]
# ## 1. Labels: Fagan et al. (2022) polygons intersecting Colombia
#
# The GeoPackage (4.29 GB) has 30 longitude-band layers, each with an R-tree, so
# only Colombia's features are read. All three classes (`regrowth`,
# `plantation`, `open`) are kept: class 1 = regrowth, and all three are
# excluded from the class-0 domain.

# %%
FAGAN_URL = ("https://gfw2-data.s3.amazonaws.com/plantations/pantropical_tree_plantation_expansion/"
             "pantropical_tree_plantation_expansion_2000_2012.gpkg")
fagan_out = RAW / "fagan" / "fagan2022_colombia.parquet"
fagan_out.parent.mkdir(parents=True, exist_ok=True)
if not fagan_out.exists():
    ds = ogr.Open("/vsicurl/" + FAGAN_URL)
    colg = ogr.CreateGeometryFromWkb(shapely.to_wkb(COL_GEOM))
    parts = []
    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayer(i)
        xmin, xmax, ymin, ymax = layer.GetExtent()
        if xmax < W or xmin > E or ymax < S or ymin > N:
            continue
        layer.SetSpatialFilter(colg)
        recs = [
            (f.GetField("sys_index"), f.GetField("pred3class"), f.GetField("conf_rank"),
             f.GetField("fin_conf"), f.GetField("area4326"), layer.GetName(), bytes(f.GetGeometryRef().ExportToWkb()))
            for f in layer
        ]
        if recs:
            df = pd.DataFrame(recs, columns=["sys_index", "pred3class", "conf_rank", "fin_conf", "area4326", "layer", "wkb"])
            parts.append(gpd.GeoDataFrame(df.drop(columns="wkb"), geometry=shapely.from_wkb(df.wkb), crs="EPSG:4326"))
        print(layer.GetName(), len(recs))
    fagan = pd.concat(parts, ignore_index=True)
    fagan.to_parquet(fagan_out)
fagan = gpd.read_parquet(fagan_out)
print(fagan.groupby(["pred3class", "conf_rank"]).size())
etag = requests.head(FAGAN_URL, timeout=60).headers.get("ETag")
SOURCES.append(dict(name="Fagan et al. 2022 pantropical tree plantation expansion 2000-2012 (Colombia subset)",
                    doi="10.1038/s41893-022-00904-w", url=FAGAN_URL, license="CC-BY-NC-4.0",
                    accessed_on=TODAY, source_etag=etag, n_features=len(fagan),
                    counts=fagan.pred3class.value_counts().to_dict()))

# %% [markdown]
# ## 2. Hansen Global Forest Change v1.13 (tree cover 2000, gain, loss year)

# %%
GFC = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/Hansen_GFC-2025-v1.13_{layer}_{tile}.tif"


def gfc_tiles(geom) -> list[str]:
    tiles = []
    for top in range(-10, 40, 10):
        for left in range(-90, -50, 10):
            if box(left, top - 10, left + 10, top).intersects(geom):
                ns = f"{abs(top):02d}{'N' if top >= 0 else 'S'}"
                tiles.append(f"{ns}_{abs(left):03d}W")
    return tiles


HANSEN_TILES = gfc_tiles(COL_GEOM)
print("Hansen tiles:", HANSEN_TILES)
hansen_files = []
for layer in ["treecover2000", "gain", "lossyear"]:
    for tile in HANSEN_TILES:
        url = GFC.format(layer=layer, tile=tile)
        md5 = gcs_md5(url)
        p = download(url, RAW / "hansen" / Path(url).name, md5)
        hansen_files.append(dict(file=p.name, md5=md5))
        print("ok", p.name)
SOURCES.append(dict(name="Hansen Global Forest Change 2000-2024 v1.13", doi="10.1126/science.1244693",
                    url=GFC, license="CC-BY-4.0", accessed_on=TODAY, files=hansen_files))

# %% [markdown]
# ## 3. ESA CCI Land Cover v2.0.7 (CEDA, no login)
#
# 2000 (training, as in the paper), 2015 (prediction, as in the paper), and
# 1992 / 1999 (step-3 robustness: land cover recorded before the outcome period).

# %%
ESA = "https://dap.ceda.ac.uk/neodc/esacci/land_cover/data/land_cover_maps/v2.0.7/ESACCI-LC-L4-LCCS-Map-300m-P1Y-{y}-v2.0.7.tif"
esa_files = []
for year in [1992, 1999, 2000, 2015]:
    p = window_to_gtiff("/vsicurl/" + ESA.format(y=year), RAW / "esacci_lc" / f"esacci_lc_{year}_colombia.tif")
    esa_files.append(dict(file=p.name, sha256=file_hash(p, "sha256")))
    print("ok", p.name)
SOURCES.append(dict(name="ESA CCI Land Cover v2.0.7 (Colombia window)", doi="10.5285/4761751d7c844e228ec2f5fe11b2e3b0",
                    url=ESA, license="ESA CCI data policy (free, attribution)", accessed_on=TODAY,
                    window_wsen=BBOX, files=esa_files))

# %% [markdown]
# ## 4. SoilGrids250m v2017: OCDENS and PHIHOX at 0, 5, 15, 30 cm (sl1–sl4)

# %%
SG = "https://files.isric.org/soilgrids/former/2017-03-10/data/{var}_M_sl{d}_250m_ll.tif"
sg_files = []
for var in ["OCDENS", "PHIHOX"]:
    for d in [1, 2, 3, 4]:
        p = window_to_gtiff("/vsicurl/" + SG.format(var=var, d=d), RAW / "soilgrids2017" / f"{var}_M_sl{d}_colombia.tif")
        sg_files.append(dict(file=p.name, sha256=file_hash(p, "sha256")))
        print("ok", p.name)
SOURCES.append(dict(name="SoilGrids250m 2017-03-10 (Colombia window)", doi="10.1371/journal.pone.0169748",
                    url=SG, license="CC-BY-4.0", accessed_on=TODAY, window_wsen=BBOX, files=sg_files))

# %% [markdown]
# ## 5. Biomes: RESOLVE Ecoregions 2017

# %%
ECO_URL = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"
eco = download(ECO_URL, RAW / "ecoregions" / "Ecoregions2017.zip", gcs_md5(ECO_URL))
SOURCES.append(dict(name="RESOLVE Ecoregions 2017", doi="10.1093/biosci/bix014", url=ECO_URL, license="CC-BY-4.0",
                    accessed_on=TODAY, md5=file_hash(eco)))

# %% [markdown]
# ## 6. NPP: MOD17A3 (collection 5.5) mean 2000–2015, 30 arc-seconds (NTSG)

# %%
NPP_URL = ("http://files.ntsg.umt.edu/data/NTSG_Products/MOD17/GeoTIFF/MOD17A3/GeoTIFF_30arcsec/"
           "MOD17A3_Science_NPP_mean_00_15.tif")
npp = window_to_gtiff("/vsicurl/" + NPP_URL, RAW / "npp" / "MOD17A3_NPP_mean_00_15_colombia.tif")
SOURCES.append(dict(name="MOD17A3 NPP mean 2000-2015 (NTSG, Colombia window)", doi="10.1016/j.rse.2004.12.011",
                    url=NPP_URL, license="NASA/NTSG open data", accessed_on=TODAY, sha256=file_hash(npp, "sha256")))

# %% [markdown]
# ## 7. Road density: GRIP4 total density (m km⁻², 5 arc-minutes)

# %%
GRIP_URL = "https://dataportaal.pbl.nl/data/GRIP4/GRIP4_density_total.zip"
grip = download(GRIP_URL, RAW / "grip4" / "GRIP4_density_total.zip")
SOURCES.append(dict(name="GRIP4 road density, total", doi="10.1088/1748-9326/aabd42", url=GRIP_URL,
                    license="open with attribution (PBL/GLOBIO)", accessed_on=TODAY, sha256=file_hash(grip, "sha256")))

# %% [markdown]
# ## 8. Burned area: GlobFire (Artés et al. 2019), monthly shapefiles 2001–2017
#
# Each global monthly zip (a tar of a shapefile) is downloaded, the Colombia
# window is extracted to one GeoParquet per month, and the zip is deleted.

# %%
GF_URL = "https://hs.pangaea.de/Maps/MCD64A1_burnt-areas/MODIS_BA_GLOBAL_1_{m}_{y}.zip"
gf_dir = RAW / "globfire"
gf_dir.mkdir(parents=True, exist_ok=True)
MONTHS = [(2005, 1), (2005, 2)] if SMOKE else [(y, m) for y in range(2001, 2018) for m in range(1, 13)]
gf_months, gf_failed = [], []
for y, m in MONTHS:
    out = gf_dir / f"globfire_{y}_{m:02d}_colombia.parquet"
    if not out.exists():
        try:
            z = download(GF_URL.format(m=m, y=y), gf_dir / f"tmp_{y}_{m}_{os.getpid()}.zip", retries=8)
        except Exception as err:  # noqa: BLE001
            print(f"GlobFire {y}-{m:02d} failed: {err}")
            gf_failed.append((y, m))
            break
        name = f"MODIS_BA_GLOBAL_1_{m}_{y}"
        g = gpd.read_file(f"/vsitar//vsizip/{z.resolve()}/{name}.tar/{name}.shp", bbox=BBOX)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        g.to_parquet(tmp)
        tmp.rename(out)
        z.unlink()
    gf_months.append(out.name)
print(len(gf_months), "GlobFire months; failed:", gf_failed)
use_globfire = not gf_failed and len(gf_months) == len(MONTHS)
if use_globfire:
    SOURCES.append(dict(name="GlobFire / GWIS global wildfire database, monthly shapefiles (Colombia window)",
                        doi="10.1594/PANGAEA.895898", url=GF_URL, license="CC-BY-SA-4.0", accessed_on=TODAY,
                        window_wsen=BBOX, months=len(gf_months)))

# %% [markdown]
# ### Burned-area fallback: MODIS MCD64A1 v061 (Earthdata)
#
# Monthly 500 m `Burn Date` (> 0 = burned) on the MODIS sinusoidal tiles covering
# Colombia (h10–h11, v07–v09). Each month's granules are reprojected (nearest) to
# a 15″ lon/lat grid over the Colombia window and summed into a count of burned
# months; HDF files are deleted after use. In smoke mode it always runs on 2
# months and one tile, to test the path.

# %%
import earthaccess  # noqa: E402

earthaccess.login(strategy="netrc")
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402

MCD_TILES = ["h10v08"] if SMOKE else ["h10v07", "h10v08", "h10v09", "h11v07", "h11v08", "h11v09"]
mcd_out = RAW / "mcd64a1" / ("mcd64a1_burned_months_smoke.tif" if SMOKE else "mcd64a1_burned_months_colombia.tif")
if (SMOKE or not use_globfire) and not mcd_out.exists():
    mcd_out.parent.mkdir(parents=True, exist_ok=True)
    res = 1 / 240
    dst_tr = from_origin(BBOX[0], BBOX[3], res, res)
    shape = (int(round((BBOX[3] - BBOX[1]) / res)), int(round((BBOX[2] - BBOX[0]) / res)))
    count = np.zeros(shape, dtype="uint16")
    nmonths = 0
    for y, m in MONTHS:
        start = f"{y}-{m:02d}-01"
        end = f"{y + (m == 12)}-{(m % 12) + 1:02d}-01"
        grans = earthaccess.search_data(short_name="MCD64A1", version="061", bounding_box=(W, S, E, N),
                                        temporal=(start, end), count=-1)
        grans = [g for g in grans if g["umm"]["GranuleUR"].split(".")[2] in MCD_TILES
                 and g["umm"]["GranuleUR"].split(".")[1] == f"A{y}{pd.Timestamp(start).dayofyear:03d}"]
        tmpdir = RAW / "mcd64a1" / f"tmp_{os.getpid()}"
        files = earthaccess.download(grans, str(tmpdir), threads=6)
        month = np.zeros(shape, dtype="uint8")
        for f in files:
            sds = f'HDF4_EOS:EOS_GRID:"{f}":MOD_Grid_Monthly_500m_DB_BA:"Burn Date"'
            with rasterio.open(sds) as src:
                burned = (src.read(1) > 0).astype("uint8")
                tmp = np.zeros(shape, dtype="uint8")
                reproject(burned, tmp, src_transform=src.transform, src_crs=src.crs, dst_transform=dst_tr,
                          dst_crs="EPSG:4326", resampling=Resampling.nearest)
                month |= tmp
        count += month
        nmonths += 1
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("MCD64A1", y, m, len(files), "granules", flush=True)
    with rasterio.open(mcd_out, "w", driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="uint16",
                       crs="EPSG:4326", transform=dst_tr, compress="deflate", tiled=True) as dst:
        dst.write(count, 1)
        dst.update_tags(months=nmonths, tiles=",".join(MCD_TILES))
if not use_globfire:
    SOURCES.append(dict(name="MODIS MCD64A1 v061 burned area (fallback for GlobFire)", doi="10.5067/MODIS/MCD64A1.061",
                        url="https://lpdaac.usgs.gov", license="NASA open data", accessed_on=TODAY,
                        months=len(MONTHS), tiles=MCD_TILES))
burned_source = "globfire" if use_globfire else ("mcd64a1" if mcd_out.exists() else "none")
json.dump({"source": burned_source, "months": len(MONTHS), "globfire_failed": gf_failed,
           "mcd64a1_file": mcd_out.name if mcd_out.exists() else None},
          open(RAW / ("burned_source_smoke.json" if SMOKE else "burned_source.json"), "w"), indent=2)
print("burned-area source:", burned_source)

# %% [markdown]
# ## 9. Elevation for slope: SRTMGL1 v003 (Earthdata login)

# %%
srtm_dir = RAW / "srtm"
srtm_dir.mkdir(parents=True, exist_ok=True)
granules = earthaccess.search_data(short_name="SRTMGL1", version="003", bounding_box=(W, S, E, N), count=-1)


def srtm_box(g) -> shapely.Geometry:
    name = g["umm"]["GranuleUR"].split(".")[0]  # e.g. N04W075
    lat = int(name[1:3]) * (1 if name[0] == "N" else -1)
    lon = int(name[4:7]) * (1 if name[3] == "E" else -1)
    return box(lon, lat, lon + 1, lat + 1)


granules = [g for g in granules if srtm_box(g).intersects(shapely.box(*SMOKE_BBOX) if SMOKE else COL_GEOM)]
missing = [g for g in granules if not (srtm_dir / Path(g.data_links()[0]).name).exists()]
print(len(granules), "SRTM tiles intersect Colombia;", len(missing), "to download")
if missing:
    earthaccess.download(missing, str(srtm_dir), threads=8)
srtm_files = sorted(p.name for p in srtm_dir.glob("*.hgt.zip"))
SOURCES.append(dict(name="NASA SRTMGL1 v003", doi="10.5067/MEaSUREs/SRTM/SRTMGL1.003", url="https://lpdaac.usgs.gov",
                    license="NASA open data", accessed_on=TODAY, n_files=len(srtm_files)))

# %% [markdown]
# ## 10. WorldClim v2.1 bioclim, 30 s (global zip, 10.4 GB)
#
# The bioclim PCA must be re-derived on a global sample of land points (the
# loadings are not published), so the global files are needed. Downloaded last
# because it is the largest file.

# %%
WC_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_bio.zip"
wc_size = int(requests.head(WC_URL, timeout=60).headers["Content-Length"])
wc = RAW / "worldclim" / "wc2.1_30s_bio.zip"
if not (wc.exists() and wc.stat().st_size == wc_size):
    wc.unlink(missing_ok=True)
    download(WC_URL, wc)
assert wc.stat().st_size == wc_size, "WorldClim zip size differs from Content-Length"
with zipfile.ZipFile(wc) as zf:
    print(len(zf.namelist()), "files in WorldClim zip")
SOURCES.append(dict(name="WorldClim v2.1 bioclimatic variables 1970-2000, 30 s", doi="10.1002/joc.5086", url=WC_URL,
                    license="WorldClim terms (research / non-commercial)", accessed_on=TODAY,
                    size=wc.stat().st_size))

# %% [markdown]
# ## Source log

# %%
with open(RAW / ("sources_predictors_smoke.json" if SMOKE else "sources_predictors.json"), "w") as f:
    json.dump({"sources": SOURCES}, f, indent=2, default=str)
print(f"Logged {len(SOURCES)} sources")
