# DOMAIN.md — domain flavour

This file encodes domain-specific conventions for replications in **biodiversity + earth observation**. It is loaded by `CLAUDE.md` and applied automatically.

To swap to a different domain (e.g. genomics, social science, materials), copy the matching file from `docs/domain-flavours/` over this file, or write a new flavour using `docs/domain-flavours/_template.md` as a scaffold and ask Claude to help you fill it in.

## Domain: biodiversity + earth observation

This template was originally built for replications at the intersection of biological diversity and remote-sensing / climate data — typical examples: pollinator extirpation under climate change, marine heatwave biodiversity exposure, satellite-derived vegetation indices, sphere-aware machine learning on earth-observation data.

If your replication is in a different domain, replace this section by following the "Adapting to a new domain" guide at the bottom of this file.

## Default tooling stack

When the user asks Claude to set up a typical analysis, the default tools to suggest are:

| Concern | Default tool | Notes |
|---|---|---|
| Multi-dimensional arrays | `xarray` | NetCDF/Zarr-aware, lazy via `dask` |
| Climate reanalysis | `cdsapi` (Copernicus C3S) | ERA5, CRU TS, CMIP6 |
| Marine reanalysis | `copernicusmarine` | credentials at `~/.copernicusmarine/.copernicusmarine-credentials` |
| Climate Digital Twin (~5 km, **hourly**) | `polytope-client` or `earthkit-data` | Destination Earth Climate DT — needs DestinE Data Lake account. See § Destination Earth Climate DT below before writing a request. |
| Biodiversity occurrences | `pygbif` | always mint a download DOI per query |
| HEALPix on geographic data (Earth) | `healpix-geo` (EOPF-DGGS) | **default for biodiversity / climate / EO work** — geo-aware: WGS84 ellipsoidal corrections, cartopy integration, xarray-friendly. Always pass `nest=True` — see below. |
| HEALPix on pure-sphere (astrophysics, sky maps) | `healpy` | The astrophysics-original library. Theta/phi colatitudes, no CRS. Use only when the data has no terrestrial coordinate system (e.g. CMB, sky surveys). |
| HEALPix + cartopy visualisation | `healpix-plot` (EOPF-DGGS) | Replaces ad-hoc ang2pix + pcolormesh bridges. |
| Discrete Global Grid Systems | `xdggs`, `dggrid4py`, `h3`, `rhealpixdggs` | DGGRID v8.41 only (v8.42+ breaks under modern GCC) |
| Map projections | `pyproj`, `cartopy` | |
| Raster / vector I/O | `rasterio`, `geopandas` | |
| Scattering on the sphere | `foscat>=2026.4.1` | upstream PyPI; CPU auto-detection |
| Intermediate / archival arrays | `netCDF4` (small ≤2 GB), `zarr` (larger / cloud) | **never `.npz`** — see Data formats convention below |
| HEALPix-indexed EO archival | EOPF Zarr (Earth Observation Processing Framework profile) | Standardised metadata for HEALPix dim-naming, NESTED ordering, projection. See [`docs/eopf-zarr-conversion.md`](docs/eopf-zarr-conversion.md). |

Pin every dependency in `pixi.toml` and commit the regenerated `pixi.lock` — pangeo dev environments hide missing deps locally and CI then silently fails with empty notebook cells.

## Domain conventions

### Data formats — prefer NetCDF / Zarr over `.npz`

For intermediate artefacts between notebooks, **never use `.npz`** as the storage format. NumPy's `.npz` is pickle-based, Python-specific, and not self-describing — it loses dimension labels, units, CRS metadata, and the kind of information downstream tools (other notebooks, other languages, FAIR-RDM platforms, the FORRT chain's audit trail) actually need.

Use the following hierarchy:

- **NetCDF (`.nc`)** — for arrays up to ~2 GB. Self-describing via CF conventions, language-agnostic, the standard for terrestrial climate and EO data. Read/write via `xarray.Dataset.to_netcdf()` / `xr.open_dataset()`. Default choice for most intermediate artefacts in this domain.
- **Zarr (`.zarr`)** — for larger arrays, cloud-native workflows, or when chunked I/O matters. Self-describing, lazy via `dask`, the standard for petabyte-scale EO archives. Read/write via `xarray.Dataset.to_zarr()` / `xr.open_zarr()`. Use when arrays exceed ~2 GB or live in object storage (S3, GCS).
- **EOPF Zarr** (Earth Observation Processing Framework Zarr profile, also known as the **GRID4EARTH** DGGS Zarr convention) — for HEALPix-indexed EO data. Standardises HEALPix dimension naming, NESTED ordering declaration, ellipsoid reference, and multiscale layout so the archive is reusable across EOPF-aware tooling (xdggs, healpix-geo, healpix-plot, healpix-resample). **See [`docs/eopf-zarr-conversion.md`](docs/eopf-zarr-conversion.md)** for the convention's structure, a minimal write/read example using `xarray + zarr v3`, and how it differs from plain Zarr. That doc also covers **which resampler to use for which variable** — conservative for fluxes, reconstruction kernels only for smooth signed fields — which matters because the wrong one silently changes the values.

**Anti-patterns:**

- `np.savez(...)` / `np.savez_compressed(...)` — drops all metadata; brittle across NumPy versions; not citable as a self-describing artefact.
- `pickle.dump(...)` for arrays — language-locked, version-locked, not interoperable with FAIR-RDM platforms.
- Custom HDF5 layouts when NetCDF would do — adds reader complexity for no gain.
- Plain CSV for arrays >100k rows or >10 columns — unindexable, slow, no dim metadata.

**Quick decision tree:**

| Artefact | Format |
|---|---|
| Tabular results (rank lists, summary tables, GLMM coefficients) | CSV or Parquet |
| Posterior draws (multi-dim, with named dims) | NetCDF (`idata.nc`) |
| Per-cell × per-time arrays (TEI, climate fields) ≤2 GB | NetCDF |
| Per-cell × per-time arrays >2 GB or cloud-archived | Zarr |
| HEALPix-indexed climate / EO inputs | EOPF Zarr (or NetCDF if the source delivers it) |
| Figures | PNG (display) + PDF (publication) |

If you find yourself reaching for `.npz`, you've picked the wrong tool — pause and use one of the formats above.

### HEALPix is always NESTED, never RING

Every HEALPix call in this domain MUST use NESTED ordering. NESTED is the only ordering that supports hierarchical bit-shift refinement (parent = `pix >> 2`, children = `pix << 2 | k`) and is the optimal ordering for zoom-in / zoom-out operations across spatial scales.

For **geographic** data (the typical case in this domain — biodiversity, climate, earth observation), prefer `healpix-geo` over `healpy`. `healpix-geo` is geo-aware (WGS84 ellipsoidal corrections, cartopy integration, xarray-friendly); `healpy` is the astrophysics-original library and assumes a pure mathematical sphere.

```python
# Geographic / Earth data — use healpix-geo (default in this domain):
import healpix_geo as hg
hg.ang2pix(nside, lon, lat, nest=True)        # geographic lon/lat, not theta/phi
hg.pix2ang(nside, pix, nest=True)             # returns geographic lon/lat

# Pure-sphere / astrophysics — use healpy with nest=True:
import healpy as hp
hp.ang2pix(nside, theta, phi, nest=True)
hp.pix2ang(nside, pix, nest=True)
hp.boundaries(nside, pix, step=N, nest=True)
```

Mixing RING and NESTED in the same workflow makes cell indices incompatible and breaks tile-based / hierarchical operations. Mixing `healpy` and `healpix-geo` in the same workflow is also a mistake — they index pixels the same way (NESTED is consistent across both), but the coordinate semantics differ (theta/phi vs lon/lat); pick one and stay consistent. Default to `healpix-geo` for any work over Earth's surface.

If a notebook needs RING (e.g. for spherical-harmonic transforms via `hp.map2alm`), do the SHT work in a clearly-scoped block, but keep the on-disk pixelisation NESTED.

### Destination Earth Climate DT — get the request keys from the examples, never from memory

Every fact below was verified against [`destination-earth-digital-twins/polytope-examples`](https://github.com/destination-earth-digital-twins/polytope-examples) (`climate-dt/`, read 2026-09-06). Re-check it there before writing a request: a wrong key does not fail loudly, it silently retrieves a different simulation.

**Generation 2 is not generation 1 with newer dates.** The keys differ:

| | generation 1 | generation 2 |
|---|---|---|
| Period | 2020–2050 | **1990–2049**, one continuous transient |
| Atmosphere | ~10 km (`standard` ≈ L7/H128) | **~5 km**; `resolution: "high"` is the native HEALPix delivery, level 10 (nside 1024, ~6.3 km) |
| Cadence | 6-hourly / daily in practice | **hourly** for the atmosphere (`clte`, `levtype` sfc/pl/hl/sol); ocean and sea-ice (`o2d`, `o3d`) are daily means |
| Scenario | `activity: "ScenarioMIP"` | `activity: "projections"`, `experiment: "SSP3-7.0"` |
| Historical | — | `activity: "baseline"`, `experiment: "hist"` |
| Endpoint | `polytope.lumi.apps.dte.destination-earth.eu` | `polytope.mn5.apps.dte.destination-earth.eu` |

Three coupled models — `IFS-NEMO`, `IFS-FESOM`, `ICON` — each a separate `model` value, all harmonised onto HEALPix.

The hourly stream (`clte`) takes `date` + `time` (`"0000/to/2300/by/0100"` for every hour); the monthly-mean stream (`clmn`) takes `year` + `month` instead, and its parameters are named `avg_*` (`avg_2t`, `avg_tprate`).

**MARS cannot crop HEALPix with a lat/lon box.** An `area` key raises `Representation::croppedRepresentation() not implemented for HEALPixNested`. A regional study therefore either downloads the globe and subsets locally, or uses Polytope's **server-side feature extraction** — `feature: {"type": "polygon", "shape": [[lat, lon], …]}`, also `timeseries`, `boundingbox`, `trajectory`, `verticalprofile`. For anything hourly at `resolution: "high"` the feature is not an optimisation, it is the only tractable route: one year of hourly global level-10 fields is hundreds of gigabytes.

Commit the polygon rather than fetching a boundary at runtime, so the study area cannot drift with an upstream dataset.

**Don't resample extremes.** The Climate DT arrives already on HEALPix, so there is usually nothing to regrid — and for extreme-value work you actively want to keep it that way. `healpix-resample`'s `PSFResampler` (a Gaussian kernel plus damped least squares) is the best reconstruction for smooth fields and exactly the wrong choice for annual maxima: it damps the peaks being measured. If a flux genuinely must be moved onto another grid, `ConservativeResampler` preserves the total; if only areas or a domain mask are needed, no resampling is required at all, because HEALPix cells are equal-area by construction and the authalic definition keeps them equal-area on the WGS84 ellipsoid.

What the ellipsoid *does* change is **membership**: a server-side polygon clip is applied on the sphere, and authalic and geodetic latitude differ by up to 0.128° (~14 km) at mid-latitudes, so border cells can fall on either side. Measure how many and report it as a deviation; don't silently "correct" it.

### Biodiversity is high-precision climate-impact science

Don't dismiss small systematic biases in biodiversity work as "sub-noise" or "second-order". Biodiversity replications in this template are typically climate-impact attribution, restoration monitoring, Habitats Directive / Natura 2000 reporting, and species-distribution work — domains where a 0.3 % systematic bias compounds across millions of cells and decades of data into real attribution errors.

When choosing pixelisations, projections, or aggregation methods, default to the option with smaller systematic bias even if it costs implementation effort. The "good enough for biodiversity" framing is wrong here.

### GBIF download DOIs are mandatory

Every GBIF query that feeds a replication must mint a download DOI (not just a URL). Issue an occurrence download via the GBIF API rather than the search UI, record the resulting DOI (`10.15468/dl.…`), and cite it in `CITATION.cff` and the Replication Study's "Methodology" field. Without a download DOI, the dataset is not citable and the replication is not reusable.

### Copernicus credentials in CI

`copernicusmarine login` prompts interactively and fails in CI. Create the credentials file directly from secrets:

```yaml
- name: Set up Copernicus Marine credentials
  run: |
    mkdir -p ~/.copernicusmarine
    echo "${{ secrets.COPERNICUS_CREDENTIALS_BASE64 }}" | base64 -d \
      > ~/.copernicusmarine/.copernicusmarine-credentials
```

The secret is a base64-encoded INI file containing `[credentials]\nusername=…\npassword=…`.

### Self-contained data downloads

Every notebook MUST include code to download its input data — never assume data exists locally. The repo should be cloneable + runnable without manual data preparation. Use Zenodo, GBIF API, Copernicus API, or direct URLs. Files >100 MB go behind `actions/cache@v4` in CI.

## Style conventions

### Spell out "percentage points"

When writing scientific narrative in nanopub fields (Outcome conclusion, evidence, limitations; Quote-with-comment personal comments), spell out **"percentage points"** rather than abbreviating as **"pp"**. The abbreviation reads as cryptic and is confusable with page numbers in citations.

- ✅ "8.4 percentage points"
- ❌ "8.4 pp"

The abbreviation is acceptable in dense tables when the column header makes it unambiguous (e.g., `Δ (pp)`). In prose / textarea fields, spell it out.

### Honest negative results

A replication that contradicts the original paper's headline is publishable. A replication that overclaims to make the result look stronger is not. If your numerical replication gives a weaker effect than the paper, write it that way; if it contradicts the paper, label the FORRT Outcome as `Contradicted` and let the CiTO citation type be `disputes`. The platform's value is research integrity — overclaiming undermines the platform.

### Vision-piece framing for advocacy posts

When drafting LinkedIn / blog posts about a replication, lead with a **vision** about what's changing in research practice (Open Science, FAIR4RS, atomic claims), and treat the worked replication as the proof-of-concept payoff. Don't lead with "we replicated X". Don't write marketing-fluff openings ("excited to announce"). Don't tag the original paper authors directly — they'll find it via citation pipelines if it's notable.

## Adapting to a new domain

To use this template for a domain other than biodiversity + earth observation:

1. Pick the closest existing flavour from `docs/domain-flavours/`.
2. Copy it over `DOMAIN.md` (in the repo root) — the file Claude reads.
3. Update three things at minimum:
   - **Default tooling stack** — list the libraries your domain reaches for first.
   - **Domain conventions** — the analogue of "HEALPix always NESTED" for your field. What's a non-obvious convention that, if violated, breaks downstream interop?
   - **Style conventions** — write-up rules specific to your domain.
4. If no flavour matches, start from `docs/domain-flavours/_template.md` and ask Claude to help you fill it in based on a few example papers from your field.

A domain flavour is a contract: by including it, you're telling Claude *"in this field, do X by default, and flag if the user violates Y"*. Keep it short and load-bearing — under ~300 lines is a healthy ceiling. Domain rules are not personal style (those go in `USER_PREFERENCES.md`).
