# Snakefile — orchestrates the replication pipeline end-to-end.
#
# Each rule wraps a jupytext notebook (the .py file is the source of truth).
#
# Usage:
#   snakemake --cores 1                  # run everything
#   snakemake --cores 1 -n               # dry run
#
# Step 1 (reproduction from the authors' published map) is implemented.
# Steps 2-3 (independent random-forest replication, robustness check) will use
# 02_data_clean / 03_analysis / 04_figures and are not wired in yet.

NOTEBOOKS = "notebooks"
DATA = "data"
RESULTS = "results"
FIGURES = "figures"


rule all:
    input:
        f"{RESULTS}/step1_reproduction_colombia.csv",
        f"{RESULTS}/step1_authors_map_healpix_d8.nc",


# ---------- 01: Data download ----------
# Authors' 30 m tiles (Zenodo 10.5281/zenodo.7428804) + GADM 4.1 Colombia.
# Every file is md5-checked; re-runs skip files already present.
rule data_download:
    output:
        f"{DATA}/raw/sources.json",
        f"{DATA}/raw/gadm/gadm41_COL.gpkg",
    log:
        f"{RESULTS}/logs/01_data_download.log",
    shell:
        f"cd {{NOTEBOOKS}} && jupytext --to notebook --execute 01_data_download.py 2>&1 | tee ../{{log}}"


# ---------- 03a: Step 1 reproduction (area from the published map) ----------
rule reproduction_area:
    input:
        f"{DATA}/raw/sources.json",
        f"{DATA}/raw/gadm/gadm41_COL.gpkg",
    output:
        f"{RESULTS}/step1_reproduction_colombia.csv",
        f"{RESULTS}/step1_pct_histogram_colombia.csv",
        f"{RESULTS}/step1_authors_map_healpix_d8.nc",
        f"{FIGURES}/step1_authors_map_healpix.png",
    log:
        f"{RESULTS}/logs/03a_reproduction_area.log",
    shell:
        f"cd {{NOTEBOOKS}} && jupytext --to notebook --execute 03a_reproduction_area.py 2>&1 | tee ../{{log}}"
