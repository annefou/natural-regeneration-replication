# Snakefile — orchestrates the replication pipeline end-to-end.
#
# Each rule wraps a jupytext notebook (the .py file is the source of truth).
#
# Usage:
#   snakemake --cores 12                       # full run (resumes from whatever is missing)
#   snakemake --cores 12 -n                    # dry run
#   snakemake --cores 12 --config smoke=1      # tiny end-to-end test (sub-window, few thousand samples)
#
# Step 1: reproduction from the authors' published map (01, 03a).
# Steps 2-3: independent random-forest replication and robustness checks
#   01b -> 02 -> 02b -> 03 (step 2) -> 03c [a,b,c] -> 03c [d,e optional] -> 04.
#
# Smoke mode writes to data/clean_smoke, data/derived_smoke, results/smoke and
# figures/smoke, and shares the download cache in data/raw.
# Heavy rules request all cores, so they run one at a time; priorities make the
# main robustness checks (a)-(c) run before the optional variants (d)-(e).

SMOKE = str(config.get("smoke", "0")) == "1"
SFX = "_smoke" if SMOKE else ""
ENV = "SMOKE=1 " if SMOKE else ""

NOTEBOOKS = "notebooks"
DATA = "data"
RAW = f"{DATA}/raw"
CLEAN = f"{DATA}/clean{SFX}"
DERIVED = f"{DATA}/derived{SFX}"
RESULTS = "results/smoke" if SMOKE else "results"
FIGURES = "figures/smoke" if SMOKE else "figures"
LOGS = f"{RESULTS}/logs"


def nb(name, extra_env=""):
    """Shell command executing a jupytext notebook with a distinct output .ipynb name."""
    out = name.replace(".py", f"{SFX}{'_' + extra_env.split('=')[1].strip() if extra_env else ''}.ipynb")
    return (f"cd {NOTEBOOKS} && {ENV}{extra_env} jupytext --to notebook --execute {name} --output {out} "
            f"2>&1 | tee ../{{log}}")


STAGE_B_FINAL = [
    f"{RESULTS}/step2_replication_colombia.csv",
    f"{RESULTS}/step3_robustness_colombia.csv",
    f"{RESULTS}/step3_optional_variants_colombia.csv",
    f"{FIGURES}/main_result.png",
]

if SMOKE:
    rule all:
        input:
            STAGE_B_FINAL,
else:
    rule all:
        input:
            "results/step1_reproduction_colombia.csv",
            "results/step1_authors_map_healpix_d8.nc",
            "results/step1_reproduction_cri.csv",
            STAGE_B_FINAL,


# ---------- 01: Step-1 data (authors' tiles + GADM) ----------
rule data_download:
    output:
        f"{RAW}/sources.json",
        f"{RAW}/gadm/gadm41_COL.gpkg",
        f"{RAW}/gadm/gadm41_CRI.gpkg",
    log:
        "results/logs/01_data_download.log",
    shell:
        "cd notebooks && jupytext --to notebook --execute 01_data_download.py 2>&1 | tee ../{log}"


# ---------- 03a: Step 1 reproduction (Colombia, Costa Rica) ----------
rule reproduction_area:
    input:
        f"{RAW}/sources.json",
        f"{RAW}/gadm/gadm41_COL.gpkg",
    output:
        "results/step1_reproduction_colombia.csv",
        "results/step1_pct_histogram_colombia.csv",
        "results/step1_authors_map_healpix_d8.nc",
        "figures/step1_authors_map_healpix.png",
    log:
        "results/logs/03a_reproduction_area.log",
    threads: workflow.cores
    shell:
        "cd notebooks && jupytext --to notebook --execute 03a_reproduction_area.py 2>&1 | tee ../{log}"


rule reproduction_area_cri:
    input:
        f"{RAW}/sources.json",
        f"{RAW}/gadm/gadm41_CRI.gpkg",
    output:
        "results/step1_reproduction_cri.csv",
        "results/step1_pct_histogram_cri.csv",
        "results/step1_authors_map_healpix_d8_cri.nc",
        "figures/step1_authors_map_healpix_cri.png",
    log:
        "results/logs/03a_reproduction_area_cri.log",
    threads: workflow.cores
    shell:
        "cd notebooks && ISO3=CRI jupytext --to notebook --execute 03a_reproduction_area.py "
        "--output 03a_reproduction_area_cri.ipynb 2>&1 | tee ../{log}"


# ---------- 01b: Labels + predictors (SRTM / MCD64A1 fallback need Earthdata ~/.netrc) ----------
rule data_download_predictors:
    input:
        f"{RAW}/gadm/gadm41_COL.gpkg",
    output:
        f"{RAW}/sources_predictors{SFX}.json",
        f"{RAW}/burned_source{SFX}.json",
    log:
        f"{LOGS}/01b_data_download_predictors.log",
    shell:
        nb("01b_data_download_predictors.py")


# ---------- 02: Predictor layers on native grids (incl. WorldClim + CHELSA PCA) ----------
rule data_clean:
    input:
        f"{RAW}/sources_predictors{SFX}.json",
        f"{RAW}/burned_source{SFX}.json",
    output:
        f"{CLEAN}/bioclim_pca.nc",
        f"{CLEAN}/chelsa_pca.nc",
        f"{CLEAN}/soil_0_30cm.nc",
        f"{CLEAN}/landcover.nc",
        f"{CLEAN}/npp.nc",
        f"{CLEAN}/roads.nc",
        f"{CLEAN}/burned.nc",
        f"{CLEAN}/biomes_colombia.parquet",
        f"{CLEAN}/clean_meta.json",
    log:
        f"{LOGS}/02_data_clean.log",
    threads: workflow.cores
    shell:
        nb("02_data_clean.py")


# ---------- 02b: 30 m masks, sampling, feature extraction ----------
rule feature_extraction:
    input:
        f"{CLEAN}/clean_meta.json",
        f"{RAW}/sources.json",
    output:
        f"{CLEAN}/samples.parquet",
        f"{CLEAN}/pred_grid.parquet",
        f"{CLEAN}/tile_sums.csv",
        f"{CLEAN}/features_meta.json",
    log:
        f"{LOGS}/02b_feature_extraction.log",
    threads: workflow.cores
    shell:
        nb("02b_feature_extraction.py")


# ---------- 03: Step 2 replication (incl. calibration extension) ----------
rule analysis:
    input:
        f"{CLEAN}/samples.parquet",
        f"{CLEAN}/pred_grid.parquet",
        f"{CLEAN}/tile_sums.csv",
        "results/step1_authors_map_healpix_d8.nc",
    output:
        f"{RESULTS}/step2_replication_colombia.csv",
        f"{RESULTS}/step2_reliability.csv",
        f"{RESULTS}/step2_healpix_d8.nc",
        f"{DERIVED}/step2_meta.json",
        f"{DERIVED}/step2_predictions_grid.parquet",
    log:
        f"{LOGS}/03_analysis.log",
    threads: workflow.cores
    priority: 100
    shell:
        nb("03_analysis.py")


# ---------- 03c: Step 3 robustness (a) land cover, (b) spatial CV, (c) variable selection ----------
rule robustness:
    input:
        f"{CLEAN}/samples.parquet",
        f"{DERIVED}/step2_meta.json",
    output:
        f"{RESULTS}/step3_robustness_colombia.csv",
        f"{RESULTS}/step3_spatial_cv.csv",
        f"{RESULTS}/step3_accuracy_by_distance.csv",
        f"{RESULTS}/step3_variable_selection_curves.csv",
    log:
        f"{LOGS}/03c_robustness.log",
    threads: workflow.cores
    priority: 50
    shell:
        nb("03c_robustness.py", "VARIANTS=abc")


# ---------- 03c (optional): (d) gradient boosting, (e) CHELSA climate sensitivity ----------
rule optional_variants:
    input:
        f"{CLEAN}/samples.parquet",
        f"{DERIVED}/step2_meta.json",
        f"{RESULTS}/step3_robustness_colombia.csv",  # forces (a)-(c) to finish first
    output:
        f"{RESULTS}/step3_optional_variants_colombia.csv",
        f"{RESULTS}/step3_spatial_cv_chelsa.csv",
    log:
        f"{LOGS}/03c_optional_variants.log",
    threads: workflow.cores
    priority: 10
    shell:
        nb("03c_robustness.py", "VARIANTS=de")


# ---------- 04: Main figure ----------
rule figures:
    input:
        "results/step1_reproduction_colombia.csv",
        f"{RESULTS}/step2_replication_colombia.csv",
        f"{RESULTS}/step3_robustness_colombia.csv",
        f"{RESULTS}/step3_spatial_cv.csv",
    output:
        f"{FIGURES}/main_result.png",
        f"{RESULTS}/summary.csv",
    log:
        f"{LOGS}/04_figures.log",
    priority: 5  # after the optional variants when both are ready (the figure includes them if present)
    shell:
        nb("04_figures.py")
