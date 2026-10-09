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
# Diagnostic 3 (transferability): 02c (Neotropical sample) -> 03d (transfer test).
#   Run on its own with `snakemake diag3`.
# Area of applicability (Meyer & Pebesma 2021, CAST port): 03e, after diagnostic 3.
#   Run on its own with `snakemake aoa`.
# MapBiomas labels (independent regrowth labels, experiments E1-E4): 01c -> 02d -> 03f.
#   Run on its own with `snakemake mapbiomas`.
#
# Sampling design (Cloud 2026: training prevalence, paired non-regrowth, forward test): 03h,
#   after 03f. Run on its own with `snakemake sampling`.
#
# Every rule lists its notebook as an input, so editing a notebook re-runs its rule
# (and, through the outputs, everything downstream) under `--rerun-triggers mtime`.
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


DIAG3_FINAL = [
    f"{RESULTS}/diag3_transfer_colombia.csv",
    f"{RESULTS}/diag3_transfer_curve.csv",
    f"{FIGURES}/diag3_transfer.png",
    f"{FIGURES}/diag3_training_cells.png",
]

AOA_FINAL = [
    f"{RESULTS}/aoa_summary.csv",
    f"{RESULTS}/aoa_di_accuracy.csv",
    f"{RESULTS}/aoa_healpix_d8.nc",
    f"{FIGURES}/aoa_colombia.png",
]

MAPBIOMAS_FINAL = [
    f"{RESULTS}/mapbiomas_e1_labels.csv",
    f"{RESULTS}/mapbiomas_e1_model.csv",
    f"{RESULTS}/mapbiomas_e1_persistence.csv",
    f"{RESULTS}/mapbiomas_e2_forward.csv",
    f"{RESULTS}/mapbiomas_e2_reliability.csv",
    f"{RESULTS}/mapbiomas_e3_history.csv",
    f"{RESULTS}/mapbiomas_e4_cross.csv",
    f"{FIGURES}/mapbiomas_labels.png",
    f"{FIGURES}/mapbiomas_forward_test.png",
]

STAGE_B_FINAL = [
    f"{RESULTS}/step2_replication_colombia.csv",
    f"{RESULTS}/step3_robustness_colombia.csv",
    f"{RESULTS}/step3_optional_variants_colombia.csv",
    f"{FIGURES}/main_result.png",
]

FULLRES_FINAL = [
    f"{DERIVED}/fullres_healpix_d15.parquet",
    f"{DERIVED}/fullres_healpix_d15.json",
    f"{RESULTS}/fullres_summary.csv",
]

ARCHIVE_DIR = "archive_smoke" if SMOKE else "archive"
ARCHIVE_FINAL = [
    f"{ARCHIVE_DIR}/checksums.csv",
    f"{RESULTS}/archive_manifest.json",
]

SAMPLING_FINAL = [
    f"{RESULTS}/step3_sampling_design.csv",
    f"{RESULTS}/step3_sampling_importance.csv",
    f"{FIGURES}/sampling_design.png",
]

if SMOKE:
    rule all:
        input:
            STAGE_B_FINAL,
            DIAG3_FINAL,
            AOA_FINAL,
            MAPBIOMAS_FINAL,
            FULLRES_FINAL,
            ARCHIVE_FINAL,
            f"{RESULTS}/step3_blocking_sensitivity.csv",
            SAMPLING_FINAL,
else:
    rule all:
        input:
            "results/step1_reproduction_colombia.csv",
            "results/step1_authors_map_healpix_d8.nc",
            "results/step1_reproduction_cri.csv",
            STAGE_B_FINAL,
            DIAG3_FINAL,
            AOA_FINAL,
            MAPBIOMAS_FINAL,
            FULLRES_FINAL,
            ARCHIVE_FINAL,
            f"{RESULTS}/step3_blocking_sensitivity.csv",
            SAMPLING_FINAL,


rule archive:
    input:
        FULLRES_FINAL,
        ARCHIVE_FINAL,


rule diag3:
    input:
        DIAG3_FINAL,


rule mapbiomas:
    input:
        MAPBIOMAS_FINAL,


rule sampling:
    input:
        SAMPLING_FINAL,


# ---------- 01: Step-1 data (authors' tiles + GADM) ----------
rule data_download:
    input:
        f"{NOTEBOOKS}/01_data_download.py",
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
        f"{NOTEBOOKS}/03a_reproduction_area.py",
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
        f"{NOTEBOOKS}/03a_reproduction_area.py",
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
        f"{NOTEBOOKS}/01b_data_download_predictors.py",
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
        f"{NOTEBOOKS}/02_data_clean.py",
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
        f"{NOTEBOOKS}/02b_feature_extraction.py",
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
        f"{NOTEBOOKS}/03_analysis.py",
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
        f"{NOTEBOOKS}/03c_robustness.py",
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
        f"{NOTEBOOKS}/03c_robustness.py",
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
        f"{NOTEBOOKS}/04_figures.py",
        "results/step1_reproduction_colombia.csv",
        f"{RESULTS}/step2_replication_colombia.csv",
        f"{RESULTS}/step3_robustness_colombia.csv",
        f"{RESULTS}/step3_spatial_cv.csv",
        f"{RESULTS}/diag3_transfer_colombia.csv",
        f"{RESULTS}/diag3_transfer_curve.csv",
        f"{RESULTS}/mapbiomas_e2_forward.csv",
        f"{RESULTS}/mapbiomas_e3_history.csv",
        f"{RESULTS}/fullres_summary.csv",
    output:
        f"{FIGURES}/main_result.png",
        f"{FIGURES}/step3_overview.png",
        f"{RESULTS}/summary.csv",
    log:
        f"{LOGS}/04_figures.log",
    priority: 5  # after the optional variants when both are ready (the figure includes them if present)
    shell:
        nb("04_figures.py")


# ---------- Diagnostic 3: Neotropical training sample (02c) and transfer test (03d) ----------
# 02c downloads Hansen tree-cover tiles one group at a time and deletes them (keeps >= 10 GB
# free); SoilGrids / ESA CCI / Fagan are read remotely at the points. Uses the full-run PCA
# loadings and LC classes of data/clean (also in smoke mode).
rule neotropics_sample:
    input:
        f"{NOTEBOOKS}/02c_neotropics_sample.py",
        "data/clean/clean_meta.json",
        "data/raw/sources_predictors.json",
        f"{RAW}/gadm/gadm41_COL.gpkg",
    output:
        f"{CLEAN}/neotropics_samples.parquet",
        f"{CLEAN}/neotropics_cells.parquet",
        f"{CLEAN}/neotropics_meta.json",
        f"{FIGURES}/diag3_training_cells.png",
    log:
        f"{LOGS}/02c_neotropics_sample.log",
    threads: workflow.cores
    priority: 1
    shell:
        nb("02c_neotropics_sample.py")


rule transfer_test:
    input:
        f"{NOTEBOOKS}/03d_transfer_test.py",
        f"{CLEAN}/samples.parquet",
        f"{CLEAN}/pred_grid.parquet",
        f"{CLEAN}/tile_sums.csv",
        f"{CLEAN}/neotropics_samples.parquet",
    output:
        f"{RESULTS}/diag3_transfer_colombia.csv",
        f"{RESULTS}/diag3_transfer_curve.csv",
        f"{RESULTS}/diag3_meta.json",
        f"{FIGURES}/diag3_transfer.png",
    log:
        f"{LOGS}/03d_transfer_test.log",
    threads: workflow.cores
    priority: 1
    shell:
        nb("03d_transfer_test.py")


# ---------- 03e: Area of applicability (CAST trainDI/aoa port) of the step-2 and diagnostic-3 models ----------
rule aoa:
    input:
        f"{NOTEBOOKS}/03e_area_of_applicability.py",
        f"{CLEAN}/samples.parquet",
        f"{CLEAN}/pred_grid.parquet",
        f"{CLEAN}/tile_sums.csv",
        f"{CLEAN}/neotropics_samples.parquet",
        f"{RESULTS}/diag3_transfer_colombia.csv",  # runs after diagnostic 3
    output:
        AOA_FINAL,
    log:
        f"{LOGS}/03e_area_of_applicability.log",
    threads: workflow.cores
    priority: 1
    shell:
        nb("03e_area_of_applicability.py")


# ---------- MapBiomas Colombia C3 labels: download (01c), labels + 2012 predictors (02d), experiments (03f) ----------
# 01c downloads the 40 annual rasters (~4.5 GB; stops below 10 GB free disk); in smoke mode 02d reads the
# smoke window remotely instead.
rule mapbiomas_download:
    input:
        f"{NOTEBOOKS}/01c_mapbiomas_download.py",
        f"{RAW}/gadm/gadm41_COL.gpkg",
    output:
        f"{RAW}/sources_mapbiomas{SFX}.json",
    log:
        f"{LOGS}/01c_mapbiomas_download.log",
    shell:
        nb("01c_mapbiomas_download.py")


rule mapbiomas_labels:
    input:
        f"{NOTEBOOKS}/02d_mapbiomas_labels.py",
        f"{RAW}/sources_mapbiomas{SFX}.json",
        f"{CLEAN}/samples.parquet",
        f"{CLEAN}/pred_grid.parquet",
    output:
        f"{CLEAN}/mapbiomas_grid.parquet",
        f"{CLEAN}/mapbiomas_samples.parquet",
        f"{CLEAN}/mapbiomas_codes.parquet",
        f"{CLEAN}/mapbiomas_meta.json",
    log:
        f"{LOGS}/02d_mapbiomas_labels.log",
    threads: workflow.cores
    priority: 1
    shell:
        nb("02d_mapbiomas_labels.py")


rule mapbiomas_experiments:
    input:
        f"{NOTEBOOKS}/03f_mapbiomas_experiments.py",
        f"{CLEAN}/mapbiomas_grid.parquet",
        f"{CLEAN}/mapbiomas_samples.parquet",
        f"{CLEAN}/samples.parquet",
        f"{CLEAN}/pred_grid.parquet",
        f"{CLEAN}/tile_sums.csv",
    output:
        MAPBIOMAS_FINAL,
        f"{RESULTS}/mapbiomas_meta.json",
    log:
        f"{LOGS}/03f_mapbiomas_experiments.log",
    threads: workflow.cores
    priority: 1
    shell:
        nb("03f_mapbiomas_experiments.py")


# ---------- 05a: full-resolution prediction, conservative HEALPix depth-15 sums ----------
rule full_prediction:
    input:
        f"{NOTEBOOKS}/05a_full_prediction.py",
        f"{CLEAN}/samples.parquet",
        f"{CLEAN}/tile_sums.csv",
        f"{DERIVED}/step2_predictions_grid.parquet",
        f"{RESULTS}/step2_replication_colombia.csv",
    output:
        FULLRES_FINAL,
    log:
        f"{LOGS}/05a_full_prediction.log",
    threads: workflow.cores
    shell:
        nb("05a_full_prediction.py")


# ---------- 05b: dataset archive (GRID4EARTH Zarr + Parquet tables) for Zenodo ----------
rule archive_dataset:
    input:
        f"{NOTEBOOKS}/05b_archive_dataset.py",
        f"{DERIVED}/fullres_healpix_d15.parquet",
        f"{CLEAN}/pred_grid.parquet",
        f"{CLEAN}/mapbiomas_grid.parquet",
        f"{RESULTS}/aoa_healpix_d8.nc",
        "docs/archive_README.md",
        "docs/archive_zenodo.json",
    output:
        ARCHIVE_FINAL,
    log:
        f"{LOGS}/05b_archive_dataset.log",
    shell:
        nb("05b_archive_dataset.py")


# ---------- 03g: blocked CV sensitivity to the blocking grid (HEALPix WGS84 / sphere / lat-lon squares) ----------
rule blocking_sensitivity:
    input:
        f"{NOTEBOOKS}/03g_blocking_sensitivity.py",
        f"{CLEAN}/samples.parquet",
        f"{RESULTS}/step3_spatial_cv.csv",  # runs after step 3(b), whose numbers it must reproduce
    output:
        f"{RESULTS}/step3_blocking_sensitivity.csv",
    log:
        f"{LOGS}/03g_blocking_sensitivity.log",
    threads: workflow.cores
    shell:
        nb("03g_blocking_sensitivity.py")


# ---------- 03h: training-sample design (prevalence sweep, paired non-regrowth, forward test) ----------
rule sampling_design:
    input:
        f"{NOTEBOOKS}/03h_sampling_design.py",
        f"{CLEAN}/samples.parquet",
        f"{CLEAN}/pred_grid.parquet",
        f"{CLEAN}/mapbiomas_grid.parquet",
        f"{CLEAN}/tile_sums.csv",
        f"{RESULTS}/mapbiomas_e2_forward.csv",  # runs after 03f, whose E2 numbers it must reproduce
    output:
        SAMPLING_FINAL,
    log:
        f"{LOGS}/03h_sampling_design.log",
    threads: workflow.cores
    shell:
        nb("03h_sampling_design.py")
