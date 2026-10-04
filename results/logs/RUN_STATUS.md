# Full run — status and resume note

- **Started:** 2026-10-03T20:53:56Z (UTC)
- **PID:** 2642160 (`pixi run snakemake …`; own session/process group 2642160, started with `setsid nohup`)
- **Command:** `pixi run snakemake --cores 12 --keep-going --rerun-incomplete --rerun-triggers mtime`
- **Log:** `results/logs/full_run_20261003T2053.log` (Snakemake); per-rule notebook logs in `results/logs/<rule>.log`
- **Smoke test:** all 8 stage-B rules passed on 2026-10-03 (`--config smoke=1`; outputs in `results/smoke/`, `figures/smoke/`, `data/clean_smoke/`).

`--rerun-triggers mtime` is deliberate: the Snakefile was rewritten, and the default
"code changed" trigger would otherwise re-run the finished step-1 rules.

## Check progress

```bash
tail -n 40 results/logs/full_run_20261003T2053.log          # which rule is running / finished
grep -E "Finished jobid|Error in rule" results/logs/full_run_20261003T2053.log
tail -n 20 results/logs/<rule>.log                          # e.g. 01b_data_download_predictors.log
ps -o pid,etime,pcpu,rss,cmd -g 2642160                     # is it still running?
pixi run snakemake --summary --rerun-triggers mtime | column -t | less -S   # per-output status (read-only)
df -h /opt/vth                                              # keep >= 10 GB free
```

Don't start a second Snakemake while this one runs: the working directory is locked.
`--summary` and `-n` are read-only and safe.

## Rules, in order, and expected outputs

| Rule | Notebook | Outputs | Expected time (rough) |
|---|---|---|---|
| `data_download_predictors` | `01b` | `data/raw/sources_predictors.json`, `data/raw/burned_source.json` (+ GlobFire monthly parquet, SRTM tiles, NPP, GRIP4, RESOLVE; MCD64A1 count raster only if GlobFire fails) | 0.5–2 h (GlobFire on PANGAEA is the slow, flaky part) |
| `data_clean` | `02` | `data/clean/{bioclim_pca,chelsa_pca,soil_0_30cm,landcover,npp,roads,burned}.nc`, `biomes_colombia.parquet`, `clean_meta.json` | ~10–20 min (PCA samples are cached) |
| `feature_extraction` | `02b` | `data/clean/{samples,pred_grid}.parquet`, `tile_sums.csv`, `features_meta.json` | ~20–60 min |
| `analysis` (step 2) | `03` | `results/step2_replication_colombia.csv`, `step2_reliability.csv`, `step2_healpix_d8.nc`, `data/derived/step2_*` | ~10–30 min |
| `robustness` (3a–c) | `03c` (`VARIANTS=abc`) | `results/step3_robustness_colombia.csv`, `step3_spatial_cv.csv`, `step3_accuracy_by_distance.csv`, `step3_variable_selection_curves.csv` | ~20–60 min |
| `optional_variants` (3d–e) | `03c` (`VARIANTS=de`) | `results/step3_optional_variants_colombia.csv`, `step3_spatial_cv_chelsa.csv` | ~10–30 min |
| `figures` | `04` | `figures/main_result.png`, `results/summary.csv` | < 1 min |

Priorities: `analysis` 100 > `robustness` 50 > `optional_variants` 10 > `figures` 5.
Heavy rules take all 12 cores, so they run one at a time. `figures` does not need
the optional variants: if (d)/(e) fail, the figure is still made without them.

## Known risks

- **PANGAEA (GlobFire) HTTP 503.** Seen repeatedly on 2026-10-03. Each month is
  tried 8 times; if one still fails, 01b switches the whole burned-area series to
  MODIS MCD64A1 (Earthdata), the tested fallback. Check
  `data/raw/burned_source.json` (`"source"`: `globfire` | `mcd64a1` | `none`). With
  `none`, burned area is dropped from the 3c candidates and this is printed in the
  `robustness` log.
- **Earthdata** (SRTM, MCD64A1 fallback) uses `~/.netrc` via `earthaccess`. This was
  verified from a detached process in the smoke test. A URS outage would fail
  `data_download_predictors`.
- **Remote reads** (CEDA ESA CCI, ISRIC SoilGrids, CHELSA) are already cached in
  `data/raw` / `data/clean` (`chelsa_pca_sample.parquet`, `chelsa_window.nc`,
  `bioclim_pca_sample.parquet`). They are not re-read unless deleted.
- **Disk.** About 25 GB was free at launch. The WorldClim zip (10.4 GB) is needed
  only by `data_clean`, and only if `data/clean/bioclim_pca_sample.parquet` or
  `bioclim_pca.nc` must be rebuilt. Temporary MCD64A1 HDFs are deleted per month.
  Keep at least 10 GB free. Safe to delete if needed: `data/clean_smoke/`,
  `data/derived_smoke/`, `results/smoke/`.
- **Memory.** `feature_extraction` runs 12 workers (about 1–2 GB each). If it is
  OOM-killed, rerun with fewer: `N_WORKERS=6 pixi run snakemake …`.

## Resume after a failure

```bash
cd /opt/vth/OpenAIRE_Alien_AI_Hackathon/natural-regeneration-replication
grep -n "Error in rule" -A15 results/logs/full_run_*.log | head -60   # what failed
tail -n 60 results/logs/<failed_rule>.log                             # notebook traceback
pixi run snakemake --unlock                                           # only if a stale lock remains
setsid nohup pixi run snakemake --cores 12 --keep-going --rerun-incomplete --rerun-triggers mtime \
  > results/logs/full_run_$(date +%Y%m%dT%H%M).log 2>&1 < /dev/null &
```

Downloads and expensive intermediates are cached, so a rerun resumes from
whatever is missing.

---

# Diagnostic 3 — transferability run (Neotropical training sample)

- **Rules:** `neotropics_sample` (`notebooks/02c_neotropics_sample.py`) → `transfer_test`
  (`notebooks/03d_transfer_test.py`); target `diag3` (also part of `all`).
- **Command:** `setsid nohup pixi run snakemake diag3 --cores 12 --keep-going --rerun-incomplete --rerun-triggers mtime > results/logs/diag3_<UTC>.log 2>&1 < /dev/null &`
- **Log:** newest `results/logs/diag3_*.log` (Snakemake); notebook logs
  `results/logs/02c_neotropics_sample.log`, `results/logs/03d_transfer_test.log`
  (written when each notebook finishes).
- **Outputs:** `data/clean/neotropics_{samples,cells,cells_all,soil_points}.parquet`,
  `data/clean/neotropics_meta.json`, `results/diag3_transfer_colombia.csv` (Test 1),
  `results/diag3_transfer_curve.csv` (Test 2), `results/diag3_meta.json`,
  `figures/diag3_transfer.png`, `figures/diag3_training_cells.png`.
- **Expected time:** 02c ~1–1.5 h (Fagan reads ~10–20 min; ~280 1° tiles with
  Hansen tree-cover downloads ~30–45 min; SoilGrids remote point reads ~20–30 min);
  03d ~20–40 min (3 + 20 forest fits, 3 predictions on the Colombian grid).
- **Every rule now lists its notebook as an input.** Editing a notebook re-runs its
  rule and everything downstream (mtime trigger).

## Check progress

```bash
LOG=$(ls -t results/logs/diag3_*.log | head -1); tail -n 30 $LOG
grep -E "Finished jobid|Error in rule" $LOG
ls data/clean/neotropics_tiles | wc -l                 # 1° tiles done (of ~280; cache, resumable)
ls -la data/raw/hansen_tmp_neotropics 2>/dev/null       # Hansen tiles currently on disk (deleted after use)
ls -la data/clean/neotropics_soil_points.parquet        # soil reads done
pgrep -af "snakemake diag3"                             # still running?
df -h /opt/vth                                          # 02c refuses downloads below 10 GB free
```

## Resume after a failure

02c caches the Fagan polygons (`data/raw/fagan/fagan2022_neotropics_cells.parquet`),
every 1° tile (`data/clean/neotropics_tiles/*.parquet`) and the soil point values,
so a rerun continues where it stopped:

```bash
cd /opt/vth/OpenAIRE_Alien_AI_Hackathon/natural-regeneration-replication
grep -n "Error in rule" -A15 $(ls -t results/logs/diag3_*.log | head -1)
tail -n 60 results/logs/02c_neotropics_sample.log      # or 03d_transfer_test.log
pixi run snakemake --unlock                              # only if a stale lock remains
setsid nohup pixi run snakemake diag3 --cores 12 --keep-going --rerun-incomplete --rerun-triggers mtime \
  > results/logs/diag3_$(date -u +%Y%m%dT%H%M).log 2>&1 < /dev/null &
```

If 02c is OOM-killed in the tile stage, rerun with `N_WORKERS=6` in front of
`pixi run`. Leftover temporary Hansen tiles in `data/raw/hansen_tmp_neotropics/`
are safe to delete. The tile cache and `neotropics_soil_points.parquet` can be
deleted once `neotropics_samples.parquet` exists; they only speed up reruns.

---

# Area of applicability (AOA) — `03e_area_of_applicability.py`

- **Rule:** `aoa` (part of `all`; runs after `transfer_test`). Inputs: `samples`, `pred_grid`, `tile_sums`,
  `neotropics_samples`, `results/diag3_transfer_colombia.csv`.
- **Outputs:** `results/aoa_summary.csv`, `results/aoa_di_accuracy.csv`, `results/aoa_healpix_d8.nc`,
  `figures/aoa_colombia.png`.
- **Smoke test:** passed on 2026-10-04 (`SMOKE=1`, run directly with jupytext because the smoke DAG would also
  re-run smoke 02c/03d, whose notebooks changed after their smoke run).
- **Full run:** started 2026-10-04 12:22 UTC with
  `setsid nohup pixi run snakemake aoa --cores 12 --rerun-triggers mtime > results/logs/aoa_20261004T1222.log 2>&1 < /dev/null &`.
  The dry run showed only `aoa`. Expected time: ~10–20 min (3 + 15 forest fits, permutation importance,
  KD-tree DI for 2.47 M grid points × 3 models). Notebook log: `results/logs/03e_area_of_applicability.log`.
- **Finished** 12:25 UTC (3.5 min). Rerun at 12:33 UTC (`results/logs/aoa_20261004T1233.log`) after adding
  per-class and balanced accuracy. Both runs completed. The refitted models reproduce the 03/03d validation accuracies
  (0.8867 / 0.7671 / 0.7835).
- Method choices: `docs/deviations.md` § Area of applicability.
