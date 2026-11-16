# Changelog

All notable changes to JORO-HPC are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-11-15

Second major scientific release: full 92-day seasonal atmospheric dispersion (May–July in `FP16`), tripartite citizen-science & invasive-species data fusion (`GBIF + iNaturalist + EDDMapS`), exogenous target-group background effort, mechanistic `cloglog` colonization with road-weighted human gravity dispersal (`K_mixto`), and non-stationary digital observation process (`f_est(i,t)`).

### Added in 2.0.0

- **Multi-month seasonal ERA5 NetCDF4 reader (`src/io/era5_netcdf_reader.*`).** Loads and concatenates May, June, and July (`2,208 h/year`, 92 days) with automatic CF-convention `scale_factor` / `add_offset` unpacking (`NC_SHORT`, `NC_FLOAT`, `NC_DOUBLE`) and configurable synoptic stride (`JORO_ERA5_TIME_STRIDE=3`, keeping all 6 calibration years resident in `3.64 GiB` of VRAM on Intel Arc A380 with `0 B` H2D transfer per evaluation).
- **VRAM `FP16` quantization (`src/lpdm/device_math.hpp`, `src/lpdm/era5_view.hpp`).** IEEE 754 half-precision (`Wind4Half`, 8 bytes/voxel) storage in SYCL Device USM with on-the-fly `FP32` promotion during tetralinear interpolation (`2x` memory reduction, verified `< 1.7 %` jump-distance difference between `3 h` synoptic and `1 h` hourly steps).
- **Phenological and convective launch gate (`src/lpdm/lpdm_sycl_kernel.cpp`, `src/lpdm/lpdm_cpu_baseline.cpp`).** Enforces calm boundary-layer wind ($u_{10} < 3.0\text{ m/s}$), diurnal release window (`14–22 UTC`), and convective boundary-layer indicator $S_{\text{conv}}$ (`blh`) over the 92-day season.
- **Road-weighted human gravity dispersal kernel (`src/inference/dispersal_kernels.hpp`).** Stratified dispersal combining 3D atmospheric LPDM with a TIGER/Line road-network and Census population gravity kernel ($K_{\text{mixto}}$, $\lambda_h = 150\text{ km}$), weighting propagule emission from occupied cells by their normalized road density $R_j$ and reception by $R_i H_i \exp(-d_{ji}/\lambda_h)$.
- **Non-stationary observation process (`src/inference/occupancy_model.hpp`).** Models media salience ($M_t / M_{2019}$, reaching $\times 3.48$ in 2024) and local observer habituation ($\tau_{i,t}$, years since first detection in cell $i$) via $f_{\text{est}}(i,t) = \lambda_M \log(1 + M_t/M_{2019}) - \kappa_\tau \log(1 + \tau_{i,t})$ with externally anchored $(\lambda_M, \kappa_\tau) = (1.15, 0.25)$.
- **Tripartite occurrence ingestion & exogenous effort.** Deduplicated fusion of GBIF, iNaturalist, and EDDMapS (`1,359` unique new records), exogenous target-group effort $E_{i,t}^{-J} = \max(0, C_{i,t}^{\text{Araneae}} - C_{i,t}^J)$, and a clean contiguous founder seed polygon in Braselton/Hoschton (`125` cells, radius `37.6 km`), excluding the isolated Anderson, SC cell (`cell_id = 118702`) from $t_0$ so it must be dynamically colonized.
- **Expanded hypothesis testing (`src/joro_evaluate.cpp`, `docs/reports/04_Out_of_Sample_Validation_2020_2024.md`).** Sectoral $P_{95}$ front radii (NE, SE, SW, NW) and $>300\text{ km}$ jump tail (H2b), Spearman rank correlations of observed CPUE against winter damage $\overline{D}_{i,t}$ (H1b) and carrying capacity $K_i$ (H3a), 4-way observation-process ablation with Cox logistic calibration slope/intercept (H3b), and `1 h` vs `3 h` temporal sensitivity benchmark.

### Changed in 2.0.0

- **Mechanistic `cloglog` colonization without spontaneous generation (`src/inference/occupancy_model.hpp`).** Replaced the envelope basal colonization floor with $\gamma_{i,t} = 1 - \exp(-\lambda_i A_{i,t}^{\text{total}})$, strictly enforcing $\gamma_{i,t} = 0$ whenever $A_{i,t}^{\text{total}} = 0$.
- **Support-centered habitat index (`src/abm/carrying_capacity.hpp`).** Standardized $H_i^{\text{hab}}$ on the informative calibration support $S$ (`4,967` cells; $m_S = 1.1729, s_S = 0.5205$), reducing posterior collinearity $\rho(K_0, \gamma_{\text{hab}})$ from `-0.92` (`v1.0.0`) to `-0.38` (`v2.0.0`) with zero parameters truncated at prior bounds.
- **Urban & canopy microclimate buffering (`src/abm/abm_engine.*`).** Added urban imperviousness thermal buffering ($\delta_{\text{imp}} = 0.50\text{ °C/SD}$) alongside forest canopy buffering ($\delta_{\text{dosel}} = 0.35\text{ °C/SD}$) and logistic spring emergence ($DD_{\text{emerg}} = 0.12, \kappa_{DD} = 8.0$).
- **Exact Ornstein–Uhlenbeck vertical Langevin step (`src/lpdm/langevin_turbulence.hpp`).** Replaced raw Euler–Maruyama vertical velocity stepping in the transport kernel with exact OU exponential decay plus bounded Thomson (1987) well-mixed drift correction.
- **Inverted evaluation loop (`src/joro_evaluate.cpp`).** Outer loop over years (`2014..2024`) and inner loop over ensemble configurations (`343` configs), recycling a single GPU VRAM slot once per year (`0.215 s` per 11-year member).

### Fixed in 2.0.0

- Resolved all 16 audit findings (`M1`–`M16`) documented in the pre-run scientific code audit.
- Fixed `compute_validation_metrics.py` calibration slope to use Cox logistic regression (`LogisticRegression`) on $\operatorname{logit}(\hat{p})$ and unified the founder seed centroid across C++ and Python (`src/common/model_geometry.hpp`).
- Updated `--transect` (`src/main.cpp`) to load $\hat{\theta}_{\text{MAP}}$ from `calibration_summary_M1.json` automatically.

## [1.0.0] - 2026-10-19

First scientific release: calibrated on 2014–2019 and validated blind on 2020–2024.

### Added in 1.0.0

- **LPDM on GPU (SYCL / Level-Zero).** 3D Lagrangian particle dispersion driven by hourly ERA5 reanalysis (12 isobaric
  levels, 1000–700 hPa), Langevin vertical turbulence (Thomson well-mixed criterion), aerodynamic silk settling, zero-copy
  Device USM, Philox counter-based RNG and `atomic_ref` arrival accumulation. Resident per-year atmospheric fields.
- **ABM on CPU (AVX-512 / OpenMP).** Structure-of-Arrays cohorts; Q10 degree-day development; cumulative winter chill
  damage `D_{j,t}`; Beverton–Holt density dependence on the carrying capacity `K_i`; three carrying-capacity models
  (`M0`, `M1` principal, `M3`).
- **PostGIS schema (EPSG:5070).** 5 km grid (256,020 cells), NLCD covariates, Araneae target-group-background effort,
  EPA Level III ecoregion blocks, and the frozen `v_calibration_set` (2014–2019) / `v_evaluation_set` (2020–2024) views.
- **Multi-start simulated-annealing calibration** with a balanced Brier cost, `Θ_comp` ensemble extraction and
  identifiability diagnostics (`joro_hpc --calibrate`).
- **Blind out-of-sample validation (`joro_evaluate`).** Frozen parameters, 11-year continuous simulation, comparison against
  a static climatic SDM (H1) and an isotropic exponential kernel (H2), rule-based verdicts and a prevalence-offset
  analysis calibrated only on 2014–2019.
- **Diagnostics.** Real latitudinal transect and saturation experiments (`--transect`) and Morris screening (`--morris`);
  `--diagnostics` runs both.
- **Reports and operations.** `docs/reports/01..04`, `docs/RUNBOOK.md`, hardware telemetry profile, LaTeX research proposal.
- **Release metadata.** `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `.env.example`.

### Changed in 1.0.0

- Semantic renames: `phase5_runner` → `joro_evaluate`; CLI `--phase4` → `--calibrate`, `--phase3` → `--diagnostics`
  (old flags kept as aliases); `JORO_UNLOCK_PHASE5` → `JORO_UNLOCK_EVALUATION` (old name kept as alias); report scripts renamed
  to `generate_calibration_report.py`, `generate_validation_report.py`, `export_validation_datasets.py`.
- Azimuth of the invasion front is now measured from the geometric centroid of the 124 seed cells and only over cells colonized
  outside the initial seed radius (the former fixed constant lay south-west of the seed polygon). Front distances and the occupancy
  envelope keep the calibrated reference, so no frozen parameter or prediction of the occupancy model changed.
- Reports are generated from run artifacts only; verdicts are computed with fixed criteria.

### Fixed in 1.0.0

- AUC-ROC tie handling (average ranks) and the log-loss clipping threshold.
- LPDM horizon aligned to 168 h of ERA5; M1/M0/M3 synchronization.
- Report generators no longer contain hard-coded numerical results; LaTeX escapes corrected.
- `--transect` now simulates real cells with the ABM and per-year ERA5 forcing instead of printing prescribed values.

### Security in 1.0.0

- Removed the embedded PostGIS password from sources, configuration and scripts. Connection parameters are read from
  `JORO_DB_HOST`, `JORO_DB_PORT`, `JORO_DB_NAME`, `JORO_DB_USER` and `JORO_DB_PASSWORD` (see `.env.example`); `.env` is git-ignored.
  The previous password remains in the Git history and **must be rotated** on any database that is reachable from a network.

### Known limitations in 1.0.0

- The ensemble reproduces ranking (AUC-ROC 0.971 vs 0.904 for the SDM) but underestimates absolute detection totals; the
  5–95 % ensemble band does not cover observed annual totals in any year, even after the prevalence offset (see report 04).
- Frontal direction is not improved by the LPDM: the mean angular error of `M1` is larger than that of the isotropic kernel
  (report 04, Table 3).
- Parameters `gamma_hab`, `k0` and `a0` are strongly collinear (|ρ| ≥ 0.9; report 03).
- With nominal demographic parameters the latitudinal transect fails to persist north of ≈ 34 °N (report 02).
- Benchmarks and validations were run on the development GPU (Intel Arc A380); the Arc A770 target has not been measured yet.

[2.0.0]: https://github.com/oscar-bio-dev/joro-hpc/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/oscar-bio-dev/joro-hpc/releases/tag/v1.0.0
