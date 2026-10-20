# Changelog

All notable changes to JORO-HPC are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-19

First scientific release: calibrated on 2014–2019 and validated blind on 2020–2024.

### Added
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

### Changed
- Semantic renames: `phase5_runner` → `joro_evaluate`; CLI `--phase4` → `--calibrate`, `--phase3` → `--diagnostics`
  (old flags kept as aliases); `JORO_UNLOCK_PHASE5` → `JORO_UNLOCK_EVALUATION` (old name kept as alias); report scripts renamed
  to `generate_calibration_report.py`, `generate_validation_report.py`, `export_validation_datasets.py`.
- Azimuth of the invasion front is now measured from the geometric centroid of the 124 seed cells and only over cells colonized
  outside the initial seed radius (the former fixed constant lay south-west of the seed polygon). Front distances and the occupancy
  envelope keep the calibrated reference, so no frozen parameter or prediction of the occupancy model changed.
- Reports are generated from run artifacts only; verdicts are computed with fixed criteria.

### Fixed
- AUC-ROC tie handling (average ranks) and the log-loss clipping threshold.
- LPDM horizon aligned to 168 h of ERA5; M1/M0/M3 synchronization.
- Report generators no longer contain hard-coded numerical results; LaTeX escapes corrected.
- `--transect` now simulates real cells with the ABM and per-year ERA5 forcing instead of printing prescribed values.

### Security
- Removed the embedded PostGIS password from sources, configuration and scripts. Connection parameters are read from
  `JORO_DB_HOST`, `JORO_DB_PORT`, `JORO_DB_NAME`, `JORO_DB_USER` and `JORO_DB_PASSWORD` (see `.env.example`); `.env` is git-ignored.
  The previous password remains in the Git history and **must be rotated** on any database that is reachable from a network.

### Known limitations
- The ensemble reproduces ranking (AUC-ROC 0.971 vs 0.904 for the SDM) but underestimates absolute detection totals; the
  5–95 % ensemble band does not cover observed annual totals in any year, even after the prevalence offset (see report 04).
- Frontal direction is not improved by the LPDM: the mean angular error of `M1` is larger than that of the isotropic kernel
  (report 04, Table 3).
- Parameters `gamma_hab`, `k0` and `a0` are strongly collinear (|ρ| ≥ 0.9; report 03).
- With nominal demographic parameters the latitudinal transect fails to persist north of ≈ 34 °N (report 02).
- Benchmarks and validations were run on the development GPU (Intel Arc A380); the Arc A770 target has not been measured yet.

[1.0.0]: https://github.com/oscar-bio-dev/joro-hpc/releases/tag/v1.0.0
