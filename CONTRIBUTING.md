# Contributing to JORO-HPC

Thank you for helping improve JORO-HPC. This is a scientific code base: reproducibility and the integrity of the
out-of-sample validation matter more than speed of merging.

## Ground rules

1. **No data leakage into 2020–2024.** The partition `v_evaluation_set` (2020–2024) and the ERA5 files for those years must not
   influence calibration, model selection, thresholds or comparator fitting. Calibration (`joro_hpc --calibrate`) runs in strict
   mode and rejects any year after 2019; only `joro_evaluate` (with `JORO_UNLOCK_EVALUATION=1`) reads 2020–2024. Any value fitted
   on data (offsets, thresholds, kernel scales) must be fitted on 2014–2019 and documented as such.
2. **Frozen artifacts do not change silently.** Changing `calibration_set_2014_2019.csv`, `evaluation_set_2020_2024.csv` or the
   `Θ_comp` ensembles requires a new SHA-256 in the report manifest and an explanation in the pull request.
3. **`ctest` must pass 4/4** before every pull request:
   ```bash
   cmake -S . -B build && cmake --build build -j"$(nproc)"
   ctest --test-dir build --output-on-failure
   ```
4. **Zero credentials in code.** Passwords, tokens and API keys never appear in sources, configs, notebooks or logs. Read them from the
   environment (`JORO_DB_*`, see `.env.example`). Check with `git grep -i "password\s*=" -- ':!.env.example'` before pushing.
5. **No hard-coded results.** Numbers in reports come from artifacts (`validation_metrics.json`, `calibration_summary_*.json`, ...)
   through the generator scripts. Never type a result by hand into a report or into code that prints a table.
6. **Report limitations honestly.** If a hypothesis criterion fails, the generated verdict must say so.

## Code style

- **C++20**, built with Intel `icpx` (`-fsycl -qopenmp`); the code must also compile with GCC/Clang for the CPU-only CI matrix
  (SYCL parts behind the existing guards).
- Keep hot loops allocation-free and vectorizable (SoA, `#pragma omp simd`); keep the CPU baseline and the GPU kernel physically
  equivalent (`test_cpu_gpu_equivalence`).
- Prefer `std::` algorithms and `constexpr` constants; no raw `new`/`delete`.
- **Python** (scripts): 3.10+, `black` and `flake8` (see `.github/workflows/python-qc.yml`); use the project virtual environment
  (`.venv`), never a global `pip install`.
- Do not delete unrelated comments or docstrings in the files you touch.

## Workflow

1. Open an issue describing the change (scientific motivation for model changes).
2. Branch from `main`: `feature/<short-name>` or `fix/<short-name>`.
3. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
4. If a change modifies model behavior, regenerate the affected reports (`docs/reports/`) and update `CHANGELOG.md`.
5. Open a pull request using the checklist below.

### Pull-request checklist

- [ ] `ctest` 4/4 passes locally
- [ ] No access to 2020–2024 data outside `joro_evaluate`; no fitting on those years
- [ ] No credentials or secrets (`git grep` clean)
- [ ] Reports regenerated from artifacts, if results changed
- [ ] `CHANGELOG.md` updated

## Reporting bugs and security issues

Open a GitHub issue with the command line, the commit hash, the oneAPI version and the relevant log. Report security problems
(for example an exposed credential) privately to the maintainer through GitHub rather than in a public issue.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0 (see `LICENSE`).
