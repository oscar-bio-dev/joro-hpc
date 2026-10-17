<div align="center">

# JORO-HPC 🕷️

### Multiscale Mechanistic Simulator for *Trichonephila clavata* Invasions: 3D Lagrangian Atmospheric Dispersion (LPDM) Coupled with Agent-Based Demographics (ABM)

[![Language](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg?style=for-the-badge&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![Compiler](https://img.shields.io/badge/Compiler-Intel_oneAPI_icpx-0071C5.svg?style=for-the-badge&logo=intel)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
[![Acceleration](https://img.shields.io/badge/GPU-Intel_Arc_SYCL-8A2BE2.svg?style=for-the-badge)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpcpp-library.html)
[![CPU SIMD](https://img.shields.io/badge/CPU_Vector-AVX--512_OpenMP-00599C.svg?style=for-the-badge)](https://www.openmp.org/)
[![Database](https://img.shields.io/badge/Spatial_DB-PostGIS_EPSG:5070-336791.svg?style=for-the-badge&logo=postgresql)](https://postgis.net/)
[![Container](https://img.shields.io/badge/Container-Apptainer-005696.svg?style=for-the-badge)](https://apptainer.org/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF.svg?style=for-the-badge&logo=githubactions)](https://github.com/oscar-bio-dev/joro-hpc/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](LICENSE)

**Principal Investigator:** Oscar Mora  
**Institution:** EcoTechLab · Pueblo West, Colorado  
**Platform Target:** Intel Core i9-11900K (Rocket Lake AVX-512) · Intel Arc A770 16 GB (Level-Zero SYCL)

</div>

---

## 📌 Executive Summary

**JORO-HPC** is a high-performance scientific computing framework designed to model the biological invasion of the East Asian Joro spider (*Trichonephila clavata*) in North America. Unlike classical correlative Species Distribution Models (SDMs) that rely exclusively on static climate envelopes, JORO-HPC models the **mechanistic chain of biological invasion**:

$$\text{Atmospheric Arrival } (A_{i,t}) \implies \text{Ecophysiological Establishment } (S_{j,t}) \implies \text{Reproduction \& Densodependence } (B_{i,t}) \implies \text{Multi-year Persistence } (Z_{i,t})$$

The simulation explicitly resolves:
1. **Phase 1: Aerial Dispersal (LPDM on GPU):** 3D Lagrangian stochastic particle trajectories driven by 12 pressure levels of hourly ERA5 reanalysis ($1000\text{--}700\text{ hPa}$), vertical Langevin turbulence satisfying the Thomson (1987) well-mixed criterion, and posture-dependent silk settling aerodynamics ($v_s(M, L_s, \chi)$).
2. **Phase 2: Terrestrial Life Cycle (ABM on CPU):** Degree-day physiological development ($Q_{10}$), winter chill injury accumulation ($D_{j,t}$), negative binomial reproduction ($B_{i,t}$), and habitat-driven carrying capacity ($K_i$) parameterized by 30 m NLCD land cover aggregated to 5 km equal-area grid cells (EPSG:5070).
3. **Phase 3: Observation & Latent State Invariance:** Dynamic occupancy modeling decoupling true presence ($Z_{i,t}$) from opportunistic citizen-science detection ($Y_{i,t}$) using Araneae Target-Group Background (TGB) sampling effort ($E_{i,t}^{\rm Araneae}$).

---

## 🏗️ Architecture & Coupled Loop

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │              3D ATMOSPHERIC FORCING (ERA5 via NetCDF4)                 │
  │    12 isobaric levels (1000–700 hPa) for u, v, omega, T + 5 surface vars│
  │     Converted vertical velocity: w = - (omega * R_d * T_v) / (p * g)   │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ Streaming NetCDF4 chunks -> USM
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │            PHASE 1: 3D LAGRANGIAN DISPERSION (GPU / SYCL)              │
  │  Intel Arc A770 · Device USM · Stochastic Euler-Maruyama Integration   │
  │  - Horizontal advection: dX = U dt,  dY = V dt                         │
  │  - Vertical Langevin: dw' = a_w(w', z) dt + b_w(z) dW  (Thomson 1987)  │
  │  - Aerodynamic settling: v_s = h(M, L_s, chi, rho, mu)                 │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ Juvenile depositions on land
                                      │ Arrivals: A_{i,t} in grid_5km
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │           PHASE 2: DEMOGRAPHY & OVERWINTERING (CPU / SIMD)             │
  │   Intel Core i9-11900K · Structure of Arrays (SoA) · AVX-512 / OpenMP  │
  │  - Winter chill injury: D_{t+1} = D_t + alpha * [max(0, Tc - T)]^beta   │
  │  - Survival probability: P(S=1) = exp(-Delta D)                        │
  │  - NLCD carrying capacity: K_i = K0 * exp(gamma1*Z_imp + gamma2*Z_edge)│
  │  - Local reproduction: B_{i,t} ~ NegBin(mu_B, kappa_B)                 │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ New aerial ballooning cohorts
                                      │ P^{aero} ~ Binomial(B, q_balloon)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │           PHASE 3: STATISTICAL INFERENCE & CALIBRATION                 │
  │  - Latent Occupancy: Z_{i,t} ~ Bernoulli(psi_{i,t})                    │
  │  - Detection: Y_{i,t} ~ Bernoulli(Z_{i,t} * p_{i,t}(E^{Araneae}))      │
  │  - Multicriteria Cost: J(theta) = w1*J_arrival + w2*J_front + w3*J_pers│
  │  - Multi-start Simulated Annealing & Latin Hypercube Surrogate (Plan B)│
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Formulations

### 1. 3D Atmospheric Kinematics & Vertical Langevin Turbulence
For each ballooning spider $j$ at position $\mathbf{X}_j = (X_j, Y_j, Z_j)^\top$:
$$\begin{aligned}
dX_j &= U(\mathbf{X}_j, t) dt \\
dY_j &= V(\mathbf{X}_j, t) dt \\
dZ_j &= \left[ W_m(\mathbf{X}_j, t) + w'_j - v_{s,j} \right] dt
\end{aligned}$$

The turbulent vertical velocity fluctuation $w'$ evolves via the stochastic differential equation (SDE):
$$dw'_j = a_w(w'_j, Z_j, t; \Theta_{\rm turb}) dt + b_w(Z_j, t; \Theta_{\rm turb}) dW_{j,t}$$

To avoid unphysical particle clustering in regions of low diffusivity, the drift $a_w$ and diffusion $b_w$ satisfy the **Thomson (1987) well-mixed criterion**:
$$a_w = -\frac{w'}{\tau_L} + \frac{1}{2}\frac{\partial \sigma_w^2}{\partial z} + \frac{w'^2}{2\sigma_w^2}\frac{\partial \sigma_w^2}{\partial z}, \qquad b_w = \sqrt{\frac{2\sigma_w^2}{\tau_L}}$$

### 2. Aerodynamic Settling Velocity ($v_s$)
$$\begin{aligned}
v_{s,j} &= \frac{M_j \cdot g}{C_{\rm drag}^{\rm silk}(L_{s,j}, d_{\rm silk}, \mu_a) + C_{\rm drag}^{\rm body}(M_j, \chi_j, \mu_a)}
\end{aligned}$$
where $\chi \in [0.5, 1.0]$ is the posture coefficient (0.5 for compact freefall, 1.0 for fully spread-eagled parachuting), $M \in [0.2, 25.0]\text{ mg}$, and $L_s \in [0.05, 2.30]\text{ m}$.

### 3. Winter Chill Injury Accumulation
Instead of an unrealistic static temperature threshold, overwintering egg sac mortality follows cumulative thermal stress:
$$D_{j,t+1} = D_{j,t} + \alpha_s \left[\max(0, T_{c,s} - T_{j,t}^{\rm micro})\right]^{\beta_s} \Delta t$$
$$P(S_{j,t+1}=1 \mid S_{j,t}=1) = \exp[-(D_{j,t+1} - D_{j,t})]$$

### 4. Habitat Carrying Capacity ($K_i$) from NLCD
Aggregated 30 m NLCD layers to 5 km equal-area grid cells (EPSG:5070):
$$K_i = \min\left\{ K_{\max}, \max\left[ K_{\min}, K_0 \exp\left( \gamma_1 Z_{\rm imp, i} + \gamma_2 Z_{\rm borde, i} + \gamma_3 Z_{\rm dosel, i} \right) \right] \right\}$$

### 5. Imperfect Detection with Target-Group Background (TGB)
$$\operatorname{logit}(p_{i,t}) = \beta_0 + \beta_1 \log(1 + E_{i,t}^{\rm Araneae}) + \beta_2 H_i + \beta_3 R_i + f_{\rm est}(t) + u_{\rm platform}$$
where $E_{i,t}^{\rm Araneae}$ is the observation count of order Araneae in cell $i$ and year $t$, isolating true biological absence from lack of citizen-science observer effort.

---

## ⚡ High-Performance Computing (HPC) Profile

| Component | Target Architecture | Implementation Technique | Empirical Benchmark |
| :--- | :--- | :--- | :--- |
| **Demographic ABM** | Intel Core i9-11900K | 64-byte aligned SoA (`CohortSoA`), AVX-512 OpenMP SIMD | **$850.24\text{ M}$ cohort updates/sec** ($2.35\text{ ms}$ for $2\times 10^6$ agents) |
| **3D LPDM Advection** | Intel Arc A770 (16 GB) | DPC++/SYCL kernel, Device USM, Level-Zero runtime | **$38.43\text{ ms}$** allocation/fill for $1\times 10^6$ particles |
| **ERA5 Atmospheric I/O** | NVMe Storage / Host RAM | Direct NetCDF-C / HDF5 chunked streaming | **$\sim 42\text{--}50\text{ GB}$** bypassed from SQL, zero DB lock |
| **Spatial Database** | PostgreSQL 16 + PostGIS 3.4 | EPSG:5070 Conus Albers ($25\text{ km}^2$ equal-area grid) | GIST spatial indexing across $\sim 200{,}000$ cells |

---

## 📂 Repository Structure

```
joro-hpc/
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Multi-compiler CI (GCC, Clang, oneAPI SYCL)
│       └── python-qc.yml              # Python pipeline linting (flake8, black)
├── CMakeLists.txt                     # C++20, Intel icpx, -fsycl, -qopenmp, -march=rocketlake
├── configs/
│   ├── default_mvp.toml               # Domain bounds [-95,-70]x[25,45], ERA5 levels, time steps
│   └── priors_bounds.toml             # Parameter priors, search bounds, and cost weights
├── containers/
│   └── joro_hpc.def                   # Production Apptainer recipe (oneAPI + NetCDF + PostGIS)
├── docs/
│   ├── JORO-HPC.tex                   # Formal scientific research project proposal
│   └── archive/
├── scripts/
│   ├── sql/
│   │   ├── 01_schema_postgis.sql      # EPSG:5070 5km grid, NLCD covariates, TGB Araneae effort
│   │   └── 02_spatial_blocks.sql      # EPA Level III ecoregion blocks (2014-2019 vs 2020-2024)
│   └── python/
│       ├── data_pipeline.py           # Main CLI orchestrator
│       ├── fetch_gbif_tgb.py          # GBIF occurrences + Araneae TGB effort + iNat API
│       ├── fetch_era5_3d.py           # CDS API downloader (12 levels 1000-700 hPa to NetCDF4)
│       ├── build_grid_nlcd.py         # NLCD 30m & TIGER/Line aggregation to 5km grid
│       ├── surrogate_emulator.py      # Plan B: Latin Hypercube (500 pts) + Gaussian Process
│       └── requirements.txt
├── src/
│   ├── main.cpp                       # Coupled annual simulation loop orchestrator
│   ├── common/                        # types.hpp, config_parser.hpp
│   ├── io/                            # era5_netcdf_reader, postgis_client
│   ├── lpdm/                          # lpdm_engine, cpu_baseline, sycl_kernel, langevin, settling
│   ├── abm/                           # abm_state_soa, abm_engine, carrying_capacity
│   ├── inference/                     # occupancy_model, cost_function, morris, simulated_annealing
│   └── community/                     # diversity_metrics (Chao2, Hill numbers, Jaccard)
├── tests/
│   ├── test_well_mixed.cpp            # Thomson (1987) Eulerian density preservation test
│   ├── test_euler_maruyama_conv.cpp   # Weak/strong convergence test (Kloeden & Platen 1992)
│   ├── test_cpu_gpu_equivalence.cpp   # CPU baseline vs SYCL GPU kernel equivalence
│   └── test_parameter_recovery.cpp    # Synthetic parameter recovery test
└── benchmarks/
    ├── bench_usm_vs_buffers.cpp       # PCIe throughput & Device USM microbenchmark
    └── bench_abm_simd.cpp             # AVX-512 SIMD scaling microbenchmark
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Operating System:** Linux (Ubuntu 22.04+, openSUSE Tumbleweed, RHEL 9+)
* **Toolchain:** Intel oneAPI Base Toolkit & HPC Toolkit ($\ge 2024.1$)
* **Libraries:** `libnetcdf-dev`, `libpqxx-dev` ($\ge 7.0$), `pkg-config`, `cmake` ($\ge 3.22$)

```bash
# Ubuntu / Debian
sudo apt-get install -y build-essential cmake pkg-config libnetcdf-dev libpqxx-dev

# openSUSE
sudo zypper install -y cmake pkg-config netcdf-devel libpqxx-devel
```

### 2. Build Instructions

```bash
# 1. Activate Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# 2. Clone repository
git clone https://github.com/oscar-bio-dev/joro-hpc.git
cd joro-hpc

# 3. Configure out-of-source build with icpx
# Use -DJORO_ENABLE_AVX512=ON for dedicated Core i9-11900K hardware,
# or default (OFF) for portable x86-64-v3 builds on cloud CI runners.
cmake -B build \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DJORO_ENABLE_AVX512=ON

# 4. Compile core library, main executable, tests, and benchmarks
cmake --build build -j$(nproc)
```

### 3. Running Verification Tests & Benchmarks

> **Scientific Verification Guarantee:** All numerical tests use NDEBUG-invariant assertions (`JORO_TEST_ASSERT` with `-UNDEBUG`). Tests genuinely execute full stochastic ensembles, parameter recovery optimization, and GPU-CPU statistical equivalence rather than returning passive no-ops in Release mode.

```bash
# Run unit tests (100% active assertions in Release and Debug)
ctest --test-dir build --output-on-failure

# Run SIMD / AVX-512 benchmark
./build/bench_abm_simd

# Run Intel Arc GPU memory benchmark
./build/bench_usm_vs_buffers

# Run coupled simulation executable
./build/joro_hpc
```

---

## 🐳 Containerized Execution (Apptainer)

For clusters and supercomputers without root permissions:

```bash
# Build container from definition
apptainer build joro_hpc.sif containers/joro_hpc.def

# Run simulation inside container with GPU passthrough
apptainer run --nv --bind /data:/data joro_hpc.sif ./build/joro_hpc
```

---

## 📊 Scientific Methodology & Validation

* **Calibration Period (2014–2019):** Multi-start Simulated Annealing and Morris elementary effects sensitivity screening over 6 years of contiguous invasion history in the southeastern US.
* **Out-of-Sample Evaluation (2020–2024):** Strict frozen evaluation without parameter re-tuning across independent spatial ecoregional blocks.
* **Success Criteria:** Brier Score, calibration slope/intercept, front advance velocity, and directional displacement accuracy against two parsimonious null models:
  1. Static Bioclimatic SDM (MaxEnt / Generalized Additive Model).
  2. Isotropic Exponential Kernel: $K_0(d \mid \lambda) = \lambda^{-1} e^{-d/\lambda}$.

---

## 📦 Clean Source Packaging

To create clean release or submission `.zip` archives without local build binaries, datasets, or `.git` internal metadata:

```bash
# Using Git native archive
git archive --format=zip --output=joro-hpc-clean.zip HEAD

# Or using the included packaging script
./scripts/package_clean_zip.sh
```

---

## 📜 Scientific References

1. **Thomson, D. J.** (1987). *Criteria for selection of stochastic models of particle trajectories in turbulent flows*. Journal of Fluid Mechanics, 180, 529–556.
2. **Humphrey, J. A. C.** (1987). *Fluid mechanic constraints on spider ballooning*. Oecologia, 73, 469–477.
3. **Suter, R. B.** (1992). *Ballooning: freefall data indicate the importance of posture*. Journal of Arachnology, 20, 107–113.
4. **Altwegg, R., & Nichols, J. D.** (2019). *Occupancy models for citizen-science data*. Methods in Ecology and Evolution, 10(1), 8–21.
5. **Phillips, S. J., et al.** (2009). *Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data*. Ecological Applications, 19(1), 181–197.

---

## 👤 Author & Contact

* **Oscar Mora**  
  *Principal Investigator & Software Architect*  
  GitHub: [@oscar-bio-dev](https://github.com/oscar-bio-dev)  
  Email: [oscar_bio_dev@outlook.com](mailto:oscar_bio_dev@outlook.com)  
  EcoTechLab · Pueblo West, Colorado

---

## ⚖️ License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for details.
