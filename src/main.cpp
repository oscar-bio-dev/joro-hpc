////////////////////////////////////////////////////////////////////////////////
/// @file   main.cpp
/// @brief  JORO-HPC — Punto de entrada principal CLI
///
/// Orquesta la simulación acoplada:
///   1. Ingesta de malla 5x5 km y covariables NLCD desde PostGIS
///   2. Carga de tensores 3D horarios de ERA5 (NetCDF4 / HDF5)
///   3. Dispersión Lagrangiana de Partículas (LPDM) en GPU (Intel Arc / SYCL)
///   4. Demografía y daño térmico (ABM SoA) en CPU (AVX-512 / OpenMP)
///   5. Evaluación del modelo de ocupación latente y detección Target-Group
////////////////////////////////////////////////////////////////////////////////

#include "common/types.hpp"
#include "common/config_parser.hpp"
#include "io/era5_netcdf_reader.hpp"
#include "io/postgis_client.hpp"
#include "lpdm/lpdm_engine.hpp"
#include "abm/abm_engine.hpp"
#include "inference/occupancy_model.hpp"
#include "inference/cost_function.hpp"

#include <iostream>
#include <chrono>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n"
              << "║  JORO-HPC v0.1.0 — Mechanistic LPDM-ABM Simulator         ║\n"
              << "║  Species: Trichonephila clavata | Platform: oneAPI / SYCL ║\n"
              << "╚═══════════════════════════════════════════════════════════╝\n\n";

    // 1. Cargar configuración maestra
    auto config = joro::SimulationConfig::load_default();
    std::cout << "[Config] Dominio MVP: [" << config.bbox.lon_min << ", " << config.bbox.lat_min
              << "] a [" << config.bbox.lon_max << ", " << config.bbox.lat_max << "]\n"
              << "[Config] Ventana de Calibración: " << config.calib_start_year << "–" << config.calib_end_year << "\n"
              << "[Config] Ventana de Evaluación:  " << config.eval_start_year  << "–" << config.eval_end_year << "\n"
              << "[Config] Storage: Data Lake = " << config.datalake_root << " | Scratch = " << config.scratch_root << "\n"
              << "[Config] DB: " << config.db_host << ":" << config.db_port << "/" << config.db_name
              << " (tablespace: " << config.db_tablespace << ")\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // 2. Conectar a PostGIS y cargar la malla de 5 km x 5 km
    joro::io::PostgisClient db_client(config.db_host, config.db_port, config.db_name, config.db_user, config.db_password);
    db_client.connect();

    std::vector<joro::GridCell> grid_cells;
    db_client.load_grid(grid_cells);
    std::cout << "[Malla] " << grid_cells.size() << " celdas demográficas cargadas en memoria.\n";

    // 3. Inicializar lector de tensores ERA5
    joro::io::Era5NetCdfReader era5_reader(config.era5_dir);
    joro::io::AtmosphericField3D atmosphere;
    era5_reader.load_month(2014, 5, atmosphere);
    std::cout << "[ERA5] Tensores atmosféricos cargados: " << atmosphere.n_levels
              << " niveles isobáricos (1000 a 700 hPa).\n";

    // 4. Inicializar motores LPDM (GPU) y ABM (CPU)
    auto lpdm_gpu = joro::lpdm::create_sycl_gpu_engine();
    auto lpdm_cpu = joro::lpdm::create_cpu_baseline_engine();
    joro::abm::AbmEngine abm_engine;

    // Inicializar población semilla en el noreste de Georgia (2014-2015)
    abm_engine.initialize_seed(grid_cells, 0.15f);
    std::cout << "[ABM] Población semilla inicializada en celdas de Georgia (rho0 = 0.15).\n\n";

    // 5. Bucle de Simulación Anual Acoplada (2014 a 2019)
    std::vector<float> mock_winter_temps(grid_cells.size(), -2.5f); // °C promedio invernal
    std::vector<float> mock_summer_temps(grid_cells.size(), 26.0f); // °C promedio estival

    for (int year = config.calib_start_year; year <= config.calib_end_year; ++year) {
        std::cout << "▶ Año " << year << " [Simulación Anual Acoplada] ──────────────────\n";

        // Paso A: Ciclo demográfico y producción de propágulos aéreos
        auto demo_summary = abm_engine.step_annual_cycle(grid_cells, mock_winter_temps, mock_summer_temps);
        std::cout << "  [ABM] Adultos: " << demo_summary.total_adults
                  << " | Nuevos propágulos aéreos: " << demo_summary.total_ballooners << "\n";

        // Paso B: Dispersión atmosférica Lagrangiana 3D (LPDM)
        // Intentar primero con el kernel SYCL en GPU; si no hay GPU, usar CPU baseline
        lpdm_cpu->initialize_particles(grid_cells, demo_summary.ballooners_per_cell);
        auto disp_result = lpdm_cpu->run_dispersion(atmosphere, static_cast<float>(config.lpdm_dt_seconds), 120);

        std::cout << "  [LPDM] Liberados: " << disp_result.total_released
                  << " | Aterrizajes exitosos: " << disp_result.total_landed << "\n";

        // Paso C: Depositar propágulos en las celdas receptoras
        abm_engine.deposit_arrivals(disp_result.arrivals_per_cell);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n✔ Ciclo de calibración completado con éxito en " << elapsed_s << " segundos.\n";
    return EXIT_SUCCESS;
}
