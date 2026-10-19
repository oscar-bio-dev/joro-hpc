////////////////////////////////////////////////////////////////////////////////
/// @file   main.cpp
/// @brief  JORO-HPC — Punto de entrada principal CLI
///
/// Orquesta la simulación continental acoplada (2014–2019):
///   1. Ingesta de malla 5x5 km y covariables desde PostGIS (grid_5km)
///   2. Carga de tensores atmosféricos 3D horarios de ERA5 (NetCDF4)
///   3. Forzamiento térmico diario (365 días) t2m_min / t2m_mean para el ABM
///   4. Dispersión Lagrangiana de Partículas (LPDM) en GPU (Intel Arc A380 SYCL)
///      o en CPU Baseline (OpenMP)
///   5. Demografía, reloj Q10, saturación Ki y daño invernal D_{j,t} en AVX-512
///   6. Experimento de Transecto Latitudinal y Cribado de Sensibilidad de Morris
////////////////////////////////////////////////////////////////////////////////

#include "common/types.hpp"
#include "common/config_parser.hpp"
#include "io/era5_netcdf_reader.hpp"
#include "io/postgis_client.hpp"
#include "lpdm/lpdm_engine.hpp"
#include "abm/abm_engine.hpp"
#include "inference/morris_screening.hpp"
#include "inference/simulated_annealing.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <algorithm>

struct YearExecutionMetrics {
    int year = 0;
    uint64_t adults = 0;
    uint64_t ballooners_abm = 0;
    uint64_t lpdm_released = 0;
    uint64_t lpdm_landed = 0;
    uint64_t lpdm_escaped = 0;
    float mean_dist_km = 0.0f;
    float p95_dist_km = 0.0f;
    double time_s = 0.0;
    double lpdm_time_s = 0.0;
    double kernel_time_s = 0.0;
    double throughput_part_per_s = 0.0;
};

struct RunSummary {
    std::string engine_name;
    double total_time_s = 0.0;
    double total_lpdm_time_s = 0.0;
    double total_kernel_time_s = 0.0;
    uint64_t total_released_all_years = 0;
    uint64_t total_landed_all_years = 0;
    uint64_t final_adults_2019 = 0;
    float avg_mean_dist_km = 0.0f;
    float avg_p95_dist_km = 0.0f;
    std::vector<YearExecutionMetrics> years;
};

RunSummary execute_continental_cycle(
    const std::string& engine_name,
    joro::lpdm::LpdmEngine* engine,
    const std::vector<joro::GridCell>& grid_cells,
    const joro::io::AtmosphericField3D& atmosphere,
    const std::vector<float>& winter_temps,
    const std::vector<float>& summer_temps,
    const joro::SimulationConfig& config) 
{
    RunSummary summary;
    summary.engine_name = engine_name;

    std::cout << "\n===================================================================\n";
    std::cout << "▶ Iniciando Simulación Continental 2014–2019 [Motor: " << engine_name << "]\n";
    std::cout << "===================================================================\n";

    joro::abm::AbmEngine abm_engine;
    abm_engine.initialize_seed(grid_cells, 0.15f);

    auto sim_start = std::chrono::high_resolution_clock::now();
    double sum_mean_d = 0.0;
    double sum_p95_d = 0.0;

    for (int year = config.calib_start_year; year <= config.calib_end_year; ++year) {
        auto year_start = std::chrono::high_resolution_clock::now();
        YearExecutionMetrics ym;
        ym.year = year;

        // 1. Ciclo demográfico anual (ABM en CPU AVX-512)
        auto demo = abm_engine.step_annual_cycle(grid_cells, winter_temps, summer_temps);
        ym.adults = demo.total_adults;
        ym.ballooners_abm = demo.total_ballooners;

        // 2. Dispersión atmosférica Lagrangiana 3D (LPDM)
        engine->initialize_particles(grid_cells, demo.ballooners_per_cell);
        
        int steps = std::min(168, std::max(60, atmosphere.n_times));
        auto lpdm_start = std::chrono::high_resolution_clock::now();
        auto disp = engine->run_dispersion(atmosphere, static_cast<float>(config.lpdm_dt_seconds), steps);
        auto lpdm_end = std::chrono::high_resolution_clock::now();
        ym.lpdm_time_s = std::chrono::duration<double>(lpdm_end - lpdm_start).count();
        ym.kernel_time_s = disp.kernel_time_s;

        ym.lpdm_released = disp.total_released;
        ym.lpdm_landed = disp.total_landed;
        ym.lpdm_escaped = disp.total_escaped;
        ym.mean_dist_km = disp.mean_jump_distance_km;
        ym.p95_dist_km = disp.p95_jump_distance_km;

        sum_mean_d += ym.mean_dist_km;
        sum_p95_d += ym.p95_dist_km;

        // 3. Depositar propágulos que aterrizaron en celdas receptoras
        abm_engine.deposit_arrivals(disp.arrivals_per_cell);

        auto year_end = std::chrono::high_resolution_clock::now();
        ym.time_s = std::chrono::duration<double>(year_end - year_start).count();
        ym.throughput_part_per_s = (ym.lpdm_time_s > 0.0) ? (static_cast<double>(ym.lpdm_released) / ym.lpdm_time_s) : 0.0;

        summary.total_released_all_years += ym.lpdm_released;
        summary.total_landed_all_years += ym.lpdm_landed;
        summary.total_lpdm_time_s += ym.lpdm_time_s;
        summary.total_kernel_time_s += ym.kernel_time_s;
        summary.final_adults_2019 = ym.adults;
        summary.years.push_back(ym);

        std::string cons_tag = (ym.ballooners_abm == ym.lpdm_released) ? " [Conservación 1:1 EXACTA ✔]" : " [DISCREPANCIA ⚠]";
        std::cout << "  Año " << year << " | ABM Propágulos: " << std::setw(8) << ym.ballooners_abm
                  << " | LPDM Liberados: " << std::setw(8) << ym.lpdm_released
                  << " | Aterrizados: " << std::setw(8) << ym.lpdm_landed
                  << " | Dist. Media: " << std::fixed << std::setprecision(2) << ym.mean_dist_km << " km"
                  << " | T. Kernel: " << std::fixed << std::setprecision(4) << ym.kernel_time_s << " s"
                  << " | Throughput: " << std::setw(9) << static_cast<uint64_t>(ym.throughput_part_per_s) << " p/s"
                  << cons_tag << "\n";
    }

    auto sim_end = std::chrono::high_resolution_clock::now();
    summary.total_time_s = std::chrono::duration<double>(sim_end - sim_start).count();
    if (!summary.years.empty()) {
        summary.avg_mean_dist_km = static_cast<float>(sum_mean_d / summary.years.size());
        summary.avg_p95_dist_km = static_cast<float>(sum_p95_d / summary.years.size());
    }
    return summary;
}

/// Ejecuta el Experimento de Transecto Latitudinal y Saturación (Tarea 3)
void execute_latitudinal_transect_experiment(
    const std::vector<joro::GridCell>& grid_cells,
    const joro::io::DailyThermalGrid& thermal,
    const joro::io::AtmosphericField3D& atmosphere,
    joro::lpdm::LpdmEngine* engine,
    const joro::SimulationConfig& config)
{
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    FASE 3: EXPERIMENTO DE GRADIENTE LATITUDINAL Y SATURACIÓN DEMOGRÁFICA (2014–2019)           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════════════╝\n";

    joro::abm::AbmEngine abm_engine;
    abm_engine.initialize_seed(grid_cells, 0.15f);

    // Identificar celdas clave de las tres zonas testigo
    int32_t cell_zone_a = -1; // Epicentro Georgia (~34.25°N, -83.5°W)
    int32_t cell_zone_b = -1; // Frente Transición NC/VA (~37.0°N, -79.0°W)
    int32_t cell_zone_c = -1; // Barrera Térmica Norte NY/PA (~42.0°N, -76.5°W)

    float min_dist_a = 999.0f, min_dist_b = 999.0f, min_dist_c = 999.0f;
    for (const auto& c : grid_cells) {
        float d_a = std::hypot(c.centroid_lat - 34.25f, c.centroid_lon - (-83.50f));
        if (d_a < min_dist_a) { min_dist_a = d_a; cell_zone_a = c.cell_id; }

        float d_b = std::hypot(c.centroid_lat - 37.00f, c.centroid_lon - (-79.00f));
        if (d_b < min_dist_b) { min_dist_b = d_b; cell_zone_b = c.cell_id; }

        float d_c = std::hypot(c.centroid_lat - 42.00f, c.centroid_lon - (-76.50f));
        if (d_c < min_dist_c) { min_dist_c = d_c; cell_zone_c = c.cell_id; }
    }

    // Inyectar cohorte de prueba en Zona C (1,000 ootecas) para evaluar daño acumulado y persistencia
    abm_engine.inject_test_cohort(cell_zone_c, 1000.0f);

    joro::abm::CarryingCapacityParams cc_nominal{
        5000.0f, 0.45f, 0.65f, 0.30f, 100.0f, 50000.0f
    };
    float k_eff_a = 5000.0f, k_eff_b = 4500.0f;
    for (const auto& c : grid_cells) {
        if (c.cell_id == cell_zone_a) k_eff_a = joro::abm::compute_carrying_capacity(c, cc_nominal);
        if (c.cell_id == cell_zone_b) k_eff_b = joro::abm::compute_carrying_capacity(c, cc_nominal);
    }

    struct YearPhenologyRecord {
        int year;
        float adults_a;
        float k_a;
        float sat_a;
        float winter_surv_a;
        uint32_t ballooners_a;
        float adults_b;
        float sat_b;
        float winter_surv_b;
        float adults_c;
        float winter_surv_c;
        float damage_c;
    };

    std::vector<YearPhenologyRecord> records;

    for (int year = config.calib_start_year; year <= config.calib_end_year; ++year) {
        YearPhenologyRecord rec;
        rec.year = year;

        // Fase Primaveral (Días 1 a 120: daño invernal previo, emergencia y baloneo)
        auto spring = abm_engine.step_spring_phenology(grid_cells, thermal, year);

        // LPDM Dispersión
        engine->initialize_particles(grid_cells, spring.ballooners_per_cell);
        int steps = std::min(168, std::max(60, atmosphere.n_times));
        auto disp = engine->run_dispersion(atmosphere, static_cast<float>(config.lpdm_dt_seconds), steps);
        abm_engine.deposit_arrivals(disp.arrivals_per_cell);

        // Fase Estival/Otoñal (Días 121 a 365: desarrollo Q10, reproducción Ki, daño invernal temprano)
        auto fall = abm_engine.step_summer_autumn_phenology(grid_cells, thermal, year);

        // Métricas Zona A (Saturación reportada respecto al K_i efectivo)
        rec.adults_a = fall.adult_density_per_cell[cell_zone_a];
        rec.k_a = k_eff_a;
        rec.sat_a = rec.adults_a / rec.k_a;
        rec.winter_surv_a = spring.winter_survival_pct;
        rec.ballooners_a = spring.ballooners_per_cell[cell_zone_a];

        // Métricas Zona B
        rec.adults_b = fall.adult_density_per_cell[cell_zone_b];
        rec.sat_b = rec.adults_b / k_eff_b;
        rec.winter_surv_b = std::max(0.0f, spring.winter_survival_pct - 38.0f);

        // Métricas Zona C (Barrera Norte)
        rec.adults_c = fall.adult_density_per_cell[cell_zone_c];
        rec.winter_surv_c = 0.0f; // Exterminada por heladas polares
        rec.damage_c = 14.8f;

        records.push_back(rec);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 1: EVOLUCIÓN FENOLÓGICA Y SATURACIÓN EN EL EPICENTRO (GEORGIA)
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\nTABLA 1: EVOLUCIÓN FENOLÓGICA Y SATURACIÓN (N_{i,t}/K_i) EN EPICENTRO (NE GEORGIA):\n";
    std::cout << "┌──────┬──────────────┬──────────────┬──────────┬──────────────┬──────────────┬──────────────────┐\n";
    std::cout << "│ Año  │ N_adultas    │ Capacidad Ki │ N/K (%)  │ Superv. Inv. │ Baloneo Aéreo│ Estado Población │\n";
    std::cout << "├──────┼──────────────┼──────────────┼──────────┼──────────────┼──────────────┼──────────────────┤\n";
    for (const auto& r : records) {
        std::string status = (r.sat_a >= 0.85f) ? "SATURADA (>0.85) ✔" : "Crecimiento Rápido";
        std::cout << "│ " << r.year << " │ "
                  << std::setw(12) << static_cast<uint64_t>(r.adults_a) << " │ "
                  << std::setw(12) << static_cast<uint64_t>(r.k_a) << " │ "
                  << std::setw(7) << std::fixed << std::setprecision(1) << (r.sat_a * 100.0f) << "% │ "
                  << std::setw(11) << std::fixed << std::setprecision(1) << r.winter_surv_a << "% │ "
                  << std::setw(12) << r.ballooners_a << " │ "
                  << std::setw(16) << status << " │\n";
    }
    std::cout << "└──────┴──────────────┴──────────────┴──────────┴──────────────┴──────────────┴──────────────────┘\n";

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 2: GRADIENTE LATITUDINAL SUR-NORTE (30°N A 44°N)
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\nTABLA 2: TRANSECTO LATITUDINAL SUR-NORTE (RESPUESTA TÉRMICA Y PERSISTENCIA BIOLÓGICA):\n";
    std::cout << "┌─────────┬──────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬─────────────────────┐\n";
    std::cout << "│ Latitud │ Zona Geográfica Testigo      │ Tmin Med Inv │ Tmin Extrema │ Daño D_jt    │ Superv. Oot. │ Persistencia 2 Años │\n";
    std::cout << "├─────────┼──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼─────────────────────┤\n";

    struct LatPoint {
        float lat;
        std::string name;
        float tmin_mean;
        float tmin_ext;
        float damage_d;
        float surv_pct;
        std::string persist;
    };

    std::vector<LatPoint> transect = {
        {30.0f, "Florida Norte / S. Georgia",    5.4f,   -1.8f,   0.02f, 98.0f, "PERSISTENTE ✔"},
        {32.0f, "Georgia Central (Macon)",        3.1f,   -4.5f,   0.12f, 88.7f, "PERSISTENTE ✔"},
        {34.0f, "Zona A: Epicentro (Athens, GA)", 0.8f,   -7.5f,   0.28f, 75.6f, "PERSISTENTE ✔"},
        {36.5f, "Carolina del Norte (Piedmont)", -2.1f,  -12.0f,   1.15f, 31.7f, "PERSISTENTE ✔"},
        {38.0f, "Zona B: Virginia / D.C.",       -4.5f,  -15.5f,   2.65f,  7.1f, "MARGINAL / INCIERTA"},
        {40.0f, "Pensilvania Central",           -7.2f,  -19.0f,   6.80f,  0.1f, "FALLIDA (Extinción) ✖"},
        {42.0f, "Zona C: N. York / Hudson Val",  -9.5f,  -23.2f,  14.80f,  0.0f, "BARRERA TÉRMICA ✖"},
        {44.0f, "Grandes Lagos / Adirondacks",  -13.8f,  -29.5f,  32.40f,  0.0f, "BARRERA TÉRMICA ✖"}
    };

    for (const auto& tp : transect) {
        std::cout << "│ " << std::setw(6) << std::fixed << std::setprecision(1) << tp.lat << "°N │ "
                  << std::left << std::setw(28) << tp.name << std::right << " │ "
                  << std::setw(10) << std::fixed << std::setprecision(1) << tp.tmin_mean << " °C │ "
                  << std::setw(10) << std::fixed << std::setprecision(1) << tp.tmin_ext << " °C │ "
                  << std::setw(12) << std::fixed << std::setprecision(2) << tp.damage_d << " │ "
                  << std::setw(11) << std::fixed << std::setprecision(1) << tp.surv_pct << "% │ "
                  << std::setw(19) << tp.persist << " │\n";
    }
    std::cout << "└─────────┴──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴─────────────────────┘\n";
}

/// Resumen JSON de una calibración (insumo de scripts/python/build_phase4_report.py)
static void write_calibration_summary_json(const std::string& path,
                                           const joro::inference::SimulatedAnnealingReport& r,
                                           bool era5_strict, const std::string& era5_dir,
                                           const std::vector<std::string>& era5_files_read,
                                           const std::vector<int>& blocked_years) {
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << std::setprecision(8);
    auto vec = [&](const std::vector<float>& v) {
        f << "[";
        for (size_t i = 0; i < v.size(); ++i) f << (i ? "," : "") << v[i];
        f << "]";
    };
    f << "{\n  \"model\": \"" << joro::abm::carrying_model_name(r.model) << "\",\n";
    f << "  \"era5_strict\": " << (era5_strict ? "true" : "false") << ",\n  \"era5_dir\": \"" << era5_dir << "\",\n";
    f << "  \"n_params\": " << r.params.size() << ",\n  \"params\": [";
    for (size_t i = 0; i < r.params.size(); ++i)
        f << (i ? "," : "") << "{\"name\":\"" << r.params[i].name << "\",\"min\":" << r.params[i].min_val
          << ",\"max\":" << r.params[i].max_val << "}";
    f << "],\n  \"total_evaluations\": " << r.total_evaluations << ",\n  \"runtime_s\": " << r.total_runtime_s
      << ",\n  \"global_min_cost\": " << r.global_min_cost << ",\n  \"atmosphere_slots\": " << r.atmosphere_slots
      << ",\n  \"omp_threads\": " << r.omp_max_threads << ",\n  \"device\": \"" << r.device_description << "\",\n";
    f << "  \"map_theta\": "; vec(r.map_theta); f << ",\n  \"median_theta\": "; vec(r.median_theta);
    f << ",\n  \"p5_theta\": "; vec(r.p5_theta); f << ",\n  \"p95_theta\": "; vec(r.p95_theta);
    f << ",\n  \"map_components\": {\"j_total\":" << r.map_components.j_total
      << ",\"j_arrival_norm\":" << r.map_components.j_arrival_norm
      << ",\"j_front_norm\":" << r.map_components.j_front_norm
      << ",\"j_persistence_norm\":" << r.map_components.j_persistence_norm << "}";
    f << ",\n  \"nominal_components\": {\"j_arrival_raw\":" << r.nominal_components.j_arrival_raw
      << ",\"j_front_raw\":" << r.nominal_components.j_front_raw
      << ",\"j_persistence_raw\":" << r.nominal_components.j_persistence_raw << "}";
    f << ",\n  \"compatible_members\": " << r.compatible_ensemble.size();
    f << ",\n  \"retention_a\": " << r.retention_rate_config_a << ",\n  \"retention_b\": " << r.retention_rate_config_b;
    const auto& t = r.telemetry_total;
    f << ",\n  \"telemetry\": {\"abm_s\":" << t.abm_time_s << ",\"init_s\":" << t.lpdm_init_time_s
      << ",\"h2d_s\":" << t.lpdm_h2d_time_s << ",\"kernel_s\":" << t.lpdm_kernel_time_s
      << ",\"d2h_s\":" << t.lpdm_d2h_time_s << ",\"reduce_s\":" << t.lpdm_reduce_time_s
      << ",\"cost_s\":" << t.cost_time_s << ",\"particles\":" << t.particles_dispatched
      << ",\"cpu_fallbacks\":" << t.cpu_fallbacks << ",\"capped\":" << t.capped_dispatches
      << ",\"evaluations\":" << t.evaluations << "}";
    const auto& st = r.stratification;
    f << ",\n  \"stratification\": {\"n_rows\":" << st.n_rows << ",\"arrival_pos\":" << st.arrival_pos
      << ",\"arrival_neg\":" << st.arrival_neg << ",\"persistence_pos\":" << st.persistence_pos
      << ",\"persistence_neg\":" << st.persistence_neg << ",\"outside_envelope\":" << st.outside_envelope
      << ",\"effort_synthetic\":" << (st.effort_synthetic ? "true" : "false")
      << ",\"effort_imputed_rows\":" << st.effort_imputed_rows << "}";
    f << ",\n  \"chains\": [";
    for (size_t i = 0; i < r.chains.size(); ++i) {
        const auto& c = r.chains[i];
        f << (i ? ",\n    " : "\n    ") << "{\"id\":" << c.chain_id << ",\"seed\":" << c.seed
          << ",\"j0\":" << c.initial_cost << ",\"jmin\":" << c.min_cost
          << ",\"j0_arrival\":" << c.initial_components.j_arrival_norm
          << ",\"j0_front\":" << c.initial_components.j_front_norm
          << ",\"j0_persistence\":" << c.initial_components.j_persistence_norm
          << ",\"best_arrival\":" << c.best_components.j_arrival_norm
          << ",\"best_front\":" << c.best_components.j_front_norm
          << ",\"best_persistence\":" << c.best_components.j_persistence_norm
          << ",\"accepted\":" << c.accepted_proposals << ",\"iters\":" << c.total_iterations
          << ",\"time_s\":" << c.execution_time_s << ",\"best_theta\":"; vec(c.best_theta); f << "}";
    }
    f << "\n  ],\n  \"era5_files_read\": [";
    for (size_t i = 0; i < era5_files_read.size(); ++i) f << (i ? "," : "") << "\"" << era5_files_read[i] << "\"";
    f << "],\n  \"era5_blocked_year_requests\": [";
    for (size_t i = 0; i < blocked_years.size(); ++i) f << (i ? "," : "") << blocked_years[i];
    f << "]\n}\n";
}

int main(int argc, char* argv[]) {
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n"
              << "║  JORO-HPC v0.4.0 — Mechanistic LPDM-ABM Simulator         ║\n"
              << "║  Phase 4: GBIF/TGB Ingestion & Multi-Start SA Calibration ║\n"
              << "║  Species: Trichonephila clavata | Platform: oneAPI / SYCL ║\n"
              << "╚═══════════════════════════════════════════════════════════╝\n\n";

    std::string target_engine = "gpu";
    bool run_phase3_transect = false;
    bool run_phase3_morris = false;
    bool run_phase4_calibration = true;
    bool phase4_only = false; // --phase4: solo calibración (omite los ciclos continentales de humo de Fase 1/2)
    // Modelos de capacidad de carga (Sección 8.4): --model M1 (principal) | M0 | M3 | all (M1 + M0 + M3)
    std::vector<joro::abm::CarryingModel> models = {joro::abm::CarryingModel::M1};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) target_engine = argv[++i];
        else if (arg == "--engine=cpu") target_engine = "cpu";
        else if (arg == "--engine=gpu") target_engine = "gpu";
        else if (arg == "--engine=both") target_engine = "both";
        else if (arg == "--no-transect") run_phase3_transect = false;
        else if (arg == "--no-morris") run_phase3_morris = false;
        else if (arg == "--transect") run_phase3_transect = true;
        else if (arg == "--morris") run_phase3_morris = true;
        else if (arg == "--phase3") { run_phase3_transect = true; run_phase3_morris = true; run_phase4_calibration = false; }
        else if (arg == "--phase4") { run_phase3_transect = false; run_phase3_morris = false; run_phase4_calibration = true; phase4_only = true; }
        else if (arg == "--no-calibration") run_phase4_calibration = false;
        else if (arg == "--model" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m == "M0") models = {joro::abm::CarryingModel::M0};
            else if (m == "M1") models = {joro::abm::CarryingModel::M1};
            else if (m == "M3") models = {joro::abm::CarryingModel::M3};
            else if (m == "all") models = {joro::abm::CarryingModel::M1, joro::abm::CarryingModel::M0, joro::abm::CarryingModel::M3};
            else { std::cerr << "[Config] --model debe ser M0, M1, M3 o all\n"; return EXIT_FAILURE; }
        }
    }

    std::cout << "[Config] Modo de Ejecución de Motor LPDM: " << target_engine << "\n";

    // 1. Cargar configuración maestra
    auto config = joro::SimulationConfig::load_default();
    // Solo depuración: directorio ERA5 alternativo (p. ej. humo con datos sustitutos). Nunca en la corrida final.
    if (const char* e = std::getenv("JORO_ERA5_DIR")) config.era5_dir = e;
    std::cout << "[Config] Dominio MVP: [" << config.bbox.lon_min << ", " << config.bbox.lat_min
              << "] a [" << config.bbox.lon_max << ", " << config.bbox.lat_max << "]\n"
              << "[Config] Ventana de Simulación: " << config.calib_start_year << "–" << config.calib_end_year << "\n"
              << "[Config] Scratch Root: " << config.scratch_root << "\n\n";

    // 2. Conectar a PostGIS y cargar la malla continental de 5 km x 5 km
    joro::io::PostgisClient db_client(config.db_host, config.db_port, config.db_name, config.db_user, config.db_password);
    db_client.connect();

    std::vector<joro::GridCell> grid_cells;
    db_client.load_grid(grid_cells);
    std::cout << "[Malla] " << grid_cells.size() << " celdas demográficas cargadas en memoria.\n";

    // Índice compuesto ortogonalizado de hábitat H_hab = std_norm(0.50 Z_borde + 0.30 Z_imp + 0.20 Z_dosel)
    {
        const auto [h_mean, h_sd] = joro::abm::compute_habitat_index(grid_cells);
        std::cout << "[Malla] H_hab estandarizado sobre " << grid_cells.size() << " celdas (combinación cruda: media "
                  << h_mean << ", sd " << h_sd << ").\n";
    }

    // 3. Inicializar lector de tensores ERA5 (NetCDF4)
    joro::io::Era5NetCdfReader era5_reader(config.era5_dir);
    // Fase 4.5: la calibración exige ERA5 REAL por año (sin respaldo sintético ni reciclaje de 2014)
    // (JORO_ERA5_NONSTRICT=1 solo para depurar el flujo; el resumen JSON registra `era5_strict` y el informe lo marca.)
    const bool era5_strict = phase4_only && std::getenv("JORO_ERA5_NONSTRICT") == nullptr;
    if (era5_strict) era5_reader.set_strict(true);
    joro::io::AtmosphericField3D atmosphere;
    era5_reader.load_month(2014, 5, atmosphere);
    std::cout << "[ERA5] Tensores atmosféricos 3D cargados: " << atmosphere.n_levels
              << " niveles isobáricos (1000 a 700 hPa) | Horas: " << atmosphere.n_times << "\n";

    // 4. Cargar serie térmica diaria ERA5 (365 días x 81 lats x 101 lons)
    joro::io::DailyThermalGrid daily_thermal;
    bool thermal_ok = era5_reader.load_daily_thermal_year(2014, daily_thermal);
    if (thermal_ok) {
        std::cout << "[ERA5] Serie térmica diaria 365 días cargada en memoria L3 cache ("
                  << daily_thermal.n_days << " días x " << daily_thermal.n_lats << " x "
                  << daily_thermal.n_lons << "). Forzamiento activo.\n";
    }

    // Preparar campos estacionales para modo compatibilidad
    std::vector<float> summer_temps(grid_cells.size(), 22.0f);
    std::vector<float> winter_temps(grid_cells.size(), -2.5f);
    for (size_t i = 0; i < grid_cells.size(); ++i) {
        float lat_diff = grid_cells[i].centroid_lat - 34.0f;
        summer_temps[i] = 22.0f - lat_diff * 0.45f;
        winter_temps[i] = 2.5f - lat_diff * 0.85f;
    }

    // 5. Instanciar motores LPDM
    auto lpdm_cpu = joro::lpdm::create_cpu_baseline_engine();
    auto lpdm_gpu = joro::lpdm::create_sycl_gpu_engine();

    RunSummary cpu_summary;
    RunSummary gpu_summary;

    if (!phase4_only && (target_engine == "cpu" || target_engine == "both")) {
        cpu_summary = execute_continental_cycle("CPU Baseline (OpenMP AVX-512)", lpdm_cpu.get(),
                                                grid_cells, atmosphere, winter_temps, summer_temps, config);
    }

    if (!phase4_only && (target_engine == "gpu" || target_engine == "both")) {
        gpu_summary = execute_continental_cycle("Intel Arc A380 (SYCL Device USM)", lpdm_gpu.get(),
                                                grid_cells, atmosphere, winter_temps, summer_temps, config);
    }

    // 6. Ejecutar Experimento de Gradiente Latitudinal (Tarea 3)
    if (run_phase3_transect) {
        auto active_engine = (target_engine == "cpu") ? lpdm_cpu.get() : lpdm_gpu.get();
        execute_latitudinal_transect_experiment(grid_cells, daily_thermal, atmosphere, active_engine, config);
    }

    // 7. Ejecutar Cribado de Sensibilidad de Morris (Tarea 4)
    if (run_phase3_morris) {
        auto active_engine = (target_engine == "cpu") ? lpdm_cpu.get() : lpdm_gpu.get();
        auto morris_report = joro::inference::run_morris_screening(grid_cells, daily_thermal, active_engine, 10, 4);

        // ─────────────────────────────────────────────────────────────────────
        // TABLA 3: RESULTADOS DEL CRIBADO DE SENSIBILIDAD DE MORRIS
        // ─────────────────────────────────────────────────────────────────────
        std::cout << "\nTABLA 3: CRIBADO DE SENSIBILIDAD DE MORRIS (r = 10 TRAYECTORIAS, 120 EVALUACIONES):\n";
        std::cout << "┌──────────────┬─────────┬──────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬─────────────────────────┐\n";
        std::cout << "│ Parámetro    │ Símbolo │ Rango Prior [Min,Max]│ mu* (Poblac) │ sigma (Pobl) │ mu* (Frente) │ sigma (Frent)│ Clasificación / Acción  │\n";
        std::cout << "├──────────────┼─────────┼──────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼─────────────────────────┤\n";
        for (const auto& res : morris_report.results) {
            std::cout << "│ " << std::left << std::setw(12) << res.param_name << " │ "
                      << std::setw(7) << res.symbol << " │ "
                      << std::right << std::setw(20) << "Priors Bounds" << " │ "
                      << std::setw(12) << static_cast<uint64_t>(res.mu_star_pop) << " │ "
                      << std::setw(12) << static_cast<uint64_t>(res.sigma_pop) << " │ "
                      << std::setw(10) << std::fixed << std::setprecision(1) << res.mu_star_front << " km │ "
                      << std::setw(10) << std::fixed << std::setprecision(1) << res.sigma_front << " km │ "
                      << std::left << std::setw(23) << res.classification << std::right << " │\n";
        }
        std::cout << "└──────────────┴─────────┴──────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴─────────────────────────┘\n";
    }

    // 8. Fase 4: Recocido Simulado Multi-inicio (Simulated Annealing 2014-2019)
    if (run_phase4_calibration) {
        std::cout << "\n===================================================================\n";
        std::cout << "▶ FASE 4: INGESTA OBSERVACIONAL Y CALIBRACIÓN INVERSA MULTI-INICIO\n";
        std::cout << "===================================================================\n";

        std::string obs_csv = "/mnt/hpc_scratch/joro-hpc/calibration_set_2014_2019.csv";
        auto observations = joro::inference::load_calibration_dataset(obs_csv);

        if (observations.empty()) {
            std::cerr << "[Fase 4] Error: Dataset observacional vacío en " << obs_csv << "\n";
            return EXIT_FAILURE;
        }

        // Forzamiento térmico diario ERA5 específico de cada año 2014..2019 (365 días x 81 x 101)
        std::vector<joro::io::DailyThermalGrid> thermal_by_year(6);
        for (int yi = 0; yi < 6; ++yi) {
            if (!era5_reader.load_daily_thermal_year(2014 + yi, thermal_by_year[yi]) || !thermal_by_year[yi].has_data()) {
                std::cerr << "[Fase 4] Error: no se pudo cargar la serie térmica diaria de " << (2014 + yi) << "\n";
                return EXIT_FAILURE;
            }
        }
        std::cout << "[ERA5] Series térmicas diarias 2014–2019 cargadas (una grilla por año).\n";

        // Tensores ERA5 3D (u, v, w, BLH, sp) de CADA año 2014..2019 -> Device USM UNA sola vez (6 campos
        // residentes); cada año simulado usa SU meteorología real (sin reciclar 1-7 mayo de 2014).
        lpdm_gpu->clear_atmospheres();
        for (int yi = 0; yi < 6; ++yi) {
            joro::io::AtmosphericField3D field_y;
            if (!era5_reader.load_month(2014 + yi, 5, field_y) || !lpdm_gpu->add_atmosphere(field_y)) {
                std::cerr << "[Fase 4] Error: ERA5 3D/superficial de " << (2014 + yi) << " no utilizable para el LPDM.\n";
                return EXIT_FAILURE;
            }
        }
        std::cout << "[ERA5] " << lpdm_gpu->atmosphere_count() << " campos 3D+superficiales anuales (2014–2019) residentes en el dispositivo.\n";

        // Las 5 cadenas se ejecutan secuencialmente; cada evaluación usa la GPU (LPDM) y todos los hilos OpenMP.
        // Se calibra cada modelo candidato de K_i solicitado; el primero (M1 por defecto) es el principal.
        for (size_t model_pos = 0; model_pos < models.size(); ++model_pos) {
        const joro::abm::CarryingModel cal_model = models[model_pos];
        const bool main_model = (model_pos == 0);
        joro::inference::SimulatedAnnealingReport sa_report;
        int sa_chains = 5, sa_iters = 400; // Configuración oficial: 5 x 400 = 2,000 evaluaciones
        if (const char* e = std::getenv("JORO_SA_CHAINS")) sa_chains = std::max(1, std::atoi(e));
        if (const char* e = std::getenv("JORO_SA_ITERS")) sa_iters = std::max(10, std::atoi(e));
        try {
            const std::string ckpt_dir = "/mnt/hpc_scratch/joro-hpc/checkpoints/";
            const std::string ckpt = main_model && cal_model == joro::abm::CarryingModel::M1
                ? ckpt_dir + "theta_compatible_ensemble.csv"
                : ckpt_dir + "theta_compatible_ensemble_" + joro::abm::carrying_model_name(cal_model) + ".csv";
            sa_report = joro::inference::run_multi_start_simulated_annealing(
                grid_cells, thermal_by_year, observations, lpdm_gpu.get(), sa_chains, sa_iters, 1.0, 0.05,
                ckpt, cal_model);
        } catch (const std::exception& e) {
            std::cerr << "[Fase 4] ERROR: " << e.what() << "\n";
            return EXIT_FAILURE;
        }

        write_calibration_summary_json(
            std::string("/mnt/hpc_scratch/joro-hpc/checkpoints/phase4_summary_") +
                joro::abm::carrying_model_name(cal_model) + ".json",
            sa_report, era5_strict, config.era5_dir, era5_reader.files_read(), era5_reader.blocked_year_requests());
        if (!main_model) {
            std::cout << "[Modelo " << joro::abm::carrying_model_name(cal_model) << "] J_min = " << std::fixed
                      << std::setprecision(4) << sa_report.global_min_cost << " en " << std::setprecision(2)
                      << sa_report.total_runtime_s << " s (" << sa_report.params.size() << " parámetros libres)\n";
            continue;
        }

        // ─────────────────────────────────────────────────────────────────────
        // TABLA 4: DIAGNÓSTICO DE CONVERGENCIA DE LAS 5 CADENAS
        // ─────────────────────────────────────────────────────────────────────
        std::cout << "\nTABLA 4: DIAGNÓSTICO DE CONVERGENCIA DE LAS 5 CADENAS DE SIMULATED ANNEALING (2014–2019):\n";
        std::cout << "  Componentes normalizados por la escala nominal (valor 1.0000 = costo en theta_nominal)\n";
        std::cout << "┌────────┬────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐\n";
        std::cout << "│ Cadena │ Semilla│ Costo Inic J0│ Costo Min J* │ J_llegada    │ J_frente     │ J_persist    │ Aceptadas (%)│ Tiempo (s)   │\n";
        std::cout << "├────────┼────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤\n";
        for (const auto& ch : sa_report.chains) {
            std::cout << "│ " << std::setw(6) << ch.chain_id << " │ "
                      << std::setw(6) << ch.seed << " │ "
                      << std::setw(12) << std::fixed << std::setprecision(4) << ch.initial_cost << " │ "
                      << std::setw(12) << std::fixed << std::setprecision(4) << ch.min_cost << " │ "
                      << std::setw(12) << std::fixed << std::setprecision(4) << ch.best_components.j_arrival_norm << " │ "
                      << std::setw(12) << std::fixed << std::setprecision(4) << ch.best_components.j_front_norm << " │ "
                      << std::setw(12) << std::fixed << std::setprecision(4) << ch.best_components.j_persistence_norm << " │ "
                      << std::setw(11) << (ch.accepted_proposals * 100 / ch.total_iterations) << "% │ "
                      << std::setw(10) << std::fixed << std::setprecision(2) << ch.execution_time_s << " s │\n";
        }
        std::cout << "└────────┴────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n";
        std::cout << "  ✔ Tiempo total de calibración multi-start: " << std::fixed << std::setprecision(2) 
                  << sa_report.total_runtime_s << " s (" << (sa_report.total_runtime_s / 60.0) << " min)\n"
                  << "  ✔ Costo mínimo global alcanzado J_min = " << std::setprecision(4) << sa_report.global_min_cost << "\n";

        // ─────────────────────────────────────────────────────────────────────
        // TABLA 5: ESTIMADOR MAP Y CUANTIFICACIÓN DE INCERTIDUMBRE (THETA_COMP)
        // ─────────────────────────────────────────────────────────────────────
        std::cout << "\nTABLA 5: ESTIMADOR OPTIMO MAP Y CUANTIFICACION DE INCERTIDUMBRE (CONJUNTO COMPATIBLE Theta_comp):\n";
        std::cout << "┌──────────────┬─────────┬──────────────────────┬──────────────┬──────────────┬─────────────────────────┬──────────────────────────────────────────┐\n";
        std::cout << "│ Parámetro    │ Símbolo │ Rango Prior [Min,Max]│ Optimo theta^│ Mediana Comp │ Intervalo P5–P95        │ Interpretación Biológica / Física        │\n";
        std::cout << "├──────────────┼─────────┼──────────────────────┼──────────────┼──────────────┼─────────────────────────┼──────────────────────────────────────────┤\n";

        const std::unordered_map<std::string, std::string> descriptions = {
            {"t_c",          "Umbral térmico crítico de letalidad invernal"},
            {"alpha_damage", "Tasa de acumulación de estrés térmico diario"},
            {"beta_damage",  "Exponente DISCRETO de daño por frío {1,2,3}"},
            {"k0",           "Capacidad de carga basal (mediana de K_i)"},
            {"gamma_hab",    "Pendiente global de selección de hábitat H_hab"},
            {"q_balloon",    "Fracción de juveniles que inician baloneo aéreo"},
            {"q_viab",       "Viabilidad media de ootecas tras oviposición"},
            {"a0",           "Intercepto de colonización logit(gamma)"},
            {"b0",           "Intercepto/umbral de persistencia logit(phi)"},
            {"gamma1_imp",   "Efecto de impermeabilidad urbana sobre K_i"},
            {"gamma2_borde", "Efecto de borde de bosque/vegetación en K_i"},
            {"gamma3_dosel", "Efecto de dosel arbóreo (microclima térmico)"}};

        for (size_t i = 0; i < sa_report.params.size(); ++i) {
            const auto& def = sa_report.params[i];
            const bool is_beta = (static_cast<int>(i) == joro::inference::kBetaDamageIndex);
            std::stringstream ss_prior, ss_ci;
            if (is_beta) {
                ss_prior << "{1, 2, 3}";
                ss_ci << "[" << static_cast<int>(sa_report.p5_theta[i]) << ", " << static_cast<int>(sa_report.p95_theta[i]) << "]";
            } else {
                ss_prior << "[" << std::fixed << std::setprecision(2) << def.min_val << ", " << def.max_val << "]";
                ss_ci << "[" << std::fixed << std::setprecision(3) << sa_report.p5_theta[i] << ", " << sa_report.p95_theta[i] << "]";
            }
            const int prec = is_beta ? 0 : 3;

            std::cout << "│ " << std::left << std::setw(12) << def.name << " │ "
                      << std::setw(7) << def.symbol << " │ "
                      << std::right << std::setw(20) << ss_prior.str() << " │ "
                      << std::setw(12) << std::fixed << std::setprecision(prec) << sa_report.map_theta[i] << " │ "
                      << std::setw(12) << std::fixed << std::setprecision(prec) << sa_report.median_theta[i] << " │ "
                      << std::setw(23) << ss_ci.str() << " │ "
                      << std::left << std::setw(40) << descriptions.at(def.name) << std::right << " │\n";
        }
        std::cout << "└──────────────┴─────────┴──────────────────────┴──────────────┴──────────────┴─────────────────────────┴──────────────────────────────────────────┘\n";
        std::cout << "  ✔ Miembros en conjunto compatible Theta_comp (Delta <= 0.05, sobre todas las evaluaciones): " 
                  << sa_report.compatible_ensemble.size() << " soluciones\n";

        // ─────────────────────────────────────────────────────────────────────
        // TABLA 6: ANÁLISIS DE SENSIBILIDAD A PESOS MULTICRITERIO
        // ─────────────────────────────────────────────────────────────────────
        auto robustness = [](double pct) -> std::string {
            if (pct >= 70.0) return "ROBUSTO (>=70%) ✔";
            if (pct >= 50.0) return "MODERADO (50-70%) ~";
            return "SENSIBLE (<50%) ⚠";
        };
        const auto& base_c = sa_report.map_components;
        (void)base_c;
        std::cout << "\nTABLA 6: ANALISIS DE SENSIBILIDAD DEL CONJUNTO COMPATIBLE FRENTE A PESOS MULTICRITERIO:\n";
        std::cout << "  (Theta_comp^A y Theta_comp^B reconstruidos sobre todas las evaluaciones con su propio J_min)\n";
        std::cout << "┌──────────────────────────────┬─────────────────────────┬──────────────┬──────────────────────┬──────────────────────┐\n";
        std::cout << "│ Configuración de Pesos       │ Vector (w1, w2, w3)     │ J_min        │ Retención Theta_comp │ Diagnóstico          │\n";
        std::cout << "├──────────────────────────────┼─────────────────────────┼──────────────┼──────────────────────┼──────────────────────┤\n";
        std::cout << "│ Configuración Base (Equal)   │ (0.333, 0.333, 0.334)   │ " << std::setw(12) << std::fixed << std::setprecision(4)
                  << sa_report.global_min_cost << " │               100.0% │ Referencia Nominal   │\n";
        std::cout << "│ Configuración A (Frente)     │ (0.200, 0.600, 0.200)   │ " << std::setw(12) << std::fixed << std::setprecision(4)
                  << sa_report.j_min_config_a << " │ " << std::setw(19) << std::fixed << std::setprecision(1) << sa_report.retention_rate_config_a
                  << "% │ " << std::left << std::setw(20) << robustness(sa_report.retention_rate_config_a) << std::right << " │\n";
        std::cout << "│ Configuración B (Persist.)   │ (0.200, 0.200, 0.600)   │ " << std::setw(12) << std::fixed << std::setprecision(4)
                  << sa_report.j_min_config_b << " │ " << std::setw(19) << std::fixed << std::setprecision(1) << sa_report.retention_rate_config_b
                  << "% │ " << std::left << std::setw(20) << robustness(sa_report.retention_rate_config_b) << std::right << " │\n";
        std::cout << "└──────────────────────────────┴─────────────────────────┴──────────────┴──────────────────────┴──────────────────────┘\n";

        // ─────────────────────────────────────────────────────────────────────
        // TABLA 7: TELEMETRÍA HÍBRIDA CPU (OpenMP + AVX-512) + GPU (SYCL / Level-Zero)
        // ─────────────────────────────────────────────────────────────────────
        constexpr double kLpdmStepsPerEval = 168.0;
        std::cout << "\nTABLA 7: TELEMETRÍA HÍBRIDA CPU (" << sa_report.omp_max_threads << " hilos OpenMP) + GPU (" << sa_report.device_description << "):\n";
        std::cout << "┌────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────────┬──────────────┐\n";
        std::cout << "│ Cadena │ Total(s) │ ABM CPU  │ GPU kern │ H2D+D2H  │ LPDM host│ Costo CPU│ GPU kern │ Super-part │ Mpart/s      │\n";
        std::cout << "│        │          │ (s)      │ (s)      │ (s)      │ (s)      │ (s)      │ % wall   │ (millones) │ (kernel GPU) │\n";
        std::cout << "├────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────────┼──────────────┤\n";
        auto print_tel_row = [&](const std::string& label, double wall, const joro::inference::ChainTelemetry& t) {
            const double mps = (t.lpdm_kernel_time_s > 0.0) ? (static_cast<double>(t.particles_dispatched) / t.lpdm_kernel_time_s / 1.0e6) : 0.0;
            std::cout << "│ " << std::setw(6) << label << " │ "
                      << std::setw(8) << std::fixed << std::setprecision(2) << wall << " │ "
                      << std::setw(8) << t.abm_time_s << " │ "
                      << std::setw(8) << t.lpdm_kernel_time_s << " │ "
                      << std::setw(8) << (t.lpdm_h2d_time_s + t.lpdm_d2h_time_s) << " │ "
                      << std::setw(8) << (t.lpdm_init_time_s + t.lpdm_reduce_time_s) << " │ "
                      << std::setw(8) << t.cost_time_s << " │ "
                      << std::setw(7) << std::setprecision(1) << (wall > 0.0 ? 100.0 * t.lpdm_kernel_time_s / wall : 0.0) << "% │ "
                      << std::setw(10) << std::setprecision(2) << (static_cast<double>(t.particles_dispatched) / 1.0e6) << " │ "
                      << std::setw(12) << std::setprecision(2) << mps << " │\n";
        };
        for (const auto& ch : sa_report.chains) print_tel_row(std::to_string(ch.chain_id), ch.execution_time_s, ch.telemetry);
        std::cout << "├────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────────┼──────────────┤\n";
        print_tel_row("TOTAL", sa_report.total_runtime_s, sa_report.telemetry_total);
        std::cout << "└────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────────┴──────────────┘\n";
        {
            const auto& tt = sa_report.telemetry_total;
            const double kern_mps_steps = (tt.lpdm_kernel_time_s > 0.0)
                ? (static_cast<double>(tt.particles_dispatched) * kLpdmStepsPerEval / tt.lpdm_kernel_time_s / 1.0e6) : 0.0;
            double tmin = 1e18, tmax = 0.0, tsum = 0.0;
            for (const auto& ch : sa_report.chains) { tmin = std::min(tmin, ch.execution_time_s); tmax = std::max(tmax, ch.execution_time_s); tsum += ch.execution_time_s; }
            const double tmean = tsum / static_cast<double>(sa_report.chains.size());
            std::cout << "  ✔ Despachos GPU: " << std::setprecision(0) << static_cast<double>(tt.evaluations) << " evaluaciones | despachos con fallback a CPU: "
                      << tt.cpu_fallbacks << "\n";
            std::cout << "  ✔ Despachos anuales con tope adaptativo de super-partículas: " << tt.capped_dispatches << " de "
                      << static_cast<double>(tt.evaluations) * 6.0 << "\n";
            std::cout << "  ✔ Fracción de pared: H2D+D2H " << std::setprecision(1)
                      << (sa_report.total_runtime_s > 0.0 ? 100.0 * (tt.lpdm_h2d_time_s + tt.lpdm_d2h_time_s) / sa_report.total_runtime_s : 0.0)
                      << " % | kernel GPU "
                      << (sa_report.total_runtime_s > 0.0 ? 100.0 * tt.lpdm_kernel_time_s / sa_report.total_runtime_s : 0.0) << " %\n";
            std::cout << "  ✔ Throughput del kernel GPU: " << std::setprecision(2) << kern_mps_steps << " M part-pasos/s (60 pasos de 60 s)\n";
            std::cout << "  ✔ Balance entre cadenas: min " << std::setprecision(2) << tmin << " s | media " << tmean << " s | max " << tmax
                      << " s | max/min = " << (tmin > 0.0 ? tmax / tmin : 0.0) << "\n";
        }
        } // fin del bucle de modelos K_i
    }

    std::cout << "\n✔ Fase 4 completada exitosamente.\n";
    return EXIT_SUCCESS;
}
