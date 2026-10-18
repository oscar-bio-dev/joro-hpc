#pragma once

#include "common/types.hpp"
#include "abm/abm_engine.hpp"
#include "lpdm/lpdm_engine.hpp"
#include "io/era5_netcdf_reader.hpp"
#include "inference/cost_function.hpp"
#include "inference/occupancy_model.hpp"
#include <string>
#include <vector>

namespace joro::inference {

struct CalibrationParamDef {
    std::string name;
    std::string symbol;
    float min_val;
    float max_val;
    float default_val;
};

extern const std::vector<CalibrationParamDef> CALIBRATION_PARAMS;

struct ChainConvergenceRecord {
    int chain_id = 0;
    uint32_t seed = 0;
    int total_iterations = 400;
    int accepted_proposals = 0;
    double initial_cost = 0.0;
    double min_cost = 0.0;
    double execution_time_s = 0.0;
    std::vector<float> best_theta;
    CostComponents best_components;
};

struct CompatibleEnsembleMember {
    int chain_id = 0;
    int iteration = 0;
    double temperature = 0.0;
    double j_total = 0.0;
    double j_arrival_norm = 0.0;
    double j_front_norm = 0.0;
    double j_persistence_norm = 0.0;
    double j_config_a = 0.0; // w = (0.20, 0.60, 0.20)
    double j_config_b = 0.0; // w = (0.20, 0.20, 0.60)
    std::vector<float> theta;
};

struct SimulatedAnnealingReport {
    int n_chains = 5;
    int iter_per_chain = 400;
    int total_evaluations = 2000;
    double total_runtime_s = 0.0;
    double global_min_cost = 1e6;
    std::vector<float> map_theta;
    CostComponents map_components;
    std::vector<ChainConvergenceRecord> chains;
    std::vector<CompatibleEnsembleMember> compatible_ensemble;
    
    // Estadísticas de incertidumbre P5, Mediana, P95
    std::vector<float> p5_theta;
    std::vector<float> median_theta;
    std::vector<float> p95_theta;

    // Sensibilidad a pesos alternativos
    double retention_rate_config_a = 0.0;
    double retention_rate_config_b = 0.0;
};

/// Carga el conjunto de calibración observacional (2014-2019)
struct CalibrationObservation {
    int32_t cell_id;
    int16_t year;
    uint8_t y_detected;
    int32_t effort_araneae;
    float centroid_lon;
    float centroid_lat;
    float z_imp;
    float z_borde;
    float z_dosel;
    float pop_density_h;
    float road_density_r;
};

std::vector<CalibrationObservation> load_calibration_dataset(const std::string& csv_path);

/// Ejecuta el algoritmo de Recocido Simulado Multi-inicio (5 cadenas independientes)
SimulatedAnnealingReport run_multi_start_simulated_annealing(
    const std::vector<GridCell>& grid_cells,
    const io::DailyThermalGrid& thermal,
    const std::vector<CalibrationObservation>& observations,
    int n_chains = 5,
    int iter_per_chain = 400,
    double initial_temp = 1.0,
    double delta_threshold = 0.05,
    const std::string& checkpoint_csv_path = "/mnt/hpc_scratch/joro-hpc/checkpoints/theta_compatible_ensemble.csv");

} // namespace joro::inference
