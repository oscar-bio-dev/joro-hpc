#pragma once

#include "common/types.hpp"
#include "abm/abm_engine.hpp"
#include "lpdm/lpdm_engine.hpp"
#include "io/era5_netcdf_reader.hpp"
#include <string>
#include <vector>

namespace joro::inference {

struct MorrisParamDef {
    std::string name;
    std::string symbol;
    float min_val;
    float max_val;
    float default_val;
};

struct MorrisResult {
    std::string param_name;
    std::string symbol;
    double mu_star_pop = 0.0;     ///< Media de efectos elementales absolutos (Población total)
    double sigma_pop = 0.0;       ///< Desviación estándar (Efectos no lineales / interacciones - Población)
    double mu_star_front = 0.0;   ///< Media de efectos elementales absolutos (Radio de frente en km)
    double sigma_front = 0.0;     ///< Desviación estándar (Radio de frente)
    std::string classification;   ///< "Dominante", "Interacción Moderada", "Despreciable"
};

struct MorrisScreeningReport {
    int n_trajectories = 10;
    int n_parameters = 11;
    int total_evaluations = 120;
    double execution_time_s = 0.0;
    std::vector<MorrisResult> results;
};

/// Ejecuta el análisis de efectos elementales de Morris (1991) con r trayectorias
MorrisScreeningReport run_morris_screening(
    const std::vector<GridCell>& grid_cells,
    const io::DailyThermalGrid& thermal,
    lpdm::LpdmEngine* lpdm_engine,
    int n_trajectories = 10,
    int p_levels = 4);

} // namespace joro::inference
