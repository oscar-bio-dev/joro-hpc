#pragma once

#include "common/types.hpp"
#include <string>
#include <vector>

namespace joro {

struct SimulationConfig {
    std::string project_name = "JORO-HPC";
    std::string version = "0.1.0";
    uint32_t seed = 42819;
    
    BoundingBox bbox;
    double cell_size_m = 5000.0;
    
    int calib_start_year = 2014;
    int calib_end_year   = 2019;
    int eval_start_year  = 2020;
    int eval_end_year    = 2024;
    
    int ballooning_start_month = 5;
    int ballooning_end_month   = 7;
    
    std::string era5_dir = "data/era5_nc";
    double lpdm_dt_seconds = 60.0;
    
    std::string db_host = "localhost";
    int db_port = 5432;
    std::string db_name = "joro_db";
    std::string db_user = "joro_admin";
    
    // Parámetros de calibración por defecto
    double w1_llegada = 0.333333;
    double w2_frente = 0.333333;
    double w3_persistencia = 0.333334;
    
    static SimulationConfig load_default() {
        return SimulationConfig{};
    }
};

} // namespace joro
