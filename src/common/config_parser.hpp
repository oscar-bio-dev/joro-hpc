#pragma once

#include "common/types.hpp"
#include <cstdlib>
#include <string>
#include <vector>

namespace joro {

/// Lee una variable de entorno; devuelve `fallback` si no existe o está vacía.
inline std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

/// Lee un entero desde el entorno; devuelve `fallback` si no existe o no es válido.
inline int env_int_or(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    return (end && *end == 0) ? static_cast<int>(x) : fallback;
}

struct SimulationConfig {
    std::string project_name = "JORO-HPC";
    std::string version = "0.1.0";
    uint32_t seed = 42819;
    
    // Almacenamiento HPC
    std::string datalake_root = "/mnt/ecotech_datalake";
    std::string scratch_root = "/mnt/hpc_scratch/joro-hpc";
    
    BoundingBox bbox;
    double cell_size_m = 5000.0;
    
    int calib_start_year = 2014;
    int calib_end_year   = 2019;
    int eval_start_year  = 2020;
    int eval_end_year    = 2024;
    
    int ballooning_start_month = 5;
    int ballooning_end_month   = 7;
    
    std::string era5_dir = "/mnt/hpc_scratch/joro-hpc/era5_nc_active";
    double lpdm_dt_seconds = 60.0;
    
    // PostgreSQL / PostGIS: parámetros leídos del entorno (ver `.env.example`).
    // La contraseña NUNCA se almacena en el código: se lee de JORO_DB_PASSWORD (por defecto vacía).
    std::string db_host = env_or("JORO_DB_HOST", "localhost");
    int db_port = env_int_or("JORO_DB_PORT", 5433);
    std::string db_name = env_or("JORO_DB_NAME", "joro_db");
    std::string db_user = env_or("JORO_DB_USER", "joro_admin");
    std::string db_password = env_or("JORO_DB_PASSWORD", "");
    std::string db_tablespace = "hpc_fast";
    
    // Parámetros de calibración por defecto
    double w1_llegada = 0.333333;
    double w2_frente = 0.333333;
    double w3_persistencia = 0.333334;
    
    static SimulationConfig load_default() {
        return SimulationConfig{};
    }
};

} // namespace joro
