#pragma once

#include "common/types.hpp"
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

namespace joro::io {

/// Tensor 4D de variables meteorológicas en memoria: [tiempo][nivel][latitud][longitud]
struct AtmosphericField3D {
    int n_times = 0;
    int n_levels = 0;
    int n_lats = 0;
    int n_lons = 0;

    std::vector<float> pressure_levels_hpa;
    std::vector<float> lats;
    std::vector<float> lons;

    // Buffers planos contiguos en memoria host o USM
    std::vector<float> u;       ///< Componente zonal [m/s]
    std::vector<float> v;       ///< Componente meridional [m/s]
    std::vector<float> w;       ///< Velocidad vertical convertida [m/s]
    std::vector<float> t;       ///< Temperatura del aire [K]
    
    // Variables superficiales 2D [tiempo][latitud][longitud]
    std::vector<float> sp;      ///< Presión superficial [Pa]
    std::vector<float> t2m;     ///< Temperatura del aire a 2m [°C]
    std::vector<float> u10;     ///< Viento a 10m zonal [m/s]
    std::vector<float> v10;     ///< Viento a 10m meridional [m/s]
    std::vector<float> blh;     ///< Altura de capa límite [m]
};

/// Malla superficial diaria (365 días x 81 lats x 101 lons) para forzamiento estacional del ABM
struct DailyThermalGrid {
    int n_days = 365;
    int n_lats = 81;
    int n_lons = 101;
    float lat_max = 45.0f;
    float lat_min = 25.0f;
    float lon_min = -95.0f;
    float lon_max = -70.0f;
    float dlat = 0.25f;
    float dlon = 0.25f;

    std::vector<float> t2m_min;  ///< [365 * 81 * 101] en °C
    std::vector<float> t2m_mean; ///< [365 * 81 * 101] en °C

    inline void get_temp(int day_0idx, float lat, float lon, float& out_tmin, float& out_tmean) const noexcept {
        if (t2m_min.empty() || t2m_mean.empty()) {
            // Climatología de respaldo analítica si no hay NetCDF cargado
            float lat_diff = lat - 34.0f;
            float doy = static_cast<float>(day_0idx + 1);
            float seasonal = std::sin(2.0f * 3.14159265f * (doy - 110.0f) / 365.0f);
            out_tmean = 16.5f - lat_diff * 0.85f + (10.5f + lat_diff * 0.35f) * seasonal;
            out_tmin = out_tmean - 8.5f;
            return;
        }
        int lat_idx = static_cast<int>((lat_max - lat) / dlat + 0.5f);
        int lon_idx = static_cast<int>((lon - lon_min) / dlon + 0.5f);
        if (lat_idx < 0) lat_idx = 0;
        if (lat_idx >= n_lats) lat_idx = n_lats - 1;
        if (lon_idx < 0) lon_idx = 0;
        if (lon_idx >= n_lons) lon_idx = n_lons - 1;
        size_t idx = (static_cast<size_t>(day_0idx) * n_lats + lat_idx) * n_lons + lon_idx;
        out_tmin = t2m_min[idx];
        out_tmean = t2m_mean[idx];
    }
};

class Era5NetCdfReader {
public:
    explicit Era5NetCdfReader(std::string data_dir);
    ~Era5NetCdfReader();

    /// Carga un mes de datos horarios en un tensor 3D continuo
    bool load_month(int year, int month, AtmosphericField3D& field);

    /// Carga la serie térmica diaria completa de 365 días para un año dado
    bool load_daily_thermal_year(int year, DailyThermalGrid& grid);

    /// Convierte velocidad vertical de presión omega [Pa/s] a velocidad física w [m/s]
    static float omega_to_w(float omega, float pressure_pa, float temp_k);

    /// Interpola trilinealmente el viento (u, v, w) en una posición (x, y, z) dada
    void interpolate_wind_3d(const AtmosphericField3D& field,
                             float lon, float lat, float altitude_m, float time_h,
                             float& u_out, float& v_out, float& w_out) const;

private:
    std::string data_dir_;
};

} // namespace joro::io
