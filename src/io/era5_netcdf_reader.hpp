#pragma once

#include "common/types.hpp"
#include <string>
#include <vector>
#include <memory>

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
    std::vector<float> u10;     ///< Viento a 10m zonal [m/s]
    std::vector<float> v10;     ///< Viento a 10m meridional [m/s]
    std::vector<float> blh;     ///< Altura de capa límite [m]
};

class Era5NetCdfReader {
public:
    explicit Era5NetCdfReader(std::string data_dir);
    ~Era5NetCdfReader();

    /// Carga un mes de datos horarios en un tensor 3D continuo
    bool load_month(int year, int month, AtmosphericField3D& field);

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
