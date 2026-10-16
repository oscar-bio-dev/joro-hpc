#include "io/era5_netcdf_reader.hpp"
#include <iostream>
#include <cmath>

#if defined(JORO_HAS_NETCDF)
#include <netcdf.h>
#endif

namespace joro::io {

Era5NetCdfReader::Era5NetCdfReader(std::string data_dir)
    : data_dir_(std::move(data_dir)) {}

Era5NetCdfReader::~Era5NetCdfReader() = default;

float Era5NetCdfReader::omega_to_w(float omega, float pressure_pa, float temp_k) {
    if (pressure_pa <= 100.0f || temp_k <= 0.0f) {
        return 0.0f;
    }
    // w = - (omega * R_d * T_v) / (p * g)
    // Asumiendo temperatura seca T_v ~ T para primera aproximación
    const double num = -static_cast<double>(omega) * constants::R_d * static_cast<double>(temp_k);
    const double den = static_cast<double>(pressure_pa) * constants::g;
    return static_cast<float>(num / den);
}

bool Era5NetCdfReader::load_month(int year, int month, AtmosphericField3D& field) {
#if defined(JORO_HAS_NETCDF)
    std::string filename = data_dir_ + "/era5_uvwt_" + std::to_string(year) +
                           "_" + (month < 10 ? "0" : "") + std::to_string(month) + ".nc";
    int ncid = 0;
    int status = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    if (status != NC_NOERR) {
        std::cerr << "[NetCDF] Advertencia: No se pudo abrir " << filename
                  << " (" << nc_strerror(status) << "). Usando inicialización sintética.\n";
        return false;
    }
    // TODO: Leer dimensiones y variables con nc_inq_varid y nc_get_vara_float
    nc_close(ncid);
    return true;
#else
    // Si NetCDF no está disponible en tiempo de compilación, generar malla meteorológica base
    field.n_times = 744; // 31 días * 24 horas
    field.n_levels = 12;
    field.n_lats = 81;
    field.n_lons = 101;
    field.pressure_levels_hpa = {1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700};
    
    size_t total_elements = static_cast<size_t>(field.n_times) * field.n_levels * field.n_lats * field.n_lons;
    field.u.assign(total_elements, 5.0f);     // 5 m/s hacia el este
    field.v.assign(total_elements, 2.0f);     // 2 m/s hacia el norte
    field.w.assign(total_elements, 0.05f);    // 0.05 m/s ascenso débil
    field.t.assign(total_elements, 295.15f);  // 22 °C
    
    size_t sfc_elements = static_cast<size_t>(field.n_times) * field.n_lats * field.n_lons;
    field.u10.assign(sfc_elements, 3.5f);
    field.v10.assign(sfc_elements, 1.5f);
    field.sp.assign(sfc_elements, 101325.0f);
    field.blh.assign(sfc_elements, 1200.0f);
    return true;
#endif
}

void Era5NetCdfReader::interpolate_wind_3d(
    const AtmosphericField3D& field,
    float /*lon*/, float /*lat*/, float /*altitude_m*/, float /*time_h*/,
    float& u_out, float& v_out, float& w_out) const 
{
    // Interpolación trilineal rápida (stub representativo)
    u_out = 4.5f;
    v_out = 2.0f;
    w_out = 0.02f;
}

} // namespace joro::io
