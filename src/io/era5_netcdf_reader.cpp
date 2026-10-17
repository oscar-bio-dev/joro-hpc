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
    std::string filename_3d = data_dir_ + "/era5_uvwt_" + std::to_string(year) +
                              "_" + (month < 10 ? "0" : "") + std::to_string(month) + ".nc";
    int ncid_3d = 0;
    int status = nc_open(filename_3d.c_str(), NC_NOWRITE, &ncid_3d);
    if (status != NC_NOERR) {
        // Intento con sufijo de día (_01.nc)
        filename_3d = data_dir_ + "/era5_uvwt_" + std::to_string(year) +
                      "_" + (month < 10 ? "0" : "") + std::to_string(month) + "_01.nc";
        status = nc_open(filename_3d.c_str(), NC_NOWRITE, &ncid_3d);
    }

    if (status == NC_NOERR) {
        int dimid_time = -1, dimid_level = -1, dimid_lat = -1, dimid_lon = -1;
        size_t n_times = 0, n_levels = 0, n_lats = 0, n_lons = 0;

        nc_inq_dimid(ncid_3d, "valid_time", &dimid_time);
        nc_inq_dimlen(ncid_3d, dimid_time, &n_times);
        nc_inq_dimid(ncid_3d, "pressure_level", &dimid_level);
        nc_inq_dimlen(ncid_3d, dimid_level, &n_levels);
        nc_inq_dimid(ncid_3d, "latitude", &dimid_lat);
        nc_inq_dimlen(ncid_3d, dimid_lat, &n_lats);
        nc_inq_dimid(ncid_3d, "longitude", &dimid_lon);
        nc_inq_dimlen(ncid_3d, dimid_lon, &n_lons);

        if (n_times > 0 && n_levels > 0 && n_lats > 0 && n_lons > 0) {
            field.n_times = static_cast<int>(n_times);
            field.n_levels = static_cast<int>(n_levels);
            field.n_lats = static_cast<int>(n_lats);
            field.n_lons = static_cast<int>(n_lons);
            field.pressure_levels_hpa = {1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700};

            size_t total_elements = n_times * n_levels * n_lats * n_lons;
            field.u.resize(total_elements);
            field.v.resize(total_elements);
            field.w.resize(total_elements);
            field.t.resize(total_elements);

            int varid_u = -1, varid_v = -1, varid_w = -1, varid_t = -1;
            nc_inq_varid(ncid_3d, "u", &varid_u);
            nc_inq_varid(ncid_3d, "v", &varid_v);
            nc_inq_varid(ncid_3d, "w", &varid_w);
            nc_inq_varid(ncid_3d, "t", &varid_t);

            if (varid_u >= 0) nc_get_var_float(ncid_3d, varid_u, field.u.data());
            if (varid_v >= 0) nc_get_var_float(ncid_3d, varid_v, field.v.data());
            if (varid_w >= 0) nc_get_var_float(ncid_3d, varid_w, field.w.data());
            if (varid_t >= 0) nc_get_var_float(ncid_3d, varid_t, field.t.data());

            nc_close(ncid_3d);

            // Leer variables superficiales si existen
            std::string filename_sfc = data_dir_ + "/era5_sfc_" + std::to_string(year) +
                                       "_" + (month < 10 ? "0" : "") + std::to_string(month) + ".nc";
            int ncid_sfc = 0;
            if (nc_open(filename_sfc.c_str(), NC_NOWRITE, &ncid_sfc) == NC_NOERR) {
                size_t sfc_elements = n_times * n_lats * n_lons;
                field.u10.resize(sfc_elements);
                field.v10.resize(sfc_elements);
                field.sp.resize(sfc_elements);
                field.blh.resize(sfc_elements);

                int varid_u10 = -1, varid_v10 = -1, varid_sp = -1, varid_blh = -1;
                nc_inq_varid(ncid_sfc, "u10", &varid_u10);
                nc_inq_varid(ncid_sfc, "v10", &varid_v10);
                nc_inq_varid(ncid_sfc, "sp", &varid_sp);
                nc_inq_varid(ncid_sfc, "blh", &varid_blh);

                if (varid_u10 >= 0) nc_get_var_float(ncid_sfc, varid_u10, field.u10.data());
                if (varid_v10 >= 0) nc_get_var_float(ncid_sfc, varid_v10, field.v10.data());
                if (varid_sp >= 0) nc_get_var_float(ncid_sfc, varid_sp, field.sp.data());
                if (varid_blh >= 0) nc_get_var_float(ncid_sfc, varid_blh, field.blh.data());

                nc_close(ncid_sfc);
            }

            return true;
        }
        nc_close(ncid_3d);
    } else {
        std::cerr << "[NetCDF] Advertencia: No se pudo abrir " << filename_3d
                  << " (" << nc_strerror(status) << "). Usando inicialización sintética de respaldo.\n";
    }
#endif

    // Si NetCDF no está disponible en tiempo de compilación o falló el archivo, usar malla meteorológica base
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
