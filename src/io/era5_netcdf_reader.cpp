#include "io/era5_netcdf_reader.hpp"
#include <cstdlib>
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

bool Era5NetCdfReader::year_allowed(int year) {
    if (year <= kLastCalibrationYear) return true;
    const char* unlock = std::getenv("JORO_UNLOCK_PHASE5");
    if (unlock != nullptr && std::string(unlock) == "1") return true;
    blocked_years_.push_back(year);
    std::cerr << "[ERA5] BLOQUEADO: el año " << year << " pertenece a la ventana 2020-2024 reservada para la Fase 5 "
              << "(no se carga en memoria).\n";
    return false;
}

bool Era5NetCdfReader::load_month(int year, int month, AtmosphericField3D& field) {
    if (!year_allowed(year)) return false;
#if defined(JORO_HAS_NETCDF)
    std::string filename_3d = data_dir_ + "/era5_uvwt_" + std::to_string(year) +
                              "_" + (month < 10 ? "0" : "") + std::to_string(month) + ".nc";
    int ncid_3d = 0;
    int status = nc_open(filename_3d.c_str(), NC_NOWRITE, &ncid_3d);
    if (status != NC_NOERR && !strict_) {
        // Intento con sufijo de día (_01.nc)
        filename_3d = data_dir_ + "/era5_uvwt_" + std::to_string(year) +
                      "_" + (month < 10 ? "0" : "") + std::to_string(month) + "_01.nc";
        status = nc_open(filename_3d.c_str(), NC_NOWRITE, &ncid_3d);
    }
    if (status == NC_NOERR) files_read_.push_back(filename_3d);
    if (status != NC_NOERR && strict_) {
        std::cerr << "[NetCDF] ERROR (modo estricto): no se pudo abrir " << filename_3d << " ("
                  << nc_strerror(status) << ").\n";
        return false;
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

            // Coordenadas reales de la malla (el muestreador 3D las usa para indexar)
            field.lats.assign(n_lats, 0.0f);
            field.lons.assign(n_lons, 0.0f);
            int varid_lat = -1, varid_lon = -1;
            if (nc_inq_varid(ncid_3d, "latitude", &varid_lat) == NC_NOERR) nc_get_var_float(ncid_3d, varid_lat, field.lats.data());
            if (nc_inq_varid(ncid_3d, "longitude", &varid_lon) == NC_NOERR) nc_get_var_float(ncid_3d, varid_lon, field.lons.data());

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

            if (strict_) {
                // Validación: variables presentes, almacenadas en float32 (sin scale_factor) y niveles en el orden esperado
                bool ok = varid_u >= 0 && varid_v >= 0 && varid_w >= 0 && varid_t >= 0;
                for (int vid : {varid_u, varid_v, varid_w, varid_t}) {
                    nc_type xt = NC_NAT;
                    if (vid >= 0 && (nc_inq_vartype(ncid_3d, vid, &xt) != NC_NOERR || xt != NC_FLOAT)) ok = false;
                }
                int varid_pl = -1;
                if (nc_inq_varid(ncid_3d, "pressure_level", &varid_pl) == NC_NOERR && n_levels == field.pressure_levels_hpa.size()) {
                    std::vector<double> pl(n_levels, 0.0);
                    nc_get_var_double(ncid_3d, varid_pl, pl.data());
                    for (size_t k = 0; k < n_levels; ++k) {
                        if (std::abs(pl[k] - static_cast<double>(field.pressure_levels_hpa[k])) > 1e-3) ok = false;
                    }
                } else {
                    ok = false;
                }
                if (!ok) {
                    std::cerr << "[NetCDF] ERROR (modo estricto): " << filename_3d
                              << " no contiene u,v,w,t en float32 sobre los 12 niveles 1000-700 hPa esperados.\n";
                    nc_close(ncid_3d);
                    return false;
                }
            }

            if (varid_u >= 0) nc_get_var_float(ncid_3d, varid_u, field.u.data());
            if (varid_v >= 0) nc_get_var_float(ncid_3d, varid_v, field.v.data());
            if (varid_w >= 0) nc_get_var_float(ncid_3d, varid_w, field.w.data());
            if (varid_t >= 0) nc_get_var_float(ncid_3d, varid_t, field.t.data());

            // Conversión hidrostática explícita: omega (Pa/s) -> velocidad geométrica vertical w (m/s)
            // w = - (omega * R_d * T) / (p * g)
            for (int t_idx = 0; t_idx < field.n_times; ++t_idx) {
                for (int l_idx = 0; l_idx < field.n_levels; ++l_idx) {
                    float p_pa = field.pressure_levels_hpa[l_idx] * 100.0f;
                    for (int lat_idx = 0; lat_idx < field.n_lats; ++lat_idx) {
                        for (int lon_idx = 0; lon_idx < field.n_lons; ++lon_idx) {
                            size_t idx = ((static_cast<size_t>(t_idx) * field.n_levels + l_idx) * field.n_lats + lat_idx) * field.n_lons + lon_idx;
                            field.w[idx] = omega_to_w(field.w[idx], p_pa, field.t[idx]);
                        }
                    }
                }
            }

            nc_close(ncid_3d);

            // Leer variables superficiales si existen
            std::string filename_sfc = data_dir_ + "/era5_sfc_" + std::to_string(year) +
                                       "_" + (month < 10 ? "0" : "") + std::to_string(month) + ".nc";
            int ncid_sfc = 0;
            const bool sfc_open = nc_open(filename_sfc.c_str(), NC_NOWRITE, &ncid_sfc) == NC_NOERR;
            if (!sfc_open && strict_) {
                std::cerr << "[NetCDF] ERROR (modo estricto): falta el archivo superficial " << filename_sfc << ".\n";
                return false;
            }
            if (sfc_open) {
                files_read_.push_back(filename_sfc);
                size_t sfc_elements = n_times * n_lats * n_lons;
                field.u10.resize(sfc_elements);
                field.v10.resize(sfc_elements);
                field.sp.resize(sfc_elements);
                field.t2m.resize(sfc_elements);
                field.blh.resize(sfc_elements);

                int varid_u10 = -1, varid_v10 = -1, varid_sp = -1, varid_t2m = -1, varid_blh = -1;
                nc_inq_varid(ncid_sfc, "u10", &varid_u10);
                nc_inq_varid(ncid_sfc, "v10", &varid_v10);
                nc_inq_varid(ncid_sfc, "sp", &varid_sp);
                nc_inq_varid(ncid_sfc, "t2m", &varid_t2m);
                nc_inq_varid(ncid_sfc, "blh", &varid_blh);
                if (strict_) {
                    bool ok = varid_u10 >= 0 && varid_v10 >= 0 && varid_sp >= 0 && varid_t2m >= 0 && varid_blh >= 0;
                    for (int vid : {varid_u10, varid_v10, varid_sp, varid_t2m, varid_blh}) {
                        nc_type xt = NC_NAT;
                        if (vid >= 0 && (nc_inq_vartype(ncid_sfc, vid, &xt) != NC_NOERR || xt != NC_FLOAT)) ok = false;
                    }
                    if (!ok) {
                        std::cerr << "[NetCDF] ERROR (modo estricto): " << filename_sfc
                                  << " no contiene sp,t2m,u10,v10,blh en float32.\n";
                        nc_close(ncid_sfc);
                        return false;
                    }
                }

                if (varid_u10 >= 0) nc_get_var_float(ncid_sfc, varid_u10, field.u10.data());
                if (varid_v10 >= 0) nc_get_var_float(ncid_sfc, varid_v10, field.v10.data());
                if (varid_sp >= 0) nc_get_var_float(ncid_sfc, varid_sp, field.sp.data());
                if (varid_t2m >= 0) {
                    nc_get_var_float(ncid_sfc, varid_t2m, field.t2m.data());
                    // Conversión Kelvin -> Celsius [°C]
                    for (size_t i = 0; i < sfc_elements; ++i) {
                        field.t2m[i] -= 273.15f;
                    }
                }
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

    if (strict_) {
        std::cerr << "[NetCDF] ERROR (modo estricto): carga ERA5 " << year << "-" << month
                  << " fallida; no se admite la malla sintética de respaldo.\n";
        return false;
    }

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
    field.t2m.assign(sfc_elements, 22.0f);   // 22 °C promedio estival
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

bool Era5NetCdfReader::load_daily_thermal_year(int year, DailyThermalGrid& grid) {
    if (!year_allowed(year)) return false;
    grid.n_days = 365;
    grid.n_lats = 81;
    grid.n_lons = 101;
    size_t total_elements = static_cast<size_t>(grid.n_days) * grid.n_lats * grid.n_lons;
    grid.t2m_min.resize(total_elements);
    grid.t2m_mean.resize(total_elements);

#if defined(JORO_HAS_NETCDF)
    // 1) Serie REAL por año (ERA5 daily statistics t2m): era5_daily_t2m_{year}.nc [día][lat][lon] en °C
    {
        const std::string path = data_dir_ + "/era5_daily_t2m_" + std::to_string(year) + ".nc";
        int ncid = 0;
        if (nc_open(path.c_str(), NC_NOWRITE, &ncid) == NC_NOERR) {
            files_read_.push_back(path);
            int varid_tmin = -1, varid_tmean = -1;
            nc_inq_varid(ncid, "t2m_min", &varid_tmin);
            nc_inq_varid(ncid, "t2m_mean", &varid_tmean);
            bool ok = varid_tmin >= 0 && varid_tmean >= 0;
            if (ok) {
                ok = nc_get_var_float(ncid, varid_tmin, grid.t2m_min.data()) == NC_NOERR &&
                     nc_get_var_float(ncid, varid_tmean, grid.t2m_mean.data()) == NC_NOERR;
            }
            nc_close(ncid);
            if (ok) return true;
            std::cerr << "[NetCDF] ERROR: " << path << " ilegible o incompleto.\n";
        }
    }
    if (strict_) {
        std::cerr << "[NetCDF] ERROR (modo estricto): falta la serie térmica diaria real de " << year << ".\n";
        return false;
    }

    // 2) Archivo agregado heredado (Fases 1-3). OJO: la versión antigua era SINTÉTICA; si se conserva, debe
    //    renombrarse. Solo se usa si no existe la serie real por año y el modo no es estricto.
    std::string thermal_nc = data_dir_ + "/era5_daily_t2m_2014_2019.nc";
    int ncid = 0;
    int status = nc_open(thermal_nc.c_str(), NC_NOWRITE, &ncid);
    if (status == NC_NOERR) {
        files_read_.push_back(thermal_nc);
        int varid_tmin = -1, varid_tmean = -1;
        nc_inq_varid(ncid, "t2m_min", &varid_tmin);
        nc_inq_varid(ncid, "t2m_mean", &varid_tmean);

        int y_idx = year - 2014;
        if (y_idx < 0) y_idx = 0;
        if (y_idx > 5) y_idx = 5;

        size_t start[4] = {static_cast<size_t>(y_idx), 0, 0, 0};
        size_t count[4] = {1, static_cast<size_t>(grid.n_days), static_cast<size_t>(grid.n_lats), static_cast<size_t>(grid.n_lons)};

        bool ok_min = false, ok_mean = false;
        if (varid_tmin >= 0) {
            status = nc_get_vara_float(ncid, varid_tmin, start, count, grid.t2m_min.data());
            ok_min = (status == NC_NOERR);
        }
        if (varid_tmean >= 0) {
            status = nc_get_vara_float(ncid, varid_tmean, start, count, grid.t2m_mean.data());
            ok_mean = (status == NC_NOERR);
        }
        nc_close(ncid);

        if (ok_min && ok_mean) {
            return true;
        }
    }
#endif

    if (strict_) return false;

    // Climatología sintética si no hay NetCDF disponible
    for (int d = 0; d < grid.n_days; ++d) {
        float doy = static_cast<float>(d + 1);
        float seasonal = std::sin(2.0f * 3.14159265f * (doy - 110.0f) / 365.0f);
        for (int la = 0; la < grid.n_lats; ++la) {
            float lat = grid.lat_max - la * grid.dlat;
            float lat_diff = lat - 34.0f;
            float tmean = 16.5f - lat_diff * 0.85f + (10.5f + lat_diff * 0.35f) * seasonal;
            float tmin = tmean - 8.5f;
            for (int lo = 0; lo < grid.n_lons; ++lo) {
                size_t idx = (static_cast<size_t>(d) * grid.n_lats + la) * grid.n_lons + lo;
                grid.t2m_mean[idx] = tmean;
                grid.t2m_min[idx] = tmin;
            }
        }
    }
    return true;
}

} // namespace joro::io
