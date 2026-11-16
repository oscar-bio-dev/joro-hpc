#include "io/era5_netcdf_reader.hpp"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <filesystem>

#if defined(JORO_HAS_NETCDF)
#include <netcdf.h>
#endif

namespace joro::io {

#if defined(JORO_HAS_NETCDF)
namespace {

[[nodiscard]] bool is_supported_nc_numeric(nc_type xt) noexcept {
    return xt == NC_FLOAT || xt == NC_DOUBLE || xt == NC_SHORT ||
           xt == NC_INT || xt == NC_USHORT || xt == NC_UINT ||
           xt == NC_BYTE || xt == NC_UBYTE;
}

void query_cf_scale_offset(int ncid, int varid, bool& has_cf, double& scale, double& offset) noexcept {
    scale = 1.0;
    offset = 0.0;
    const bool has_s = (nc_get_att_double(ncid, varid, "scale_factor", &scale) == NC_NOERR);
    const bool has_o = (nc_get_att_double(ncid, varid, "add_offset", &offset) == NC_NOERR);
    has_cf = (has_s && std::abs(scale - 1.0) > 1e-12) || (has_o && std::abs(offset) > 1e-12);
}

bool read_nc_var_cf(int ncid, int varid, size_t n_elems, float* dst,
                    const size_t* start = nullptr, const size_t* count = nullptr) noexcept
{
    if (varid < 0 || !dst) return false;
    nc_type xt = NC_NAT;
    if (nc_inq_vartype(ncid, varid, &xt) != NC_NOERR || !is_supported_nc_numeric(xt)) return false;

    const int rc = (start && count)
                       ? nc_get_vara_float(ncid, varid, start, count, dst)
                       : nc_get_var_float(ncid, varid, dst);
    if (rc != NC_NOERR) return false;

    bool has_cf = false;
    double scale = 1.0, offset = 0.0;
    query_cf_scale_offset(ncid, varid, has_cf, scale, offset);
    if (has_cf) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_elems; ++i) {
            dst[i] = static_cast<float>(static_cast<double>(dst[i]) * scale + offset);
        }
    }
    return true;
}

} // namespace
#endif

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
    // JORO_UNLOCK_EVALUATION es el nombre de producción; JORO_UNLOCK_PHASE5 se mantiene como alias heredado.
    const char* unlock = std::getenv("JORO_UNLOCK_EVALUATION");
    if (unlock == nullptr) unlock = std::getenv("JORO_UNLOCK_PHASE5");
    if (unlock != nullptr && std::string(unlock) == "1") return true;
    blocked_years_.push_back(year);
    std::cerr << "[ERA5] BLOQUEADO: el año " << year << " pertenece a la ventana 2020-2024 reservada para la validación fuera de muestra "
              << "(no se carga en memoria).\n";
    return false;
}

bool Era5NetCdfReader::read_months(int year, const std::vector<int>& months, bool require_sfc, bool resilient,
                                   AtmosphericField3D& field, int* months_loaded)
{
    if (months_loaded) *months_loaded = 0;
#if defined(JORO_HAS_NETCDF)
    struct Probe {
        int month = 0;
        std::string f3d, fsfc;
        size_t nt = 0, nlev = 0, nlat = 0, nlon = 0;
        bool has_sfc = false;
    };
    auto mm = [](int m) { return std::string(m < 10 ? "0" : "") + std::to_string(m); };
    auto probe_dims = [](int ncid, size_t& nt, size_t& nlev, size_t& nlat, size_t& nlon) {
        int d = -1;
        bool ok = nc_inq_dimid(ncid, "valid_time", &d) == NC_NOERR && nc_inq_dimlen(ncid, d, &nt) == NC_NOERR;
        if (ok) ok = nc_inq_dimid(ncid, "pressure_level", &d) == NC_NOERR && nc_inq_dimlen(ncid, d, &nlev) == NC_NOERR;
        if (ok) ok = nc_inq_dimid(ncid, "latitude", &d) == NC_NOERR && nc_inq_dimlen(ncid, d, &nlat) == NC_NOERR;
        if (ok) ok = nc_inq_dimid(ncid, "longitude", &d) == NC_NOERR && nc_inq_dimlen(ncid, d, &nlon) == NC_NOERR;
        return ok && nt > 0 && nlev > 0 && nlat > 0 && nlon > 0;
    };

    // ── Fase 1: sondeo de los meses disponibles (solo archivos cerrados .nc; los .part nunca coinciden con el patrón)
    std::vector<Probe> probes;
    for (size_t mi = 0; mi < months.size(); ++mi) {
        Probe pr;
        pr.month = months[mi];
        pr.f3d = data_dir_ + "/era5_uvwt_" + std::to_string(year) + "_" + mm(pr.month) + ".nc";
        pr.fsfc = data_dir_ + "/era5_sfc_" + std::to_string(year) + "_" + mm(pr.month) + ".nc";

        int nc3 = 0;
        int st = nc_open(pr.f3d.c_str(), NC_NOWRITE, &nc3);
        if (st != NC_NOERR && !strict_ && mi == 0) {
            const std::string alt = data_dir_ + "/era5_uvwt_" + std::to_string(year) + "_" + mm(pr.month) + "_01.nc";
            if (nc_open(alt.c_str(), NC_NOWRITE, &nc3) == NC_NOERR) { pr.f3d = alt; st = NC_NOERR; }
        }
        bool usable = (st == NC_NOERR);
        if (usable) {
            usable = probe_dims(nc3, pr.nt, pr.nlev, pr.nlat, pr.nlon);
            nc_close(nc3);
        } else if (mi == 0 || !resilient) {
            if (strict_) {
                std::cerr << "[NetCDF] ERROR (modo estricto): no se pudo abrir " << pr.f3d << " (" << nc_strerror(st) << ").\n";
            } else {
                std::cerr << "[NetCDF] Advertencia: No se pudo abrir " << pr.f3d << " (" << nc_strerror(st)
                          << "). Usando inicialización sintética de respaldo.\n";
            }
        }
        if (usable) {
            int ncs = 0;
            if (nc_open(pr.fsfc.c_str(), NC_NOWRITE, &ncs) == NC_NOERR) {
                int dvt = -1, dla = -1, dlo = -1;
                size_t nts = 0, nla = 0, nlo = 0;
                pr.has_sfc = nc_inq_dimid(ncs, "valid_time", &dvt) == NC_NOERR && nc_inq_dimlen(ncs, dvt, &nts) == NC_NOERR &&
                             nc_inq_dimid(ncs, "latitude", &dla) == NC_NOERR && nc_inq_dimlen(ncs, dla, &nla) == NC_NOERR &&
                             nc_inq_dimid(ncs, "longitude", &dlo) == NC_NOERR && nc_inq_dimlen(ncs, dlo, &nlo) == NC_NOERR &&
                             nts == pr.nt && nla == pr.nlat && nlo == pr.nlon;
                nc_close(ncs);
            }
            if (!pr.has_sfc && (require_sfc || strict_)) {
                std::cerr << "[NetCDF] " << (strict_ ? "ERROR (modo estricto): " : "Aviso: ")
                          << "falta o es incompatible el archivo superficial " << pr.fsfc << ".\n";
                usable = false;
            }
        }
        if (usable && !probes.empty()) {
            const Probe& f = probes.front();
            if (pr.nlev != f.nlev || pr.nlat != f.nlat || pr.nlon != f.nlon) {
                std::cerr << "[NetCDF] Aviso: la malla de " << pr.f3d << " difiere de la del primer mes.\n";
                usable = false;
            }
        }
        if (!usable) {
            if (mi == 0 || !resilient) return false;
            std::cout << "[ERA5] AVISO: mes " << pr.month << " de " << year
                      << " aún no disponible en disco (descarga en curso); se omiten los meses posteriores.\n";
            break;
        }
        probes.push_back(pr);
    }
    if (probes.empty()) return false;

    // ── Fase 2: reserva y lectura cronológica por bloques temporales (conversión omega -> w al vuelo)
    const size_t nlev = probes.front().nlev, nlat = probes.front().nlat, nlon = probes.front().nlon;
    const size_t vol = nlev * nlat * nlon, hp = nlat * nlon;
    size_t nt_total = 0;
    for (const auto& pr : probes) nt_total += pr.nt;
    bool want_sfc = true;
    for (const auto& pr : probes) want_sfc = want_sfc && pr.has_sfc;

    field = AtmosphericField3D{};
    field.n_levels = static_cast<int>(nlev);
    field.n_lats = static_cast<int>(nlat);
    field.n_lons = static_cast<int>(nlon);
    field.u.resize(nt_total * vol);
    field.v.resize(nt_total * vol);
    field.w.resize(nt_total * vol);
    // field.t permanece vacío: la temperatura solo se necesita por bloque para omega -> w
    if (want_sfc) {
        field.u10.resize(nt_total * hp);
        field.v10.resize(nt_total * hp);
        field.sp.resize(nt_total * hp);
        field.t2m.resize(nt_total * hp);
        field.blh.resize(nt_total * hp);
    }
    std::vector<float> t_block(vol);

    auto fail_truncate = [&](size_t keep_times) {
        field.u.resize(keep_times * vol);
        field.v.resize(keep_times * vol);
        field.w.resize(keep_times * vol);
        if (want_sfc) {
            field.u10.resize(keep_times * hp); field.v10.resize(keep_times * hp); field.sp.resize(keep_times * hp);
            field.t2m.resize(keep_times * hp); field.blh.resize(keep_times * hp);
        }
    };

    size_t t_off = 0;
    int loaded = 0;
    for (size_t pi = 0; pi < probes.size(); ++pi) {
        const Probe& pr = probes[pi];
        bool ok = true;
        int nc3 = 0;
        if (nc_open(pr.f3d.c_str(), NC_NOWRITE, &nc3) != NC_NOERR) ok = false;
        if (ok) {
            int vid_u = -1, vid_v = -1, vid_w = -1, vid_t = -1;
            nc_inq_varid(nc3, "u", &vid_u);
            nc_inq_varid(nc3, "v", &vid_v);
            nc_inq_varid(nc3, "w", &vid_w);
            nc_inq_varid(nc3, "t", &vid_t);
            ok = vid_u >= 0 && vid_v >= 0 && vid_w >= 0 && vid_t >= 0;
            for (int vid : {vid_u, vid_v, vid_w, vid_t}) {
                int nd = 0; nc_type xt = NC_NAT;
                if (vid >= 0 && (nc_inq_varndims(nc3, vid, &nd) != NC_NOERR || nd != 4 ||
                                 nc_inq_vartype(nc3, vid, &xt) != NC_NOERR || !is_supported_nc_numeric(xt))) ok = false;
            }

            // Niveles isobáricos: se leen del archivo; en modo estricto deben ser los 12 niveles 1000-700 hPa esperados
            const std::vector<float> expected_levels = {1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700};
            std::vector<float> levels;
            int vid_pl = -1;
            if (ok && nc_inq_varid(nc3, "pressure_level", &vid_pl) == NC_NOERR) {
                std::vector<double> pl(nlev, 0.0);
                if (nc_get_var_double(nc3, vid_pl, pl.data()) == NC_NOERR) {
                    for (double v : pl) levels.push_back(static_cast<float>(v));
                }
            }
            if (levels.size() != nlev) levels = (nlev == expected_levels.size()) ? expected_levels : std::vector<float>{};
            if (strict_ || levels.empty()) {
                if (levels.size() != expected_levels.size()) ok = false;
                else for (size_t k = 0; k < nlev; ++k) if (std::abs(levels[k] - expected_levels[k]) > 1e-3f) ok = false;
            }
            if (!ok) {
                std::cerr << "[NetCDF] ERROR" << (strict_ ? " (modo estricto)" : "") << ": " << pr.f3d
                          << " no contiene u,v,w,t numéricas [tiempo,nivel,lat,lon] sobre los 12 niveles 1000-700 hPa esperados.\n";
            }
            if (ok && pi == 0) {
                field.pressure_levels_hpa = levels;
                field.lats.assign(nlat, 0.0f);
                field.lons.assign(nlon, 0.0f);
                int vid_lat = -1, vid_lon = -1;
                if (nc_inq_varid(nc3, "latitude", &vid_lat) == NC_NOERR) nc_get_var_float(nc3, vid_lat, field.lats.data());
                if (nc_inq_varid(nc3, "longitude", &vid_lon) == NC_NOERR) nc_get_var_float(nc3, vid_lon, field.lons.data());
            }
            // Los archivos CDS (zlib) usan chunks de ~248 h x 4 niveles x 27 x 34 (~3.6 MB): con la caché HDF5 por defecto
            // (1 MB) cada lectura horaria re-descomprimiría el chunk completo (x248). Se amplía la caché por variable para
            // que los 27 chunks espaciales de un instante queden residentes y cada chunk se descomprima UNA sola vez.
            if (ok) {
                constexpr size_t kChunkCacheBytes = 192u * 1024u * 1024u;
                for (int vid : {vid_u, vid_v, vid_w, vid_t}) nc_set_var_chunk_cache(nc3, vid, kChunkCacheBytes, 521, 1.0f);
            }
            for (size_t ti = 0; ok && ti < pr.nt; ++ti) {
                const size_t start[4] = {ti, 0, 0, 0};
                const size_t count[4] = {1, nlev, nlat, nlon};
                const size_t base = (t_off + ti) * vol;
                ok = read_nc_var_cf(nc3, vid_u, vol, field.u.data() + base, start, count) &&
                     read_nc_var_cf(nc3, vid_v, vol, field.v.data() + base, start, count) &&
                     read_nc_var_cf(nc3, vid_w, vol, field.w.data() + base, start, count) &&
                     read_nc_var_cf(nc3, vid_t, vol, t_block.data(), start, count);
                if (!ok) break;
                // Conversión hidrostática al vuelo: omega (Pa/s) -> w geométrica (m/s) = -omega R_d T / (p g)
                float* wb = field.w.data() + base;
                #pragma omp parallel for schedule(static)
                for (std::ptrdiff_t l = 0; l < static_cast<std::ptrdiff_t>(nlev); ++l) {
                    const float p_pa = field.pressure_levels_hpa[static_cast<size_t>(l)] * 100.0f;
                    const size_t lb = static_cast<size_t>(l) * hp;
                    for (size_t c = 0; c < hp; ++c) wb[lb + c] = omega_to_w(wb[lb + c], p_pa, t_block[lb + c]);
                }
            }
            nc_close(nc3);
            if (ok) files_read_.push_back(pr.f3d);
        }

        if (ok && want_sfc) {
            int ncs = 0;
            ok = nc_open(pr.fsfc.c_str(), NC_NOWRITE, &ncs) == NC_NOERR;
            if (ok) {
                const size_t start[3] = {0, 0, 0};
                const size_t count[3] = {pr.nt, nlat, nlon};
                const size_t n = pr.nt * hp, off = t_off * hp;
                struct SfcVar { const char* name; std::vector<float>* dst; };
                const SfcVar vars[] = {{"u10", &field.u10}, {"v10", &field.v10}, {"sp", &field.sp},
                                       {"t2m", &field.t2m}, {"blh", &field.blh}};
                for (const auto& sv : vars) {
                    int vid = -1, nd = 0;
                    ok = ok && nc_inq_varid(ncs, sv.name, &vid) == NC_NOERR &&
                         nc_inq_varndims(ncs, vid, &nd) == NC_NOERR && nd == 3 &&
                         read_nc_var_cf(ncs, vid, n, sv.dst->data() + off, start, count);
                }
                if (ok) {
                    float* t2 = field.t2m.data() + off;   // Kelvin -> Celsius
                    for (size_t i = 0; i < n; ++i) t2[i] -= 273.15f;
                }
                nc_close(ncs);
                if (ok) files_read_.push_back(pr.fsfc);
                else std::cerr << "[NetCDF] ERROR: " << pr.fsfc << " no contiene sp,t2m,u10,v10,blh numéricas [tiempo,lat,lon].\n";
            }
        }

        if (!ok) {
            if (pi == 0 || !resilient) return false;
            std::cout << "[ERA5] AVISO: mes " << pr.month << " de " << year
                      << " ilegible/incompleto; se conservan solo los meses previos.\n";
            fail_truncate(t_off);
            break;
        }
        t_off += pr.nt;
        ++loaded;
    }
    field.n_times = static_cast<int>(t_off);
    if (months_loaded) *months_loaded = loaded;
    return t_off > 0;
#else
    (void)year; (void)months; (void)require_sfc; (void)resilient; (void)field;
    return false;
#endif
}

bool Era5NetCdfReader::load_month(int year, int month, AtmosphericField3D& field) {
    if (!year_allowed(year)) return false;
    if (read_months(year, {month}, /*require_sfc=*/strict_, /*resilient=*/false, field, nullptr)) return true;

    if (strict_) {
        std::cerr << "[NetCDF] ERROR (modo estricto): carga ERA5 " << year << "-" << month
                  << " fallida; no se admite la malla sintética de respaldo.\n";
        return false;
    }

    // Si NetCDF no está disponible en tiempo de compilación o falló el archivo, usar malla meteorológica base
    field = AtmosphericField3D{};
    field.n_times = 744; // 31 días * 24 horas
    field.n_levels = 12;
    field.n_lats = 81;
    field.n_lons = 101;
    field.pressure_levels_hpa = {1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700};
    
    size_t total_elements = static_cast<size_t>(field.n_times) * field.n_levels * field.n_lats * field.n_lons;
    field.u.assign(total_elements, 5.0f);     // 5 m/s hacia el este
    field.v.assign(total_elements, 2.0f);     // 2 m/s hacia el norte
    field.w.assign(total_elements, 0.05f);    // 0.05 m/s ascenso débil
    
    size_t sfc_elements = static_cast<size_t>(field.n_times) * field.n_lats * field.n_lons;
    field.u10.assign(sfc_elements, 3.5f);
    field.v10.assign(sfc_elements, 1.5f);
    field.sp.assign(sfc_elements, 101325.0f);
    field.t2m.assign(sfc_elements, 22.0f);   // 22 °C promedio estival
    field.blh.assign(sfc_elements, 1200.0f);
    return true;
}

bool Era5NetCdfReader::load_season(int year, const std::vector<int>& months, AtmosphericField3D& out) {
    if (months.empty()) return false;
    if (!year_allowed(year)) return false;

    int loaded = 0;
    if (read_months(year, months, /*require_sfc=*/true, /*resilient=*/true, out, &loaded)) {
        std::cout << "[ERA5] Temporada " << year << ": " << loaded << " de " << months.size()
                  << " mes(es) cargado(s) = " << out.n_times << " h";
        if (loaded < static_cast<int>(months.size())) {
            std::cout << "  (AVISO: temporada incompleta, faltan meses por descargar; horas esperadas "
                      << "168/744/1464/2208 según meses cerrados)";
        }
        std::cout << "\n";
        return true;
    }
    if (strict_) {
        std::cerr << "[NetCDF] ERROR (modo estricto): ni siquiera el primer mes (" << months.front() << ") de "
                  << year << " está disponible.\n";
        return false;
    }
    return load_month(year, months.front(), out);
}

void Era5NetCdfReader::interpolate_wind_3d(
    const AtmosphericField3D& /*field*/,
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
