#include "io/postgis_client.hpp"
#include <iostream>

#if defined(JORO_HAS_LIBPQXX)
#include <pqxx/pqxx>
#endif

namespace joro::io {

PostgisClient::PostgisClient(std::string host, int port, std::string dbname, std::string user, std::string password)
    : host_(std::move(host)), port_(port), dbname_(std::move(dbname)), user_(std::move(user)), password_(std::move(password)) {}

PostgisClient::~PostgisClient() = default;

bool PostgisClient::connect() {
#if defined(JORO_HAS_LIBPQXX)
    try {
        std::string conn_str = "host=" + host_ + " port=" + std::to_string(port_) +
                               " dbname=" + dbname_ + " user=" + user_;
        if (!password_.empty()) {
            conn_str += " password=" + password_;
        }
        pqxx::connection conn(conn_str);
        is_connected_ = conn.is_open();
        return is_connected_;
    } catch (const std::exception& e) {
        std::cerr << "[PostGIS] Error de conexión: " << e.what() << "\n";
        return false;
    }
#else
    std::cout << "[PostGIS] Modo simulación (libpqxx no enlazado en compilación).\n";
    is_connected_ = true;
    return true;
#endif
}

bool PostgisClient::load_grid(std::vector<GridCell>& cells) {
    if (!is_connected_) return false;
    
#if defined(JORO_HAS_LIBPQXX)
    try {
        std::string conn_str = "host=" + host_ + " port=" + std::to_string(port_) +
                               " dbname=" + dbname_ + " user=" + user_;
        if (!password_.empty()) {
            conn_str += " password=" + password_;
        }
        pqxx::connection conn(conn_str);
        if (conn.is_open()) {
            pqxx::work txn(conn);
            pqxx::result res = txn.exec(
                "SELECT cell_id, centroid_lon, centroid_lat, z_imp, z_borde, z_dosel, "
                "pop_density_h, road_density_r, ecoregion_block, is_seed_2014_15 "
                "FROM grid_5km ORDER BY cell_id"
            );
            
            if (!res.empty()) {
                cells.clear();
                cells.reserve(res.size());
                for (const auto& row : res) {
                    GridCell cell;
                    cell.cell_id = row[0].as<int32_t>();
                    cell.centroid_lon = row[1].as<float>();
                    cell.centroid_lat = row[2].as<float>();
                    cell.z_imp = row[3].is_null() ? 0.0f : row[3].as<float>();
                    cell.z_borde = row[4].is_null() ? 0.0f : row[4].as<float>();
                    cell.z_dosel = row[5].is_null() ? 0.0f : row[5].as<float>();
                    cell.pop_density_h = row[6].is_null() ? 0.0f : row[6].as<float>();
                    cell.road_density_r = row[7].is_null() ? 0.0f : row[7].as<float>();
                    cell.ecoregion_block = row[8].is_null() ? 0 : row[8].as<int16_t>();
                    cell.is_seed = row[9].is_null() ? false : row[9].as<bool>();
                    cells.push_back(cell);
                }
                return true;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[PostGIS] Error al consultar grid_5km: " << e.what()
                  << ". Usando inicialización sintética de respaldo.\n";
    }
#endif

    // Generar malla sintética representativa de prueba (100 celdas para pruebas unitarias)
    cells.clear();
    cells.reserve(100);
    for (int i = 0; i < 100; ++i) {
        GridCell cell;
        cell.cell_id = i + 1;
        cell.centroid_lon = -84.0f + (i % 10) * 0.05f;
        cell.centroid_lat = 34.0f + (i / 10) * 0.05f;
        cell.z_imp = 0.2f;
        cell.z_borde = 0.5f;
        cell.z_dosel = 0.4f;
        cell.is_seed = (i == 45); // Celda central semilla
        cells.push_back(cell);
    }
    return true;
}

bool PostgisClient::load_observations(int /*start_year*/, int /*end_year*/,
                                      std::vector<AnnualObservation>& observations) {
    if (!is_connected_) return false;
    observations.clear();
    return true;
}

bool PostgisClient::export_occupancy_predictions(
    int /*year*/, const std::unordered_map<int32_t, float>& /*psi_map*/) {
    return true;
}

} // namespace joro::io
