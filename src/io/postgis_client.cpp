#include "io/postgis_client.hpp"
#include <iostream>

#if defined(JORO_HAS_LIBPQXX)
#include <pqxx/pqxx>
#endif

namespace joro::io {

PostgisClient::PostgisClient(std::string host, int port, std::string dbname, std::string user)
    : host_(std::move(host)), port_(port), dbname_(std::move(dbname)), user_(std::move(user)) {}

PostgisClient::~PostgisClient() = default;

bool PostgisClient::connect() {
#if defined(JORO_HAS_LIBPQXX)
    try {
        std::string conn_str = "host=" + host_ + " port=" + std::to_string(port_) +
                               " dbname=" + dbname_ + " user=" + user_;
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
    // Query a la tabla grid_5km
    return true;
#else
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
#endif
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
