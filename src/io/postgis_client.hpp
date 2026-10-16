#pragma once

#include "common/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace joro::io {

class PostgisClient {
public:
    PostgisClient(std::string host, int port, std::string dbname, std::string user);
    ~PostgisClient();

    /// Conecta con la base de datos PostgreSQL
    bool connect();

    /// Carga la malla demográfica de 5 km x 5 km en memoria
    bool load_grid(std::vector<GridCell>& cells);

    /// Carga observaciones anuales y esfuerzo Araneae para un rango de años
    bool load_observations(int start_year, int end_year,
                           std::vector<AnnualObservation>& observations);

    /// Exporta las probabilidades de ocupación predichas a la base de datos
    bool export_occupancy_predictions(int year,
                                     const std::unordered_map<int32_t, float>& psi_map);

private:
    std::string host_;
    int port_;
    std::string dbname_;
    std::string user_;
    bool is_connected_ = false;
};

} // namespace joro::io
