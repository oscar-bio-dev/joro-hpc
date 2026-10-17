#pragma once

#include "common/types.hpp"
#include <cmath>
#include <vector>
#include <unordered_map>
#include <random>

namespace joro::lpdm {

/// Modelo nulo isotrópico exponencial para la hipótesis H2 (Sección 6.2 de JORO-HPC.tex):
/// K_0(d | lambda) = (lambda / (2 * pi)) * exp(-lambda * d)
/// Sirve como referencia base no mecanística frente a la dispersión eólica 3D (LPDM).
class NullKernelDisperser {
public:
    explicit NullKernelDisperser(double lambda = 0.0001) // lambda en m^-1 (~10 km media)
        : lambda_(lambda) {}

    /// Densidad de probabilidad de dispersión a distancia d [m]
    [[nodiscard]] double evaluate_density(double distance_m) const noexcept {
        if (distance_m < 0.0) return 0.0;
        constexpr double two_pi = 2.0 * M_PI;
        return (lambda_ / two_pi) * std::exp(-lambda_ * distance_m);
    }

    /// Dispersa propágulos desde celdas de origen según un kernel exponencial isotrópico
    std::unordered_map<int32_t, uint32_t> disperse_null(
        const std::vector<GridCell>& cells,
        const std::unordered_map<int32_t, uint32_t>& release,
        uint32_t random_seed = 42819) const
    {
        std::unordered_map<int32_t, uint32_t> arrivals;
        std::mt19937 rng(random_seed);
        std::exponential_distribution<double> exp_dist(lambda_);
        std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

        // Mapa de celdas por posición
        for (const auto& [origin_id, count] : release) {
            if (count == 0) continue;

            // Encontrar centroide de origen
            float orig_lon = 0.0f;
            float orig_lat = 0.0f;
            for (const auto& cell : cells) {
                if (cell.cell_id == origin_id) {
                    orig_lon = cell.centroid_lon;
                    orig_lat = cell.centroid_lat;
                    break;
                }
            }

            for (uint32_t i = 0; i < count; ++i) {
                double dist_m = exp_dist(rng);
                double theta = angle_dist(rng);

                // Convertir distancia y ángulo a desplazamiento lon/lat
                double d_lat = (dist_m * std::sin(theta)) / 111320.0;
                double d_lon = (dist_m * std::cos(theta)) / (111320.0 * std::cos(orig_lat * M_PI / 180.0));

                double dest_lon = orig_lon + d_lon;
                double dest_lat = orig_lat + d_lat;

                // Buscar celda receptora más cercana
                int32_t best_cell = -1;
                double min_dist_sq = 1e9;
                for (const auto& c : cells) {
                    double dx = c.centroid_lon - dest_lon;
                    double dy = c.centroid_lat - dest_lat;
                    double d2 = dx * dx + dy * dy;
                    if (d2 < min_dist_sq) {
                        min_dist_sq = d2;
                        best_cell = c.cell_id;
                    }
                }

                // Asignar si cae dentro del radio de influencia de la malla (~5 km = 0.05 deg)
                if (best_cell != -1 && min_dist_sq < (0.05 * 0.05 * 2.0)) {
                    arrivals[best_cell]++;
                }
            }
        }
        return arrivals;
    }

private:
    double lambda_;
};

} // namespace joro::lpdm
