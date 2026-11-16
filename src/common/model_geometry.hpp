#pragma once

#include "common/types.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace joro {

/// Geometría UNIFICADA del polígono semilla fundador (celdas `is_seed`, 2014-2015): centroide geométrico y radio inicial.
/// Es la ÚNICA referencia de los radios P50/P90/P95 del frente, de la envolvente y del azimut, tanto en calibración
/// (simulated_annealing.cpp) como en evaluación (joro_evaluate.cpp). Retira el antiguo punto heredado kEpiLat/kEpiLon
/// (34.13816, -83.75723), desplazado ~27 km al suroeste del centroide semilla (discrepancia M15).
struct SeedGeometry {
    double lat = 34.13816;   ///< Respaldo heredado SOLO si la malla no contiene celdas semilla
    double lon = -83.75723;
    double radius_km = 0.0;
    size_t n_cells = 0;

    [[nodiscard]] double dist_km(float lat_c, float lon_c) const noexcept {
        const double dlat = (lat_c - lat) * 111.0;
        const double dlon = (lon_c - lon) * 111.0 * std::cos(lat * 0.0174533);
        return std::sqrt(dlat * dlat + dlon * dlon);
    }
    /// Azimut (0 = Norte, 90 = Este) respecto al centroide semilla, en [0, 360)
    [[nodiscard]] double azimuth_deg(float lat_c, float lon_c) const noexcept {
        const double dlat = (lat_c - lat) * 111.0;
        const double dlon = (lon_c - lon) * 111.0 * std::cos(lat * 0.0174533);
        double deg = std::atan2(dlon, dlat) * 180.0 / M_PI;
        if (deg < 0.0) deg += 360.0;
        return deg;
    }

    [[nodiscard]] static SeedGeometry from_cells(const std::vector<GridCell>& cells) {
        SeedGeometry g;
        double sum_lat = 0.0, sum_lon = 0.0;
        size_t n = 0;
        for (const auto& c : cells) {
            if (!c.is_seed) continue;
            sum_lat += c.centroid_lat; sum_lon += c.centroid_lon; ++n;
        }
        if (n == 0) return g;
        g.lat = sum_lat / static_cast<double>(n);
        g.lon = sum_lon / static_cast<double>(n);
        g.n_cells = n;
        double rmax = 0.0;
        for (const auto& c : cells) if (c.is_seed) rmax = std::max(rmax, g.dist_km(c.centroid_lat, c.centroid_lon));
        g.radius_km = rmax;
        return g;
    }
};

/// Límites de normalización min-max de las covariables humanas (H = densidad poblacional, R = densidad vial) sobre la
/// malla completa Ω: idénticos en calibración y evaluación (discrepancia M15).
struct CovariateNorm {
    float h_min = 0.0f, h_span = 1.0f, r_min = 0.0f, r_span = 1.0f;

    [[nodiscard]] float h(float pop_density) const noexcept { return (pop_density - h_min) / h_span; }
    [[nodiscard]] float r(float road_density) const noexcept { return (road_density - r_min) / r_span; }

    [[nodiscard]] static CovariateNorm from_cells(const std::vector<GridCell>& cells) {
        CovariateNorm n;
        if (cells.empty()) return n;
        float h_min = 1e30f, h_max = -1e30f, r_min = 1e30f, r_max = -1e30f;
        for (const auto& c : cells) {
            h_min = std::min(h_min, c.pop_density_h); h_max = std::max(h_max, c.pop_density_h);
            r_min = std::min(r_min, c.road_density_r); r_max = std::max(r_max, c.road_density_r);
        }
        n.h_min = h_min; n.r_min = r_min;
        n.h_span = (h_max - h_min) > 0.0f ? (h_max - h_min) : 1.0f;
        n.r_span = (r_max - r_min) > 0.0f ? (r_max - r_min) : 1.0f;
        return n;
    }
};

} // namespace joro

