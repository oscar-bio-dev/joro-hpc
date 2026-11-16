#pragma once

#include "abm/abm_engine.hpp"
#include "common/spatial_grid_lookup.hpp"
#include "common/types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace joro::inference {

/// Dispersor nulo isotrópico exponencial K0(d | lambda): distancia ~ Exp(1/lambda), azimut uniforme (comparador de H2).
/// Muestreo ponderado (peso por muestra) para celdas con muchos propágulos. O(1) por muestra con SpatialGridView.
struct IsotropicKernelDisperser {
    double lambda_m = 3500.0;
    SpatialGridView grid_view{};

    [[nodiscard]] std::unordered_map<int32_t, uint32_t> disperse(
        const std::unordered_map<int32_t, uint32_t>& release,
        const std::vector<GridCell>& cells,
        const abm::CellIndexTable& cidx,
        uint32_t seed) const
    {
        std::unordered_map<int32_t, uint32_t> arrivals;
        std::mt19937 rng(seed);
        std::exponential_distribution<double> exp_dist(1.0 / lambda_m);
        std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

        // Orden canónico por cell_id: el recorrido de unordered_map depende del orden de inserción (reproducibilidad)
        std::vector<std::pair<int32_t, uint32_t>> ordered(release.begin(), release.end());
        std::sort(ordered.begin(), ordered.end());
        for (const auto& [origin_id, count] : ordered) {
            if (count == 0) continue;
            const int32_t pos = cidx.find(origin_id);
            if (pos < 0) continue;
            const float orig_lon = cells[static_cast<size_t>(pos)].centroid_lon;
            const float orig_lat = cells[static_cast<size_t>(pos)].centroid_lat;
            const double cos_lat = std::cos(static_cast<double>(orig_lat) * (M_PI / 180.0));

            uint32_t samples = count;
            uint32_t weight = 1;
            if (samples > 200) {
                weight = (samples + 199) / 200;
                samples = (samples + weight - 1) / weight;
            }
            for (uint32_t i = 0; i < samples; ++i) {
                const double dist_m = exp_dist(rng);
                const double theta = angle_dist(rng);
                const double d_lat = (dist_m * std::cos(theta)) / 111320.0;
                const double d_lon = (dist_m * std::sin(theta)) / (111320.0 * cos_lat);
                const int32_t cid = grid_view.find(static_cast<float>(orig_lon + d_lon), static_cast<float>(orig_lat + d_lat));
                if (cid >= 0) arrivals[cid] += weight;
            }
        }
        return arrivals;
    }
};

/// Kernel gravitacional antropogénico K_h(i | j) ∝ w_i exp(-c_ij / lambda_h) con atracción del receptor w_i = R_i H_i
/// (densidad vial x población urbana normalizadas a [0,1]), que captura los saltos satelitales (> 300 km) mediados por
/// vehículos por corredores viales. El costo de viaje c_ij se aproxima por la distancia geodésica (no se dispone de la
/// red TIGER/Line en el motor). Se muestrea directamente (Poisson(p_h P_j) propágulos por celda fuente j):
///   * distancia ~ Gamma(2, lambda_h)  (densidad por área ∝ exp(-r / lambda_h) en el plano => densidad radial r e^{-r/λ}),
///   * azimut uniforme,
///   * aceptación del destino con probabilidad w_i / w_max (se remuestrea si se rechaza).
/// Las posiciones fuera del dominio son pérdidas L_t (se conserva Σ_i A_i + L = p_h Σ_j P_j).
class HumanTransportKernel {
public:
    static constexpr float kMinWeight = 0.02f;   ///< Piso de atracción: evita celdas rurales con probabilidad exactamente 0

    /// `h_norm`/`r_norm` en [0,1] por POSICIÓN de celda (mismo orden que `cells`)
    void build(const std::vector<GridCell>& cells, const std::vector<float>& h_norm, const std::vector<float>& r_norm,
               const SpatialGridView& view, const abm::CellIndexTable* cidx)
    {
        cells_ = &cells;
        view_ = view;
        cidx_ = cidx;
        r_norm_ = r_norm;
        w_.resize(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            w_[i] = std::max(kMinWeight, h_norm[i]) * std::max(kMinWeight, r_norm[i]);
        }
        w_max_ = w_.empty() ? 1.0f : *std::max_element(w_.begin(), w_.end());
        if (w_max_ <= 0.0f) w_max_ = 1.0f;
    }

    struct Result {
        std::unordered_map<int32_t, uint32_t> arrivals;   ///< A^hum por celda (ya incluye el factor p_h)
        uint64_t transported = 0;                         ///< Propágulos muestreados (Σ Poisson)
        uint64_t lost = 0;                                ///< Perdidos fuera del dominio
        uint64_t beyond_300km = 0;                        ///< Depositados a más de 300 km de su origen
    };

    /// `sources`: (cell_id, ootecas transportables) ordenado por cell_id. p_h = 0 => resultado vacío.
    /// Emisión ponderada por el producto de la abundancia (ootecas) y la densidad vial de la celda origen R_j.
    [[nodiscard]] Result transport(const std::vector<std::pair<int32_t, float>>& sources, double p_h,
                                   double lambda_km, uint32_t seed) const
    {
        Result res;
        if (p_h <= 0.0 || cells_ == nullptr || sources.empty()) return res;
        std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ seed);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        constexpr int kMaxTries = 64;
        constexpr uint64_t kMaxPerSource = 2000000;

        for (const auto& [src_id, weight] : sources) {
            if (weight <= 0.0f) continue;
            const int32_t pos = cidx_ ? cidx_->find(src_id) : -1;
            if (pos < 0) continue;
            const GridCell& src = (*cells_)[static_cast<size_t>(pos)];
            const double r_src = (static_cast<size_t>(pos) < r_norm_.size())
                                     ? std::max<double>(kMinWeight, r_norm_[static_cast<size_t>(pos)])
                                     : 1.0;
            const double mean = p_h * static_cast<double>(weight) * r_src;
            std::poisson_distribution<uint64_t> pois(std::min(mean, 1.0e9));
            const uint64_t n = std::min<uint64_t>(pois(rng), kMaxPerSource);
            const double cos_lat = std::max(0.05, std::cos(static_cast<double>(src.centroid_lat) * (M_PI / 180.0)));

            for (uint64_t k = 0; k < n; ++k) {
                ++res.transported;
                bool placed = false;
                for (int t = 0; t < kMaxTries; ++t) {
                    const double r_km = -lambda_km * std::log(std::max(1e-12, u01(rng) * u01(rng)));  // Gamma(2, lambda)
                    const double th = 2.0 * M_PI * u01(rng);
                    const double dlat = r_km * std::cos(th) / 111.32;
                    const double dlon = r_km * std::sin(th) / (111.32 * cos_lat);
                    const int32_t cid = view_.find(static_cast<float>(src.centroid_lon + dlon),
                                                   static_cast<float>(src.centroid_lat + dlat));
                    if (cid < 0) { ++res.lost; placed = true; break; }       // fuera del dominio: pérdida L_t
                    const int32_t dpos = cidx_->find(cid);
                    if (dpos < 0) continue;
                    if (u01(rng) * w_max_ <= w_[static_cast<size_t>(dpos)]) {   // aceptación ∝ w_i
                        ++res.arrivals[cid];
                        if (r_km > 300.0) ++res.beyond_300km;
                        placed = true;
                        break;
                    }
                }
                if (!placed) ++res.lost;
            }
        }
        return res;
    }

private:
    const std::vector<GridCell>* cells_ = nullptr;
    SpatialGridView view_{};
    const abm::CellIndexTable* cidx_ = nullptr;
    std::vector<float> r_norm_;
    std::vector<float> w_;
    float w_max_ = 1.0f;
};

} // namespace joro::inference

