#pragma once

#include "common/types.hpp"
#include <vector>
#include <cmath>

namespace joro::inference {

struct CostWeights {
    double w1_arrival = 0.333333;
    double w2_front = 0.333333;
    double w3_persistence = 0.333334;
};

/// Escalas de referencia iniciales para la normalización adimensional de componentes (Sección 11)
struct CostReferenceScales {
    double s_arrival = 0.005;    ///< Escala de referencia inicial para Brier score
    double s_front = 40.0;       ///< Escala de referencia inicial para error de frente (km)
    double s_persistence = 0.10; ///< Escala de referencia inicial para discrepancia de persistencia
};

/// Desglose de componentes de la función de costo
struct CostComponents {
    double j_total = 0.0;
    double j_arrival_raw = 0.0;
    double j_front_raw = 0.0;
    double j_persistence_raw = 0.0;
    double j_arrival_norm = 0.0;
    double j_front_norm = 0.0;
    double j_persistence_norm = 0.0;
};

/// Evalúa la función de costo multicriterio normalizada (Secciones 9 y 11 de JORO-HPC.tex):
/// J(theta) = w1 * J_tilde_llegada + w2 * J_tilde_frente + w3 * J_tilde_persistencia
[[nodiscard]] inline CostComponents evaluate_multicriteria_cost(
    const std::vector<float>& predicted_p_marginal,
    const std::vector<uint8_t>& observed_y,
    const std::vector<double>& sim_front_km,
    const std::vector<double>& obs_front_km,
    double persistence_discrepancy,
    const CostWeights& weights = CostWeights{},
    const CostReferenceScales& scales = CostReferenceScales{}) noexcept
{
    CostComponents comp;
    if (predicted_p_marginal.empty() || predicted_p_marginal.size() != observed_y.size()) {
        comp.j_total = 1e6;
        return comp;
    }

    // 1. Componente de llegada / detección: Brier Score
    double brier_sum = 0.0;
    const size_t n = predicted_p_marginal.size();
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(predicted_p_marginal[i]) - static_cast<double>(observed_y[i]);
        brier_sum += diff * diff;
    }
    comp.j_arrival_raw = brier_sum / static_cast<double>(n);

    // 2. Componente de frente de expansión (distancia radial desde epicentro Braselton GA)
    double front_err_sum = 0.0;
    size_t n_years = std::min(sim_front_km.size(), obs_front_km.size());
    if (n_years > 0) {
        for (size_t t = 0; t < n_years; ++t) {
            front_err_sum += std::abs(sim_front_km[t] - obs_front_km[t]);
        }
        comp.j_front_raw = front_err_sum / static_cast<double>(n_years);
    } else {
        comp.j_front_raw = 0.0;
    }

    // 3. Componente de persistencia biológica interanual
    comp.j_persistence_raw = std::abs(persistence_discrepancy);

    // Normalización adimensional por escalas de referencia
    comp.j_arrival_norm = comp.j_arrival_raw / std::max(scales.s_arrival, 1e-6);
    comp.j_front_norm = comp.j_front_raw / std::max(scales.s_front, 1e-6);
    comp.j_persistence_norm = comp.j_persistence_raw / std::max(scales.s_persistence, 1e-6);

    // Función de costo total ponderada
    comp.j_total = weights.w1_arrival * comp.j_arrival_norm +
                   weights.w2_front * comp.j_front_norm +
                   weights.w3_persistence * comp.j_persistence_norm;

    return comp;
}

/// Compatibilidad con baseline previo
[[nodiscard]] inline double compute_cost_function(
    const std::vector<float>& predicted_p_detect,
    const std::vector<uint8_t>& observed_y,
    double front_error_km = 0.0,
    double persistence_discrepancy = 0.0,
    const CostWeights& weights = CostWeights{}) noexcept
{
    if (predicted_p_detect.empty() || predicted_p_detect.size() != observed_y.size()) {
        return 1e6;
    }

    double brier_sum = 0.0;
    const size_t n = predicted_p_detect.size();
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(predicted_p_detect[i]) - static_cast<double>(observed_y[i]);
        brier_sum += diff * diff;
    }
    double j_arrival = brier_sum / static_cast<double>(n);
    double j_front = front_error_km / 100.0;
    double j_persistence = persistence_discrepancy;

    return weights.w1_arrival * j_arrival +
           weights.w2_front * j_front +
           weights.w3_persistence * j_persistence;
}

} // namespace joro::inference
