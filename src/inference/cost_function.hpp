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

/// Evalúa la función de pérdida compuesta J(theta) comparando predicciones vs observaciones reales
[[nodiscard]] inline double compute_cost_function(
    const std::vector<float>& predicted_p_detect,
    const std::vector<uint8_t>& observed_y,
    double front_error_km = 0.0,
    double persistence_discrepancy = 0.0,
    const CostWeights& weights = CostWeights{}) noexcept
{
    if (predicted_p_detect.empty() || predicted_p_detect.size() != observed_y.size()) {
        return 1e6; // Penalización por dimensiones inválidas
    }

    // 1. Componente de llegada / detección: Brier Score cuadrático
    // Brier = (1/N) * sum (p_i - y_i)^2
    double brier_sum = 0.0;
    const size_t n = predicted_p_detect.size();
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(predicted_p_detect[i]) - static_cast<double>(observed_y[i]);
        brier_sum += diff * diff;
    }
    double j_arrival = brier_sum / static_cast<double>(n);

    // 2. Normalización de componentes de frente y persistencia
    double j_front = front_error_km / 100.0; // Normalizado con respecto a 100 km de error de escala
    double j_persistence = persistence_discrepancy;

    // Función de costo total ponderada
    return weights.w1_arrival * j_arrival +
           weights.w2_front * j_front +
           weights.w3_persistence * j_persistence;
}

} // namespace joro::inference
