#include "inference/occupancy_model.hpp"
#include "inference/cost_function.hpp"
#include <iostream>
#include <vector>
#include <cassert>

/// Prueba de recuperación de parámetros sintéticos (Sección 11 de JORO-HPC.tex).
int main() {
    std::cout << "[TEST] Ejecutando test_parameter_recovery...\n";

    joro::inference::OccupancyParameters true_params;
    true_params.beta1_effort = 0.85f;

    // Generar datos sintéticos de detección bajo un gradiente de esfuerzo
    std::vector<float> p_pred;
    std::vector<uint8_t> y_obs;

    for (int effort = 0; effort < 100; effort += 10) {
        float p = joro::inference::compute_detection_prob(effort, 0.5f, 0.2f, true_params);
        p_pred.push_back(p);
        y_obs.push_back((p > 0.5f) ? 1 : 0);
    }

    double cost = joro::inference::compute_cost_function(p_pred, y_obs);
    assert(cost >= 0.0);
    assert(cost < 1.0);

    std::cout << "[TEST] test_parameter_recovery PASÓ (costo sintético = " << cost << ").\n";
    return 0;
}
