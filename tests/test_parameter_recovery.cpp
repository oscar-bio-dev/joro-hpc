#include "inference/occupancy_model.hpp"
#include "inference/cost_function.hpp"
#include "test_common.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

/// Prueba de recuperación de parámetros sintéticos (Sección 11 de JORO-HPC.tex).
/// Genera observaciones sintéticas con un parámetro objetivo conocido (theta*)
/// y ejecuta una búsqueda de optimización unidimensional para verificar
/// que el mínimo de la función de costo recupera el valor verdadero.
int main() {
    std::cout << "─────────────────────────────────────────────────────────────\n";
    std::cout << "[TEST] test_parameter_recovery: Identificabilidad de Parámetros\n";
    std::cout << "─────────────────────────────────────────────────────────────\n";

    constexpr float true_beta1 = 0.85f;
    joro::inference::OccupancyParameters true_params;
    true_params.beta1_effort = true_beta1;

    // 1. Generación de observaciones sintéticas
    std::vector<int> efforts;
    std::vector<uint8_t> y_obs;
    for (int effort = 1; effort <= 200; effort += 2) {
        efforts.push_back(effort);
        float p = joro::inference::compute_detection_prob(effort, 0.4f, 0.3f, true_params);
        y_obs.push_back((p >= 0.5f) ? 1 : 0);
    }

    // Función lambda para evaluar el costo J(theta) para un valor candidato de beta1_effort
    auto evaluate_candidate_cost = [&](float candidate_beta1) -> double {
        joro::inference::OccupancyParameters cand_params;
        cand_params.beta1_effort = candidate_beta1;

        std::vector<float> p_pred;
        p_pred.reserve(efforts.size());
        for (int effort : efforts) {
            p_pred.push_back(joro::inference::compute_detection_prob(effort, 0.4f, 0.3f, cand_params));
        }
        return joro::inference::compute_cost_function(p_pred, y_obs);
    };

    // 2. Evaluar costo en punto perturbado inicial (theta_initial = 0.15)
    constexpr float initial_guess = 0.15f;
    double cost_initial = evaluate_candidate_cost(initial_guess);
    std::cout << "  Parámetro inicial (perturbado): " << initial_guess 
              << " | Costo inicial = " << cost_initial << "\n";

    // 3. Búsqueda de sección áurea (Golden Section Search) para optimizar beta1 en [0.10, 1.50]
    double a = 0.10;
    double b = 1.50;
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double resphi = 2.0 - phi;

    double c = a + resphi * (b - a);
    double d = b - resphi * (b - a);
    double fc = evaluate_candidate_cost(static_cast<float>(c));
    double fd = evaluate_candidate_cost(static_cast<float>(d));

    for (int iter = 0; iter < 40; ++iter) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = a + resphi * (b - a);
            fc = evaluate_candidate_cost(static_cast<float>(c));
        } else {
            a = c;
            c = d;
            fc = fd;
            d = b - resphi * (b - a);
            fd = evaluate_candidate_cost(static_cast<float>(d));
        }
    }

    float recovered_beta1 = static_cast<float>((a + b) / 2.0);
    double cost_recovered = evaluate_candidate_cost(recovered_beta1);

    std::cout << "  Parámetro verdadero (theta*)  : " << true_beta1 << "\n";
    std::cout << "  Parámetro recuperado (theta^) : " << recovered_beta1 
              << " | Costo final = " << cost_recovered << "\n";

    double recovery_error = std::abs(recovered_beta1 - true_beta1);
    std::cout << "  Error de recuperación |theta^ - theta*|: " << recovery_error << "\n";

    // Criterios de validación rigurosos
    JORO_TEST_ASSERT(cost_recovered < cost_initial, "La optimización no redujo el costo respecto al guess inicial");
    JORO_TEST_ASSERT(recovery_error < 0.30, "El optimizador no convergió al entorno del parámetro verdadero");

    std::cout << "✔ test_parameter_recovery PASÓ: Identificabilidad numérica confirmada.\n";
    return 0;
}
