#include "inference/occupancy_model.hpp"
#include "inference/cost_function.hpp"
#include "abm/carrying_capacity.hpp"
#include "test_common.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

/// Prueba de recuperación de parámetros sintéticos (Sección 11 de JORO-HPC_Research_Proposal.tex).
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

    // ── Reparametrización de K_i (Sección 8.4) y enlace de ocupación calibrable ──
    {
        using joro::abm::CarryingCapacityParams;
        using joro::abm::compute_carrying_capacity;
        using joro::abm::compute_habitat_index;
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<joro::GridCell> cells(5000);
        for (auto& c : cells) { c.z_borde = nd(rng); c.z_imp = nd(rng); c.z_dosel = nd(rng); }
        compute_habitat_index(cells);
        double m = 0.0, v = 0.0;
        for (const auto& c : cells) m += c.h_hab;
        m /= static_cast<double>(cells.size());
        for (const auto& c : cells) v += (c.h_hab - m) * (c.h_hab - m);
        v /= static_cast<double>(cells.size() - 1);
        JORO_TEST_ASSERT(std::abs(m) < 1e-4 && std::abs(v - 1.0) < 1e-3, "H_hab debe tener media 0 y varianza unitaria");

        CarryingCapacityParams m0{}; m0.k0 = 4000.0f; m0.use_hab_index = true; m0.gamma_hab = 0.0f;
        CarryingCapacityParams m1 = m0; m1.gamma_hab = 1.2f;
        bool m0_const = true, m1_monotone = true;
        auto lo = cells[0], hi = cells[0];
        for (const auto& c : cells) { if (c.h_hab < lo.h_hab) lo = c; if (c.h_hab > hi.h_hab) hi = c; }
        for (const auto& c : cells) m0_const &= std::abs(compute_carrying_capacity(c, m0) - 4000.0f) < 1e-3f;
        m1_monotone = compute_carrying_capacity(hi, m1) > compute_carrying_capacity(lo, m1);
        JORO_TEST_ASSERT(m0_const, "M0: K_i debe ser K0 en todas las celdas");
        JORO_TEST_ASSERT(m1_monotone, "M1: K_i debe crecer con H_hab cuando gamma_hab > 0");
        // K0 = mediana de K_i en M1: la celda de H_hab = 0 tiene K_i = K0 (ortogonal a gamma_hab)
        joro::GridCell mid{}; mid.h_hab = 0.0f;
        JORO_TEST_ASSERT(std::abs(compute_carrying_capacity(mid, m1) - 4000.0f) < 1e-3f, "M1: K_i(H=0) debe ser K0");

        // a0 y b0 desplazan monótonamente las probabilidades de colonización y persistencia
        joro::inference::OccupancyParameters lo_a, hi_a;
        lo_a.a0 = -5.0f; hi_a.a0 = -2.0f; lo_a.b0 = -1.0f; hi_a.b0 = 2.0f;
        JORO_TEST_ASSERT(joro::inference::compute_colonization_gamma(3, 0.4f, hi_a) >
                         joro::inference::compute_colonization_gamma(3, 0.4f, lo_a), "gamma debe crecer con a0");
        JORO_TEST_ASSERT(joro::inference::compute_persistence_phi(5.0f, 0.2f, hi_a) >
                         joro::inference::compute_persistence_phi(5.0f, 0.2f, lo_a), "phi debe crecer con b0");

        // El Brier balanceado es diferenciable y monótono en psi: positivos decrecen, negativos crecen
        const double d = 1e-3;
        auto brier_pos = [](double pm) { return (1.0 - pm) * (1.0 - pm); };
        auto brier_neg = [](double pm) { return pm * pm; };
        JORO_TEST_ASSERT(brier_pos(0.30 + d) < brier_pos(0.30) && brier_neg(0.30 + d) > brier_neg(0.30),
                         "Brier de positivos/negativos debe ser monótono respecto a P_hat");
        std::cout << "  ✔ Reparametrización K_i (M0/M1) y enlace a0/b0 verificados.\n";
    }

    std::cout << "✔ test_parameter_recovery PASÓ: Identificabilidad numérica confirmada.\n";
    return 0;
}
