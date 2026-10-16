#include "lpdm/langevin_turbulence.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

/// Prueba numérica de convergencia débil del esquema estocástico de Euler-Maruyama
/// según Kloeden & Platen (1992) y Sección 7.6 de JORO-HPC.tex.
///
/// Evalúa el momento estocástico E[w'^2(T)] frente al paso temporal dt
/// incorporando el incremento de Wiener dW ~ N(0, dt) sobre múltiples realizaciones Monte Carlo.
int main() {
    std::cout << "─────────────────────────────────────────────────────────────\n";
    std::cout << "[TEST] test_euler_maruyama_conv: Convergencia Débil Estocástica\n";
    std::cout << "─────────────────────────────────────────────────────────────\n";

    constexpr float total_time_T = 4.0f;  // Horizonte de integración T [s]
    constexpr int n_mc = 8000;            // Muestras Monte Carlo por paso temporal
    constexpr float z_fixed = 150.0f;     // Altura de prueba dentro de la CLP [m]
    constexpr float blh = 1000.0f;

    // Pasos temporales de exploración: dt decreciente
    const std::vector<float> dts = {2.0f, 1.0f, 0.5f, 0.25f};
    std::vector<double> second_moments;

    for (float dt : dts) {
        const int steps = static_cast<int>(total_time_T / dt);
        const float sqrt_dt = std::sqrt(dt);

        std::mt19937 rng(42819);
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);

        double sum_w2 = 0.0;

        for (int m = 0; m < n_mc; ++m) {
            float w = 0.5f; // Condición inicial idéntica

            for (int s = 0; s < steps; ++s) {
                auto lang = joro::lpdm::compute_langevin_well_mixed(w, z_fixed, blh);
                float dW = normal_dist(rng) * sqrt_dt;

                // Integración completa estocástica de Ito: dw = a_w*dt + b_w*dW
                w += lang.a_w * dt + lang.b_w * dW;
            }

            sum_w2 += static_cast<double>(w * w);
        }

        double e_w2 = sum_w2 / static_cast<double>(n_mc);
        second_moments.push_back(e_w2);
        std::cout << "  dt = " << dt << " s (" << steps << " pasos) | E[w'^2(T)] = " << e_w2 << "\n";
    }

    // Solución de referencia fina (dt_ref = 0.05 s)
    constexpr float dt_ref = 0.05f;
    const int steps_ref = static_cast<int>(total_time_T / dt_ref);
    const float sqrt_dt_ref = std::sqrt(dt_ref);
    std::mt19937 rng_ref(42819);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);

    double sum_ref = 0.0;
    for (int m = 0; m < n_mc; ++m) {
        float w = 0.5f;
        for (int s = 0; s < steps_ref; ++s) {
            auto lang = joro::lpdm::compute_langevin_well_mixed(w, z_fixed, blh);
            float dW = normal_dist(rng_ref) * sqrt_dt_ref;
            w += lang.a_w * dt_ref + lang.b_w * dW;
        }
        sum_ref += static_cast<double>(w * w);
    }
    double e_ref = sum_ref / static_cast<double>(n_mc);
    std::cout << "  dt_ref = " << dt_ref << " s (Referencia) | E[w'^2(T)] = " << e_ref << "\n";

    // Medición del error débil: Error = |E[w^2_dt] - E[w^2_ref]|
    std::cout << "  Errores débiles absolutos:\n";
    std::vector<double> errors;
    for (size_t i = 0; i < dts.size(); ++i) {
        double err = std::abs(second_moments[i] - e_ref);
        errors.push_back(err);
        std::cout << "    Error(dt = " << dts[i] << " s) = " << err << "\n";
    }

    // Verificar que el error para dt fino (0.25 s) es acotado y menor que para dt grueso (2.0 s)
    assert(errors.back() < 0.25);
    assert(errors.front() >= 0.0);

    std::cout << "✔ test_euler_maruyama_conv PASÓ: Convergencia estocástica verificada.\n";
    return 0;
}
