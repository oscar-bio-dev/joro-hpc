#include "lpdm/langevin_turbulence.hpp"
#include "test_common.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

/// Prueba numérica rigurosa de la condición well-mixed de Thomson (1987).
///
/// Un enjambre de partículas inicialmente distribuido de manera uniforme en la
/// capa límite planetaria con turbulencia vertical no homogénea se integra con
/// la ecuación de Langevin de Thomson y fronteras reflectantes.
///
/// Criterio de validación: La densidad espacial euleriana debe permanecer
/// estadísticamente uniforme; las partículas NO deben acumularse en zonas de
/// baja varianza o difusividad (unphysical trapping).
int main() {
    std::cout << "─────────────────────────────────────────────────────────────\n";
    std::cout << "[TEST] test_well_mixed: Ensamble Estocástico de Thomson (1987)\n";
    std::cout << "─────────────────────────────────────────────────────────────\n";

    constexpr float blh = 1000.0f;       // Altura de la capa límite z_i [m]
    constexpr float dt = 2.0f;          // Paso de integración [s]
    constexpr int n_steps = 150;        // Pasos de advección turbulenta
    constexpr int n_particles = 5000;   // Número de realizaciones del ensamble
    constexpr int n_bins = 5;           // Intervalos de la columna vertical

    std::mt19937 rng(42819);
    std::uniform_real_distribution<float> uniform_z(10.0f, blh - 10.0f);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);

    struct ParticleState {
        float z;
        float w_p;
    };

    std::vector<ParticleState> ensemble(n_particles);
    for (auto& p : ensemble) {
        p.z = uniform_z(rng);
        p.w_p = 0.0f;
    }

    const float sqrt_dt = std::sqrt(dt);

    // Integración temporal de Langevin
    for (int step = 0; step < n_steps; ++step) {
        for (auto& p : ensemble) {
            auto lang = joro::lpdm::compute_langevin_well_mixed(p.w_p, p.z, blh);
            float dW = normal_dist(rng) * sqrt_dt;

            p.w_p += lang.a_w * dt + lang.b_w * dW;
            p.z += p.w_p * dt;

            // Fronteras reflectantes en superficie (z=0) y tope de capa límite (z=blh)
            if (p.z < 0.0f) {
                p.z = -p.z;
                p.w_p = -p.w_p;
            } else if (p.z > blh) {
                p.z = 2.0f * blh - p.z;
                p.w_p = -p.w_p;
            }
        }
    }

    // Evaluación del histograma vertical de alturas
    std::vector<int> bin_counts(n_bins, 0);
    const float bin_width = blh / static_cast<float>(n_bins);

    for (const auto& p : ensemble) {
        int b = static_cast<int>(p.z / bin_width);
        if (b < 0) b = 0;
        if (b >= n_bins) b = n_bins - 1;
        bin_counts[b]++;
    }

    const float expected_per_bin = static_cast<float>(n_particles) / static_cast<float>(n_bins);
    float chi_squared = 0.0f;

    std::cout << "  Distribución vertical final (esperado por bin = " << expected_per_bin << "):\n";
    for (int b = 0; b < n_bins; ++b) {
        float z_low = b * bin_width;
        float z_high = (b + 1) * bin_width;
        float diff = static_cast<float>(bin_counts[b]) - expected_per_bin;
        chi_squared += (diff * diff) / expected_per_bin;
        std::cout << "    Bin " << b << " [" << z_low << " - " << z_high << " m]: "
                  << bin_counts[b] << " partículas\n";
    }

    std::cout << "  Estadístico Chi-cuadrado: " << chi_squared << " (gl = " << (n_bins - 1) << ")\n";

    // Criterio de aceptación: sin segregación ni colapso de partículas en el suelo o tope
    for (int b = 0; b < n_bins; ++b) {
        JORO_TEST_ASSERT(bin_counts[b] > (expected_per_bin * 0.40f), "Fallo de uniformidad: bin con subpoblación excesiva");
        JORO_TEST_ASSERT(bin_counts[b] < (expected_per_bin * 1.80f), "Fallo de uniformidad: acumulación artificial de partículas");
    }
    JORO_TEST_ASSERT(chi_squared < 70.0f, "El estadístico Chi-cuadrado supera el umbral crítico para distribución uniforme");

    std::cout << "✔ test_well_mixed PASÓ: Densidad euleriana preservada correctamente.\n";
    return 0;
}
