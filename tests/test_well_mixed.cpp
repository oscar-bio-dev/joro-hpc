#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include "test_common.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>

#if defined(_OPENMP)
  #if defined(__has_include)
    #if __has_include(<omp.h>)
      #include <omp.h>
      #define JORO_HAS_OMP_HEADER 1
    #endif
  #else
    #include <omp.h>
    #define JORO_HAS_OMP_HEADER 1
  #endif
#endif

/// ============================================================================
/// FASE 2: Verificación Física del LPDM en 2 Sub-fases (2A y 2B)
/// ============================================================================
int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  JORO-HPC — FASE 2: Verificación Física del LPDM (168 Horas)      ║\n";
    std::cout << "║  Sub-fase 2A: Well-Mixed Pura (vs = 0)                            ║\n";
    std::cout << "║  Sub-fase 2B: Vuelo Real con Sedimentación Aerodinámica (vs > 0)  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";

    constexpr float blh = 1200.0f;                    // Altura de capa límite z_i [m]
    constexpr float total_duration_s = 168.0f * 3600.0f; // 168 horas = 604,800 segundos
    constexpr float dt = 20.0f;                       // Paso temporal dt [s] (estable para tau_L ~ 100s)
    constexpr int n_steps = static_cast<int>(total_duration_s / dt); // 30,240 pasos
    constexpr int n_particles = 10000;                // 10,000 realizaciones
    constexpr int n_bins = 10;                        // 10 bins verticales de 120 m

    // ─────────────────────────────────────────────────────────────────────────
    // SUB-FASE 2A: Condición Well-Mixed Pura de Thomson (1987), vs = 0
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "▶ SUB-FASE 2A: Condición Well-Mixed Pura (vs = 0, Fronteras Reflectantes)\n";
    std::cout << "  Partículas: " << n_particles << " | Duración: 168 horas (" << n_steps 
              << " pasos @ dt = " << dt << " s) | BLH = " << blh << " m\n";

    struct Particle2A {
        float z;
        float w_prime;
    };

    std::vector<Particle2A> ensemble_2a(n_particles);
    {
        std::mt19937 init_rng(42819);
        std::uniform_real_distribution<float> uniform_z(10.0f, blh - 10.0f);
        for (auto& p : ensemble_2a) {
            p.z = uniform_z(init_rng);
            p.w_prime = 0.0f;
        }
    }

    const float sqrt_dt = std::sqrt(dt);

    #pragma omp parallel
    {
        #if defined(JORO_HAS_OMP_HEADER)
        std::mt19937 rng(42819 + omp_get_thread_num() * 10007);
        #else
        std::mt19937 rng(42819);
        #endif
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);

        #pragma omp for schedule(static)
        for (int i = 0; i < n_particles; ++i) {
            auto& p = ensemble_2a[i];
            for (int step = 0; step < n_steps; ++step) {
                auto lang = joro::lpdm::compute_langevin_well_mixed(p.w_prime, p.z, blh);
                float dW = normal_dist(rng) * sqrt_dt;

                p.w_prime += lang.a_w * dt + lang.b_w * dW;
                p.z += p.w_prime * dt; // vs = 0

                // Fronteras reflectantes en superficie (z=0) y techo de capa límite (z=BLH)
                if (p.z < 0.0f) {
                    p.z = -p.z;
                    p.w_prime = -p.w_prime;
                } else if (p.z > blh) {
                    p.z = 2.0f * blh - p.z;
                    p.w_prime = -p.w_prime;
                }
            }
        }
    }

    // Histograma vertical de 10 bins
    std::vector<int> bin_counts(n_bins, 0);
    const float bin_width = blh / static_cast<float>(n_bins);

    for (const auto& p : ensemble_2a) {
        JORO_TEST_ASSERT(!std::isnan(p.z) && !std::isnan(p.w_prime), "NaN detectado en trayectoria Sub-fase 2A");
        int b = static_cast<int>(p.z / bin_width);
        if (b < 0) b = 0;
        if (b >= n_bins) b = n_bins - 1;
        bin_counts[b]++;
    }

    const float expected_per_bin = static_cast<float>(n_particles) / static_cast<float>(n_bins); // 1000
    float chi_squared = 0.0f;

    std::cout << "  Histograma Vertical de 10 Bins (Esperado uniforme = " << expected_per_bin << "):\n";
    for (int b = 0; b < n_bins; ++b) {
        float z_low = b * bin_width;
        float z_high = (b + 1) * bin_width;
        float diff = static_cast<float>(bin_counts[b]) - expected_per_bin;
        chi_squared += (diff * diff) / expected_per_bin;
        std::cout << "    Bin " << std::setw(2) << b << " [" << std::setw(4) << static_cast<int>(z_low)
                  << " - " << std::setw(4) << static_cast<int>(z_high) << " m]: "
                  << std::setw(5) << bin_counts[b] << " partículas\n";
    }

    std::cout << "  Estadístico Chi-cuadrado: " << chi_squared << " (gl = " << (n_bins - 1) << ")\n";

    // Criterios de aceptación:
    // 1. Ausencia de acumulación espuria en el suelo (Bin 0) o techo (Bin 9)
    JORO_TEST_ASSERT(bin_counts[0] > (expected_per_bin * 0.50f), "Fallo 2A: Bin 0 (suelo) subpoblado");
    JORO_TEST_ASSERT(bin_counts[0] < (expected_per_bin * 1.60f), "Fallo 2A: Acumulación artificial en suelo (Bin 0)");
    JORO_TEST_ASSERT(bin_counts[n_bins - 1] > (expected_per_bin * 0.50f), "Fallo 2A: Bin 9 (techo) subpoblado");
    JORO_TEST_ASSERT(bin_counts[n_bins - 1] < (expected_per_bin * 1.60f), "Fallo 2A: Acumulación artificial en techo (Bin 9)");
    JORO_TEST_ASSERT(chi_squared < 60.0f, "Fallo 2A: Chi-cuadrado supera umbral crítico para condición Well-Mixed");
    std::cout << "  ✔ Sub-fase 2A superada con éxito (condición Well-Mixed validada).\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // SUB-FASE 2B: Vuelo Real con Sedimentación Aerodinámica (vs > 0)
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "▶ SUB-FASE 2B: Vuelo Real con Sedimentación (vs > 0, Frontera Inferior Absorbente)\n";
    std::cout << "  Partículas: " << n_particles << " | Juveniles en Baloneo (M = 0.6 mg, Ls = 1.2 m, chi = 0.85)\n";

    struct Particle2B {
        float x_km;       // Desplazamiento horizontal X [km]
        float y_km;       // Desplazamiento horizontal Y [km]
        float z;          // Altura [m]
        float w_prime;    // Turbulencia [m/s]
        float flight_time_s;
        bool landed;
    };

    // Parámetros alométricos de juveniles ninfales de mayo
    constexpr float mass_mg = 0.6f;
    constexpr float silk_m = 1.2f;
    constexpr float chi = 0.85f;
    const float vs_aero = joro::lpdm::compute_settling_velocity(mass_mg, silk_m, chi);
    std::cout << "  Velocidad de sedimentación calculada vs = " << vs_aero << " m/s\n";

    std::vector<Particle2B> ensemble_2b(n_particles);
    {
        std::mt19937 init_rng(84920);
        // Altura de entrada al régimen de baloneo libre tras el impulso convectivo de despegue
        std::uniform_real_distribution<float> takeoff_z(30.0f, 250.0f);
        for (auto& p : ensemble_2b) {
            p.x_km = 0.0f;
            p.y_km = 0.0f;
            p.z = takeoff_z(init_rng);
            p.w_prime = 0.0f;
            p.flight_time_s = 0.0f;
            p.landed = false;
        }
    }

    #pragma omp parallel
    {
        #if defined(JORO_HAS_OMP_HEADER)
        std::mt19937 rng(84920 + omp_get_thread_num() * 10007);
        #else
        std::mt19937 rng(84920);
        #endif
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);

        #pragma omp for schedule(static)
        for (int i = 0; i < n_particles; ++i) {
            auto& p = ensemble_2b[i];

            for (int step = 0; step < n_steps; ++step) {
                if (p.landed) break;

                float sim_time_s = step * dt;
                // Ciclo diurno convectivo en la CLP (térmicas diurnas entre las 10:00 y 18:00)
                float diurnal_phase = std::sin(2.0f * static_cast<float>(M_PI) * (sim_time_s / 86400.0f));
                float w_convective = (diurnal_phase > 0.0f) ? (0.20f * diurnal_phase) : 0.0f;

                // Viento medio horizontal (ERA5 en mayo: ~4.27 m/s)
                float u_wind = 4.0f;  // m/s
                float v_wind = 1.5f;  // m/s
                float w_mean = 0.02f; // m/s ascenso sinóptico débil

                auto lang = joro::lpdm::compute_langevin_well_mixed(p.w_prime, p.z, blh);
                float dW = normal_dist(rng) * sqrt_dt;

                p.w_prime += lang.a_w * dt + lang.b_w * dW;
                p.z += (w_mean + w_convective + p.w_prime - vs_aero) * dt;

                p.x_km += (u_wind * dt) / 1000.0f;
                p.y_km += (v_wind * dt) / 1000.0f;
                p.flight_time_s += dt;

                // Techo de capa límite reflectante
                if (p.z > blh) {
                    p.z = 2.0f * blh - p.z;
                    p.w_prime = -p.w_prime;
                }

                // Frontera inferior absorbente (Aterrizaje en vegetación/suelo)
                if (p.z <= 0.0f) {
                    p.z = 0.0f;
                    p.landed = true;
                    break;
                }
            }
        }
    }

    // Análisis de distancias de salto euclidianas [km]
    std::vector<float> jump_distances_km(n_particles);
    int total_landed = 0;

    for (int i = 0; i < n_particles; ++i) {
        const auto& p = ensemble_2b[i];
        if (p.landed) total_landed++;
        float d = std::sqrt(p.x_km * p.x_km + p.y_km * p.y_km);
        jump_distances_km[i] = d;
    }

    std::sort(jump_distances_km.begin(), jump_distances_km.end());

    double sum_dist = std::accumulate(jump_distances_km.begin(), jump_distances_km.end(), 0.0);
    float mean_dist = static_cast<float>(sum_dist / n_particles);
    float p50 = jump_distances_km[static_cast<size_t>(n_particles * 0.50)];
    float p90 = jump_distances_km[static_cast<size_t>(n_particles * 0.90)];
    float p99 = jump_distances_km[static_cast<size_t>(n_particles * 0.99)];

    float landing_rate_pct = (static_cast<float>(total_landed) / n_particles) * 100.0f;

    std::cout << "  Resultados de Dispersión Sub-fase 2B:\n";
    std::cout << "    Tasa de Asentamiento: " << landing_rate_pct << "% (" << total_landed << " / " << n_particles << ")\n";
    std::cout << "    Distancia Media de Salto: " << mean_dist << " km\n";
    std::cout << "    Cuantil P50 (Mediana):    " << p50 << " km\n";
    std::cout << "    Cuantil P90:              " << p90 << " km\n";
    std::cout << "    Cuantil P99 (Cola larga): " << p99 << " km\n";

    // Validaciones físicas para Trichonephila clavata:
    JORO_TEST_ASSERT(landing_rate_pct > 75.0f, "Fallo 2B: Menos del 75% de partículas aterrizaron tras 168 horas");
    JORO_TEST_ASSERT(mean_dist >= 1.0f && mean_dist <= 150.0f, "Fallo 2B: Distancia media fuera del rango biológico realista");
    JORO_TEST_ASSERT(p50 <= p90 && p90 <= p99, "Fallo 2B: Monotonía de cuantiles violada");
    JORO_TEST_ASSERT(p99 <= 650.0f, "Fallo 2B: P99 excede el límite físico superior de 7 días");

    std::cout << "  ✔ Sub-fase 2B superada con éxito (dispersión aerodinámica biológicamente coherente).\n\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "✔ FASE 2: TODAS LAS PRUEBAS FÍSICAS (2A Y 2B) PASARON SATISFACTORIAMENTE.\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";

    return 0;
}
