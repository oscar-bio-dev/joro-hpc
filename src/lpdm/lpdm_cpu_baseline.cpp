#include "lpdm/lpdm_engine.hpp"
#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include <random>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace joro::lpdm {

class LpdmCpuBaseline : public LpdmEngine {
public:
    void initialize_particles(
        const std::vector<GridCell>& cells,
        const std::unordered_map<int32_t, uint32_t>& propagules_per_cell) override 
    {
        particles_.clear();
        for (const auto& cell : cells) {
            auto it = propagules_per_cell.find(cell.cell_id);
            if (it != propagules_per_cell.end() && it->second > 0) {
                uint32_t n = it->second;
                for (uint32_t i = 0; i < n; ++i) {
                    Particle3D p;
                    p.x = cell.centroid_lon;
                    p.y = cell.centroid_lat;
                    p.z = 5.0f; // Altura de despegue inicial (copa arbórea/vegetación) [m]
                    p.w_prime = 0.0f;
                    p.mass_mg = 1.2f;
                    p.silk_length_m = 0.8f;
                    p.posture_chi = 0.85f;
                    p.origin_cell = cell.cell_id;
                    p.is_active = true;
                    particles_.push_back(p);
                }
            }
        }
    }

    DispersionResult run_dispersion(
        const io::AtmosphericField3D& /*atmosphere*/,
        float dt_seconds,
        int total_steps) override 
    {
        DispersionResult result;
        result.total_released = particles_.size();

        if (particles_.empty()) {
            return result;
        }

        const float sqrt_dt = std::sqrt(dt_seconds);

        #pragma omp parallel
        {
            // Generador de números pseudoaleatorios independiente por hilo
            #if defined(_OPENMP)
            std::mt19937 rng(42819 + omp_get_thread_num() * 10007);
            #else
            std::mt19937 rng(42819);
            #endif
            std::normal_distribution<float> norm_dist(0.0f, 1.0f);

            #pragma omp for schedule(guided)
            for (size_t i = 0; i < particles_.size(); ++i) {
                auto& p = particles_[i];
                if (!p.is_active) continue;

                for (int step = 0; step < total_steps; ++step) {
                    // Viento medio aproximado
                    float u_wind = 4.0f;
                    float v_wind = 1.5f;
                    float w_mean = 0.02f;

                    // Velocidad de sedimentación aerodinámica
                    float v_s = compute_settling_velocity(p.mass_mg, p.silk_length_m, p.posture_chi);

                    // Langevin vertical
                    auto lang = compute_langevin_well_mixed(p.w_prime, p.z, 1200.0f);
                    float dW = norm_dist(rng) * sqrt_dt;
                    p.w_prime += lang.a_w * dt_seconds + lang.b_w * dW;

                    // Integración Euler-Maruyama 3D
                    // 1 grado lon/lat ~ 111,000 m * cos(lat)
                    float meters_per_deg_lat = 111320.0f;
                    float meters_per_deg_lon = meters_per_deg_lat * std::cos(p.y * static_cast<float>(M_PI) / 180.0f);

                    p.x += (u_wind * dt_seconds) / meters_per_deg_lon;
                    p.y += (v_wind * dt_seconds) / meters_per_deg_lat;
                    p.z += (w_mean + p.w_prime - v_s) * dt_seconds;

                    // Frontera inferior absorbente (Aterrizaje en vegetación/suelo)
                    if (p.z <= 0.0f) {
                        p.z = 0.0f;
                        p.is_active = false;
                        break;
                    }
                }
            }
        }

        // Reducción secuencial de resultados
        for (const auto& p : particles_) {
            if (!p.is_active) {
                result.total_landed++;
                result.arrivals_per_cell[p.origin_cell]++;
            } else {
                result.total_escaped++;
            }
        }

        return result;
    }

private:
    std::vector<Particle3D> particles_;
};

std::unique_ptr<LpdmEngine> create_cpu_baseline_engine() {
    return std::make_unique<LpdmCpuBaseline>();
}

} // namespace joro::lpdm
