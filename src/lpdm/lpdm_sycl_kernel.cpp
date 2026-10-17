#include "lpdm/lpdm_engine.hpp"
#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include <iostream>
#include <cmath>

#if defined(JORO_HAS_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace joro::lpdm {

class LpdmSyclEngine : public LpdmEngine {
public:
    void initialize_particles(
        const std::vector<GridCell>& cells,
        const std::unordered_map<int32_t, uint32_t>& propagules_per_cell) override 
    {
        cells_ = cells;
        particles_.clear();
        constexpr uint32_t MAX_PARTICLES_PER_CELL = 200;

        for (const auto& cell : cells) {
            auto it = propagules_per_cell.find(cell.cell_id);
            if (it != propagules_per_cell.end() && it->second > 0) {
                uint32_t total_props = it->second;
                uint32_t n_sim = std::min(total_props, MAX_PARTICLES_PER_CELL);
                float weight_per_p = static_cast<float>(total_props) / static_cast<float>(n_sim);

                for (uint32_t i = 0; i < n_sim; ++i) {
                    Particle3D p;
                    p.x = cell.centroid_lon;
                    p.y = cell.centroid_lat;
                    p.z = 5.0f;
                    p.w_prime = 0.0f;
                    p.mass_mg = 1.2f;
                    p.silk_length_m = 0.8f;
                    p.posture_chi = 0.85f;
                    p.weight = weight_per_p;
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
        if (particles_.empty()) return result;

        bool offload_succeeded = false;

#if defined(JORO_HAS_SYCL)
        try {
            sycl::device dev;
            try {
                dev = sycl::device(sycl::gpu_selector_v);
            } catch (...) {
                dev = sycl::device(sycl::default_selector_v);
            }
            sycl::queue q(dev);
            const size_t n_particles = particles_.size();

            // Asignación de memoria en el dispositivo (Device USM)
            Particle3D* d_particles = sycl::malloc_device<Particle3D>(n_particles, q);
            q.memcpy(d_particles, particles_.data(), n_particles * sizeof(Particle3D)).wait();

            const float sqrt_dt = std::sqrt(dt_seconds);

            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(n_particles), [=](sycl::id<1> idx) {
                    Particle3D& p = d_particles[idx];
                    if (!p.is_active) return;

                    float meters_lat = 111320.0f;
                    float u_wind = 4.0f;
                    float v_wind = 1.5f;
                    float w_mean = 0.02f;
                    float v_s = 0.35f;

                    // Estado LCG por partícula
                    uint32_t lcg_state = static_cast<uint32_t>(idx[0] * 1664525u + 42819u);

                    for (int s = 0; s < total_steps; ++s) {
                        // Generador pseudoaleatorio uniforme en GPU
                        lcg_state = lcg_state * 1664525u + 1013904223u;
                        float u1 = static_cast<float>(lcg_state & 0x00FFFFFF) / 16777216.0f;
                        lcg_state = lcg_state * 1664525u + 1013904223u;
                        float u2 = static_cast<float>(lcg_state & 0x00FFFFFF) / 16777216.0f;

                        // Transformada Box-Muller para variable normal estándar N(0, 1)
                        float r = sycl::sqrt(-2.0f * sycl::log(sycl::max(1e-6f, u1)));
                        float theta = 6.2831853f * u2;
                        float dW = (r * sycl::cos(theta)) * sqrt_dt;

                        // Turbulencia vertical de Langevin
                        constexpr float tau_L = 100.0f;
                        constexpr float sigma_w = 0.5f;
                        float a_w = -p.w_prime / tau_L;
                        float b_w = sycl::sqrt(2.0f * sigma_w * sigma_w / tau_L);
                        p.w_prime += a_w * dt_seconds + b_w * dW;

                        // Advección espacial Euler-Maruyama 3D
                        float meters_lon = meters_lat * sycl::cos(p.y * 3.14159265f / 180.0f);
                        p.x += (u_wind * dt_seconds) / meters_lon;
                        p.y += (v_wind * dt_seconds) / meters_lat;
                        p.z += (w_mean + p.w_prime - v_s) * dt_seconds;

                        if (p.z <= 0.0f) {
                            p.z = 0.0f;
                            p.is_active = false;
                            break;
                        }
                    }
                });
            }).wait();

            q.memcpy(particles_.data(), d_particles, n_particles * sizeof(Particle3D)).wait();
            sycl::free(d_particles, q);
            offload_succeeded = true;

        } catch (const sycl::exception& e) {
            std::cerr << "[SYCL Note] Dispositivo SYCL no disponible (" << e.what()
                      << "). Ejecutando fallback local en CPU.\n";
            offload_succeeded = false;
        }
#endif

        if (!offload_succeeded) {
            const float sqrt_dt = std::sqrt(dt_seconds);
            for (size_t i = 0; i < particles_.size(); ++i) {
                auto& p = particles_[i];
                if (!p.is_active) continue;

                float meters_lat = 111320.0f;
                float u_wind = 4.0f;
                float v_wind = 1.5f;
                float w_mean = 0.02f;
                float v_s = 0.35f;

                uint32_t lcg_state = static_cast<uint32_t>(i * 1664525u + 42819u);

                for (int s = 0; s < total_steps; ++s) {
                    lcg_state = lcg_state * 1664525u + 1013904223u;
                    float u1 = static_cast<float>(lcg_state & 0x00FFFFFF) / 16777216.0f;
                    lcg_state = lcg_state * 1664525u + 1013904223u;
                    float u2 = static_cast<float>(lcg_state & 0x00FFFFFF) / 16777216.0f;

                    float r = std::sqrt(-2.0f * std::log(std::max(1e-6f, u1)));
                    float theta = 6.2831853f * u2;
                    float dW = (r * std::cos(theta)) * sqrt_dt;

                    constexpr float tau_L = 100.0f;
                    constexpr float sigma_w = 0.5f;
                    float a_w = -p.w_prime / tau_L;
                    float b_w = std::sqrt(2.0f * sigma_w * sigma_w / tau_L);
                    p.w_prime += a_w * dt_seconds + b_w * dW;

                    float meters_lon = meters_lat * std::cos(p.y * 3.14159265f / 180.0f);
                    p.x += (u_wind * dt_seconds) / meters_lon;
                    p.y += (v_wind * dt_seconds) / meters_lat;
                    p.z += (w_mean + p.w_prime - v_s) * dt_seconds;

                    if (p.z <= 0.0f) {
                        p.z = 0.0f;
                        p.is_active = false;
                        break;
                    }
                }
            }
        }

        // Reducción espacial: acreditar aterrizaje en la celda geográfica de destino real
        for (const auto& p : particles_) {
            uint32_t w = static_cast<uint32_t>(std::round(p.weight));
            result.total_released += w;

            if (!p.is_active) {
                result.total_landed += w;

                int32_t dest_cell_id = -1;
                double min_dist_sq = 1e9;
                for (const auto& c : cells_) {
                    double dx = c.centroid_lon - p.x;
                    double dy = c.centroid_lat - p.y;
                    double d2 = dx * dx + dy * dy;
                    if (d2 < min_dist_sq) {
                        min_dist_sq = d2;
                        dest_cell_id = c.cell_id;
                    }
                }

                // Asignar si aterrizó dentro de la malla demográfica (~15 km = radio ~0.15 deg)
                if (dest_cell_id != -1 && min_dist_sq < 0.02) {
                    result.arrivals_per_cell[dest_cell_id] += w;
                } else {
                    result.total_escaped += w;
                }
            } else {
                result.total_escaped += w;
            }
        }

        return result;
    }

private:
    std::vector<GridCell> cells_;
    std::vector<Particle3D> particles_;
};

std::unique_ptr<LpdmEngine> create_sycl_gpu_engine() {
    return std::make_unique<LpdmSyclEngine>();
}

} // namespace joro::lpdm
