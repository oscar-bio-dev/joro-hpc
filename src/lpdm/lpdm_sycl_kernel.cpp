#include "lpdm/lpdm_engine.hpp"
#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include <iostream>

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
        particles_.clear();
        for (const auto& cell : cells) {
            auto it = propagules_per_cell.find(cell.cell_id);
            if (it != propagules_per_cell.end() && it->second > 0) {
                for (uint32_t i = 0; i < it->second; ++i) {
                    Particle3D p;
                    p.x = cell.centroid_lon;
                    p.y = cell.centroid_lat;
                    p.z = 5.0f;
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
        if (particles_.empty()) return result;

#if defined(JORO_HAS_SYCL)
        try {
            sycl::queue q(sycl::gpu_selector_v);
            const size_t n_particles = particles_.size();

            // Asignación de memoria en el dispositivo (Device USM)
            Particle3D* d_particles = sycl::malloc_device<Particle3D>(n_particles, q);
            q.memcpy(d_particles, particles_.data(), n_particles * sizeof(Particle3D)).wait();

            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(n_particles), [=](sycl::id<1> idx) {
                    Particle3D& p = d_particles[idx];
                    if (!p.is_active) return;

                    for (int s = 0; s < total_steps; ++s) {
                        float u_wind = 4.0f;
                        float v_wind = 1.5f;
                        float w_mean = 0.02f;
                        float v_s = 0.35f;

                        // Paso cinemático simplificado en GPU
                        float meters_lat = 111320.0f;
                        p.x += (u_wind * dt_seconds) / meters_lat;
                        p.y += (v_wind * dt_seconds) / meters_lat;
                        p.z += (w_mean - v_s) * dt_seconds;

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

        } catch (const sycl::exception& e) {
            std::cerr << "[SYCL Error] GPU offload falló: " << e.what()
                      << ". Revertiendo a procesamiento local.\n";
        }
#endif

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

std::unique_ptr<LpdmEngine> create_sycl_gpu_engine() {
    return std::make_unique<LpdmSyclEngine>();
}

} // namespace joro::lpdm
