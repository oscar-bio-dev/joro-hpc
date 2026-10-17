#include "lpdm/lpdm_engine.hpp"
#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include <random>
#include <iostream>

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

namespace joro::lpdm {

class LpdmCpuBaseline : public LpdmEngine {
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
                uint32_t base_weight = total_props / n_sim;
                uint32_t remainder = total_props % n_sim;

                for (uint32_t i = 0; i < n_sim; ++i) {
                    Particle3D p;
                    p.x = cell.centroid_lon;
                    p.y = cell.centroid_lat;
                    p.origin_x = cell.centroid_lon;
                    p.origin_y = cell.centroid_lat;
                    p.z = 5.0f; // Altura de despegue inicial (copa arbórea/vegetación) [m]
                    p.w_prime = 0.0f;
                    p.mass_mg = 1.2f;
                    p.silk_length_m = 0.8f;
                    p.posture_chi = 0.85f;
                    // Partición entera exacta: la suma sobre las n_sim partículas es estrictamente total_props
                    p.weight = static_cast<float>(base_weight + (i < remainder ? 1 : 0));
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
        if (particles_.empty()) {
            return result;
        }

        const float sqrt_dt = std::sqrt(dt_seconds);
        constexpr double deg_to_rad = M_PI / 180.0;
        constexpr double rad_to_deg = 180.0 / M_PI;

        #pragma omp parallel
        {
            // Generador de números pseudoaleatorios independiente por hilo
            #if defined(JORO_HAS_OMP_HEADER)
            std::mt19937 rng(42819 + omp_get_thread_num() * 10007);
            #else
            std::mt19937 rng(42819);
            #endif
            std::normal_distribution<float> normal_dist(0.0f, 1.0f);

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
                    float dW = normal_dist(rng) * sqrt_dt;
                    p.w_prime += lang.a_w * dt_seconds + lang.b_w * dW;

                    // Integración Euler-Maruyama 3D considerando curvatura terrestre esférica
                    // dLon = (U * dt / (R_earth * cos(lat))) * (180 / pi)
                    // dLat = (V * dt / R_earth) * (180 / pi)
                    float cos_lat = std::max(0.01f, std::cos(static_cast<float>(p.y * deg_to_rad)));
                    float d_lon = static_cast<float>((u_wind * dt_seconds / (constants::R_earth * cos_lat)) * rad_to_deg);
                    float d_lat = static_cast<float>((v_wind * dt_seconds / constants::R_earth) * rad_to_deg);

                    p.x += d_lon;
                    p.y += d_lat;
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

        // Reducción espacial: acreditar aterrizaje en la celda geográfica de destino real
        std::vector<float> landed_distances_km;
        for (const auto& p : particles_) {
            if (p.weight <= 0.0f) continue;
            uint32_t w = static_cast<uint32_t>(std::round(p.weight));
            result.total_released += w;

            // Distancia de salto respecto a la celda de origen
            double mid_lat = 0.5 * (p.origin_y + p.y) * deg_to_rad;
            double dx_m = (p.x - p.origin_x) * deg_to_rad * constants::R_earth * std::cos(mid_lat);
            double dy_m = (p.y - p.origin_y) * deg_to_rad * constants::R_earth;
            float dist_km = static_cast<float>(std::sqrt(dx_m * dx_m + dy_m * dy_m) / 1000.0);

            if (!p.is_active) {
                result.total_landed += w;
                landed_distances_km.push_back(dist_km);

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

                // Asignar si aterrizó dentro de la malla demográfica (~15 km radio = 0.02 deg^2)
                if (dest_cell_id != -1 && min_dist_sq < 0.02) {
                    result.arrivals_per_cell[dest_cell_id] += w;
                } else {
                    result.total_escaped += w;
                }
            } else {
                result.total_escaped += w;
            }
        }

        if (!landed_distances_km.empty()) {
            std::sort(landed_distances_km.begin(), landed_distances_km.end());
            double sum_d = 0.0;
            for (float d : landed_distances_km) sum_d += d;
            result.mean_jump_distance_km = static_cast<float>(sum_d / landed_distances_km.size());
            size_t p95_idx = static_cast<size_t>(landed_distances_km.size() * 0.95);
            if (p95_idx >= landed_distances_km.size()) p95_idx = landed_distances_km.size() - 1;
            result.p95_jump_distance_km = landed_distances_km[p95_idx];
        }

        return result;
    }

private:
    std::vector<GridCell> cells_;
    std::vector<Particle3D> particles_;
};

std::unique_ptr<LpdmEngine> create_cpu_baseline_engine() {
    return std::make_unique<LpdmCpuBaseline>();
}

} // namespace joro::lpdm
