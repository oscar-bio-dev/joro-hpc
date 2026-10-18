#include "lpdm/lpdm_engine.hpp"
#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include "common/spatial_grid_lookup.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>

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
        if (spatial_lookup_.empty() || cells_.size() != cells.size()) {
            spatial_lookup_.build(cells_);
        }

        constexpr uint32_t MAX_PARTICLES_PER_CELL = 25000;

        for (const auto& cell : cells) {
            auto it = propagules_per_cell.find(cell.cell_id);
            if (it != propagules_per_cell.end() && it->second > 0) {
                uint32_t total_props = it->second;
                uint32_t n_sim = std::min(total_props, MAX_PARTICLES_PER_CELL);
                uint32_t base_weight = total_props / n_sim;
                uint32_t remainder = total_props % n_sim;

                // Generador para inyección convectiva inicial en despegue
                std::mt19937 init_rng(84920 + cell.cell_id);
                std::uniform_real_distribution<float> takeoff_z(30.0f, 250.0f);
                std::uniform_real_distribution<float> takeoff_w(0.35f, 0.50f);

                for (uint32_t i = 0; i < n_sim; ++i) {
                    Particle3D p;
                    p.x = cell.centroid_lon;
                    p.y = cell.centroid_lat;
                    p.origin_x = cell.centroid_lon;
                    p.origin_y = cell.centroid_lat;
                    p.z = takeoff_z(init_rng); // Altura de entrada al régimen de baloneo libre
                    p.w_prime = takeoff_w(init_rng); // Impulso convectivo ascendente inicial (w'_0 > vs ~0.295 m/s)
                    p.mass_mg = 0.6f;          // Juvenil ninfal típico de mayo
                    p.silk_length_m = 1.2f;    // Línea de baloneo
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
        constexpr float R_earth = 6371000.0f;

        // Velocidad de sedimentación biológica exacta para Trichonephila clavata (~0.295 m/s)
        const float vs_aero = compute_settling_velocity(0.6f, 1.2f, 0.85f);

        auto k_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel
        {
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
                    float sim_time_s = step * dt_seconds;
                    // Ciclo diurno convectivo diurno (térmicas de ascenso)
                    float diurnal_phase = std::sin(2.0f * static_cast<float>(M_PI) * (sim_time_s / 86400.0f));
                    float w_convective = (diurnal_phase > 0.0f) ? (0.20f * diurnal_phase) : 0.0f;

                    // Viento advectivo medio horizontal y vertical sinóptico
                    float u_wind = 4.0f;
                    float v_wind = 1.5f;
                    float w_mean = 0.02f;

                    // Turbulencia vertical de Langevin
                    auto lang = compute_langevin_well_mixed(p.w_prime, p.z, 1200.0f);
                    float dW = normal_dist(rng) * sqrt_dt;
                    p.w_prime += lang.a_w * dt_seconds + lang.b_w * dW;

                    // Cinemática horizontal esférica Euler-Maruyama
                    float cos_lat = std::max(0.01f, std::cos(static_cast<float>(p.y * deg_to_rad)));
                    float d_lon = static_cast<float>((u_wind * dt_seconds / (R_earth * cos_lat)) * rad_to_deg);
                    float d_lat = static_cast<float>((v_wind * dt_seconds / R_earth) * rad_to_deg);

                    p.x += d_lon;
                    p.y += d_lat;
                    p.z += (w_mean + w_convective + p.w_prime - vs_aero) * dt_seconds;

                    // Techo de capa límite reflectante
                    if (p.z > 1200.0f) {
                        p.z = 2400.0f - p.z;
                        p.w_prime = -p.w_prime;
                    }

                    // Frontera inferior absorbente (Aterrizaje en vegetación/suelo)
                    if (p.z <= 0.0f) {
                        p.z = 0.0f;
                        p.is_active = false;
                        break;
                    }
                }
            }
        }
        auto k_end = std::chrono::high_resolution_clock::now();
        result.kernel_time_s = std::chrono::duration<double>(k_end - k_start).count();

        // Reducción espacial multihilo O(1) con SpatialGridLookup
        result.arrivals_per_cell.reserve(4096);
        std::vector<float> landed_distances_km;
        landed_distances_km.reserve(particles_.size());

        uint64_t total_released_acc = 0;
        uint64_t total_landed_acc = 0;
        uint64_t total_escaped_acc = 0;

        #pragma omp parallel
        {
            std::vector<float> local_distances;
            local_distances.reserve(particles_.size() / 8);
            std::unordered_map<int32_t, uint32_t> local_arrivals;
            local_arrivals.reserve(256);

            uint64_t loc_rel = 0;
            uint64_t loc_land = 0;
            uint64_t loc_esc = 0;

            #pragma omp for schedule(static)
            for (size_t i = 0; i < particles_.size(); ++i) {
                const auto& p = particles_[i];
                if (p.weight <= 0.0f) continue;
                uint32_t w = static_cast<uint32_t>(std::round(p.weight));
                loc_rel += w;

                double mid_lat = 0.5 * (p.origin_y + p.y) * deg_to_rad;
                double dx_m = (p.x - p.origin_x) * deg_to_rad * R_earth * std::cos(mid_lat);
                double dy_m = (p.y - p.origin_y) * deg_to_rad * R_earth;
                float dist_km = static_cast<float>(std::sqrt(dx_m * dx_m + dy_m * dy_m) / 1000.0);

                if (!p.is_active) {
                    loc_land += w;
                    local_distances.push_back(dist_km);

                    int32_t dest_cell_id = spatial_lookup_.find_cell(p.x, p.y);
                    if (dest_cell_id != -1) {
                        local_arrivals[dest_cell_id] += w;
                    } else {
                        loc_esc += w;
                    }
                } else {
                    loc_esc += w;
                }
            }

            #pragma omp critical
            {
                total_released_acc += loc_rel;
                total_landed_acc += loc_land;
                total_escaped_acc += loc_esc;
                landed_distances_km.insert(landed_distances_km.end(), local_distances.begin(), local_distances.end());
                for (const auto& [cid, count] : local_arrivals) {
                    result.arrivals_per_cell[cid] += count;
                }
            }
        }

        result.total_released = total_released_acc;
        result.total_landed = total_landed_acc;
        result.total_escaped = total_escaped_acc;

        if (!landed_distances_km.empty()) {
            double sum_d = 0.0;
            for (float d : landed_distances_km) sum_d += d;
            result.mean_jump_distance_km = static_cast<float>(sum_d / landed_distances_km.size());
            size_t p95_idx = static_cast<size_t>(landed_distances_km.size() * 0.95);
            if (p95_idx >= landed_distances_km.size()) p95_idx = landed_distances_km.size() - 1;
            std::nth_element(landed_distances_km.begin(), landed_distances_km.begin() + p95_idx, landed_distances_km.end());
            result.p95_jump_distance_km = landed_distances_km[p95_idx];
        }

        return result;
    }

private:
    std::vector<GridCell> cells_;
    std::vector<Particle3D> particles_;
    SpatialGridLookup spatial_lookup_;
};

std::unique_ptr<LpdmEngine> create_cpu_baseline_engine() {
    return std::make_unique<LpdmCpuBaseline>();
}

} // namespace joro::lpdm
