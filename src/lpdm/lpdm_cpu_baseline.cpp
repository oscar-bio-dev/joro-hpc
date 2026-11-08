#include "lpdm/lpdm_engine.hpp"
#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include "common/spatial_grid_lookup.hpp"
#include "lpdm/era5_view.hpp"
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
    bool set_atmosphere(const io::AtmosphericField3D& atmosphere) override {
        clear_atmospheres();
        return add_atmosphere(atmosphere);
    }

    void clear_atmospheres() override {
        eras_.clear();
        active_ = 0;
        atmosphere_resident_ = false;
    }

    bool add_atmosphere(const io::AtmosphericField3D& atmosphere) override {
        Era5Pack pack;
        if (!pack.build(atmosphere)) return false;
        eras_.push_back(std::move(pack));
        eras_.back().refresh_view();
        active_ = eras_.size() - 1;
        atmosphere_resident_ = true;
        return true;
    }

    bool select_atmosphere(size_t idx) override {
        if (idx >= eras_.size()) return false;
        active_ = idx;
        return true;
    }

    [[nodiscard]] size_t atmosphere_count() const noexcept override { return eras_.size(); }

    [[nodiscard]] bool is_atmosphere_resident() const noexcept override { return atmosphere_resident_; }

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
        const Era5View atm = (atmosphere_resident_ && !eras_.empty()) ? eras_[active_].view : Era5View{};

        auto k_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel
        {
            #if defined(JORO_HAS_OMP_HEADER)
            std::mt19937 rng(42819 + omp_get_thread_num() * 10007);
            #else
            std::mt19937 rng(42819);
            #endif
            std::normal_distribution<float> normal_dist(0.0f, 1.0f);
            std::uniform_real_distribution<float> unif01_dist(0.0f, 1.0f);
            auto unif01 = [&](std::mt19937& g) { return unif01_dist(g); };

            #pragma omp for schedule(guided)
            for (size_t i = 0; i < particles_.size(); ++i) {
                auto& p = particles_[i];
                if (!p.is_active) continue;

                // Hora de liberación dentro de la ventana ERA5 (igual que el kernel GPU)
                float t0_h = 0.0f;
                if (atm.valid) {
                    const float span = static_cast<float>(atm.n_t - 1) - static_cast<float>(total_steps) * dt_seconds / 3600.0f;
                    t0_h = span > 0.0f ? span * unif01(rng) : 0.0f;
                }

                for (int step = 0; step < total_steps; ++step) {
                    float sim_time_s = step * dt_seconds;
                    // Ciclo diurno convectivo diurno (térmicas de ascenso)
                    float diurnal_phase = std::sin(2.0f * static_cast<float>(M_PI) * (sim_time_s / 86400.0f));
                    float w_convective = (diurnal_phase > 0.0f) ? (0.20f * diurnal_phase) : 0.0f;

                    // Viento advectivo medio horizontal y vertical sinóptico (ERA5 residente o uniforme heredado)
                    float u_wind = 4.0f;
                    float v_wind = 1.5f;
                    float w_mean = 0.02f;
                    float blh = 1200.0f;
                    if (atm.valid) {
                        float b = 0.0f;
                        atm.sample(p.x, p.y, p.z, t0_h + sim_time_s / 3600.0f, u_wind, v_wind, w_mean, b);
                        blh = std::clamp(b, 250.0f, 3000.0f);
                    }

                    // Turbulencia vertical de Langevin
                    auto lang = compute_langevin_well_mixed(p.w_prime, p.z, blh);
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
                    if (p.z > blh) {
                        p.z = 2.0f * blh - p.z;
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
        std::vector<std::pair<float, uint32_t>> landed_distances_km; // (distancia, propágulos representados)
        landed_distances_km.reserve(particles_.size());

        uint64_t total_released_acc = 0;
        uint64_t total_landed_acc = 0;
        uint64_t total_escaped_acc = 0;

        #pragma omp parallel
        {
            std::vector<std::pair<float, uint32_t>> local_distances;
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
                    local_distances.emplace_back(dist_km, w);

                    int32_t dest_cell_id = spatial_lookup_.find_cell(p.x, p.y);
                    if (dest_cell_id != -1) {
                        local_arrivals[dest_cell_id] += w;
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
            double sum_wd = 0.0;
            double sum_w = 0.0;
            for (const auto& [d, w] : landed_distances_km) { sum_wd += static_cast<double>(d) * w; sum_w += w; }
            result.mean_jump_distance_km = static_cast<float>(sum_wd / sum_w);
            std::sort(landed_distances_km.begin(), landed_distances_km.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            const double target = 0.95 * sum_w;
            double cum = 0.0;
            result.p95_jump_distance_km = landed_distances_km.back().first;
            for (const auto& [d, w] : landed_distances_km) {
                cum += w;
                if (cum >= target) { result.p95_jump_distance_km = d; break; }
            }
        }

        return result;
    }

private:
    std::vector<GridCell> cells_;
    std::vector<Particle3D> particles_;
    SpatialGridLookup spatial_lookup_;
    std::vector<Era5Pack> eras_;
    size_t active_ = 0;
    bool atmosphere_resident_ = false;
};

std::unique_ptr<LpdmEngine> create_cpu_baseline_engine() {
    return std::make_unique<LpdmCpuBaseline>();
}

} // namespace joro::lpdm
