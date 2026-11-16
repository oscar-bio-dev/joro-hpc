#include "lpdm/lpdm_engine.hpp"
#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include "lpdm/superparticle_policy.hpp"
#include "lpdm/lpdm_trace.hpp"
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
        if (!pack.build(atmosphere, time_stride_)) return false;
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

    [[nodiscard]] size_t atmosphere_host_bytes() const noexcept override {
        size_t b = 0;
        for (const auto& e : eras_) b += e.bytes();
        return b;
    }

    void set_emergence_offsets(const std::unordered_map<int32_t, float>& start_hours) override { emergence_ = start_hours; }
    void set_era5_time_stride(int hours) override { time_stride_ = std::max(1, hours); }
    void set_mc_seed(uint32_t mc_seed) override { mc_seed_ = mc_seed; }
    void set_year_index(uint32_t year_idx) override { year_idx_override_ = year_idx; has_year_override_ = true; }
    void set_max_super_particles(size_t budget) override { max_super_ = budget; }

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

        // Política unificada de tope y reescalado de pesos (M7): idéntica a la del motor SYCL
        std::vector<const GridCell*> emit_cells;
        std::vector<uint32_t> totals;
        for (const auto& cell : cells) {
            auto it = propagules_per_cell.find(cell.cell_id);
            if (it != propagules_per_cell.end() && it->second > 0) {
                emit_cells.push_back(&cell);
                totals.push_back(it->second);
            }
        }
        std::vector<uint32_t> n_sim_plan;
        plan_super_particles(totals, max_super_, n_sim_plan);

        for (size_t k = 0; k < emit_cells.size(); ++k) {
            const GridCell& cell = *emit_cells[k];
            const uint32_t total_props = totals[k];
            const uint32_t n_sim = n_sim_plan[k];
            const uint32_t base_weight = total_props / n_sim;
            const uint32_t remainder = total_props % n_sim;

            // Generador para inyección convectiva inicial en despegue y rasgos alométricos individuales (M6)
            std::mt19937 init_rng((84920u + static_cast<uint32_t>(cell.cell_id)) ^ mc_seed_);
            std::uniform_real_distribution<float> takeoff_z(30.0f, 250.0f);
            std::uniform_real_distribution<float> takeoff_w(0.35f, 0.50f);
            std::uniform_real_distribution<float> mass_dist(0.4f, 0.9f);
            std::uniform_real_distribution<float> silk_dist(0.8f, 1.6f);

            for (uint32_t i = 0; i < n_sim; ++i) {
                Particle3D p;
                p.x = cell.centroid_lon;
                p.y = cell.centroid_lat;
                p.origin_x = cell.centroid_lon;
                p.origin_y = cell.centroid_lat;
                p.z = takeoff_z(init_rng); // Altura de entrada al régimen de baloneo libre
                p.w_prime = takeoff_w(init_rng); // Impulso convectivo ascendente inicial (w'_0 > vs ~0.295 m/s)
                p.mass_mg = mass_dist(init_rng);      // Juvenil ninfal de mayo: M_j en [0.4, 0.9] mg
                p.silk_length_m = silk_dist(init_rng); // Longitud efectiva de seda L_s,j
                p.posture_chi = 0.85f;
                // Partición entera exacta: la suma sobre las n_sim partículas es estrictamente total_props
                p.weight = static_cast<float>(base_weight + (i < remainder ? 1 : 0));
                p.origin_cell = cell.cell_id;
                p.is_active = true;
                if (auto it = emergence_.find(cell.cell_id); it != emergence_.end()) p.start_h = it->second;
                particles_.push_back(p);
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

        constexpr double deg_to_rad = M_PI / 180.0;
        constexpr double rad_to_deg = 180.0 / M_PI;
        constexpr float R_earth = 6371000.0f;

        // Cada partícula usa su propia M_j, L_s,j (M6)
        const Era5View atm = (atmosphere_resident_ && !eras_.empty()) ? eras_[active_].view : Era5View{};
        const uint32_t year_idx = has_year_override_ ? year_idx_override_ : static_cast<uint32_t>(active_);
        const float u_crit = kUCritTakeoffDefault;
        const float flight_h = static_cast<float>(total_steps) * dt_seconds / 3600.0f;
        const float span_h = atm.valid ? static_cast<float>(atm.n_hours - 1) - flight_h : 0.0f;

        auto k_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel
        {
            // Semilla dinámica (M8): año y réplica Monte Carlo + hilo
            #if defined(JORO_HAS_OMP_HEADER)
            const uint32_t thread_id = static_cast<uint32_t>(omp_get_thread_num());
            #else
            const uint32_t thread_id = 0u;
            #endif
            std::mt19937 rng((42819u ^ (year_idx * 0x85EBCA6Bu) ^ mc_seed_) + thread_id * 10007u);
            std::normal_distribution<float> normal_dist(0.0f, 1.0f);
            std::uniform_real_distribution<float> unif01_dist(0.0f, 1.0f);
            auto unif01 = [&](std::mt19937& g) { return unif01_dist(g); };

            #pragma omp for schedule(guided)
            for (size_t i = 0; i < particles_.size(); ++i) {
                auto& p = particles_[i];
                if (!p.is_active) continue;

                const float v_sed = compute_settling_velocity(p.mass_mg, p.silk_length_m, p.posture_chi);

                // Hora y día de despegue con compuerta biomecánica u10 < u_crit (M3), igual que el kernel GPU:
                // hasta 4 días candidatos distintos (+1 a +3 d entre reintentos) antes de la retención local
                float t0_h = 0.0f;
                bool takeoff_ok = !atm.valid;
                float u_day0 = 0.0f;
                int day_offset = 0;
                if (!atm.valid) {
                    const float ud = unif01(rng);
                    const float uh = unif01(rng);
                    t0_h = candidate_takeoff_hour(p.x, span_h, false, ud, uh, 0, 0.0f);
                }
                for (int a = 0; a < kTakeoffMaxAttempts && !takeoff_ok; ++a) {
                    if (a == 0) u_day0 = unif01(rng); else day_offset += takeoff_retry_day_step(unif01(rng));
                    const float cand = candidate_takeoff_hour(p.x, span_h, true, u_day0, unif01(rng), day_offset, p.start_h);
                    if (cand < 0.0f) continue;
                    float ws10 = 0.0f, blh0 = 0.0f;
                    atm.sample_surface(p.origin_x, p.origin_y, cand, ws10, blh0);
                    if (ws10 < u_crit) { t0_h = cand; takeoff_ok = true; }
                }
                if (!takeoff_ok) {
                    // Viento fuerte en todos los días candidatos: retención local (sin vuelo distante)
                    p.z = 0.0f;
                    p.is_active = false;
                    p.retained = true;
                    continue;
                }

                for (int step = 0; step < total_steps; ++step) {
                    const float sim_time_s = step * dt_seconds;
                    const float t_h = t0_h + sim_time_s / 3600.0f;
                    // Fase convectiva con hora solar local real (M2)
                    const float phase = diurnal_phase(local_solar_hour(t_h, p.x));

                    // Viento advectivo medio horizontal y vertical sinóptico (ERA5 residente o uniforme heredado)
                    float u_wind = 4.0f;
                    float v_wind = 1.5f;
                    float w_mean = 0.02f;
                    float blh = 1200.0f;
                    if (atm.valid) {
                        float b = 0.0f;
                        atm.sample(p.x, p.y, p.z, t_h, u_wind, v_wind, w_mean, b);
                        blh = std::clamp(b, 250.0f, 3000.0f);
                    }

                    // Turbulencia vertical de Langevin inhomogénea con deriva de Thomson (M4)
                    const float speed = std::sqrt(u_wind * u_wind + v_wind * v_wind);
                    p.w_prime = langevin_advance(p.w_prime, p.z, blh, friction_velocity(speed),
                                                 convective_velocity_scale(phase, blh), dt_seconds,
                                                 normal_dist(rng));
                    const float w_convective = convective_updraft(phase, p.z, blh);

                    // Cinemática horizontal esférica Euler-Maruyama
                    float cos_lat = std::max(0.01f, std::cos(static_cast<float>(p.y * deg_to_rad)));
                    float d_lon = static_cast<float>((u_wind * dt_seconds / (R_earth * cos_lat)) * rad_to_deg);
                    float d_lat = static_cast<float>((v_wind * dt_seconds / R_earth) * rad_to_deg);

                    p.x += d_lon;
                    p.y += d_lat;
                    p.z += (w_mean + w_convective + p.w_prime - v_sed) * dt_seconds;

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
        uint64_t total_retained_acc = 0;

        #pragma omp parallel
        {
            std::vector<std::pair<float, uint32_t>> local_distances;
            local_distances.reserve(particles_.size() / 8);
            std::unordered_map<int32_t, uint32_t> local_arrivals;
            local_arrivals.reserve(256);

            uint64_t loc_rel = 0;
            uint64_t loc_land = 0;
            uint64_t loc_esc = 0;
            uint64_t loc_ret = 0;

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
                    if (p.retained) {
                        // Retención local: cuenta en el balance demográfico (celda de origen) pero NO en los
                        // diagnósticos de distancia de vuelo efectivo
                        loc_ret += w;
                        local_arrivals[p.origin_cell] += w;
                        continue;
                    }
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
                total_retained_acc += loc_ret;
                landed_distances_km.insert(landed_distances_km.end(), local_distances.begin(), local_distances.end());
                for (const auto& [cid, count] : local_arrivals) {
                    result.arrivals_per_cell[cid] += count;
                }
            }
        }

        result.total_released = total_released_acc;
        result.total_landed = total_landed_acc;
        result.total_escaped = total_escaped_acc;
        result.total_retained = total_retained_acc;

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
    size_t max_super_ = kDefaultMaxSuperPerDispatch;
    int time_stride_ = []() { const char* e = std::getenv("JORO_ERA5_TIME_STRIDE"); return e ? std::max(1, std::atoi(e)) : 1; }();
    uint32_t mc_seed_ = 0;
    uint32_t year_idx_override_ = 0;
    bool has_year_override_ = false;
    std::unordered_map<int32_t, float> emergence_;   ///< Inicio de la ventana de despegue por celda (h tras el 1 de mayo)
};

std::unique_ptr<LpdmEngine> create_cpu_baseline_engine() {
    return std::make_unique<LpdmCpuBaseline>();
}

} // namespace joro::lpdm
