#include "lpdm/lpdm_engine.hpp"
#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include "common/spatial_grid_lookup.hpp"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>
#include <chrono>

#if defined(JORO_HAS_SYCL)
#include <sycl/sycl.hpp>
#endif

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

class LpdmSyclEngine : public LpdmEngine {
public:
    LpdmSyclEngine() {
#if defined(JORO_HAS_SYCL)
        try {
            sycl::device dev;
            try {
                dev = sycl::device(sycl::gpu_selector_v);
            } catch (...) {
                dev = sycl::device(sycl::default_selector_v);
            }
            std::cout << "  [SYCL Accelerator] Dispositivo activo: "
                      << dev.get_info<sycl::info::device::name>()
                      << " | Driver: " << dev.get_info<sycl::info::device::driver_version>()
                      << " (" << (dev.is_gpu() ? "GPU Level-Zero" : "Host") << ")\n";
            is_gpu_ = dev.is_gpu();
            device_desc_ = dev.get_info<sycl::info::device::name>() + " | " +
                           dev.get_platform().get_info<sycl::info::platform::name>() +
                           (is_gpu_ ? " | GPU" : " | NO-GPU");
            q_ = std::make_unique<sycl::queue>(dev, sycl::property::queue::in_order{});
            // Warmup JIT kernel para precompilar pipeline Level-Zero y eliminar latencia inicial
            q_->single_task([=]() {}).wait();
        } catch (const sycl::exception& e) {
            std::cerr << "[SYCL Note] Error al inicializar cola GPU (" << e.what()
                      << "). Fallback a CPU disponible.\n";
        }
#endif
    }

    ~LpdmSyclEngine() override {
#if defined(JORO_HAS_SYCL)
        if (d_particles_ && q_) {
            try {
                sycl::free(d_particles_, *q_);
            } catch (...) {}
            d_particles_ = nullptr;
            d_capacity_ = 0;
        }
#endif
    }

    void initialize_particles(
        const std::vector<GridCell>& cells,
        const std::unordered_map<int32_t, uint32_t>& propagules_per_cell) override 
    {
        const auto t_init0 = std::chrono::steady_clock::now();

        // La malla (256k celdas) solo se copia/indexa cuando cambia: el SpatialGridLookup y la tabla
        // cell_id -> posición se reutilizan en las miles de evaluaciones de una calibración.
        const bool same_grid =
            !cells_.empty() && cells.size() == cells_.size() && cells.data() == grid_ptr_ &&
            cells.front().cell_id == cells_.front().cell_id &&
            cells.back().cell_id == cells_.back().cell_id &&
            cells.front().centroid_lon == cells_.front().centroid_lon &&
            cells.back().centroid_lat == cells_.back().centroid_lat;
        if (!same_grid) {
            cells_ = cells;
            grid_ptr_ = cells.data();
            spatial_lookup_.build(cells_);
            int32_t max_id = -1;
            for (const auto& c : cells_) max_id = std::max(max_id, c.cell_id);
            cell_pos_.assign(static_cast<size_t>(max_id + 1), -1);
            for (size_t i = 0; i < cells_.size(); ++i) {
                if (cells_[i].cell_id >= 0) cell_pos_[static_cast<size_t>(cells_[i].cell_id)] = static_cast<int32_t>(i);
            }
        }

        constexpr uint32_t MAX_PARTICLES_PER_CELL = 25000;

        // Un trabajo por celda emisora; el orden por posición en la malla reproduce exactamente
        // el orden de recorrido original (y por tanto la semilla de cada super-partícula).
        struct Job { size_t pos; uint32_t total; uint32_t n_sim; size_t offset; };
        std::vector<Job> jobs;
        jobs.reserve(propagules_per_cell.size());
        for (const auto& [cid, cnt] : propagules_per_cell) {
            if (cnt == 0) continue;
            const int32_t pos = (cid >= 0 && static_cast<size_t>(cid) < cell_pos_.size())
                                    ? cell_pos_[static_cast<size_t>(cid)] : -1;
            if (pos < 0) continue;
            jobs.push_back({static_cast<size_t>(pos), cnt, std::min(cnt, MAX_PARTICLES_PER_CELL), 0});
        }
        std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.pos < b.pos; });

        size_t total_particles = 0;
        for (auto& j : jobs) { j.offset = total_particles; total_particles += j.n_sim; }

        particles_.clear();
        particles_.resize(total_particles);

        #pragma omp parallel for schedule(dynamic, 16) if(total_particles > 4096)
        for (size_t jb = 0; jb < jobs.size(); ++jb) {
            const Job& job = jobs[jb];
            const GridCell& cell = cells_[job.pos];
            const uint32_t base_weight = job.total / job.n_sim;
            const uint32_t remainder = job.total % job.n_sim;

            // Generador para inyección convectiva idéntico al motor CPU
            std::mt19937 init_rng(84920 + cell.cell_id);
            std::uniform_real_distribution<float> takeoff_z(30.0f, 250.0f);
            std::uniform_real_distribution<float> takeoff_w(0.35f, 0.50f);

            for (uint32_t i = 0; i < job.n_sim; ++i) {
                Particle3D& p = particles_[job.offset + i];
                p.x = cell.centroid_lon;
                p.y = cell.centroid_lat;
                p.origin_x = cell.centroid_lon;
                p.origin_y = cell.centroid_lat;
                p.z = takeoff_z(init_rng); // Altura de entrada al régimen de baloneo libre
                p.w_prime = takeoff_w(init_rng); // Impulso convectivo ascendente inicial (w'_0 > vs ~0.295 m/s)
                p.mass_mg = 0.6f;          // Juvenil ninfal de mayo
                p.silk_length_m = 1.2f;    // Línea de baloneo
                p.posture_chi = 0.85f;
                // Partición entera exacta: suma total de pesos en la celda == total_props
                p.weight = static_cast<float>(base_weight + (i < remainder ? 1 : 0));
                p.origin_cell = cell.cell_id;
                p.is_active = true;
            }
        }

        tel_.init_time_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_init0).count();
    }

    [[nodiscard]] EngineTelemetry telemetry() const override { return tel_; }
    [[nodiscard]] bool is_gpu_offload_active() const noexcept override { return is_gpu_; }
    [[nodiscard]] std::string device_description() const override { return device_desc_; }

    DispersionResult run_dispersion(
        const io::AtmosphericField3D& /*atmosphere*/,
        float dt_seconds,
        int total_steps) override 
    {
        DispersionResult result;
        if (particles_.empty()) return result;

        tel_.dispatches++;
        tel_.particles_dispatched += particles_.size();

        bool offload_succeeded = false;
        const float vs_aero = compute_settling_velocity(0.6f, 1.2f, 0.85f); // ~0.2947 m/s

#if defined(JORO_HAS_SYCL)
        if (q_) {
            try {
                auto& q = *q_;
                const size_t n_particles = particles_.size();

                // Alinear al tamaño de work-group en GPU (ej. 64) mediante dummy padding inactivo
                constexpr size_t WG_SIZE = 64;
                const size_t padded_size = ((n_particles + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

                if (padded_size > d_capacity_) {
                    if (d_particles_) sycl::free(d_particles_, q);
                    d_capacity_ = std::max<size_t>(padded_size + padded_size / 2, size_t{1} << 20);
                    d_particles_ = sycl::malloc_device<Particle3D>(d_capacity_, q);
                }

                if (padded_size > n_particles) {
                    particles_.resize(padded_size);
                    for (size_t i = n_particles; i < padded_size; ++i) {
                        particles_[i].weight = 0.0f;
                        particles_[i].is_active = false;
                        particles_[i].origin_cell = 0;
                    }
                }

                // Transferencia a GPU (Device USM) sin copia intermedia en host
                Particle3D* d_particles = d_particles_;
                const auto t_h2d0 = std::chrono::steady_clock::now();
                q.memcpy(d_particles, particles_.data(), padded_size * sizeof(Particle3D)).wait();
                tel_.h2d_time_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_h2d0).count();

                const float sqrt_dt = std::sqrt(dt_seconds);
                constexpr float deg_to_rad = 3.14159265f / 180.0f;
                constexpr float rad_to_deg = 180.0f / 3.14159265f;
                constexpr float R_earth = 6371000.0f;

                // 1 solo q.submit por lote anual con el bucle temporal completo dentro del kernel
                auto k_start = std::chrono::high_resolution_clock::now();
                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::nd_range<1>(padded_size, WG_SIZE), [=](sycl::nd_item<1> item) {
                        size_t idx = item.get_global_id(0);
                        if (idx >= n_particles) return;

                        Particle3D& p = d_particles[idx];
                        if (!p.is_active || p.weight <= 0.0f) return;

                        float u_wind = 4.0f;
                        float v_wind = 1.5f;
                        float w_mean = 0.02f;

                        // Estado LCG por partícula
                        uint32_t lcg_state = static_cast<uint32_t>(idx * 1664525u + 42819u);

                        for (int s = 0; s < total_steps; ++s) {
                            float sim_time_s = s * dt_seconds;
                            // Ciclo diurno convectivo en la CLP (térmicas de ascenso)
                            float diurnal_phase = sycl::sin(2.0f * 3.14159265f * (sim_time_s / 86400.0f));
                            float w_convective = (diurnal_phase > 0.0f) ? (0.20f * diurnal_phase) : 0.0f;

                            // Generador pseudoaleatorio uniforme en GPU
                            lcg_state = lcg_state * 1664525u + 1013904223u;
                            float u1 = static_cast<float>(lcg_state & 0x00FFFFFF) / 16777216.0f;
                            lcg_state = lcg_state * 1664525u + 1013904223u;
                            float u2 = static_cast<float>(lcg_state & 0x00FFFFFF) / 16777216.0f;

                            // Transformada Box-Muller para variable normal estándar N(0, 1)
                            float r = sycl::sqrt(-2.0f * sycl::log(sycl::max(1e-6f, u1)));
                            float theta = 6.2831853f * u2;
                            float dW = (r * sycl::cos(theta)) * sqrt_dt;

                            // Turbulencia vertical de Langevin (Well-Mixed)
                            constexpr float tau_L = 100.0f;
                            constexpr float sigma_w = 0.5f;
                            float a_w = -p.w_prime / tau_L;
                            float b_w = sycl::sqrt(2.0f * sigma_w * sigma_w / tau_L);
                            p.w_prime += a_w * dt_seconds + b_w * dW;

                            // Cinemática esférica exacta:
                            float cos_lat = sycl::max(0.01f, sycl::cos(p.y * deg_to_rad));
                            float d_lon = (u_wind * dt_seconds / (R_earth * cos_lat)) * rad_to_deg;
                            float d_lat = (v_wind * dt_seconds / R_earth) * rad_to_deg;

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
                    });
                }).wait();
                auto k_end = std::chrono::high_resolution_clock::now();
                result.kernel_time_s = std::chrono::duration<double>(k_end - k_start).count();
                tel_.kernel_time_s += result.kernel_time_s;

                const auto t_d2h0 = std::chrono::steady_clock::now();
                q.memcpy(particles_.data(), d_particles_, n_particles * sizeof(Particle3D)).wait();
                tel_.d2h_time_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_d2h0).count();
                particles_.resize(n_particles);
                offload_succeeded = true;

            } catch (const sycl::exception& e) {
                std::cerr << "[SYCL Note] Dispositivo SYCL error (" << e.what()
                          << "). Ejecutando fallback local en CPU.\n";
                offload_succeeded = false;
            }
        }
#endif

        if (!offload_succeeded) {
            tel_.cpu_fallbacks++;
            const float sqrt_dt = std::sqrt(dt_seconds);
            constexpr float deg_to_rad = 3.14159265f / 180.0f;
            constexpr float rad_to_deg = 180.0f / 3.14159265f;
            constexpr float R_earth = 6371000.0f;

            for (size_t i = 0; i < particles_.size(); ++i) {
                auto& p = particles_[i];
                if (!p.is_active || p.weight <= 0.0f) continue;

                float u_wind = 4.0f;
                float v_wind = 1.5f;
                float w_mean = 0.02f;

                uint32_t lcg_state = static_cast<uint32_t>(i * 1664525u + 42819u);

                for (int s = 0; s < total_steps; ++s) {
                    float sim_time_s = s * dt_seconds;
                    float diurnal_phase = std::sin(2.0f * 3.14159265f * (sim_time_s / 86400.0f));
                    float w_convective = (diurnal_phase > 0.0f) ? (0.20f * diurnal_phase) : 0.0f;

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

                    float cos_lat = std::max(0.01f, std::cos(p.y * deg_to_rad));
                    float d_lon = (u_wind * dt_seconds / (R_earth * cos_lat)) * rad_to_deg;
                    float d_lat = (v_wind * dt_seconds / R_earth) * rad_to_deg;

                    p.x += d_lon;
                    p.y += d_lat;
                    p.z += (w_mean + w_convective + p.w_prime - vs_aero) * dt_seconds;

                    if (p.z > 1200.0f) {
                        p.z = 2400.0f - p.z;
                        p.w_prime = -p.w_prime;
                    }
                    if (p.z <= 0.0f) {
                        p.z = 0.0f;
                        p.is_active = false;
                        break;
                    }
                }
            }
        }

        // Reducción espacial multihilo O(1) con SpatialGridLookup
        const auto t_red0 = std::chrono::steady_clock::now();
        constexpr double deg_to_rad_red = 3.141592653589793 / 180.0;
        constexpr double R_earth_red = 6371000.0;
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
                if (p.weight <= 0.0f) continue; // Descartar dummy padding
                uint32_t w = static_cast<uint32_t>(std::round(p.weight));
                loc_rel += w;

                double mid_lat = 0.5 * (p.origin_y + p.y) * deg_to_rad_red;
                double dx_m = (p.x - p.origin_x) * deg_to_rad_red * R_earth_red * std::cos(mid_lat);
                double dy_m = (p.y - p.origin_y) * deg_to_rad_red * R_earth_red;
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
        tel_.reduce_time_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_red0).count();

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
    const GridCell* grid_ptr_ = nullptr;
    std::vector<int32_t> cell_pos_;   ///< cell_id -> posición en cells_ (-1 si no existe)
    std::vector<Particle3D> particles_;
    SpatialGridLookup spatial_lookup_;
    EngineTelemetry tel_;
    bool is_gpu_ = false;
    std::string device_desc_ = "SYCL no disponible (fallback CPU)";
#if defined(JORO_HAS_SYCL)
    std::unique_ptr<sycl::queue> q_;
    Particle3D* d_particles_ = nullptr;
    size_t d_capacity_ = 0;
#endif
};

std::unique_ptr<LpdmEngine> create_sycl_gpu_engine() {
    return std::make_unique<LpdmSyclEngine>();
}

} // namespace joro::lpdm
