#include "lpdm/lpdm_engine.hpp"
#include "lpdm/lpdm_trace.hpp"
#include "lpdm/settling_velocity.hpp"
#include "lpdm/superparticle_policy.hpp"
#include "common/spatial_grid_lookup.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

#if defined(JORO_HAS_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace joro::lpdm {

namespace {

constexpr uint64_t kMaxPropagules = 4000000000ull; ///< Los contadores de llegada son uint32
#if defined(JORO_HAS_SYCL)
constexpr size_t kWgSize = 128;
#endif
constexpr double kDefaultVramBudgetGiB = 3.8;   ///< Tope de VRAM para campos ERA5 en Device USM (Arc A380: 6 GB)
[[maybe_unused]] constexpr size_t kMaxStreamSlots = 2; ///< Años simultáneos máximos en VRAM cuando se hace streaming

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

} // namespace

/// Campo ERA5 de un año: copia empaquetada FP16 residente en la RAM del host (metadatos de la vista).
struct AtmSlot {
    Era5Pack host;
};

/// Búfer reutilizable de VRAM (Device USM) donde se copia el año requerido antes de despachar el kernel.
/// Contiene el viento FP16 y las variables superficiales contiguas [blh | zground | ws10].
struct DeviceSlot {
    Wind4Half* wind = nullptr;
    size_t wind_cap_bytes = 0;
    float* surf = nullptr;      ///< 3 * n_t * n_lat * n_lon floats
    size_t surf_cap_floats = 0;
    int owner = -1;             ///< Índice de atmósfera residente (-1 = libre)
    uint64_t last_use = 0;
};

/// Motor LPDM sobre GPU (SYCL / Level-Zero) con bucle de partículas "zero-copy":
///  * al dispositivo solo viaja la tabla compacta de celdas emisoras (EmissionJob, ~24 B por celda);
///  * cada work-item toma su super-partícula, integra la trayectoria completa en registros
///    (Philox 4x32-10 por hilo), localiza la celda de aterrizaje con el SpatialGridLookup alojado en
///    Device USM y acumula con atomic_ref en d_arrivals_[cell_id];
///  * al terminar solo regresa el arreglo compacto de llegadas (1 uint32 por celda) y unos parciales.
class LpdmSyclEngine : public LpdmEngine {
public:
    LpdmSyclEngine() {
        if (const char* env = std::getenv("JORO_MAX_SUPER")) {
            max_super_ = static_cast<size_t>(std::strtoull(env, nullptr, 10));
        }
        double budget_gib = kDefaultVramBudgetGiB;
        if (const char* env = std::getenv("JORO_VRAM_BUDGET_GIB")) budget_gib = std::strtod(env, nullptr);
        vram_budget_ = static_cast<size_t>(budget_gib * 1024.0 * 1024.0 * 1024.0);
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
            max_alloc_bytes_ = static_cast<size_t>(dev.get_info<sycl::info::device::max_mem_alloc_size>());
            q_ = std::make_unique<sycl::queue>(dev, sycl::property::queue::in_order{});
            // Warmup JIT kernel para precompilar pipeline Level-Zero y eliminar latencia inicial
            q_->single_task([=]() {}).wait();
        } catch (const sycl::exception& e) {
            std::cerr << "[SYCL Note] Error al inicializar cola GPU (" << e.what()
                      << "). Fallback a CPU disponible.\n";
        }
#endif
    }

    ~LpdmSyclEngine() override { free_device(); }

    void set_max_super_particles(size_t budget) override { max_super_ = budget; }

    bool set_atmosphere(const io::AtmosphericField3D& atmosphere) override {
        clear_atmospheres();
        return add_atmosphere(atmosphere);
    }

    void clear_atmospheres() override {
#if defined(JORO_HAS_SYCL)
        free_device_slots();
#endif
        atms_.clear();
        active_atm_ = 0;
        atmosphere_resident_ = false;
    }

    bool add_atmosphere(const io::AtmosphericField3D& atmosphere) override {
        const auto t0 = Clock::now();
        AtmSlot slot;
        if (!slot.host.build(atmosphere, time_stride_)) return false;
        slot.host.refresh_view();
        atms_.push_back(std::move(slot));
        atms_.back().host.refresh_view();
        active_atm_ = atms_.size() - 1;
        atmosphere_resident_ = true;
        const AtmSlot& s = atms_.back();
        bool on_device = false;
#if defined(JORO_HAS_SYCL)
        if (q_) {
            try {
                on_device = ensure_on_device(active_atm_) != nullptr;
            } catch (const std::exception& e) {
                std::cerr << "[SYCL Note] No se pudo subir ERA5 al dispositivo (" << e.what()
                          << "); el transporte usará el campo en host.\n";
            }
        }
#endif
        std::cout << "  [ERA5->FP16] Campo atmosférico #" << atms_.size() << ": " << s.host.view.n_hours << " h (paso " << static_cast<int>(std::lround(1.0f / s.host.view.inv_stride)) << " h, " << s.host.view.n_t << " pasos) x "
                  << s.host.view.n_lev << " niveles x " << s.host.view.n_lat << " x " << s.host.view.n_lon
                  << " | host " << (static_cast<double>(s.host.bytes()) / 1.048576e6) << " MB (FP16)"
                  << (on_device ? " | en Device USM" : " | en RAM host (VRAM bajo demanda)")
                  << " | carga " << seconds_since(t0) << " s\n";
        return true;
    }

    bool select_atmosphere(size_t idx) override {
        if (idx >= atms_.size()) return false;
        active_atm_ = idx;
        return true;
    }

    [[nodiscard]] size_t atmosphere_count() const noexcept override { return atms_.size(); }

    [[nodiscard]] bool is_atmosphere_resident() const noexcept override { return atmosphere_resident_; }

    [[nodiscard]] size_t atmosphere_host_bytes() const noexcept override { return total_host_bytes(); }

    [[nodiscard]] size_t atmosphere_device_bytes() const noexcept override {
#if defined(JORO_HAS_SYCL)
        size_t b = 0;
        for (const auto& d : dslots_) b += d.wind_cap_bytes + d.surf_cap_floats * sizeof(float);
        return b;
#else
        return 0;
#endif
    }

    void set_emergence_offsets(const std::unordered_map<int32_t, float>& start_hours) override { emergence_ = start_hours; }
    void set_era5_time_stride(int hours) override { time_stride_ = std::max(1, hours); }
    void set_mc_seed(uint32_t mc_seed) override { mc_seed_ = mc_seed; }
    void set_year_index(uint32_t year_idx) override { year_idx_override_ = year_idx; has_year_override_ = true; }

    void initialize_particles(
        const std::vector<GridCell>& cells,
        const std::unordered_map<int32_t, uint32_t>& propagules_per_cell) override
    {
        const auto t_init0 = Clock::now();

        // La malla (256k celdas) solo se copia/indexa cuando cambia: el SpatialGridLookup, la tabla
        // cell_id -> posición y su copia en Device USM se reutilizan en toda una calibración.
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
            n_ids_ = static_cast<size_t>(max_id + 1);
            cell_pos_.assign(n_ids_, -1);
            for (size_t i = 0; i < cells_.size(); ++i) {
                if (cells_[i].cell_id >= 0) cell_pos_[static_cast<size_t>(cells_[i].cell_id)] = static_cast<int32_t>(i);
            }
            h_arrivals_.assign(n_ids_, 0u);
            upload_grid();
        }

        // Política unificada de tope y reescalado de pesos (M7), idéntica a la del baseline CPU
        struct Tmp { size_t pos; uint32_t total; uint32_t n_sim; };
        std::vector<Tmp> tmp;
        tmp.reserve(propagules_per_cell.size());
        uint64_t sum_total = 0;
        for (const auto& [cid, cnt] : propagules_per_cell) {
            if (cnt == 0) continue;
            const int32_t pos = (cid >= 0 && static_cast<size_t>(cid) < cell_pos_.size())
                                    ? cell_pos_[static_cast<size_t>(cid)] : -1;
            if (pos < 0) continue;
            tmp.push_back({static_cast<size_t>(pos), cnt, 0u});
            sum_total += cnt;
        }
        if (sum_total > kMaxPropagules) {
            throw std::runtime_error("LPDM: más de 4e9 propágulos en un despacho; los contadores de llegada son uint32.");
        }
        std::sort(tmp.begin(), tmp.end(), [](const Tmp& a, const Tmp& b) { return a.pos < b.pos; });
        std::vector<uint32_t> totals(tmp.size()), n_sim_plan;
        for (size_t i = 0; i < tmp.size(); ++i) totals[i] = tmp[i].total;
        last_capped_ = plan_super_particles(totals, max_super_, n_sim_plan);
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i].n_sim = n_sim_plan[i];
        if (last_capped_) tel_.capped_dispatches++;

        jobs_.clear();
        jobs_.reserve(tmp.size());
        uint32_t offset = 0;
        for (const auto& t : tmp) {
            const GridCell& c = cells_[t.pos];
            float start_h = 0.0f;
            if (auto it = emergence_.find(c.cell_id); it != emergence_.end()) start_h = it->second;
            jobs_.push_back(EmissionJob{c.cell_id, c.centroid_lon, c.centroid_lat, t.total, t.n_sim, offset, start_h});
            offset += t.n_sim;
        }
        total_sim_ = offset;
        total_props_ = sum_total;

        tel_.init_time_s += seconds_since(t_init0);
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
        if (jobs_.empty()) return result;

        tel_.dispatches++;
        tel_.particles_dispatched += total_sim_;

        TraceParams tp;
        tp.dt = dt_seconds;
        tp.sqrt_dt = std::sqrt(dt_seconds);
        tp.steps = total_steps;
        tp.vs_aero = compute_settling_velocity(0.6f, 1.2f, 0.85f); // ~0.2947 m/s (nominal; M6 la individualiza)
        tp.vs_ref = tp.vs_aero;
        tp.year_idx = has_year_override_ ? year_idx_override_ : static_cast<uint32_t>(active_atm_);
        tp.mc_seed = mc_seed_;

        uint64_t landed_w = 0, escaped_w = 0, retained_w = 0;
        double sum_wd = 0.0;
        std::vector<uint32_t> hist(kDistHistBins, 0u);
        bool offload_succeeded = false;

#if defined(JORO_HAS_SYCL)
        if (q_) {
            try {
                auto& q = *q_;
                const size_t n_jobs = jobs_.size();
                const size_t n_wg = (static_cast<size_t>(total_sim_) + kWgSize - 1) / kWgSize;
                ensure_job_buffers(n_jobs, q);
                ensure_result_buffers(n_wg, q);
                if (!d_jobs_ || !h_jobs_pin_ || !d_res_ || !h_res_ || !d_grid_) {
                    throw std::runtime_error("asignación en Device USM fallida");
                }

                // Bloque de resultados contiguo en el dispositivo: [llegadas | histograma | parciales por work-group].
                // Así el regreso a host es UNA sola copia DMA sobre memoria anclada (pinned).
                const size_t words_cleared = n_ids_ + static_cast<size_t>(kDistHistBins);
                uint32_t* d_arrivals = d_res_;
                uint32_t* d_hist = d_res_ + n_ids_;
                uint32_t* wg_land = d_hist + kDistHistBins;
                uint32_t* wg_esc = wg_land + n_wg;
                uint32_t* wg_ret = wg_esc + n_wg;
                float* wg_swd = reinterpret_cast<float*>(wg_ret + n_wg);
                const size_t res_words = words_cleared + 4 * n_wg;

                // H2D: solo la tabla compacta de celdas emisoras (< 1 MB, vía memoria anclada) + puesta a cero de acumuladores
                const auto t_h2d0 = Clock::now();
                std::memcpy(h_jobs_pin_, jobs_.data(), n_jobs * sizeof(EmissionJob));
                q.memcpy(d_jobs_, h_jobs_pin_, n_jobs * sizeof(EmissionJob));
                q.memset(d_res_, 0, words_cleared * sizeof(uint32_t));
                q.wait();
                tel_.h2d_time_s += seconds_since(t_h2d0);

                TraceParams dtp = tp;
                dtp.grid = SpatialGridView{grid_view_host_.min_lon, grid_view_host_.min_lat, grid_view_host_.inv_cell,
                                           grid_view_host_.n_cols, grid_view_host_.n_rows, d_grid_};
                if (atmosphere_resident_ && !atms_.empty()) {
                    // Copia el año requerido al DeviceSlot de VRAM si no está ya residente (streaming/doble búfer)
                    const DeviceSlot* ds = ensure_on_device(active_atm_);
                    if (!ds) throw std::runtime_error("ERA5 no residente en el dispositivo");
                    const size_t n_surf = static_cast<size_t>(atms_[active_atm_].host.view.n_t) *
                                          static_cast<size_t>(atms_[active_atm_].host.view.n_lat) *
                                          static_cast<size_t>(atms_[active_atm_].host.view.n_lon);
                    dtp.atm = atms_[active_atm_].host.view;
                    dtp.atm.wind = ds->wind;
                    dtp.atm.blh = ds->surf;
                    dtp.atm.zground = ds->surf + n_surf;
                    dtp.atm.ws10 = ds->surf + 2 * n_surf;
                }

                const EmissionJob* jobs = d_jobs_;
                const uint32_t n_jobs_u = static_cast<uint32_t>(n_jobs);
                const uint32_t total_sim = total_sim_;

                const auto k_start = Clock::now();
                q.submit([&](sycl::handler& cgh) {
                    sycl::local_accessor<uint32_t, 1> lhist(sycl::range<1>(kDistHistBins), cgh);
                    sycl::local_accessor<uint32_t, 1> l_land(sycl::range<1>(kWgSize), cgh);
                    sycl::local_accessor<uint32_t, 1> l_esc(sycl::range<1>(kWgSize), cgh);
                    sycl::local_accessor<uint32_t, 1> l_ret(sycl::range<1>(kWgSize), cgh);
                    sycl::local_accessor<float, 1> l_swd(sycl::range<1>(kWgSize), cgh);
                    cgh.parallel_for(
                        sycl::nd_range<1>(n_wg * kWgSize, kWgSize),
                        [=](sycl::nd_item<1> item) {
                            const size_t lid = item.get_local_id(0);
                            const uint32_t gid = static_cast<uint32_t>(item.get_global_id(0));

                            for (size_t b = lid; b < static_cast<size_t>(kDistHistBins); b += kWgSize) lhist[b] = 0u;
                            sycl::group_barrier(item.get_group());

                            uint32_t w_land = 0, w_esc = 0, w_ret = 0;
                            float swd = 0.0f;
                            if (gid < total_sim) {
                                // Lote al que pertenece esta super-partícula (búsqueda binaria por offset)
                                uint32_t lo = 0, hi = n_jobs_u - 1;
                                while (lo < hi) {
                                    const uint32_t mid = (lo + hi + 1) >> 1;
                                    if (jobs[mid].offset <= gid) lo = mid; else hi = mid - 1;
                                }
                                const EmissionJob job = jobs[lo];
                                const TraceResult r = trace_super_particle(dtp, job, gid - job.offset);
                                if (r.landed) {
                                    w_land = r.weight;
                                    if (r.retained) {
                                        // Retención local: balance demográfico sí, diagnóstico de distancia de vuelo no
                                        w_ret = r.weight;
                                    } else {
                                        swd = static_cast<float>(r.weight) * r.dist_km;
                                        int bin = static_cast<int>(r.dist_km * (1.0f / kDistHistBinKm));
                                        bin = bin < kDistHistBins ? bin : kDistHistBins - 1;
                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::work_group,
                                                         sycl::access::address_space::local_space>(lhist[bin])
                                            .fetch_add(r.weight);
                                    }
                                    if (r.dest_cell >= 0) {
                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>(
                                            d_arrivals[r.dest_cell])
                                            .fetch_add(r.weight);
                                    }
                                } else {
                                    w_esc = r.weight; // sigue en suspensión al terminar la ventana
                                }
                            }
                            l_land[lid] = w_land;
                            l_esc[lid] = w_esc;
                            l_ret[lid] = w_ret;
                            l_swd[lid] = swd;
                            sycl::group_barrier(item.get_group());

                            for (size_t b = lid; b < static_cast<size_t>(kDistHistBins); b += kWgSize) {
                                const uint32_t v = lhist[b];
                                if (v != 0u) {
                                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>(d_hist[b])
                                        .fetch_add(v);
                                }
                            }
                            if (lid == 0) {
                                uint32_t s_land = 0, s_esc = 0, s_ret = 0;
                                float s_swd = 0.0f;
                                for (size_t i = 0; i < kWgSize; ++i) {
                                    s_land += l_land[i];
                                    s_esc += l_esc[i];
                                    s_ret += l_ret[i];
                                    s_swd += l_swd[i];
                                }
                                const size_t g = item.get_group_linear_id();
                                wg_land[g] = s_land;
                                wg_esc[g] = s_esc;
                                wg_ret[g] = s_ret;
                                wg_swd[g] = s_swd;
                            }
                        });
                }).wait();
                result.kernel_time_s = seconds_since(k_start);
                tel_.kernel_time_s += result.kernel_time_s;

                // D2H: UNA copia anclada del bloque compacto (n_ids x 4 B ≈ 1 MB + histograma + parciales)
                const auto t_d2h0 = Clock::now();
                q.memcpy(h_res_, d_res_, res_words * sizeof(uint32_t)).wait();
                tel_.d2h_time_s += seconds_since(t_d2h0);

                const auto t_red0 = Clock::now();
                const uint32_t* h_hist = h_res_ + n_ids_;
                const uint32_t* h_land = h_hist + kDistHistBins;
                const uint32_t* h_esc = h_land + n_wg;
                const uint32_t* h_ret = h_esc + n_wg;
                const float* h_swd = reinterpret_cast<const float*>(h_ret + n_wg);
                std::copy(h_hist, h_hist + kDistHistBins, hist.begin());
                for (size_t g = 0; g < n_wg; ++g) {
                    landed_w += h_land[g];
                    escaped_w += h_esc[g];
                    retained_w += h_ret[g];
                    sum_wd += static_cast<double>(h_swd[g]);
                }
                tel_.reduce_time_s += seconds_since(t_red0);
                offload_succeeded = true;
            } catch (const sycl::exception& e) {
                std::cerr << "[SYCL Note] Dispositivo SYCL error (" << e.what()
                          << "). Ejecutando fallback local en CPU.\n";
            } catch (const std::runtime_error& e) {
                std::cerr << "[SYCL Note] " << e.what() << ". Ejecutando fallback local en CPU.\n";
            }
        }
#endif

        if (!offload_succeeded) {
            tel_.cpu_fallbacks++;
            landed_w = escaped_w = retained_w = 0;
            sum_wd = 0.0;
            std::fill(hist.begin(), hist.end(), 0u);
            std::fill(h_arrivals_.begin(), h_arrivals_.end(), 0u);
            TraceParams htp = tp;
            htp.grid = spatial_lookup_.view();
            if (atmosphere_resident_ && !atms_.empty()) htp.atm = atms_[active_atm_].host.view;
            const auto k_start = Clock::now();
            const int64_t n = static_cast<int64_t>(total_sim_);
            const int64_t n_jobs = static_cast<int64_t>(jobs_.size());
            uint64_t land_acc = 0, esc_acc = 0, ret_acc = 0;
            double swd_acc = 0.0;
            #pragma omp parallel for schedule(static) reduction(+ : land_acc, esc_acc, ret_acc, swd_acc)
            for (int64_t gid = 0; gid < n; ++gid) {
                int64_t lo = 0, hi = n_jobs - 1;
                while (lo < hi) {
                    const int64_t mid = (lo + hi + 1) >> 1;
                    if (static_cast<int64_t>(jobs_[static_cast<size_t>(mid)].offset) <= gid) lo = mid; else hi = mid - 1;
                }
                const EmissionJob& job = jobs_[static_cast<size_t>(lo)];
                const TraceResult r = trace_super_particle(htp, job, static_cast<uint32_t>(gid - job.offset));
                if (r.landed) {
                    land_acc += r.weight;
                    if (r.retained) {
                        ret_acc += r.weight;
                    } else {
                        swd_acc += static_cast<double>(r.weight) * r.dist_km;
                        int bin = static_cast<int>(r.dist_km * (1.0f / kDistHistBinKm));
                        bin = bin < kDistHistBins ? bin : kDistHistBins - 1;
                        #pragma omp atomic
                        hist[static_cast<size_t>(bin)] += r.weight;
                    }
                    if (r.dest_cell >= 0) {
                        #pragma omp atomic
                        h_arrivals_[static_cast<size_t>(r.dest_cell)] += r.weight;
                    }
                } else {
                    esc_acc += r.weight;
                }
            }
            landed_w = land_acc;
            escaped_w = esc_acc;
            retained_w = ret_acc;
            sum_wd = swd_acc;
            result.kernel_time_s = seconds_since(k_start);
            tel_.kernel_time_s += result.kernel_time_s;
        }

        // Reducción final en host: mapa disperso de llegadas + estadísticos de salto
        const auto t_red0 = Clock::now();
        result.total_released = total_props_;
        result.total_landed = landed_w;
        result.total_escaped = escaped_w;
        result.total_retained = retained_w;
        result.arrivals_per_cell.reserve(4096);
        const uint32_t* arr = offload_succeeded ? h_res_ : h_arrivals_.data();
        for (size_t id = 0; id < n_ids_; ++id) {
            if (arr[id] != 0u) result.arrivals_per_cell[static_cast<int32_t>(id)] = arr[id];
        }
        const uint64_t flown_w = landed_w - retained_w;   // denominador del vuelo efectivo (sin retenidos)
        if (flown_w > 0) {
            result.mean_jump_distance_km = static_cast<float>(sum_wd / static_cast<double>(flown_w));
            const uint64_t target = static_cast<uint64_t>(std::ceil(0.95 * static_cast<double>(flown_w)));
            uint64_t cum = 0;
            int p95_bin = kDistHistBins - 1;
            for (int b = 0; b < kDistHistBins; ++b) {
                cum += hist[static_cast<size_t>(b)];
                if (cum >= target) { p95_bin = b; break; }
            }
            result.p95_jump_distance_km = (static_cast<float>(p95_bin) + 0.5f) * kDistHistBinKm;
        }
        tel_.reduce_time_s += seconds_since(t_red0);
        return result;
    }

private:
    [[nodiscard]] size_t total_host_bytes() const noexcept {
        size_t b = 0;
        for (const auto& a : atms_) b += a.host.bytes();
        return b;
    }

#if defined(JORO_HAS_SYCL)
    void free_slot_buffers(DeviceSlot& d) {
        if (d.wind) sycl::free(d.wind, *q_);
        if (d.surf) sycl::free(d.surf, *q_);
        d.wind = nullptr; d.surf = nullptr;
        d.wind_cap_bytes = 0; d.surf_cap_floats = 0;
        d.owner = -1;
    }

    void free_device_slots() {
        if (!q_) { dslots_.clear(); return; }
        try { for (auto& d : dslots_) free_slot_buffers(d); } catch (...) {}
        dslots_.clear();
    }

    /// Garantiza que el año `idx` resida en un DeviceSlot de VRAM, copiándolo desde la RAM host (FP16) si hace falta.
    /// Política: si Σ años cargados <= presupuesto (3.8 GiB) todos pueden residir; si no, se hace streaming con
    /// a lo sumo 2 años simultáneos y se desaloja el menos recientemente usado. Devuelve nullptr sin dispositivo.
    DeviceSlot* ensure_on_device(size_t idx) {
        if (!q_ || idx >= atms_.size()) return nullptr;
        for (auto& d : dslots_) {
            if (d.owner == static_cast<int>(idx)) { d.last_use = ++use_clock_; return &d; }
        }
        const Era5Pack& pk = atms_[idx].host;
        const size_t need = pk.bytes();
        if (pk.wind_bytes() > max_alloc_bytes_) {
            throw std::runtime_error("el tensor de viento FP16 excede max_mem_alloc_size del dispositivo");
        }
        const bool streaming = total_host_bytes() > vram_budget_;

        // Desalojo LRU hasta respetar el presupuesto de VRAM (y el máximo de 2 años en modo streaming)
        for (;;) {
            size_t resident = 0, occupied = 0;
            for (const auto& d : dslots_) {
                if (d.owner >= 0) { resident += atms_[static_cast<size_t>(d.owner)].host.bytes(); ++occupied; }
            }
            const bool over = (resident + need > vram_budget_) || (streaming && occupied >= kMaxStreamSlots);
            if (!over || occupied == 0) break;
            DeviceSlot* lru = nullptr;
            for (auto& d : dslots_) if (d.owner >= 0 && (!lru || d.last_use < lru->last_use)) lru = &d;
            lru->owner = -1;
        }
        // En modo streaming nunca se conservan más de kMaxStreamSlots búferes alojados
        if (streaming) {
            for (size_t i = 0; i < dslots_.size() && dslots_.size() > kMaxStreamSlots;) {
                if (dslots_[i].owner < 0) { free_slot_buffers(dslots_[i]); dslots_.erase(dslots_.begin() + static_cast<std::ptrdiff_t>(i)); }
                else ++i;
            }
        }

        DeviceSlot* t = nullptr;
        for (auto& d : dslots_) if (d.owner < 0) { t = &d; break; }
        if (!t) { dslots_.emplace_back(); t = &dslots_.back(); }

        const size_t n_surf = static_cast<size_t>(pk.view.n_t) * static_cast<size_t>(pk.view.n_lat) *
                              static_cast<size_t>(pk.view.n_lon);
        if (t->wind_cap_bytes < pk.wind_bytes() || !t->wind) {
            if (t->wind) sycl::free(t->wind, *q_);
            t->wind = sycl::malloc_device<Wind4Half>(pk.wind.size(), *q_);
            t->wind_cap_bytes = t->wind ? pk.wind_bytes() : 0;
        }
        if (t->surf_cap_floats < 3 * n_surf || !t->surf) {
            if (t->surf) sycl::free(t->surf, *q_);
            t->surf = sycl::malloc_device<float>(3 * n_surf, *q_);
            t->surf_cap_floats = t->surf ? 3 * n_surf : 0;
        }
        if (!t->wind || !t->surf) {
            free_slot_buffers(*t);
            throw std::runtime_error("sin memoria de dispositivo para ERA5 (DeviceSlot)");
        }

        const auto t_h2d = Clock::now();
        q_->memcpy(t->wind, pk.wind.data(), pk.wind_bytes());
        q_->memcpy(t->surf, pk.blh.data(), n_surf * sizeof(float));
        q_->memcpy(t->surf + n_surf, pk.zground.data(), n_surf * sizeof(float));
        q_->memcpy(t->surf + 2 * n_surf, pk.ws10.data(), n_surf * sizeof(float)).wait();
        tel_.h2d_time_s += seconds_since(t_h2d);
        t->owner = static_cast<int>(idx);
        t->last_use = ++use_clock_;
        return t;
    }

    void ensure_job_buffers(size_t n, sycl::queue& q) {
        if (n <= cap_jobs_ && d_jobs_ && h_jobs_pin_) return;
        if (d_jobs_) sycl::free(d_jobs_, q);
        if (h_jobs_pin_) sycl::free(h_jobs_pin_, q);
        cap_jobs_ = std::max<size_t>(n + n / 2, 1024);
        d_jobs_ = sycl::malloc_device<EmissionJob>(cap_jobs_, q);
        h_jobs_pin_ = sycl::malloc_host<EmissionJob>(cap_jobs_, q);
    }

    void ensure_result_buffers(size_t n_wg, sycl::queue& q) {
        if (n_wg <= cap_wg_ && d_res_ && h_res_) return;
        if (d_res_) sycl::free(d_res_, q);
        if (h_res_) sycl::free(h_res_, q);
        cap_wg_ = std::max<size_t>(n_wg + n_wg / 2, 1024);
        const size_t words = n_ids_ + static_cast<size_t>(kDistHistBins) + 4 * cap_wg_;
        d_res_ = sycl::malloc_device<uint32_t>(words, q);
        h_res_ = sycl::malloc_host<uint32_t>(words, q);
    }

#endif

    void upload_grid() {
        grid_view_host_ = spatial_lookup_.view();
#if defined(JORO_HAS_SYCL)
        if (!q_) return;
        try {
            const auto& raw = spatial_lookup_.raw_grid();
            if (d_grid_) sycl::free(d_grid_, *q_);
            d_grid_ = sycl::malloc_device<int32_t>(raw.size(), *q_);
            q_->memcpy(d_grid_, raw.data(), raw.size() * sizeof(int32_t)).wait();
            if (d_res_) { sycl::free(d_res_, *q_); d_res_ = nullptr; }
            if (h_res_) { sycl::free(h_res_, *q_); h_res_ = nullptr; }
            cap_wg_ = 0;
        } catch (const sycl::exception& e) {
            std::cerr << "[SYCL Note] No se pudo subir el SpatialGridLookup a Device USM (" << e.what() << ")\n";
            d_grid_ = nullptr;
        }
#endif
    }

    void free_device() {
#if defined(JORO_HAS_SYCL)
        if (!q_) return;
        try {
            free_device_slots();
            if (d_grid_) sycl::free(d_grid_, *q_);
            if (d_jobs_) sycl::free(d_jobs_, *q_);
            if (h_jobs_pin_) sycl::free(h_jobs_pin_, *q_);
            if (d_res_) sycl::free(d_res_, *q_);
            if (h_res_) sycl::free(h_res_, *q_);
        } catch (...) {}
        d_grid_ = nullptr; d_jobs_ = nullptr; h_jobs_pin_ = nullptr; d_res_ = nullptr; h_res_ = nullptr;
#endif
    }

    std::vector<GridCell> cells_;
    const GridCell* grid_ptr_ = nullptr;
    std::vector<int32_t> cell_pos_;       ///< cell_id -> posición en cells_ (-1 si no existe)
    size_t n_ids_ = 0;                    ///< max(cell_id) + 1 (tamaño del arreglo de llegadas)
    SpatialGridLookup spatial_lookup_;
    SpatialGridView grid_view_host_;

    std::vector<EmissionJob> jobs_;       ///< Tabla compacta de celdas emisoras del despacho
    uint32_t total_sim_ = 0;              ///< Super-partículas del despacho
    uint64_t total_props_ = 0;            ///< Propágulos reales del despacho (Σ pesos)
    size_t max_super_ = kDefaultMaxSuperPerDispatch;
    bool last_capped_ = false;
    int time_stride_ = []() { const char* e = std::getenv("JORO_ERA5_TIME_STRIDE"); return e ? std::max(1, std::atoi(e)) : 1; }();
    uint32_t mc_seed_ = 0;
    uint32_t year_idx_override_ = 0;
    bool has_year_override_ = false;
    std::unordered_map<int32_t, float> emergence_;   ///< Inicio de la ventana de despegue por celda (h tras el 1 de mayo)
    size_t vram_budget_ = static_cast<size_t>(kDefaultVramBudgetGiB * 1024.0 * 1024.0 * 1024.0);
    size_t max_alloc_bytes_ = static_cast<size_t>(-1);
    uint64_t use_clock_ = 0;

    std::vector<uint32_t> h_arrivals_;    ///< Llegadas en host (solo ruta de respaldo CPU)
    uint32_t* h_res_ = nullptr;           ///< Bloque de resultados anclado (pinned) para la copia D2H única
    EmissionJob* h_jobs_pin_ = nullptr;   ///< Tabla de celdas emisoras en memoria anclada (staging H2D)

    std::vector<AtmSlot> atms_;           ///< Un campo atmosférico residente por año (2014..2019)
    size_t active_atm_ = 0;
    bool atmosphere_resident_ = false;

    EngineTelemetry tel_;
    bool is_gpu_ = false;
    std::string device_desc_ = "SYCL no disponible (fallback CPU)";
#if defined(JORO_HAS_SYCL)
    std::unique_ptr<sycl::queue> q_;
    std::vector<DeviceSlot> dslots_;      ///< Búferes reutilizables de VRAM para los campos ERA5 anuales
    int32_t* d_grid_ = nullptr;
    EmissionJob* d_jobs_ = nullptr;       size_t cap_jobs_ = 0;
    uint32_t* d_res_ = nullptr;           size_t cap_wg_ = 0;  ///< [llegadas | histograma | parciales WG]
#endif
};

std::unique_ptr<LpdmEngine> create_sycl_gpu_engine() {
    return std::make_unique<LpdmSyclEngine>();
}

} // namespace joro::lpdm
