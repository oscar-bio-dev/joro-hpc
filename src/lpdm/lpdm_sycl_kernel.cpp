#include "lpdm/lpdm_engine.hpp"
#include "lpdm/lpdm_trace.hpp"
#include "lpdm/settling_velocity.hpp"
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

constexpr uint32_t kMaxSuperPerCell = 25000;       ///< Tope por celda (heredado)
constexpr size_t kDefaultMaxSuperPerDispatch = 120000; ///< Presupuesto adaptativo por despacho anual
constexpr uint64_t kMaxPropagules = 4000000000ull; ///< Los contadores de llegada son uint32
#if defined(JORO_HAS_SYCL)
constexpr size_t kWgSize = 128;
#endif

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

} // namespace

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

    ~LpdmSyclEngine() override { free_device(); }

    void set_max_super_particles(size_t budget) override { max_super_ = budget; }

    bool set_atmosphere(const io::AtmosphericField3D& atmosphere) override {
        const auto t0 = Clock::now();
        Era5Pack pack;
        if (!pack.build(atmosphere)) return false;
        era5_host_ = std::move(pack);
        era5_host_.refresh_view();
#if defined(JORO_HAS_SYCL)
        if (q_) {
            try {
                free_atmosphere_device();
                d_wind_ = sycl::malloc_device<Wind4>(era5_host_.wind.size(), *q_);
                d_blh_ = sycl::malloc_device<float>(era5_host_.blh.size(), *q_);
                d_zg_ = sycl::malloc_device<float>(era5_host_.zground.size(), *q_);
                if (!d_wind_ || !d_blh_ || !d_zg_) throw std::runtime_error("sin memoria de dispositivo para ERA5");
                q_->memcpy(d_wind_, era5_host_.wind.data(), era5_host_.wind.size() * sizeof(Wind4));
                q_->memcpy(d_blh_, era5_host_.blh.data(), era5_host_.blh.size() * sizeof(float));
                q_->memcpy(d_zg_, era5_host_.zground.data(), era5_host_.zground.size() * sizeof(float)).wait();
            } catch (const std::exception& e) {
                std::cerr << "[SYCL Note] No se pudo subir ERA5 al dispositivo (" << e.what()
                          << "); el transporte usará el campo en host.\n";
                free_atmosphere_device();
            }
        }
#endif
        atmosphere_resident_ = true;
        std::cout << "  [ERA5->USM] Campo atmosférico residente: " << era5_host_.view.n_t << " h x "
                  << era5_host_.view.n_lev << " niveles x " << era5_host_.view.n_lat << " x " << era5_host_.view.n_lon
                  << " | " << (static_cast<double>(era5_host_.bytes()) / 1.048576e6) << " MB"
                  << (d_wind_ ? " en Device USM" : " en host") << " | carga única "
                  << seconds_since(t0) << " s\n";
        return true;
    }

    [[nodiscard]] bool is_atmosphere_resident() const noexcept override { return atmosphere_resident_; }

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

        struct Tmp { size_t pos; uint32_t total; uint32_t n_sim; };
        std::vector<Tmp> tmp;
        tmp.reserve(propagules_per_cell.size());
        uint64_t sum_total = 0, sum_sim = 0;
        for (const auto& [cid, cnt] : propagules_per_cell) {
            if (cnt == 0) continue;
            const int32_t pos = (cid >= 0 && static_cast<size_t>(cid) < cell_pos_.size())
                                    ? cell_pos_[static_cast<size_t>(cid)] : -1;
            if (pos < 0) continue;
            const uint32_t n_sim = std::min(cnt, kMaxSuperPerCell);
            tmp.push_back({static_cast<size_t>(pos), cnt, n_sim});
            sum_total += cnt;
            sum_sim += n_sim;
        }
        if (sum_total > kMaxPropagules) {
            throw std::runtime_error("LPDM: más de 4e9 propágulos en un despacho; los contadores de llegada son uint32.");
        }
        std::sort(tmp.begin(), tmp.end(), [](const Tmp& a, const Tmp& b) { return a.pos < b.pos; });

        // Tope adaptativo: si el despacho excede el presupuesto de super-partículas, se reduce su
        // número por celda y su peso crece (total/n_sim); la suma de pesos por celda sigue siendo exacta.
        last_capped_ = false;
        if (max_super_ > 0 && sum_sim > max_super_) {
            const double f = static_cast<double>(max_super_) / static_cast<double>(sum_sim);
            sum_sim = 0;
            for (auto& t : tmp) {
                const uint32_t scaled = static_cast<uint32_t>(std::ceil(static_cast<double>(t.n_sim) * f));
                t.n_sim = std::max<uint32_t>(1u, std::min(t.n_sim, scaled));
                sum_sim += t.n_sim;
            }
            last_capped_ = true;
            tel_.capped_dispatches++;
        }

        jobs_.clear();
        jobs_.reserve(tmp.size());
        uint32_t offset = 0;
        for (const auto& t : tmp) {
            const GridCell& c = cells_[t.pos];
            jobs_.push_back(EmissionJob{c.cell_id, c.centroid_lon, c.centroid_lat, t.total, t.n_sim, offset});
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
        tp.vs_aero = compute_settling_velocity(0.6f, 1.2f, 0.85f); // ~0.2947 m/s

        uint64_t landed_w = 0, escaped_w = 0;
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
                float* wg_swd = reinterpret_cast<float*>(wg_esc + n_wg);
                const size_t res_words = words_cleared + 3 * n_wg;

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
                if (atmosphere_resident_ && d_wind_) {
                    dtp.atm = era5_host_.view;
                    dtp.atm.wind = d_wind_;
                    dtp.atm.blh = d_blh_;
                    dtp.atm.zground = d_zg_;
                } else if (atmosphere_resident_) {
                    throw std::runtime_error("ERA5 no residente en el dispositivo");
                }

                const EmissionJob* jobs = d_jobs_;
                const uint32_t n_jobs_u = static_cast<uint32_t>(n_jobs);
                const uint32_t total_sim = total_sim_;

                const auto k_start = Clock::now();
                q.submit([&](sycl::handler& cgh) {
                    sycl::local_accessor<uint32_t, 1> lhist(sycl::range<1>(kDistHistBins), cgh);
                    cgh.parallel_for(
                        sycl::nd_range<1>(n_wg * kWgSize, kWgSize),
                        [=](sycl::nd_item<1> item) {
                            const size_t lid = item.get_local_id(0);
                            const uint32_t gid = static_cast<uint32_t>(item.get_global_id(0));

                            for (size_t b = lid; b < static_cast<size_t>(kDistHistBins); b += kWgSize) lhist[b] = 0u;
                            sycl::group_barrier(item.get_group());

                            uint32_t w_land = 0, w_esc = 0;
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
                                    swd = static_cast<float>(r.weight) * r.dist_km;
                                    int bin = static_cast<int>(r.dist_km * (1.0f / kDistHistBinKm));
                                    bin = bin < kDistHistBins ? bin : kDistHistBins - 1;
                                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::work_group,
                                                     sycl::access::address_space::local_space>(lhist[bin])
                                        .fetch_add(r.weight);
                                    if (r.dest_cell >= 0) {
                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>(
                                            d_arrivals[r.dest_cell])
                                            .fetch_add(r.weight);
                                    } else {
                                        w_esc = r.weight;
                                    }
                                } else {
                                    w_esc = r.weight; // sigue en suspensión al terminar la ventana
                                }
                            }
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
                            const auto grp = item.get_group();
                            const uint32_t s_land = sycl::reduce_over_group(grp, w_land, sycl::plus<uint32_t>());
                            const uint32_t s_esc = sycl::reduce_over_group(grp, w_esc, sycl::plus<uint32_t>());
                            const float s_swd = sycl::reduce_over_group(grp, swd, sycl::plus<float>());
                            if (lid == 0) {
                                const size_t g = item.get_group_linear_id();
                                wg_land[g] = s_land;
                                wg_esc[g] = s_esc;
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
                const float* h_swd = reinterpret_cast<const float*>(h_esc + n_wg);
                std::copy(h_hist, h_hist + kDistHistBins, hist.begin());
                for (size_t g = 0; g < n_wg; ++g) {
                    landed_w += h_land[g];
                    escaped_w += h_esc[g];
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
            landed_w = escaped_w = 0;
            sum_wd = 0.0;
            std::fill(hist.begin(), hist.end(), 0u);
            std::fill(h_arrivals_.begin(), h_arrivals_.end(), 0u);
            TraceParams htp = tp;
            htp.grid = spatial_lookup_.view();
            if (atmosphere_resident_) htp.atm = era5_host_.view;
            const auto k_start = Clock::now();
            const int64_t n = static_cast<int64_t>(total_sim_);
            const int64_t n_jobs = static_cast<int64_t>(jobs_.size());
            uint64_t land_acc = 0, esc_acc = 0;
            double swd_acc = 0.0;
            #pragma omp parallel for schedule(static) reduction(+ : land_acc, esc_acc, swd_acc)
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
                    swd_acc += static_cast<double>(r.weight) * r.dist_km;
                    int bin = static_cast<int>(r.dist_km * (1.0f / kDistHistBinKm));
                    bin = bin < kDistHistBins ? bin : kDistHistBins - 1;
                    #pragma omp atomic
                    hist[static_cast<size_t>(bin)] += r.weight;
                    if (r.dest_cell >= 0) {
                        #pragma omp atomic
                        h_arrivals_[static_cast<size_t>(r.dest_cell)] += r.weight;
                    } else {
                        esc_acc += r.weight;
                    }
                } else {
                    esc_acc += r.weight;
                }
            }
            landed_w = land_acc;
            escaped_w = esc_acc;
            sum_wd = swd_acc;
            result.kernel_time_s = seconds_since(k_start);
            tel_.kernel_time_s += result.kernel_time_s;
        }

        // Reducción final en host: mapa disperso de llegadas + estadísticos de salto
        const auto t_red0 = Clock::now();
        result.total_released = total_props_;
        result.total_landed = landed_w;
        result.total_escaped = escaped_w;
        result.arrivals_per_cell.reserve(4096);
        const uint32_t* arr = offload_succeeded ? h_res_ : h_arrivals_.data();
        for (size_t id = 0; id < n_ids_; ++id) {
            if (arr[id] != 0u) result.arrivals_per_cell[static_cast<int32_t>(id)] = arr[id];
        }
        if (landed_w > 0) {
            result.mean_jump_distance_km = static_cast<float>(sum_wd / static_cast<double>(landed_w));
            const uint64_t target = static_cast<uint64_t>(std::ceil(0.95 * static_cast<double>(landed_w)));
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
#if defined(JORO_HAS_SYCL)
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
        const size_t words = n_ids_ + static_cast<size_t>(kDistHistBins) + 3 * cap_wg_;
        d_res_ = sycl::malloc_device<uint32_t>(words, q);
        h_res_ = sycl::malloc_host<uint32_t>(words, q);
    }

    void free_atmosphere_device() {
        if (!q_) return;
        if (d_wind_) { sycl::free(d_wind_, *q_); d_wind_ = nullptr; }
        if (d_blh_) { sycl::free(d_blh_, *q_); d_blh_ = nullptr; }
        if (d_zg_) { sycl::free(d_zg_, *q_); d_zg_ = nullptr; }
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
            free_atmosphere_device();
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

    std::vector<uint32_t> h_arrivals_;    ///< Llegadas en host (solo ruta de respaldo CPU)
    uint32_t* h_res_ = nullptr;           ///< Bloque de resultados anclado (pinned) para la copia D2H única
    EmissionJob* h_jobs_pin_ = nullptr;   ///< Tabla de celdas emisoras en memoria anclada (staging H2D)

    Era5Pack era5_host_;
    bool atmosphere_resident_ = false;

    EngineTelemetry tel_;
    bool is_gpu_ = false;
    std::string device_desc_ = "SYCL no disponible (fallback CPU)";
#if defined(JORO_HAS_SYCL)
    std::unique_ptr<sycl::queue> q_;
    int32_t* d_grid_ = nullptr;
    EmissionJob* d_jobs_ = nullptr;       size_t cap_jobs_ = 0;
    uint32_t* d_res_ = nullptr;           size_t cap_wg_ = 0;  ///< [llegadas | histograma | parciales WG]
    Wind4* d_wind_ = nullptr;
    float* d_blh_ = nullptr;
    float* d_zg_ = nullptr;
#endif
};

std::unique_ptr<LpdmEngine> create_sycl_gpu_engine() {
    return std::make_unique<LpdmSyclEngine>();
}

} // namespace joro::lpdm
