#pragma once

#include "common/types.hpp"
#include "io/era5_netcdf_reader.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>

namespace joro::lpdm {

/// Resultado de la simulación de dispersión aérea anual
struct DispersionResult {
    uint64_t total_released = 0;
    uint64_t total_landed = 0;    ///< Aterrizados = aterrizajes tras vuelo + retenidos localmente (balance demográfico del ABM)
    uint64_t total_escaped = 0;   ///< Aún en suspensión al terminar la ventana de vuelo
    uint64_t total_retained = 0;  ///< Retención local: viento >= u_crit en todos los días candidatos (sin vuelo distante)
    
    /// Diagnósticos de vuelo EFECTIVO: excluyen a los retenidos (total_landed - total_retained es el denominador)
    float mean_jump_distance_km = 0.0f;
    float p95_jump_distance_km = 0.0f;
    double kernel_time_s = 0.0; ///< Tiempo exclusivo de integración de trayectorias (GPU vs CPU)

    /// Matriz dispersa de llegadas por celda: cell_id -> número de propágulos depositados A_{i,t}
    std::unordered_map<int32_t, uint32_t> arrivals_per_cell;
};

/// Telemetría acumulada del motor (segundos de reloj de pared por fase)
struct EngineTelemetry {
    uint64_t dispatches = 0;            ///< Llamadas a run_dispersion con partículas
    uint64_t particles_dispatched = 0;  ///< Super-partículas integradas (excluye padding)
    uint64_t cpu_fallbacks = 0;         ///< Despachos que NO pudieron ejecutarse en GPU
    double init_time_s = 0.0;           ///< Preparación de lotes de emisión (host)
    double h2d_time_s = 0.0;            ///< Copia host -> Device USM (tabla compacta de celdas emisoras + puesta a cero)
    double kernel_time_s = 0.0;         ///< Kernel SYCL parallel_for (integración de trayectorias + acumulación)
    double d2h_time_s = 0.0;            ///< Copia Device USM -> host (llegadas por celda + parciales)
    double reduce_time_s = 0.0;         ///< Reducción final en host (mapa de llegadas, estadísticos)
    uint64_t capped_dispatches = 0;     ///< Despachos en los que el tope adaptativo escaló el peso de las super-partículas

    [[nodiscard]] EngineTelemetry operator-(const EngineTelemetry& o) const noexcept {
        return {dispatches - o.dispatches, particles_dispatched - o.particles_dispatched,
                cpu_fallbacks - o.cpu_fallbacks, init_time_s - o.init_time_s, h2d_time_s - o.h2d_time_s,
                kernel_time_s - o.kernel_time_s, d2h_time_s - o.d2h_time_s, reduce_time_s - o.reduce_time_s,
                capped_dispatches - o.capped_dispatches};
    }
    [[nodiscard]] double total_s() const noexcept {
        return init_time_s + h2d_time_s + kernel_time_s + d2h_time_s + reduce_time_s;
    }
};


class LpdmEngine {
public:
    virtual ~LpdmEngine() = default;

    /// Inicializa las partículas en las celdas de origen
    virtual void initialize_particles(
        const std::vector<GridCell>& cells,
        const std::unordered_map<int32_t, uint32_t>& propagules_per_cell) = 0;

    /// Ejecuta la advección lagrangiana estocástica sobre un periodo temporal
    virtual DispersionResult run_dispersion(
        const io::AtmosphericField3D& atmosphere,
        float dt_seconds,
        int total_steps) = 0;

    /// Telemetría acumulada desde la construcción (por defecto vacía)
    [[nodiscard]] virtual EngineTelemetry telemetry() const { return {}; }

    /// True si run_dispersion se ejecuta en un dispositivo GPU (SYCL)
    [[nodiscard]] virtual bool is_gpu_offload_active() const noexcept { return false; }

    /// Descripción legible del dispositivo (nombre, backend)
    [[nodiscard]] virtual std::string device_description() const { return "CPU (host)"; }

    /// Carga el campo ERA5 (u, v, w, BLH, presión superficial) UNA sola vez; a partir de aquí
    /// run_dispersion ignora el argumento `atmosphere` y usa el campo residente (Device USM en GPU).
    /// Devuelve false si el campo no es utilizable (el motor sigue en modo de viento uniforme heredado).
    virtual bool set_atmosphere(const io::AtmosphericField3D& /*atmosphere*/) { return false; }

    /// True si hay un campo ERA5 residente que gobierna el transporte
    [[nodiscard]] virtual bool is_atmosphere_resident() const noexcept { return false; }

    /// Atmósferas interanuales: `add_atmosphere` agrega un campo ERA5 residente (índice = orden de
    /// llamada, p. ej. 0 -> 2014 ... 5 -> 2019); `select_atmosphere(i)` elige el campo que gobierna los
    /// siguientes despachos. `set_atmosphere` equivale a `clear_atmospheres()` + `add_atmosphere()`.
    virtual void clear_atmospheres() {}
    virtual bool add_atmosphere(const io::AtmosphericField3D& /*atmosphere*/) { return false; }
    virtual bool select_atmosphere(size_t /*index*/) { return false; }
    [[nodiscard]] virtual size_t atmosphere_count() const noexcept { return 0; }

    /// Empaqueta el campo en FP16 y libera de inmediato el `AtmosphericField3D` temporal del llamador
    /// (evita retener ~2.6 GiB de float32 por año una vez construido el Era5Pack comprimido).
    bool add_atmosphere_and_release(io::AtmosphericField3D& atmosphere) {
        const bool ok = add_atmosphere(atmosphere);
        atmosphere.clear_and_shrink();
        return ok;
    }

    /// Huella de memoria de los campos ERA5 empaquetados en FP16: RAM host (todos los años) y VRAM Device USM
    /// actualmente ocupada por los DeviceSlot (0 en el motor CPU).
    [[nodiscard]] virtual size_t atmosphere_host_bytes() const noexcept { return 0; }
    [[nodiscard]] virtual size_t atmosphere_device_bytes() const noexcept { return 0; }

    /// Semillas dinámicas (M8): réplica Monte Carlo y desplazamiento del índice de año en el generador.
    /// Por defecto year_idx = índice de la atmósfera activa; `set_year_index` lo fija explícitamente.
    virtual void set_mc_seed(uint32_t /*mc_seed*/) {}

    /// Paso temporal (h) con que se almacenan los campos ERA5 en FP16 (1 = serie horaria completa; por defecto lo fija
    /// JORO_ERA5_TIME_STRIDE). Debe fijarse ANTES de add_atmosphere. Con 3 h los 6 años de calibración (3.64 GiB) quedan
    /// 100 % residentes en la VRAM de la Arc A380: 0 transferencias H2D por evaluación.
    virtual void set_era5_time_stride(int /*hours*/) {}

    /// Emergencia escalonada: inicio de la ventana de despegue por celda [h tras el 1 de mayo 00 UTC], producido por el
    /// ABM (desarrollo térmico acumulado). Debe fijarse ANTES de `initialize_particles`; el mapa vacío => 0 h para todas.
    virtual void set_emergence_offsets(const std::unordered_map<int32_t, float>& /*start_hours*/) {}
    virtual void set_year_index(uint32_t /*year_idx*/) {}

    /// Presupuesto máximo de super-partículas por despacho anual; si se excede, se reduce su número
    /// y se escala su peso (la suma de pesos por celda se conserva exactamente). 0 = sin presupuesto.
    /// Política idéntica en CPU y GPU (superparticle_policy.hpp); el modo v2.0 admite hasta 5e6.
    virtual void set_max_super_particles(size_t /*budget*/) {}
};


/// Crea el motor de referencia en CPU (OpenMP SIMD)
std::unique_ptr<LpdmEngine> create_cpu_baseline_engine();

/// Crea el motor acelerado en GPU Intel Arc (DPC++/SYCL)
std::unique_ptr<LpdmEngine> create_sycl_gpu_engine();

} // namespace joro::lpdm
