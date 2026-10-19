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
    uint64_t total_landed = 0;
    uint64_t total_escaped = 0;
    
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

    /// Presupuesto máximo de super-partículas por despacho anual; si se excede, se reduce su número
    /// y se escala su peso (la suma de pesos por celda se conserva exactamente). 0 = sin presupuesto.
    virtual void set_max_super_particles(size_t /*budget*/) {}
};

/// Crea el motor de referencia en CPU (OpenMP SIMD)
std::unique_ptr<LpdmEngine> create_cpu_baseline_engine();

/// Crea el motor acelerado en GPU Intel Arc (DPC++/SYCL)
std::unique_ptr<LpdmEngine> create_sycl_gpu_engine();

} // namespace joro::lpdm
