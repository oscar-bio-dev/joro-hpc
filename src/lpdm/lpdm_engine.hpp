#pragma once

#include "common/types.hpp"
#include "io/era5_netcdf_reader.hpp"
#include <vector>
#include <unordered_map>

namespace joro::lpdm {

/// Resultado de la simulación de dispersión aérea anual
struct DispersionResult {
    uint64_t total_released = 0;
    uint64_t total_landed = 0;
    uint64_t total_escaped = 0;
    
    /// Matriz dispersa de llegadas por celda: cell_id -> número de propágulos depositados A_{i,t}
    std::unordered_map<int32_t, uint32_t> arrivals_per_cell;
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
};

/// Crea el motor de referencia en CPU (OpenMP SIMD)
std::unique_ptr<LpdmEngine> create_cpu_baseline_engine();

/// Crea el motor acelerado en GPU Intel Arc (DPC++/SYCL)
std::unique_ptr<LpdmEngine> create_sycl_gpu_engine();

} // namespace joro::lpdm
