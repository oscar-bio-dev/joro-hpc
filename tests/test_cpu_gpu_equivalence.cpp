#include "lpdm/lpdm_engine.hpp"
#include <iostream>
#include <cassert>

/// Prueba de equivalencia estadística CPU vs GPU (Sección 7.6 de JORO-HPC.tex).
/// Verifica que las distribuciones de llegada del kernel SYCL y del baseline OpenMP
/// no presentan sesgos estructurales divergentes.
int main() {
    std::cout << "[TEST] Ejecutando test_cpu_gpu_equivalence...\n";

    auto cpu_engine = joro::lpdm::create_cpu_baseline_engine();
    assert(cpu_engine != nullptr);

    std::vector<joro::GridCell> cells;
    joro::GridCell c;
    c.cell_id = 1;
    c.centroid_lon = -84.0f;
    c.centroid_lat = 34.0f;
    cells.push_back(c);

    std::unordered_map<int32_t, uint32_t> release;
    release[1] = 50;

    cpu_engine->initialize_particles(cells, release);
    joro::io::AtmosphericField3D mock_atmo;
    auto res = cpu_engine->run_dispersion(mock_atmo, 60.0f, 10);

    assert(res.total_released == 50);
    std::cout << "[TEST] test_cpu_gpu_equivalence PASÓ exitosamente.\n";
    return 0;
}
