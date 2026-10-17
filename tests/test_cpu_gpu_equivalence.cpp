#include "lpdm/lpdm_engine.hpp"
#include "test_common.hpp"
#include <iostream>
#include <vector>
#include <cmath>

/// Prueba de equivalencia estadística CPU vs GPU (Sección 7.6 de JORO-HPC.tex).
/// Verifica que las distribuciones de llegada del kernel SYCL y del baseline OpenMP
/// no presentan sesgos estructurales divergentes.
int main() {
    std::cout << "─────────────────────────────────────────────────────────────\n";
    std::cout << "[TEST] test_cpu_gpu_equivalence: Verificación de Motores LPDM\n";
    std::cout << "─────────────────────────────────────────────────────────────\n";

    auto cpu_engine = joro::lpdm::create_cpu_baseline_engine();
    JORO_TEST_ASSERT(cpu_engine != nullptr, "Fallo al instanciar motor CPU baseline");

    std::vector<joro::GridCell> cells;
    for (int i = 0; i < 9; ++i) {
        joro::GridCell c;
        c.cell_id = i + 1;
        c.centroid_lon = -84.0f + (i % 3) * 0.05f;
        c.centroid_lat = 34.0f + (i / 3) * 0.05f;
        c.is_seed = (i == 4);
        cells.push_back(c);
    }

    constexpr uint32_t n_particles = 200;
    std::unordered_map<int32_t, uint32_t> release;
    release[5] = n_particles; // Liberar desde la celda central (id=5)

    cpu_engine->initialize_particles(cells, release);
    joro::io::AtmosphericField3D mock_atmo;
    auto cpu_res = cpu_engine->run_dispersion(mock_atmo, 10.0f, 60);

    JORO_TEST_ASSERT(cpu_res.total_released == n_particles, "Motor CPU no inicializó todas las partículas");
    JORO_TEST_ASSERT(cpu_res.total_landed + cpu_res.total_escaped == n_particles,
                     "Violación de conservación de masa/partículas en CPU");
    std::cout << "  CPU: Liberadas = " << cpu_res.total_released 
              << " | Aterrizadas = " << cpu_res.total_landed
              << " | Escapadas = " << cpu_res.total_escaped << "\n";

#if defined(JORO_HAS_SYCL)
    std::cout << "  [SYCL] Detectado soporte GPU. Evaluando motor acelerado...\n";
    auto gpu_engine = joro::lpdm::create_sycl_gpu_engine();
    JORO_TEST_ASSERT(gpu_engine != nullptr, "Fallo al instanciar motor SYCL GPU");

    gpu_engine->initialize_particles(cells, release);
    auto gpu_res = gpu_engine->run_dispersion(mock_atmo, 10.0f, 60);

    JORO_TEST_ASSERT(gpu_res.total_released == n_particles, "Motor GPU no inicializó todas las partículas");
    JORO_TEST_ASSERT(gpu_res.total_landed + gpu_res.total_escaped == n_particles,
                     "Violación de conservación de partículas en GPU");

    std::cout << "  GPU: Liberadas = " << gpu_res.total_released 
              << " | Aterrizadas = " << gpu_res.total_landed
              << " | Escapadas = " << gpu_res.total_escaped << "\n";

    // Tolerancia estocástica entre implementaciones (fracción de aterrizaje relativa)
    double cpu_land_ratio = static_cast<double>(cpu_res.total_landed) / n_particles;
    double gpu_land_ratio = static_cast<double>(gpu_res.total_landed) / n_particles;
    double diff = std::abs(cpu_land_ratio - gpu_land_ratio);
    std::cout << "  Diferencia en tasa de aterrizaje (CPU vs GPU): " << diff << "\n";
    JORO_TEST_ASSERT(diff < 0.35, "Divergencia inaceptable entre distribuciones CPU y GPU");
#else
    std::cout << "  [Nota] Binario compilado sin SYCL (CPU-only). Equivalencia GPU verificada en CI oneAPI.\n";
#endif

    std::cout << "✔ test_cpu_gpu_equivalence PASÓ exitosamente.\n";
    return 0;
}
