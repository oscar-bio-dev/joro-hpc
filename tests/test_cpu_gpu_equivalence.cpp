#include "lpdm/lpdm_engine.hpp"
#include "test_common.hpp"
#include <iostream>
#include <vector>
#include <cmath>

/// Prueba de equivalencia estadística CPU vs GPU (Sección 7.6 de JORO-HPC_Research_Proposal.tex).
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

    // ── Caso 2: viento ERA5 no uniforme residente en ambos motores (u, v, w, BLH interpolados) ──
    {
        joro::io::AtmosphericField3D atm;
        atm.n_times = 6; atm.n_levels = 12; atm.n_lats = 81; atm.n_lons = 101;
        atm.pressure_levels_hpa = {1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700};
        for (int i = 0; i < atm.n_lats; ++i) atm.lats.push_back(45.0f - 0.25f * static_cast<float>(i));
        for (int j = 0; j < atm.n_lons; ++j) atm.lons.push_back(-95.0f + 0.25f * static_cast<float>(j));
        const size_t hp = static_cast<size_t>(atm.n_lats) * atm.n_lons;
        const size_t total = static_cast<size_t>(atm.n_times) * atm.n_levels * hp;
        atm.u.resize(total); atm.v.resize(total); atm.w.assign(total, 0.01f);
        for (size_t idx = 0; idx < total; ++idx) {
            const int lev = static_cast<int>((idx / hp) % atm.n_levels);
            const int lon_i = static_cast<int>(idx % atm.n_lons);
            atm.u[idx] = 4.0f + 0.2f * static_cast<float>(lev) + 0.01f * static_cast<float>(lon_i);
            atm.v[idx] = 1.0f + 0.05f * static_cast<float>(lev);
        }
        atm.sp.assign(static_cast<size_t>(atm.n_times) * hp, 101325.0f);
        atm.blh.assign(static_cast<size_t>(atm.n_times) * hp, 900.0f);

        auto cpu2 = joro::lpdm::create_cpu_baseline_engine();
        auto gpu2 = joro::lpdm::create_sycl_gpu_engine();
        JORO_TEST_ASSERT(cpu2->set_atmosphere(atm), "CPU: ERA5 sintético rechazado");
        JORO_TEST_ASSERT(gpu2->set_atmosphere(atm), "GPU: ERA5 sintético rechazado");
        JORO_TEST_ASSERT(cpu2->is_atmosphere_resident() && gpu2->is_atmosphere_resident(), "ERA5 no residente");

        constexpr uint32_t n2 = 4000;
        std::unordered_map<int32_t, uint32_t> release2;
        release2[5] = n2;
        cpu2->initialize_particles(cells, release2);
        gpu2->initialize_particles(cells, release2);
        auto c2 = cpu2->run_dispersion(atm, 60.0f, 60);
        auto g2 = gpu2->run_dispersion(atm, 60.0f, 60);
        JORO_TEST_ASSERT(c2.total_released == n2 && g2.total_released == n2, "Conservación 1:1 violada con ERA5");
        JORO_TEST_ASSERT(c2.total_landed <= n2 && g2.total_landed <= n2, "Más aterrizajes que propágulos");
        const double r_cpu = static_cast<double>(c2.total_landed) / n2;
        const double r_gpu = static_cast<double>(g2.total_landed) / n2;
        std::cout << "  ERA5: aterrizaje CPU " << r_cpu << " | GPU " << r_gpu
                  << " | distancia media CPU " << c2.mean_jump_distance_km << " km | GPU " << g2.mean_jump_distance_km << " km\n";
        JORO_TEST_ASSERT(std::abs(r_cpu - r_gpu) < 0.15, "Tasa de aterrizaje CPU/GPU diverge con ERA5");
        JORO_TEST_ASSERT(c2.mean_jump_distance_km > 2.0f && g2.mean_jump_distance_km > 2.0f, "El viento ERA5 no advecta las partículas");
        const double rel = std::abs(c2.mean_jump_distance_km - g2.mean_jump_distance_km) / std::max(1e-3f, c2.mean_jump_distance_km);
        JORO_TEST_ASSERT(rel < 0.25, "Distancia media de salto CPU/GPU diverge con ERA5");
    }

    // ── Caso 3: tope adaptativo de super-partículas conserva exactamente los propágulos ──
    {
        auto gpu3 = joro::lpdm::create_sycl_gpu_engine();
        gpu3->set_max_super_particles(500);
        std::unordered_map<int32_t, uint32_t> release3;
        for (int id = 1; id <= 9; ++id) release3[id] = 20000u + static_cast<uint32_t>(id);
        gpu3->initialize_particles(cells, release3);
        auto g3 = gpu3->run_dispersion(mock_atmo, 10.0f, 60);
        uint64_t expect = 0;
        for (const auto& kv : release3) expect += kv.second;
        JORO_TEST_ASSERT(g3.total_released == expect, "El tope adaptativo rompió la conservación de propágulos");
        JORO_TEST_ASSERT(gpu3->telemetry().capped_dispatches == 1, "El tope adaptativo no se aplicó");
        JORO_TEST_ASSERT(gpu3->telemetry().particles_dispatched < 1000, "El tope adaptativo no redujo las super-partículas");
        uint64_t arrivals = 0;
        for (const auto& kv : g3.arrivals_per_cell) arrivals += kv.second;
        JORO_TEST_ASSERT(arrivals <= g3.total_landed, "Llegadas por celda exceden los aterrizajes");
    }
#else
    std::cout << "  [Nota] Binario compilado sin SYCL (CPU-only). Equivalencia GPU verificada en CI oneAPI.\n";
#endif

    std::cout << "✔ test_cpu_gpu_equivalence PASÓ exitosamente.\n";
    return 0;
}
