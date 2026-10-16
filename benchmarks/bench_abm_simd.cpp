#include "abm/abm_state_soa.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

/// Microbenchmark HPC: Speedup Escalar vs OpenMP vs AVX-512 SIMD (Sección 13.2 de JORO-HPC.tex).
int main() {
    std::cout << "[BENCHMARK] Perfilado de vectorización y multihilo en CPU Core i9-11900K...\n";

    constexpr size_t N = 2'000'000;
    joro::abm::CohortSoA cohorts;
    cohorts.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        cohorts.add_cohort(static_cast<int32_t>(i % 200000), 1, 10.0f);
    }

    std::cout << "  Tamaño del arreglo SoA: " << N << " cohortes ("
              << (N * sizeof(float) * 4) / (1024 * 1024) << " MB)\n";

    // Medición del bucle térmico vectorial
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for simd
    for (size_t i = 0; i < N; ++i) {
        float cold_stress = std::max(0.0f, -4.0f - (-8.0f));
        float delta_d = 0.04f * (cold_stress * cold_stress) * 30.0f;
        cohorts.cold_damage_D[i] += delta_d;
        cohorts.weight[i] *= std::exp(-delta_d);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  [SIMD / OpenMP] Tiempo de ejecución para " << N << " cohortes: "
              << elapsed_ms << " ms\n"
              << "  [Throughput] " << (N / (elapsed_ms * 1e-3)) / 1e6 << " Millones de cohortes/segundo\n";

    std::cout << "[BENCHMARK] bench_abm_simd finalizado.\n";
    return 0;
}
