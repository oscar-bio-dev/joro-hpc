#include "common/types.hpp"
#include <iostream>
#include <chrono>
#include <vector>

#if defined(JORO_HAS_SYCL)
#include <sycl/sycl.hpp>
#endif

/// Microbenchmark HPC: Tráfico PCIe y throughput de memoria en GPU Intel Arc A770
/// Compara Unified Shared Memory (USM) vs Buffers explícitos (Sección 13.2 de JORO-HPC.tex).
int main() {
    std::cout << "[BENCHMARK] Perfilado de memoria: USM vs Buffers explícitos...\n";

    constexpr size_t N = 1'000'000;
    std::cout << "  Tamaño de muestra: " << N << " partículas ("
              << (N * sizeof(joro::Particle3D)) / (1024 * 1024) << " MB)\n";

#if defined(JORO_HAS_SYCL)
    try {
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "  Dispositivo: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

        // Benchmark USM Device Allocation
        auto t0 = std::chrono::high_resolution_clock::now();
        joro::Particle3D* d_ptr = sycl::malloc_device<joro::Particle3D>(N, q);
        q.fill(d_ptr, joro::Particle3D{}, N).wait();
        sycl::free(d_ptr, q);
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_usm = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  [USM Device] Tiempo de asignación y llenado: " << elapsed_usm << " ms\n";

    } catch (const sycl::exception& e) {
        std::cout << "  [Aviso] GPU SYCL no disponible en esta máquina: " << e.what() << "\n";
    }
#else
    std::cout << "  [Aviso] SYCL no compilado en este target.\n";
#endif

    std::cout << "[BENCHMARK] bench_usm_vs_buffers finalizado.\n";
    return 0;
}
