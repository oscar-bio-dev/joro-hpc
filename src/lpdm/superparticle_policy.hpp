#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace joro::lpdm {

/// Tope de super-partículas por celda emisora (heredado, idéntico en CPU y GPU)
constexpr uint32_t kMaxSuperPerCell = 25000;
/// Presupuesto por defecto por despacho anual: ágil para pruebas unitarias y calibración
constexpr size_t kDefaultMaxSuperPerDispatch = 120000;
/// Presupuesto del modo v2.0 (corrida definitiva): hasta 5e6 super-partículas por despacho
constexpr size_t kMaxSuperPerDispatchV2 = 5000000;

/// Política UNIFICADA (M7) de tope y reescalado de pesos para los motores CPU (OpenMP) y GPU (SYCL).
/// Entrada: propágulos reales por celda emisora. Salida: super-partículas simuladas por celda, de modo que
/// la suma de pesos (total / n_sim por super-partícula, partición entera exacta) conserva EXACTAMENTE los
/// propágulos de cada celda. Si Σ n_sim excede `budget` se reduce n_sim de forma proporcional (mínimo 1) y el
/// peso por super-partícula crece. `budget == 0` desactiva el presupuesto global. Devuelve true si hubo reescalado.
inline bool plan_super_particles(const std::vector<uint32_t>& totals, size_t budget, std::vector<uint32_t>& n_sim)
{
    n_sim.resize(totals.size());
    uint64_t sum_sim = 0;
    for (size_t i = 0; i < totals.size(); ++i) {
        n_sim[i] = std::min(totals[i], kMaxSuperPerCell);
        sum_sim += n_sim[i];
    }
    if (budget == 0 || sum_sim <= budget) return false;
    const double f = static_cast<double>(budget) / static_cast<double>(sum_sim);
    for (size_t i = 0; i < totals.size(); ++i) {
        const uint32_t scaled = static_cast<uint32_t>(std::ceil(static_cast<double>(n_sim[i]) * f));
        n_sim[i] = std::max<uint32_t>(1u, std::min(n_sim[i], scaled));
    }
    return true;
}

} // namespace joro::lpdm

