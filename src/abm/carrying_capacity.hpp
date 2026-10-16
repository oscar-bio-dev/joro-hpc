#pragma once

#include "common/types.hpp"
#include <cmath>
#include <algorithm>

namespace joro::abm {

/// Parámetros de capacidad de carga K_i (Sección 8.4 de JORO-HPC.tex)
struct CarryingCapacityParams {
    float k0 = 5000.0f;           ///< Capacidad de carga base en hábitat promedio
    float gamma1_imp = 0.45f;     ///< Coeficiente para impermeabilidad urbana Z_imp
    float gamma2_borde = 0.65f;   ///< Coeficiente para densidad de borde de bosque Z_borde
    float gamma3_dosel = 0.30f;   ///< Coeficiente para dosel arbóreo Z_dosel
    float k_min = 100.0f;         ///< Cota inferior
    float k_max = 50000.0f;       ///< Cota superior
};

/// Calcula la capacidad de carga K_i para una celda a partir de covariables NLCD
[[nodiscard]] inline float compute_carrying_capacity(
    const GridCell& cell,
    const CarryingCapacityParams& params = CarryingCapacityParams{}) noexcept
{
    // K_i = K0 * exp(gamma1 * z_imp + gamma2 * z_borde + gamma3 * z_dosel)
    const float exponent = params.gamma1_imp * cell.z_imp +
                           params.gamma2_borde * cell.z_borde +
                           params.gamma3_dosel * cell.z_dosel;

    const float k_raw = params.k0 * std::exp(exponent);
    return std::clamp(k_raw, params.k_min, params.k_max);
}

} // namespace joro::abm
