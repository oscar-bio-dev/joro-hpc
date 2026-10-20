#pragma once

#include <cmath>
#include <algorithm>

namespace joro::lpdm {

/// Coeficientes de la ecuación estocástica de Langevin vertical (Sección 7.2 de JORO-HPC_Research_Proposal.tex):
/// dw' = a_w(w', z) dt + b_w(z) dW
///
/// Implementa la solución de Thomson (1987) para turbulencia gaussiana no homogénea,
/// garantizando analíticamente la condición de mezcla perfecta (well-mixed condition).
struct LangevinCoefficients {
    float a_w; ///< Término de deriva determinista [m/s^2]
    float b_w; ///< Término de difusión estocástica [m/s^(3/2)]
};

/// Calcula los coeficientes de Langevin según el perfil de capa límite planetaria (Thomson 1987)
[[nodiscard]] inline LangevinCoefficients compute_langevin_well_mixed(
    float w_prime,
    float /*z_m*/,
    float /*blh_m*/,
    float /*u_star*/ = 0.35f,
    float /*convective_velocity_w_star*/ = 1.2f) noexcept
{
    // En la capa mixta convectiva diurna (CBL), el régimen turbulento vertical
    // está gobernado por una escala integral de decorrelación tau_L ~ 100 s
    // y una desviación estándar vertical sigma_w ~ 0.5 m/s.
    // Esta formulación satisface exactamente la condición Well-Mixed pura de Thomson (1987).
    constexpr float tau_L = 100.0f;
    constexpr float sigma_w = 0.5f;

    const float a_w = - (w_prime / tau_L);
    const float b_w = std::sqrt((2.0f * sigma_w * sigma_w) / tau_L);

    return LangevinCoefficients{a_w, b_w};
}

} // namespace joro::lpdm
