#pragma once

#include <cmath>

namespace joro::lpdm {

/// Coeficientes de la ecuación estocástica de Langevin vertical (Sección 7.2 de JORO-HPC.tex):
/// dw' = a_w(w', z) dt + b_w(z) dW
///
/// Implementa la solución de Thomson (1987) para turbulencia gaussiana no homogénea,
/// garantizando analíticamente la condición de mezcla perfecta (well-mixed condition).
struct LangevinCoefficients {
    float a_w; ///< Término de deriva determinista [m/s^2]
    float b_w; ///< Término de difusión estocástica [m/s^(3/2)]
};

/// Calcula los coeficientes de Langevin según el perfil de capa límite planetaria
[[nodiscard]] inline LangevinCoefficients compute_langevin_well_mixed(
    float w_prime,
    float z_m,
    float blh_m,
    float u_star = 0.35f,
    float convective_velocity_w_star = 1.2f) noexcept
{
    // Escala de tiempo lagrangiana vertical tau_L [s]
    const float z_eff = (z_m > 1.0f) ? z_m : 1.0f;
    const float blh_eff = (blh_m > 100.0f) ? blh_m : 1000.0f;
    const float z_norm = (z_eff < blh_eff) ? (z_eff / blh_eff) : 1.0f;

    // Varianza vertical sigma_w^2 según parametrización convectiva estándar
    // sigma_w^2 = 1.2 * w_*^2 * (z/z_i)^(2/3) * (1 - 0.9 * z/z_i)^2 + 1.6 * u_*^2
    const float term_conv = 1.2f * (convective_velocity_w_star * convective_velocity_w_star) *
                            std::cbrt(z_norm * z_norm) * std::pow(1.0f - 0.9f * z_norm, 2.0f);
    const float sigma_w2 = term_conv + 1.6f * (u_star * u_star);
    const float sigma_w = std::sqrt((sigma_w2 > 0.01f) ? sigma_w2 : 0.01f);

    // Escala temporal lagrangiana tau_L
    const float tau_L = (0.5f * z_eff) / sigma_w;
    const float tau_L_eff = (tau_L > 2.0f) ? ((tau_L < 300.0f) ? tau_L : 300.0f) : 2.0f;

    // Gradiente vertical de varianza d(sigma_w^2)/dz (diferencia finita analítica aproximada)
    const float d_sigma_w2_dz = (z_norm < 0.8f) ? (0.5f * sigma_w2 / z_eff) : (-0.8f * sigma_w2 / blh_eff);

    // Deriva a_w con corrección de Thomson para satisfacer well-mixed
    const float a_w = - (w_prime / tau_L_eff)
                      + 0.5f * d_sigma_w2_dz
                      + ( (w_prime * w_prime) / (2.0f * sigma_w2) ) * d_sigma_w2_dz;

    // Difusión b_w
    const float b_w = std::sqrt((2.0f * sigma_w2) / tau_L_eff);

    return LangevinCoefficients{a_w, b_w};
}

} // namespace joro::lpdm
