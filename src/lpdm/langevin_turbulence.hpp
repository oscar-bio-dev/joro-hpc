#pragma once

#include "lpdm/device_math.hpp"
#include <algorithm>
#include <cmath>

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

/// Calcula los coeficientes de Langevin HOMOGÉNEOS (sigma_w, tau_L constantes).
/// Es el esquema de referencia analítico de las pruebas de verificación (well-mixed puro, convergencia
/// Euler-Maruyama); el transporte de producción usa `langevin_advance` (perfil inhomogéneo, M4).
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

/// Perfil vertical de turbulencia de la capa límite planetaria (CBL convectiva + cizalla mecánica)
struct TurbulenceProfile {
    float sigma_w;     ///< Desviación estándar de la velocidad vertical [m/s]
    float tau_L;       ///< Escala lagrangiana integral [s]
    float dsigma2_dz;  ///< Gradiente vertical de la varianza d(sigma_w^2)/dz [m/s^2]
};

/// Perfil estándar (Hanna 1982 / Weil): con zeta = clamp(z/BLH, 0.02, 0.99)
///   sigma_w^2 = 1.2 w*^2 zeta^(2/3) (1 - 0.8 zeta)^2 + (1.3 u*)^2 (1 - zeta)^(3/2)
///   tau_L     = clamp(0.15 BLH / sigma_w (1 - exp(-5 zeta)), 10, 300) s
/// Por encima de la capa límite (z >= BLH): atmósfera libre residual sigma_w = 0.05 m/s, tau_L = 60 s.
[[nodiscard]] inline TurbulenceProfile compute_turbulence_profile(
    float z_m, float blh_m, float u_star, float w_star) noexcept
{
    constexpr float sigma_free = 0.05f;
    constexpr float tau_free = 60.0f;
    const float blh = jm::maxf(blh_m, 50.0f);
    if (z_m >= blh) return TurbulenceProfile{sigma_free, tau_free, 0.0f};

    const float zeta = jm::clampf(z_m / blh, 0.02f, 0.99f);
    const float ws2 = w_star * w_star;
    const float us = 1.3f * u_star;
    const float us2 = us * us;
    const float a = jm::pow(zeta, 2.0f / 3.0f);
    const float b = 1.0f - 0.8f * zeta;
    const float c = jm::pow(1.0f - zeta, 1.5f);

    const float var = 1.2f * ws2 * a * b * b + us2 * c;
    const float sigma = jm::maxf(jm::sqrt(var), sigma_free);
    const float tau = jm::clampf(0.15f * blh / sigma * (1.0f - jm::exp(-5.0f * zeta)), 10.0f, 300.0f);

    // d(var)/d(zeta) analítico; el gradiente se anula donde actúa el piso sigma_free
    const float dvar_dzeta = 1.2f * ws2 * ((2.0f / 3.0f) * jm::pow(zeta, -1.0f / 3.0f) * b * b - 1.6f * a * b)
                             - 1.5f * us2 * jm::sqrt(1.0f - zeta);
    const float dsig2_dz = (var > sigma_free * sigma_free) ? dvar_dzeta / blh : 0.0f;
    return TurbulenceProfile{sigma, tau, dsig2_dz};
}

/// Avance de un paso dt de la fluctuación vertical w' con turbulencia gaussiana INHOMOGÉNEA (Thomson 1987):
///   dw' = [ -w'/tau_L + 0.5 (1 + w'^2/sigma_w^2) d(sigma_w^2)/dz ] dt + sqrt(2 sigma_w^2 / tau_L) dW
/// La parte de fricción/difusión se integra con la solución exacta de Ornstein-Uhlenbeck (estable aunque
/// dt > tau_L, p. ej. dt = 60 s con tau_L >= 10 s); la deriva de Thomson se añade explícitamente y acotada.
/// `xi` ~ N(0,1) provisto por el generador de la partícula.
[[nodiscard]] inline float langevin_advance(
    float w_prime, float z_m, float blh_m, float u_star, float w_star, float dt, float xi) noexcept
{
    const TurbulenceProfile p = compute_turbulence_profile(z_m, blh_m, u_star, w_star);
    const float decay = jm::exp(-dt / p.tau_L);
    const float ratio = jm::clampf(w_prime / p.sigma_w, -3.0f, 3.0f);
    const float drift = 0.5f * (1.0f + ratio * ratio) * p.dsigma2_dz * dt;
    const float drift_lim = jm::clampf(drift, -p.sigma_w, p.sigma_w);
    return w_prime * decay + p.sigma_w * jm::sqrt(1.0f - decay * decay) * xi + drift_lim;
}

} // namespace joro::lpdm
