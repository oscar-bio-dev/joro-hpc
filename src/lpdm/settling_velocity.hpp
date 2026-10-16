#pragma once

#include "common/types.hpp"
#include <cmath>

namespace joro::lpdm {

/// Calcula la velocidad terminal de sedimentación v_s [m/s] de un juvenil con seda
/// de acuerdo con la Sección 7.3 de JORO-HPC.tex (Humphrey 1987, Suter 1992).
///
/// v_s = h(M, L_s, n_s, rho_air, mu_air, chi)
///
/// Parámetros:
///   mass_mg      : Masa del juvenil [mg] (típicamente 0.2 a 5.0 mg para estadios II-III)
///   silk_len_m   : Longitud total del hilo de baloneo [m] (0.05 a 2.30 m)
///   posture_chi  : Factor aerodinámico de postura (0.5 = compacta, 1.0 = paracaídas extendido)
///   rho_air      : Densidad del aire en la altitud actual [kg/m^3]
///   mu_air       : Viscosidad dinámica del aire [Pa*s]
[[nodiscard]] inline float compute_settling_velocity(
    float mass_mg,
    float silk_len_m,
    float posture_chi = 0.85f,
    float rho_air = static_cast<float>(constants::rho_air_sfc),
    float mu_air = static_cast<float>(constants::mu_air)) noexcept
{
    // Conversión de masa a kg
    const float mass_kg = mass_mg * 1.0e-6f;
    const float weight_n = mass_kg * static_cast<float>(constants::g);

    // Diámetro característico del hilo de seda de baloneo (~2.0 micrómetros)
    constexpr float d_silk = 2.0e-6f;
    
    // Resistencia viscosa por unidad de longitud según régimen laminar de Stokes-Oseen
    // C_drag_silk ~ 4*pi*mu / (ln(L/d) + 0.5)
    const float aspect_ratio = (silk_len_m > 0.01f) ? (silk_len_m / d_silk) : 1000.0f;
    const float log_aspect = std::log(aspect_ratio);
    const float drag_coef_silk = (4.0f * static_cast<float>(M_PI) * mu_air * silk_len_m) / (log_aspect + 0.5f);

    // Resistencia del cuerpo de la araña (esfera equivalente modificada por postura chi)
    // Radio corporal r_body ~ (3 * M / (4 * pi * rho_spider))^(1/3), con rho_spider ~ 1000 kg/m^3
    const float r_body = std::cbrt((3.0f * mass_kg) / (4.0f * static_cast<float>(M_PI) * 1000.0f));
    const float drag_coef_body = 6.0f * static_cast<float>(M_PI) * mu_air * r_body * (1.0f + 0.5f * (1.0f - posture_chi));

    const float total_drag_resistance = drag_coef_silk + drag_coef_body;

    if (total_drag_resistance <= 1e-9f) {
        return 1.0f; // Caída libre límite
    }

    // Velocidad de equilibrio donde Arrastre = Peso
    float v_s = weight_n / total_drag_resistance;

    // Acotar físicamente a valores empíricos observados en arañas tejedoras (0.05 a 2.5 m/s)
    if (v_s < 0.02f) v_s = 0.02f;
    if (v_s > 3.00f) v_s = 3.00f;

    return v_s;
}

} // namespace joro::lpdm
