#pragma once

#include "common/types.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace joro::inference {

/// Modelo dinámico de ocupación latente y detección imperfecta (Sección 9 de JORO-HPC.tex)
struct OccupancyParameters {
    // Parámetros de colonización gamma: logit(gamma) = a0 + a1*log(1 + A) + a2*H
    float a0 = -4.0f;
    float a1_arrival = 1.2f;
    float a2_human = 0.35f;

    // Parámetros de persistencia phi: logit(phi) = b0 + b1*log(1 + N) - b2*D
    float b0 = 1.5f;
    float b1_density = 0.8f;
    float b2_damage = 0.45f;

    // Parámetros de detección p: logit(p) = beta0 + beta1*log(1 + E_araneae) + beta2*H + beta3*R
    float beta0 = -3.2f;
    float beta1_effort = 0.75f;
    float beta2_human = 0.25f;
    float beta3_road = 0.30f;
};

[[nodiscard]] inline float sigmoid(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

/// Calcula la probabilidad de colonización gamma_{i,t} condicional a las llegadas A_{i,t}
[[nodiscard]] inline float compute_colonization_gamma(
    uint32_t arrivals_A,
    float human_density_H,
    const OccupancyParameters& p = OccupancyParameters{}) noexcept 
{
    if (arrivals_A == 0) return 0.0f;
    float logit_val = p.a0 + p.a1_arrival * std::log1p(static_cast<float>(arrivals_A)) +
                      p.a2_human * human_density_H;
    return sigmoid(logit_val);
}

/// Calcula la probabilidad de persistencia phi_{i,t} condicional a N y al daño D
[[nodiscard]] inline float compute_persistence_phi(
    float adult_density_N,
    float mean_damage_D,
    const OccupancyParameters& p = OccupancyParameters{}) noexcept 
{
    float logit_val = p.b0 + p.b1_density * std::log1p(adult_density_N) -
                      p.b2_damage * mean_damage_D;
    return sigmoid(logit_val);
}

/// Calcula la probabilidad de detección p_{i,t} condicional al esfuerzo Araneae E (Sección 9)
[[nodiscard]] inline float compute_detection_prob(
    int32_t effort_araneae,
    float human_density_H,
    float road_density_R,
    const OccupancyParameters& p = OccupancyParameters{},
    float f_est = 0.0f) noexcept 
{
    float logit_val = p.beta0 + p.beta1_effort * std::log1p(static_cast<float>(effort_araneae)) +
                      p.beta2_human * human_density_H +
                      p.beta3_road * road_density_R +
                      f_est;
    return sigmoid(logit_val);
}

/// Actualiza la probabilidad latente de ocupación psi_{i,t} (Sección 9 de JORO-HPC.tex):
/// psi_{i,t} = (1 - psi_{i,t-1}) * gamma_{i,t} + psi_{i,t-1} * phi_{i,t}
[[nodiscard]] inline float update_dynamic_occupancy(
    float psi_prev,
    float gamma_colonization,
    float phi_persistence) noexcept 
{
    return (1.0f - psi_prev) * gamma_colonization + psi_prev * phi_persistence;
}

/// Probabilidad marginal de registro observado (Sección 9):
/// P_hat(Y_{i,t} = 1) = psi_{i,t} * p_{i,t}
[[nodiscard]] inline float marginal_observation_prob(
    float psi_occupancy,
    float p_detection) noexcept 
{
    return std::clamp(psi_occupancy * p_detection, 0.0f, 1.0f);
}

} // namespace joro::inference
