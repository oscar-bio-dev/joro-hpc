#pragma once

#include "common/types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace joro::inference {

/// Modelo dinámico de ocupación latente y detección imperfecta (Secciones 9-10 de JORO-HPC_Research_Proposal.tex, v2.0)
struct OccupancyParameters {
    // Colonización mecanística cloglog (Ec. 22): cloglog(gamma) = a0 + a2 H_hab + a1 log(A_total)
    //   a1 = 1: propágulos independientes (hipótesis nula); a1 > 1 efecto Allee, a1 < 1 interferencia
    float a0 = -4.0f;
    float a1_arrival = 1.0f;
    float a2_hab = 0.35f;

    // Parámetros de persistencia phi: logit(phi) = b0 + b1*log(1 + N) - b2*D
    float b0 = 1.5f;
    float b1_density = 0.8f;
    float b2_damage = 0.45f;

    // Parámetros de detección p: logit(p) = beta0 + beta1*log(1 + E^{-J}) + beta2*H + beta3*R + f_est
    float beta0 = -3.2f;
    float beta1_effort = 0.75f;
    float beta2_human = 0.25f;
    float beta3_road = 0.30f;

    // Observación no estacionaria f_est(i,t) = lambda_M log(1 + M_t/M_2019) - kappa_tau log(1 + tau_{i,t}) (Ec. 24-25).
    // M_t es el índice exógeno de actividad de plataforma (registros GBIF de Araneae no-Joro), NO atención mediática.
    // Valores por defecto estimados estrictamente en 2014-2019 (cero fuga temporal hacia 2020-2024; ver
    // local_pipeline/ingestion/fit_observation_nonstationarity.py y observation_fit_2014_2019.json):
    //   lambda_M  = 1.1533 ± 0.1762 (IC 95 % [0.8079, 1.4987], p < 1e-10; o 1.1464 ± 0.1763 si kappa_tau = 0)
    //   kappa_tau = 0.0797 ± 0.4791 (IC 95 % [-0.8594, +1.0187], p = 0.8679; estadísticamente no identificable en 2014-2019)
    // Se pueden sobrescribir con JORO_LAMBDA_M / JORO_KAPPA_TAU. lambda_M = kappa_tau = 0 anula f_est.
    float lambda_M = 1.1533f;
    float kappa_tau = 0.0797f;

    // Kernel mixto humano (Sección 7, Ec. atot): A_total = (1 - p_h) A_LPDM + p_h A_hum.  p_h ≡ w_h (prior log-uniforme
    // [1e-4, 1e-1]); lambda_h es la escala de la gravedad antropogénica por corredores viales. p_h = 0 => modelo atmosférico puro.
    float w_h = 0.0f;
    float lambda_h_km = 150.0f;

    /// Lee sobrescrituras desde el entorno (JORO_LAMBDA_M, JORO_KAPPA_TAU, JORO_P_HUMAN, JORO_LAMBDA_HUMAN_KM)
    static OccupancyParameters from_environment() {
        OccupancyParameters p;
        if (const char* e = std::getenv("JORO_LAMBDA_M")) p.lambda_M = std::strtof(e, nullptr);
        if (const char* e = std::getenv("JORO_KAPPA_TAU")) p.kappa_tau = std::strtof(e, nullptr);
        if (const char* e = std::getenv("JORO_P_HUMAN")) p.w_h = std::strtof(e, nullptr);
        if (const char* e = std::getenv("JORO_LAMBDA_HUMAN_KM")) p.lambda_h_km = std::strtof(e, nullptr);
        return p;
    }
};

/// Índice exógeno de atención/actividad M_t/M_2019 por año (2014..2024). Sin archivo => 1.0 (f_est solo por tau).
struct AttentionIndex {
    static constexpr int kFirstYear = 2014;
    static constexpr int kNYears = 11;
    std::array<float, kNYears> ratio{};
    bool loaded = false;

    AttentionIndex() { ratio.fill(1.0f); }

    [[nodiscard]] float ratio_for(int year) const noexcept {
        const int i = std::clamp(year - kFirstYear, 0, kNYears - 1);
        return ratio[static_cast<size_t>(i)];
    }

    /// CSV con cabecera `year,M_t,M_ratio_2019` (proxy exógeno de actividad de plataforma; ver pipeline de datos)
    bool load(const std::string& csv_path) {
        std::ifstream f(csv_path);
        if (!f.is_open()) return false;
        std::string line;
        std::getline(f, line);
        std::vector<std::string> headers;
        { std::stringstream hs(line); std::string t; while (std::getline(hs, t, ',')) headers.push_back(t); }
        int c_year = -1, c_ratio = -1;
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == "year") c_year = static_cast<int>(i);
            if (headers[i] == "M_ratio_2019") c_ratio = static_cast<int>(i);
        }
        if (c_year < 0 || c_ratio < 0) return false;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string t;
            std::vector<std::string> tok;
            while (std::getline(ss, t, ',')) tok.push_back(t);
            if (static_cast<int>(tok.size()) <= std::max(c_year, c_ratio)) continue;
            const int y = std::stoi(tok[static_cast<size_t>(c_year)]);
            if (y >= kFirstYear && y < kFirstYear + kNYears) ratio[static_cast<size_t>(y - kFirstYear)] = std::stof(tok[static_cast<size_t>(c_ratio)]);
        }
        loaded = true;
        return true;
    }
};

[[nodiscard]] inline float sigmoid(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

/// Llegadas totales A_total = (1 - p_h) A_LPDM + A_hum (Ec. atot). `human_arrivals` ya incorpora el factor p_h
/// (se muestrea explícitamente con Poisson(p_h P_j)); con p_h = 0 se recupera el modelo puramente atmosférico.
[[nodiscard]] inline float total_arrivals(float aero_arrivals, float human_arrivals, float p_h) noexcept {
    return (1.0f - p_h) * aero_arrivals + human_arrivals;
}

/// Probabilidad de colonización con enlace log-log complementario (Ec. 22):
///   gamma = 1 - exp(-lambda_c),  cloglog(gamma) = a0 + a2 H_hab + a1 log(A_total)
/// Se cumple gamma = 0 EXACTAMENTE cuando A_total = 0 (sin generación espontánea ni tasa basal en la envolvente).
[[nodiscard]] inline float compute_colonization_gamma(
    float total_arrivals_A,
    float habitat_index_H,
    const OccupancyParameters& p = OccupancyParameters{}) noexcept
{
    if (!(total_arrivals_A > 0.0f)) return 0.0f;
    const float eta = p.a0 + p.a2_hab * habitat_index_H + p.a1_arrival * std::log(total_arrivals_A);
    const float lambda_c = std::exp(std::clamp(eta, -30.0f, 12.0f));
    return -std::expm1(-lambda_c);
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

/// Término no estacionario de detección (Ec. 25): índice de actividad de plataforma M_t (Araneae no-Joro) y habituación local tau_{i,t} [años]
[[nodiscard]] inline float compute_f_est(float m_ratio_2019, float tau_invasion_age, const OccupancyParameters& p) noexcept
{
    return p.lambda_M * std::log1p(std::max(0.0f, m_ratio_2019)) - p.kappa_tau * std::log1p(std::max(0.0f, tau_invasion_age));
}

/// Probabilidad de detección p_{i,t} condicional al esfuerzo EXÓGENO E^{-J} = max(0, C^Araneae - C^J) (Ec. 24)
[[nodiscard]] inline float compute_detection_prob(
    int32_t effort_exogenous,
    float human_density_H,
    float road_density_R,
    const OccupancyParameters& p = OccupancyParameters{},
    float f_est = 0.0f) noexcept 
{
    float logit_val = p.beta0 + p.beta1_effort * std::log1p(static_cast<float>(std::max(0, effort_exogenous))) +
                      p.beta2_human * human_density_H +
                      p.beta3_road * road_density_R +
                      f_est;
    return sigmoid(logit_val);
}

/// Actualiza la probabilidad latente de ocupación psi_{i,t} (Sección 9 de JORO-HPC_Research_Proposal.tex):
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
