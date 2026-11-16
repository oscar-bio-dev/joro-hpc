#pragma once

#include "common/types.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace joro::abm {

/// Modelos candidatos de capacidad de carga (Sección 8.4 de JORO-HPC_Research_Proposal.tex:
/// "Se compararán modelos completos, univariados y K_i = K_0").
enum class CarryingModel : int {
    M0 = 0,  ///< K_i = K_0                                   (gamma_hab = 0; 8 parámetros libres)
    M1 = 1,  ///< K_i = K_0 exp(gamma_hab * H_i^hab)           (modelo reparametrizado principal; 9 libres)
    M3 = 3   ///< K_i = K_0 exp(g1 Z_imp + g2 Z_borde + g3 Z_dosel)  (completo, sobreparametrizado; 11 libres)
};

[[nodiscard]] inline const char* carrying_model_name(CarryingModel m) noexcept {
    switch (m) {
        case CarryingModel::M0: return "M0";
        case CarryingModel::M1: return "M1";
        default:                return "M3";
    }
}

/// Parámetros de capacidad de carga K_i (Sección 8.4 de JORO-HPC_Research_Proposal.tex)
struct CarryingCapacityParams {
    float k0 = 5000.0f;           ///< Capacidad de carga base en hábitat promedio
    float gamma1_imp = 0.45f;     ///< Coeficiente para impermeabilidad urbana Z_imp          (M3)
    float gamma2_borde = 0.65f;   ///< Coeficiente para densidad de borde de bosque Z_borde   (M3)
    float gamma3_dosel = 0.30f;   ///< Coeficiente para dosel arbóreo Z_dosel                 (M3)
    float k_min = 100.0f;         ///< Cota inferior
    float k_max = 50000.0f;       ///< Cota superior
    float gamma_hab = 0.0f;       ///< Pendiente global de selección de hábitat sobre H_i^hab  (M0/M1)
    bool use_hab_index = false;   ///< true: K_i = K0 exp(gamma_hab H_i^hab); false: tres gammas (M3, modo heredado)
};

/// Índice compuesto de hábitat (tejedoras orbiculares sinantrópicas y de borde):
///   H_i^hab = (h_i - m_S) / s_S,   h_i = 0.50 Z_borde + 0.30 Z_imp + 0.20 Z_dosel
/// La media m_S y la desviación s_S se calculan sobre el SOPORTE INFORMATIVO S (celdas con esfuerzo de observación en el
/// dominio de calibración; Ec. 19 y 31 del .tex), de modo que (1/|S|) Σ_{i∈S} H_i^hab = 0 y se anula la covarianza
/// estructural Cov(κ̂, γ̂_hab) ≈ -m_S Var(γ̂_hab) (corr = -0.92 al estandarizar sobre todo el continente Ω).
/// `support[i] != 0` marca las celdas de S (mismo orden que `cells`); si es nulo/vacío se usa Ω (comportamiento heredado).
/// El índice se aplica a TODAS las celdas. Devuelve {m_S, s_S} de la combinación cruda (para auditoría).
inline std::pair<double, double> compute_habitat_index(std::vector<GridCell>& cells,
                                                       const std::vector<uint8_t>* support = nullptr) {
    if (cells.empty()) return {0.0, 1.0};
    const bool use_support = support != nullptr && support->size() == cells.size();
    double sum = 0.0, sum2 = 0.0;
    size_t n_used = 0;
    for (size_t i = 0; i < cells.size(); ++i) {
        if (use_support && (*support)[i] == 0) continue;
        const auto& c = cells[i];
        const double h = 0.50 * c.z_borde + 0.30 * c.z_imp + 0.20 * c.z_dosel;
        sum += h; sum2 += h * h; ++n_used;
    }
    if (n_used < 2) {                       // soporte degenerado: se recurre a Ω
        return compute_habitat_index(cells, nullptr);
    }
    const double n = static_cast<double>(n_used);
    const double mean = sum / n;
    const double var = (sum2 - n * mean * mean) / std::max(1.0, n - 1.0);
    const double sd = std::sqrt(std::max(var, 1e-12));
    for (auto& c : cells) {
        const double h = 0.50 * c.z_borde + 0.30 * c.z_imp + 0.20 * c.z_dosel;
        c.h_hab = static_cast<float>((h - mean) / sd);
    }
    return {mean, sd};
}

/// Máscara del soporte informativo S a partir de los cell_id con esfuerzo observado (mismo orden que `cells`)
[[nodiscard]] inline std::vector<uint8_t> make_support_mask(const std::vector<GridCell>& cells,
                                                            const std::vector<int32_t>& support_cell_ids) {
    int32_t max_id = -1;
    for (const auto& c : cells) max_id = std::max(max_id, c.cell_id);
    std::vector<uint8_t> in_s(static_cast<size_t>(std::max(0, max_id + 1)), 0);
    for (int32_t id : support_cell_ids) if (id >= 0 && id <= max_id) in_s[static_cast<size_t>(id)] = 1;
    std::vector<uint8_t> mask(cells.size(), 0);
    for (size_t i = 0; i < cells.size(); ++i) {
        const int32_t id = cells[i].cell_id;
        if (id >= 0 && id <= max_id) mask[i] = in_s[static_cast<size_t>(id)];
    }
    return mask;
}

/// Calcula la capacidad de carga K_i para una celda a partir de covariables NLCD
[[nodiscard]] inline float compute_carrying_capacity(
    const GridCell& cell,
    const CarryingCapacityParams& params = CarryingCapacityParams{}) noexcept
{
    float exponent;
    if (params.use_hab_index) {
        // M1 / M0:  K_i = K0 * exp(gamma_hab * H_i^hab)   (M0 es gamma_hab = 0)
        exponent = params.gamma_hab * cell.h_hab;
    } else {
        // M3:  K_i = K0 * exp(gamma1 * z_imp + gamma2 * z_borde + gamma3 * z_dosel)
        exponent = params.gamma1_imp * cell.z_imp +
                   params.gamma2_borde * cell.z_borde +
                   params.gamma3_dosel * cell.z_dosel;
    }

    const float k_raw = params.k0 * std::exp(exponent);
    return std::clamp(k_raw, params.k_min, params.k_max);
}

} // namespace joro::abm
