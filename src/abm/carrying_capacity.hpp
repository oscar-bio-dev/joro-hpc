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

/// Índice compuesto ortogonalizado de hábitat (tejedoras orbiculares sinantrópicas y de borde):
///   H_i^hab = std_norm( 0.50 Z_borde + 0.30 Z_imp + 0.20 Z_dosel )
/// donde std_norm centra y escala a varianza unitaria sobre TODAS las celdas de grid_5km.
/// Devuelve {media, desviación estándar} de la combinación cruda (para auditoría).
inline std::pair<double, double> compute_habitat_index(std::vector<GridCell>& cells) {
    if (cells.empty()) return {0.0, 1.0};
    double sum = 0.0, sum2 = 0.0;
    for (const auto& c : cells) {
        const double h = 0.50 * c.z_borde + 0.30 * c.z_imp + 0.20 * c.z_dosel;
        sum += h; sum2 += h * h;
    }
    const double n = static_cast<double>(cells.size());
    const double mean = sum / n;
    const double var = (sum2 - n * mean * mean) / std::max(1.0, n - 1.0);
    const double sd = std::sqrt(std::max(var, 1e-12));
    for (auto& c : cells) {
        const double h = 0.50 * c.z_borde + 0.30 * c.z_imp + 0.20 * c.z_dosel;
        c.h_hab = static_cast<float>((h - mean) / sd);
    }
    return {mean, sd};
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
