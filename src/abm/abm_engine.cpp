#include "abm/abm_engine.hpp"
#include <cmath>
#include <random>
#include <algorithm>

namespace joro::abm {

AbmEngine::AbmEngine(DemographicParameters params)
    : params_(params) {}

void AbmEngine::initialize_seed(const std::vector<GridCell>& cells, float rho0) {
    cohorts_.clear();
    for (const auto& cell : cells) {
        if (cell.is_seed) {
            float k_i = compute_carrying_capacity(cell);
            float initial_n = rho0 * k_i;
            cohorts_.add_cohort(cell.cell_id, 3 /*adultos*/, initial_n);
        }
    }
}

void AbmEngine::deposit_arrivals(const std::unordered_map<int32_t, uint32_t>& arrivals) {
    for (const auto& [cell_id, count] : arrivals) {
        if (count > 0) {
            cohorts_.add_cohort(cell_id, 1 /*juveniles*/, static_cast<float>(count));
        }
    }
}

DemographyYearSummary AbmEngine::step_annual_cycle(
    const std::vector<GridCell>& cells,
    const std::vector<float>& mean_winter_temp_c,
    const std::vector<float>& /*mean_summer_temp_c*/) 
{
    DemographyYearSummary summary;
    std::mt19937 rng(42819);

    // Mapeo rápido de celda a índice para temperaturas
    std::unordered_map<int32_t, size_t> cell_idx_map;
    for (size_t i = 0; i < cells.size(); ++i) {
        cell_idx_map[cells[i].cell_id] = i;
    }

    const size_t n_cohorts = cohorts_.size();

    // 1. Simulación de Daño Térmico Invernal acumulado D_{j,t} (OpenMP SIMD friendly)
    for (size_t i = 0; i < n_cohorts; ++i) {
        if (!cohorts_.is_alive[i]) continue;

        int32_t c_id = cohorts_.cell_id[i];
        float t_winter = 0.0f;
        auto it = cell_idx_map.find(c_id);
        if (it != cell_idx_map.end() && it->second < mean_winter_temp_c.size()) {
            t_winter = mean_winter_temp_c[it->second];
        }

        // Exposición a temperaturas bajo el umbral crítico T_c
        float cold_stress = std::max(0.0f, params_.t_c - t_winter);
        float delta_damage = params_.alpha_damage * std::pow(cold_stress, static_cast<float>(params_.beta_damage)) * 30.0f; // 30 días de invierno
        cohorts_.cold_damage_D[i] += delta_damage;

        // Probabilidad de supervivencia condicional P(S=1) = exp(-Delta D)
        float p_survival = std::exp(-delta_damage);
        cohorts_.weight[i] *= p_survival;

        if (cohorts_.weight[i] < 0.01f) {
            cohorts_.is_alive[i] = 0;
            cohorts_.weight[i] = 0.0f;
        } else {
            // Promoción al estadio adulto
            cohorts_.stage[i] = 3;
            summary.adult_density_per_cell[c_id] += cohorts_.weight[i];
            summary.total_adults += static_cast<uint64_t>(cohorts_.weight[i]);
        }
    }

    // 2. Reproducción local (Fecundidad densodependiente) y Baloneo
    for (const auto& cell : cells) {
        float n_adults = summary.adult_density_per_cell[cell.cell_id];
        if (n_adults <= 0.0f) continue;

        float k_i = compute_carrying_capacity(cell);
        float density_factor = std::max(0.0f, 1.0f - (n_adults / k_i));

        // Media de descendientes viables mu_B = F * N_adultas * (1 - N/K) * q_viab
        float mu_b = params_.fecundity_F * (n_adults * 0.5f) * density_factor * params_.q_viab;
        if (mu_b <= 0.0f) continue;

        // Generar descendencia estocástica
        float total_b = mu_b; // Aproximación determinista del valor esperado de NegBin
        summary.total_offspring += static_cast<uint64_t>(total_b);

        // Fracción que emprende baloneo aéreo P^{aereos} ~ Binomial(B, q_balloon)
        uint32_t n_ballooners = static_cast<uint32_t>(total_b * params_.q_balloon);
        if (n_ballooners > 0) {
            summary.ballooners_per_cell[cell.cell_id] = n_ballooners;
            summary.total_ballooners += n_ballooners;
        }
    }

    return summary;
}

} // namespace joro::abm
