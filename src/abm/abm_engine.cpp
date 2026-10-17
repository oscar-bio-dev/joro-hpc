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
            cohorts_.add_cohort(cell_id, 1 /*juveniles/ootecas*/, static_cast<float>(count));
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

    // 1. Simulación de Daño Térmico Invernal acumulado D_{j,t} y Maduración de Juveniles a Adultos
    for (size_t i = 0; i < n_cohorts; ++i) {
        if (!cohorts_.is_alive[i]) continue;

        int32_t c_id = cohorts_.cell_id[i];
        float t_winter = 0.0f;
        auto it = cell_idx_map.find(c_id);
        if (it != cell_idx_map.end() && it->second < mean_winter_temp_c.size()) {
            t_winter = mean_winter_temp_c[it->second];
        }

        // Si la cohorte es juvenil (ooteca/inmaduro invernando), aplicar estrés térmico invernal
        if (cohorts_.stage[i] == 1) {
            float cold_stress = std::max(0.0f, params_.t_c - t_winter);
            float delta_damage = params_.alpha_damage * std::pow(cold_stress, static_cast<float>(params_.beta_damage)) * 30.0f; // 30 días de invierno
            cohorts_.cold_damage_D[i] += delta_damage;

            // Probabilidad de supervivencia condicional P(S=1) = exp(-Delta D)
            float p_survival = std::exp(-delta_damage);
            cohorts_.weight[i] *= p_survival;

            if (cohorts_.weight[i] < 0.01f) {
                cohorts_.is_alive[i] = 0;
                cohorts_.weight[i] = 0.0f;
                continue;
            }

            // Promoción al estadio adulto para la temporada de reproducción
            cohorts_.stage[i] = 3;
        }

        if (cohorts_.weight[i] >= 0.01f && cohorts_.stage[i] == 3) {
            summary.adult_density_per_cell[c_id] += cohorts_.weight[i];
            summary.total_adults += static_cast<uint64_t>(cohorts_.weight[i]);
        }
    }

    // 2. Reproducción local (Fecundidad densodependiente) y Baloneo
    std::unordered_map<int32_t, float> local_recruits_per_cell;

    for (const auto& cell : cells) {
        float n_adults = summary.adult_density_per_cell[cell.cell_id];
        if (n_adults <= 0.0f) continue;

        float k_i = compute_carrying_capacity(cell);
        // Formulación canónica de Beverton-Holt para organismos anuales semélparos:
        // mu_B = (R0 * N) / (1 + ((R0 - 1) / K_i) * N), garantizando estabilidad asintótica en K_i
        const float r0 = params_.fecundity_F * 0.5f * params_.q_viab;
        const float mu_b = (r0 * n_adults) / (1.0f + ((r0 - 1.0f) / k_i) * n_adults);
        if (mu_b <= 0.0f) continue;

        float total_b = mu_b;
        summary.total_offspring += static_cast<uint64_t>(total_b);

        // Fracción que emprende baloneo aéreo P^{aereos} ~ Binomial(B, q_balloon)
        uint32_t n_ballooners = static_cast<uint32_t>(total_b * params_.q_balloon);
        if (n_ballooners > 0) {
            summary.ballooners_per_cell[cell.cell_id] = n_ballooners;
            summary.total_ballooners += n_ballooners;
        }

        // Descendencia que permanece local (no balonea)
        float local_b = total_b - static_cast<float>(n_ballooners);
        if (local_b > 0.0f) {
            local_recruits_per_cell[cell.cell_id] = local_b;
        }
    }

    // 3. Transición de ciclo de vida: Senescencia adulta obligatoria (Semelparidad de Trichonephila clavata)
    // Los adultos completan su ciclo y mueren tras ovipositar.
    // Solo las nuevas ootecas/juveniles locales entran a la fase de invernada.
    cohorts_.clear();
    for (const auto& [cell_id, local_count] : local_recruits_per_cell) {
        if (local_count > 0.0f) {
            cohorts_.add_cohort(cell_id, 1 /*juveniles/ootecas*/, local_count);
        }
    }

    return summary;
}

} // namespace joro::abm
