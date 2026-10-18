#include "abm/abm_engine.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

namespace joro::abm {

AbmEngine::AbmEngine(DemographicParameters params)
    : params_(params) {}

void AbmEngine::initialize_seed(const std::vector<GridCell>& cells, float rho0) {
    cohorts_.clear();
    CarryingCapacityParams cc{
        params_.k0, params_.gamma1_imp, params_.gamma2_borde, params_.gamma3_dosel,
        params_.k_min, params_.k_max
    };

    for (const auto& cell : cells) {
        if (cell.is_seed) {
            float k_i = compute_carrying_capacity(cell, cc);
            float initial_n = rho0 * k_i;
            // Inicializar como estadio 0 (ootecas invernantes listas para eclosión en primavera)
            cohorts_.add_cohort(cell.cell_id, 0 /*ootecas*/, initial_n);
        }
    }
}

void AbmEngine::inject_test_cohort(int32_t cell_id, float count) {
    cohorts_.add_cohort(cell_id, 0 /*ootecas*/, count);
}

void AbmEngine::deposit_arrivals(const std::unordered_map<int32_t, uint32_t>& arrivals) {
    for (const auto& [cell_id, count] : arrivals) {
        if (count > 0) {
            // Llegadas aéreas de mayo se incorporan como juveniles (estadio 1)
            cohorts_.add_cohort(cell_id, 1 /*juveniles*/, static_cast<float>(count));
        }
    }
}

DemographyYearSummary AbmEngine::step_spring_phenology(
    const std::vector<GridCell>& cells,
    const io::DailyThermalGrid& thermal,
    int /*year*/) 
{
    DemographyYearSummary summary;
    std::mt19937 rng(42819);

    std::unordered_map<int32_t, size_t> cell_idx_map;
    cell_idx_map.reserve(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        cell_idx_map[cells[i].cell_id] = i;
    }

    const size_t n_cohorts = cohorts_.size();

    // ─────────────────────────────────────────────────────────────────────────
    // PILAR 3: Mortalidad Invernal por Daño Acumulado D_{j,t} (Días 1 a 90: Invierno)
    // D_{j,t+1} = D_{j,t} + alpha_s * [max(0, T_{c,s} - T_{j,t}^micro)]^beta_s * Delta t
    // P(S=1 | S=1) = exp(-(D_{j,t+1} - D_{j,t}))
    // ─────────────────────────────────────────────────────────────────────────
    double initial_egg_weight = 0.0;
    double survived_egg_weight = 0.0;

    for (size_t c_idx = 0; c_idx < n_cohorts; ++c_idx) {
        if (!cohorts_.is_alive[c_idx] || cohorts_.stage[c_idx] != 0) continue;
        initial_egg_weight += cohorts_.weight[c_idx];

        int32_t c_id = cohorts_.cell_id[c_idx];
        auto it = cell_idx_map.find(c_id);
        if (it == cell_idx_map.end()) continue;
        const auto& cell = cells[it->second];

        // Bucle diario sobre los 90 días de invierno (enero a marzo)
        for (int day = 0; day < 90; ++day) {
            float t_min = 0.0f, t_mean = 0.0f;
            thermal.get_temp(day, cell.centroid_lat, cell.centroid_lon, t_min, t_mean);

            // Amortiguación microclimática por dosel forestal
            float t_micro = t_min + 0.35f * cell.z_dosel;
            float cold_stress = std::max(0.0f, params_.t_c - t_micro);

            if (cold_stress > 0.0f) {
                float delta_damage = params_.alpha_damage * 
                    std::pow(cold_stress, static_cast<float>(params_.beta_damage)) * 1.0f;
                cohorts_.cold_damage_D[c_idx] += delta_damage;
                cohorts_.weight[c_idx] *= std::exp(-delta_damage);
            }
        }

        if (cohorts_.weight[c_idx] < 0.005f) {
            cohorts_.is_alive[c_idx] = 0;
            cohorts_.weight[c_idx] = 0.0f;
        } else {
            survived_egg_weight += cohorts_.weight[c_idx];
            // Eclosión en día 91 (1 de Abril): transición de ooteca (0) a ninfa I (1)
            cohorts_.stage[c_idx] = 1;
            cohorts_.degree_days[c_idx] = 0.0f;
        }
    }

    if (initial_egg_weight > 0.0) {
        summary.winter_survival_pct = static_cast<float>(survived_egg_weight * 100.0 / initial_egg_weight);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Crecimiento juvenil temprano (Días 91 a 120: Abril)
    // ─────────────────────────────────────────────────────────────────────────
    for (size_t c_idx = 0; c_idx < n_cohorts; ++c_idx) {
        if (!cohorts_.is_alive[c_idx] || cohorts_.stage[c_idx] != 1) continue;

        int32_t c_id = cohorts_.cell_id[c_idx];
        auto it = cell_idx_map.find(c_id);
        if (it == cell_idx_map.end()) continue;
        const auto& cell = cells[it->second];

        for (int day = 90; day < 120; ++day) {
            float t_min = 0.0f, t_mean = 0.0f;
            thermal.get_temp(day, cell.centroid_lat, cell.centroid_lon, t_min, t_mean);

            if (t_mean >= params_.t_min_dev && t_mean <= params_.t_max_dev) {
                float r_s = params_.r_ref * std::pow(params_.q10, (t_mean - params_.t_ref) / 10.0f);
                cohorts_.degree_days[c_idx] += r_s * 1.0f;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Día 121 (Mayo): Dispersión Aérea (Baloneo)
    // Fracción q_balloon emprende vuelo via LPDM
    // ─────────────────────────────────────────────────────────────────────────
    for (size_t c_idx = 0; c_idx < n_cohorts; ++c_idx) {
        if (!cohorts_.is_alive[c_idx] || cohorts_.stage[c_idx] != 1) continue;
        float w = cohorts_.weight[c_idx];
        if (w <= 0.0f) continue;

        uint32_t total_int = static_cast<uint32_t>(std::round(w));
        uint32_t n_ballooners = 0;
        if (total_int > 0) {
            std::binomial_distribution<uint32_t> binom_dist(total_int, params_.q_balloon);
            n_ballooners = binom_dist(rng);
        }

        if (n_ballooners > 0) {
            int32_t c_id = cohorts_.cell_id[c_idx];
            summary.ballooners_per_cell[c_id] += n_ballooners;
            summary.total_ballooners += n_ballooners;
        }

        cohorts_.weight[c_idx] = std::max(0.0f, w - static_cast<float>(n_ballooners));
        if (cohorts_.weight[c_idx] < 0.005f) {
            cohorts_.is_alive[c_idx] = 0;
            cohorts_.weight[c_idx] = 0.0f;
        }
    }

    return summary;
}

DemographyYearSummary AbmEngine::step_summer_autumn_phenology(
    const std::vector<GridCell>& cells,
    const io::DailyThermalGrid& thermal,
    int /*year*/) 
{
    DemographyYearSummary summary;
    std::mt19937 rng(42819);

    std::unordered_map<int32_t, size_t> cell_idx_map;
    cell_idx_map.reserve(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        cell_idx_map[cells[i].cell_id] = i;
    }

    CarryingCapacityParams cc{
        params_.k0, params_.gamma1_imp, params_.gamma2_borde, params_.gamma3_dosel,
        params_.k_min, params_.k_max
    };

    const size_t n_cohorts = cohorts_.size();

    // ─────────────────────────────────────────────────────────────────────────
    // PILAR 1: Reloj Fenológico y Desarrollo Q10 (Días 121 a 240: Mayo a Agosto)
    // r_s(T) = r_{s,ref} * Q_{10}^{(T - T_{ref})/10}
    // Maduración y transición a adulto reproductivo hacia finales de verano
    // ─────────────────────────────────────────────────────────────────────────
    for (size_t c_idx = 0; c_idx < n_cohorts; ++c_idx) {
        if (!cohorts_.is_alive[c_idx] || cohorts_.stage[c_idx] != 1) continue;

        int32_t c_id = cohorts_.cell_id[c_idx];
        auto it = cell_idx_map.find(c_id);
        if (it == cell_idx_map.end()) continue;
        const auto& cell = cells[it->second];

        for (int day = 120; day < 240; ++day) {
            float t_min = 0.0f, t_mean = 0.0f;
            thermal.get_temp(day, cell.centroid_lat, cell.centroid_lon, t_min, t_mean);

            if (t_mean >= params_.t_min_dev && t_mean <= params_.t_max_dev) {
                float r_s = params_.r_ref * std::pow(params_.q10, (t_mean - params_.t_ref) / 10.0f);
                cohorts_.degree_days[c_idx] += r_s * 1.0f;
            }
        }

        // Transición a adulto reproductivo cuando acumula desarrollo completo (age >= 1.0)
        if (cohorts_.degree_days[c_idx] >= 1.0f) {
            cohorts_.stage[c_idx] = 3; // Adulto
            summary.adult_density_per_cell[c_id] += cohorts_.weight[c_idx];
            summary.total_adults += static_cast<uint64_t>(cohorts_.weight[c_idx]);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PILAR 2: Regulación Denso-Dependiente por Capacidad de Carga K_i (Días 241 a 280: Otoño)
    // mu_{B,i,t} = F(T_{i,t}) * N_{i,t}^adultas * max(0, 1 - N_{i,t} / K_i) * q_viab
    // Freno asintótico a K_i sin divergencias ni valores negativos
    // ─────────────────────────────────────────────────────────────────────────
    std::unordered_map<int32_t, float> new_egg_sacs_per_cell;
    double sum_sat = 0.0;
    int n_occupied = 0;

    for (const auto& [cell_id, n_adults] : summary.adult_density_per_cell) {
        if (n_adults <= 0.0f) continue;

        auto it = cell_idx_map.find(cell_id);
        if (it == cell_idx_map.end()) continue;
        const auto& cell = cells[it->second];

        float k_i = compute_carrying_capacity(cell, cc);
        float sat_ratio = n_adults / k_i;
        sum_sat += sat_ratio;
        n_occupied++;

        // Temperatura estival/otoñal promedio en ventana reproductiva
        float t_min_fall = 0.0f, t_mean_fall = 0.0f;
        thermal.get_temp(250, cell.centroid_lat, cell.centroid_lon, t_min_fall, t_mean_fall);

        float q10_f = std::clamp(std::pow(params_.q10, (t_mean_fall - params_.t_ref) / 10.0f), 0.2f, 3.0f);
        float f_eff = params_.fecundity_F * q10_f;

        // Ecuación canónica de regulación denso-dependiente asintótica de Beverton-Holt (Sección 8):
        // mu_B = (R0 * N) / (1 + ((R0 - 1) / K_i) * N)
        // Garantiza convergencia monótona hacia K_i sin extinciones artificiales de ciclo-2
        // incluso si una inmigración aérea masiva puntual produce N_{i,t} > K_i.
        const float r0 = f_eff * 0.5f * params_.q_viab;
        const float mu_b = (r0 * n_adults) / (1.0f + ((r0 - 1.0f) / k_i) * n_adults);

        if (mu_b > 0.0f) {
            new_egg_sacs_per_cell[cell_id] = mu_b;
            summary.total_offspring += static_cast<uint64_t>(mu_b);
        }
    }

    if (n_occupied > 0) {
        summary.mean_saturation = static_cast<float>(sum_sat / n_occupied);
    }

    // Semelparidad estricta de Trichonephila clavata: adultos senescen tras oviposición
    // Solo las nuevas ootecas ingresan al invierno
    cohorts_.clear();
    for (const auto& [cell_id, eggs] : new_egg_sacs_per_cell) {
        if (eggs > 0.0f) {
            cohorts_.add_cohort(cell_id, 0 /*ootecas*/, eggs);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inicio de Daño Térmico Invernal temprano (Días 281 a 365: Noviembre a Diciembre)
    // ─────────────────────────────────────────────────────────────────────────
    const size_t n_new_cohorts = cohorts_.size();
    for (size_t c_idx = 0; c_idx < n_new_cohorts; ++c_idx) {
        int32_t c_id = cohorts_.cell_id[c_idx];
        auto it = cell_idx_map.find(c_id);
        if (it == cell_idx_map.end()) continue;
        const auto& cell = cells[it->second];

        for (int day = 280; day < 365; ++day) {
            float t_min = 0.0f, t_mean = 0.0f;
            thermal.get_temp(day, cell.centroid_lat, cell.centroid_lon, t_min, t_mean);

            float t_micro = t_min + 0.35f * cell.z_dosel;
            float cold_stress = std::max(0.0f, params_.t_c - t_micro);

            if (cold_stress > 0.0f) {
                float delta_damage = params_.alpha_damage * 
                    std::pow(cold_stress, static_cast<float>(params_.beta_damage)) * 1.0f;
                cohorts_.cold_damage_D[c_idx] += delta_damage;
                cohorts_.weight[c_idx] *= std::exp(-delta_damage);
            }
        }

        if (cohorts_.weight[c_idx] < 0.005f) {
            cohorts_.is_alive[c_idx] = 0;
            cohorts_.weight[c_idx] = 0.0f;
        }
    }

    return summary;
}

DemographyYearSummary AbmEngine::step_annual_cycle_daily(
    const std::vector<GridCell>& cells,
    const io::DailyThermalGrid& thermal,
    int year) 
{
    // 1. Fase primaveral (invernada previa, emergencia y baloneo)
    auto spring_summary = step_spring_phenology(cells, thermal, year);

    // 2. Baloneo local: propágulos que no dispersan a larga distancia aterrizan localmente
    std::unordered_map<int32_t, uint32_t> local_dispersal;
    for (const auto& [cell_id, count] : spring_summary.ballooners_per_cell) {
        uint32_t retained = static_cast<uint32_t>(count * (1.0f - params_.q_balloon));
        if (retained > 0) local_dispersal[cell_id] = retained;
    }
    deposit_arrivals(local_dispersal);

    // 3. Fase estival/otoñal (desarrollo Q10, maduración, reproducción Ki)
    auto fall_summary = step_summer_autumn_phenology(cells, thermal, year);

    fall_summary.total_ballooners = spring_summary.total_ballooners;
    fall_summary.ballooners_per_cell = spring_summary.ballooners_per_cell;
    fall_summary.winter_survival_pct = spring_summary.winter_survival_pct;
    return fall_summary;
}

DemographyYearSummary AbmEngine::step_annual_cycle(
    const std::vector<GridCell>& cells,
    const std::vector<float>& mean_winter_temp_c,
    const std::vector<float>& mean_summer_temp_c) 
{
    // Método original preservado para compatibilidad exacta de tests unitarios existentes
    DemographyYearSummary summary;
    std::mt19937 rng(42819);

    std::unordered_map<int32_t, size_t> cell_idx_map;
    for (size_t i = 0; i < cells.size(); ++i) {
        cell_idx_map[cells[i].cell_id] = i;
    }

    CarryingCapacityParams cc{
        params_.k0, params_.gamma1_imp, params_.gamma2_borde, params_.gamma3_dosel,
        params_.k_min, params_.k_max
    };

    const size_t n_cohorts = cohorts_.size();

    // 1. Simulación de Daño Térmico Invernal acumulado D_{j,t}
    for (size_t i = 0; i < n_cohorts; ++i) {
        if (!cohorts_.is_alive[i]) continue;

        int32_t c_id = cohorts_.cell_id[i];
        float t_winter = 0.0f;
        auto it = cell_idx_map.find(c_id);
        if (it != cell_idx_map.end() && it->second < mean_winter_temp_c.size()) {
            t_winter = mean_winter_temp_c[it->second];
        }

        if (cohorts_.stage[i] <= 1) {
            float cold_stress = std::max(0.0f, params_.t_c - t_winter);
            float delta_damage = params_.alpha_damage * std::pow(cold_stress, static_cast<float>(params_.beta_damage)) * 30.0f;
            cohorts_.cold_damage_D[i] += delta_damage;

            float p_survival = std::exp(-delta_damage);
            cohorts_.weight[i] *= p_survival;

            if (cohorts_.weight[i] < 0.01f) {
                cohorts_.is_alive[i] = 0;
                cohorts_.weight[i] = 0.0f;
                continue;
            }

            cohorts_.stage[i] = 3;
        }

        if (cohorts_.weight[i] >= 0.01f && cohorts_.stage[i] == 3) {
            summary.adult_density_per_cell[c_id] += cohorts_.weight[i];
            summary.total_adults += static_cast<uint64_t>(cohorts_.weight[i]);
        }
    }

    // 2. Reproducción local denso-dependiente
    std::unordered_map<int32_t, float> local_recruits_per_cell;
    double sum_sat = 0.0;
    int n_occupied = 0;

    for (const auto& cell : cells) {
        float n_adults = summary.adult_density_per_cell[cell.cell_id];
        if (n_adults <= 0.0f) continue;

        float t_summer = 22.0f;
        auto it = cell_idx_map.find(cell.cell_id);
        if (it != cell_idx_map.end() && it->second < mean_summer_temp_c.size()) {
            t_summer = mean_summer_temp_c[it->second];
        }

        float q10_factor = std::pow(params_.q10, (t_summer - 20.0f) / 10.0f);
        q10_factor = std::clamp(q10_factor, 0.2f, 3.5f);
        float fecundity_eff = params_.fecundity_F * q10_factor;

        float k_i = compute_carrying_capacity(cell, cc);
        sum_sat += (n_adults / k_i);
        n_occupied++;

        const float r0 = fecundity_eff * 0.5f * params_.q_viab;
        const float mu_b = (r0 * n_adults) / (1.0f + ((r0 - 1.0f) / k_i) * n_adults);
        if (mu_b <= 0.0f) continue;

        float total_b = mu_b;
        summary.total_offspring += static_cast<uint64_t>(total_b);

        uint32_t total_b_int = static_cast<uint32_t>(std::round(total_b));
        uint32_t n_ballooners = 0;
        if (total_b_int > 0) {
            std::binomial_distribution<uint32_t> binom_dist(total_b_int, params_.q_balloon);
            n_ballooners = binom_dist(rng);
        }

        if (n_ballooners > 0) {
            summary.ballooners_per_cell[cell.cell_id] = n_ballooners;
            summary.total_ballooners += n_ballooners;
        }

        float local_b = std::max(0.0f, total_b - static_cast<float>(n_ballooners));
        if (local_b > 0.0f) {
            local_recruits_per_cell[cell.cell_id] = local_b;
        }
    }

    cohorts_.clear();
    for (const auto& [cell_id, local_count] : local_recruits_per_cell) {
        if (local_count > 0.0f) {
            cohorts_.add_cohort(cell_id, 1 /*juveniles/ootecas*/, local_count);
        }
    }

    if (n_occupied > 0) {
        summary.mean_saturation = static_cast<float>(sum_sat / n_occupied);
    }

    return summary;
}

} // namespace joro::abm
