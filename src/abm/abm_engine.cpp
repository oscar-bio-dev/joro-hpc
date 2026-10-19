#include "abm/abm_engine.hpp"
#include "common/fast_math.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

namespace joro::abm {

// ─────────────────────────────────────────────────────────────────────────────
// Utilidades internas: kernels diarios vectorizables (AVX-512 vía #pragma omp simd)
// ─────────────────────────────────────────────────────────────────────────────
namespace {

/// Con menos cohortes que este umbral el coste de abrir una región paralela supera al trabajo útil.
constexpr size_t kParallelCohortThreshold = 128;

/// Daño térmico acumulado sobre los días [d0, d1):
///   sum_d alpha_s * [max(0, T_c - T_micro(d))]^beta_s * dt,  dt = 1 día
/// El índice espacial se calcula una sola vez; los días se recorren con paso constante
/// (gather AVX-512). beta_s ∈ {1,2,3} se evalúa con multiplicación directa (sin std::pow).
inline float winter_damage_sum(
    const io::DailyThermalGrid& th, const GridCell& cell,
    int d0, int d1, float t_c, float alpha, int beta) noexcept
{
    const float dosel_buffer = 0.35f * cell.z_dosel; // amortiguación microclimática del dosel
    float acc = 0.0f;
    if (th.has_data()) {
        const size_t stride = th.day_stride();
        const float* tmin = th.t2m_min.data() + th.locate(cell.centroid_lat, cell.centroid_lon);
        #pragma omp simd reduction(+:acc)
        for (int d = d0; d < d1; ++d) {
            const float t_micro = tmin[static_cast<size_t>(d) * stride] + dosel_buffer;
            const float cold_stress = std::max(0.0f, t_c - t_micro);
            acc += alpha * ipow123(cold_stress, beta);
        }
    } else {
        for (int d = d0; d < d1; ++d) {
            float t_min = 0.0f, t_mean = 0.0f;
            th.get_temp(d, cell.centroid_lat, cell.centroid_lon, t_min, t_mean);
            const float cold_stress = std::max(0.0f, t_c - (t_min + dosel_buffer));
            acc += alpha * ipow123(cold_stress, beta);
        }
    }
    return acc;
}

/// Desarrollo fisiológico acumulado (Q10) sobre los días [d0, d1):
///   r_s(T) = r_ref * Q10^((T - T_ref)/10)  si T_min,dev <= T <= T_max,dev; 0 en otro caso
inline float development_sum(
    const io::DailyThermalGrid& th, const GridCell& cell,
    int d0, int d1, const DemographicParameters& p, float ln_q10) noexcept
{
    float acc = 0.0f;
    const float t_lo = p.t_min_dev, t_hi = p.t_max_dev, t_ref = p.t_ref, r_ref = p.r_ref;
    if (th.has_data()) {
        const size_t stride = th.day_stride();
        const float* tmean = th.t2m_mean.data() + th.locate(cell.centroid_lat, cell.centroid_lon);
        #pragma omp simd reduction(+:acc)
        for (int d = d0; d < d1; ++d) {
            const float t = tmean[static_cast<size_t>(d) * stride];
            const float r_s = r_ref * std::exp(((t - t_ref) * 0.1f) * ln_q10);
            acc += (t >= t_lo && t <= t_hi) ? r_s : 0.0f;
        }
    } else {
        for (int d = d0; d < d1; ++d) {
            float t_min = 0.0f, t_mean = 0.0f;
            th.get_temp(d, cell.centroid_lat, cell.centroid_lon, t_min, t_mean);
            if (t_mean >= t_lo && t_mean <= t_hi) {
                acc += r_ref * std::exp(((t_mean - t_ref) * 0.1f) * ln_q10);
            }
        }
    }
    return acc;
}

} // namespace

CellIndexTable CellIndexTable::build(const std::vector<GridCell>& cells) {
    CellIndexTable table;
    int32_t max_id = -1;
    for (const auto& c : cells) max_id = std::max(max_id, c.cell_id);
    table.idx_of.assign(static_cast<size_t>(max_id + 1), -1);
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].cell_id >= 0) table.idx_of[static_cast<size_t>(cells[i].cell_id)] = static_cast<int32_t>(i);
    }
    return table;
}

namespace {
CarryingCapacityParams make_carrying_params(const DemographicParameters& p) {
    CarryingCapacityParams cc{p.k0, p.gamma1_imp, p.gamma2_borde, p.gamma3_dosel, p.k_min, p.k_max};
    cc.gamma_hab = p.gamma_hab;
    cc.use_hab_index = p.use_hab_index;
    return cc;
}
} // namespace

AbmEngine::AbmEngine(DemographicParameters params)
    : params_(params) {}

void AbmEngine::initialize_seed(const std::vector<GridCell>& cells, float rho0) {
    cohorts_.clear();
    CarryingCapacityParams cc = make_carrying_params(params_);

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
    // Orden canónico por cell_id: el orden de iteración de unordered_map depende del orden de
    // inserción (fusión OpenMP de las llegadas del LPDM) y alteraría la secuencia del RNG binomial.
    std::vector<std::pair<int32_t, uint32_t>> ordered;
    ordered.reserve(arrivals.size());
    for (const auto& kv : arrivals) {
        if (kv.second > 0) ordered.emplace_back(kv.first, kv.second);
    }
    std::sort(ordered.begin(), ordered.end());
    for (const auto& [cell_id, count] : ordered) {
        // Llegadas aéreas de mayo se incorporan como juveniles (estadio 1)
        cohorts_.add_cohort(cell_id, 1 /*juveniles*/, static_cast<float>(count));
    }
}

DemographyYearSummary AbmEngine::step_spring_phenology(
    const std::vector<GridCell>& cells,
    const io::DailyThermalGrid& thermal,
    int /*year*/) 
{
    DemographyYearSummary summary;
    std::mt19937 rng(42819);

    CellIndexTable local_index;
    const CellIndexTable* index = cell_index_ext_;
    if (index == nullptr) {
        local_index = CellIndexTable::build(cells);
        index = &local_index;
    }

    const size_t n_cohorts = cohorts_.size();
    const float ln_q10 = std::log(params_.q10);
    const bool par = n_cohorts >= kParallelCohortThreshold;

    // ─────────────────────────────────────────────────────────────────────────
    // PILAR 3: Mortalidad Invernal por Daño Acumulado D_{j,t} (Días 1 a 90: Invierno)
    // D_{j,t+1} = D_{j,t} + alpha_s * [max(0, T_{c,s} - T_{j,t}^micro)]^beta_s * Delta t
    // P(S=1 | S=1) = exp(-(D_{j,t+1} - D_{j,t}))  =>  la supervivencia acumulada es exp(-sum Delta D)
    // Cada cohorte es independiente: bucle paralelo OpenMP (static) con reducción de pesos.
    // ─────────────────────────────────────────────────────────────────────────
    double initial_egg_weight = 0.0;
    double survived_egg_weight = 0.0;

    #pragma omp parallel for schedule(static) if(par) reduction(+:initial_egg_weight, survived_egg_weight)
    for (size_t c_idx = 0; c_idx < n_cohorts; ++c_idx) {
        if (!cohorts_.is_alive[c_idx] || cohorts_.stage[c_idx] != 0) continue;
        initial_egg_weight += cohorts_.weight[c_idx];

        const int32_t pos = index->find(cohorts_.cell_id[c_idx]);
        if (pos < 0) continue;
        const auto& cell = cells[static_cast<size_t>(pos)];

        // Bucle diario sobre los 90 días de invierno (enero a marzo)
        const float d_sum = winter_damage_sum(thermal, cell, 0, 90,
                                              params_.t_c, params_.alpha_damage, params_.beta_damage);
        if (d_sum > 0.0f) {
            cohorts_.cold_damage_D[c_idx] += d_sum;
            cohorts_.weight[c_idx] *= std::exp(-d_sum);
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
    #pragma omp parallel for schedule(static) if(par)
    for (size_t c_idx = 0; c_idx < n_cohorts; ++c_idx) {
        if (!cohorts_.is_alive[c_idx] || cohorts_.stage[c_idx] != 1) continue;

        const int32_t pos = index->find(cohorts_.cell_id[c_idx]);
        if (pos < 0) continue;
        const auto& cell = cells[static_cast<size_t>(pos)];

        cohorts_.degree_days[c_idx] += development_sum(thermal, cell, 90, 120, params_, ln_q10);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Día 121 (Mayo): Dispersión Aérea (Baloneo)
    // Fracción q_balloon emprende vuelo via LPDM. Se mantiene SERIAL: el generador binomial
    // comparte un único RNG determinista (reproducibilidad independiente del nº de hilos).
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

    CellIndexTable local_index;
    const CellIndexTable* index = cell_index_ext_;
    if (index == nullptr) {
        local_index = CellIndexTable::build(cells);
        index = &local_index;
    }

    CarryingCapacityParams cc = make_carrying_params(params_);

    const size_t n_cohorts = cohorts_.size();
    const float ln_q10 = std::log(params_.q10);
    const bool par = n_cohorts >= kParallelCohortThreshold;

    // ─────────────────────────────────────────────────────────────────────────
    // PILAR 1: Reloj Fenológico y Desarrollo Q10 (Días 121 a 240: Mayo a Agosto)
    // r_s(T) = r_{s,ref} * Q_{10}^{(T - T_{ref})/10}
    // Maduración y transición a adulto reproductivo hacia finales de verano
    // ─────────────────────────────────────────────────────────────────────────
    #pragma omp parallel for schedule(static) if(par)
    for (size_t c_idx = 0; c_idx < n_cohorts; ++c_idx) {
        if (!cohorts_.is_alive[c_idx] || cohorts_.stage[c_idx] != 1) continue;

        const int32_t pos = index->find(cohorts_.cell_id[c_idx]);
        if (pos < 0) continue;
        const auto& cell = cells[static_cast<size_t>(pos)];

        cohorts_.degree_days[c_idx] += development_sum(thermal, cell, 120, 240, params_, ln_q10);

        // Transición a adulto reproductivo cuando acumula desarrollo completo (age >= 1.0)
        if (cohorts_.degree_days[c_idx] >= 1.0f) {
            cohorts_.stage[c_idx] = 3; // Adulto
        }
    }

    // Agregación por celda (serial: escritura en unordered_map). Antes de este paso no existen
    // cohortes en estadio 3 (la semelparidad vacía las cohortes cada otoño), así que stage==3
    // identifica exactamente a las cohortes que maduraron en este ciclo.
    for (size_t c_idx = 0; c_idx < n_cohorts; ++c_idx) {
        if (!cohorts_.is_alive[c_idx] || cohorts_.stage[c_idx] != 3) continue;
        if (index->find(cohorts_.cell_id[c_idx]) < 0) continue;
        summary.adult_density_per_cell[cohorts_.cell_id[c_idx]] += cohorts_.weight[c_idx];
        summary.total_adults += static_cast<uint64_t>(cohorts_.weight[c_idx]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PILAR 2: Regulación Denso-Dependiente por Capacidad de Carga K_i (Días 241 a 280: Otoño)
    // ─────────────────────────────────────────────────────────────────────────
    std::unordered_map<int32_t, float> new_egg_sacs_per_cell;
    double sum_sat = 0.0;
    int n_occupied = 0;

    for (const auto& [cell_id, n_adults] : summary.adult_density_per_cell) {
        if (n_adults <= 0.0f) continue;

        const int32_t pos = index->find(cell_id);
        if (pos < 0) continue;
        const auto& cell = cells[static_cast<size_t>(pos)];

        // Saturación SIEMPRE respecto a la capacidad efectiva de la celda:
        // K_i = K_0 exp(gamma1 Z_imp + gamma2 Z_borde + gamma3 Z_dosel)  (acotada a [K_min, K_max])
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
    {
        std::vector<std::pair<int32_t, float>> ordered_eggs(new_egg_sacs_per_cell.begin(), new_egg_sacs_per_cell.end());
        std::sort(ordered_eggs.begin(), ordered_eggs.end());
        for (const auto& [cell_id, eggs] : ordered_eggs) {
            if (eggs > 0.0f) {
                cohorts_.add_cohort(cell_id, 0 /*ootecas*/, eggs);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inicio de Daño Térmico Invernal temprano (Días 281 a 365: Noviembre a Diciembre)
    // ─────────────────────────────────────────────────────────────────────────
    const size_t n_new_cohorts = cohorts_.size();
    const bool par_new = n_new_cohorts >= kParallelCohortThreshold;

    #pragma omp parallel for schedule(static) if(par_new)
    for (size_t c_idx = 0; c_idx < n_new_cohorts; ++c_idx) {
        const int32_t pos = index->find(cohorts_.cell_id[c_idx]);
        if (pos < 0) continue;
        const auto& cell = cells[static_cast<size_t>(pos)];

        const float d_sum = winter_damage_sum(thermal, cell, 280, 365,
                                              params_.t_c, params_.alpha_damage, params_.beta_damage);
        if (d_sum > 0.0f) {
            cohorts_.cold_damage_D[c_idx] += d_sum;
            cohorts_.weight[c_idx] *= std::exp(-d_sum);
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

    CarryingCapacityParams cc = make_carrying_params(params_);

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
