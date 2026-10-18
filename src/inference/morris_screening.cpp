#include "inference/morris_screening.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

namespace joro::inference {

const std::vector<MorrisParamDef> MORRIS_PARAMS = {
    {"fecundity_F",   "F",        400.0f,   500.0f,   450.0f},
    {"q10",           "Q10",        2.0f,     3.0f,     2.2f},
    {"t_c",           "T_c,s",     -5.0f,     0.0f,    -4.0f},
    {"alpha_damage",  "alpha_s",   0.01f,    0.10f,    0.04f},
    {"beta_damage",   "beta_s",     1.0f,     3.0f,     2.0f},
    {"q_viab",        "q_viab",    0.20f,    0.90f,    0.65f},
    {"q_balloon",     "q_balloon", 0.10f,    0.60f,    0.35f},
    {"k0",            "K0",      1000.0f, 20000.0f,  5000.0f},
    {"gamma1_imp",    "gamma_1",   0.10f,    1.00f,    0.45f},
    {"gamma2_borde",  "gamma_2",   0.20f,    1.20f,    0.65f},
    {"gamma3_dosel",  "gamma_3",   0.05f,    0.80f,    0.30f}
};

struct ModelOutputs {
    double total_adults = 0.0;
    double front_radius_km = 0.0;
};

static ModelOutputs evaluate_candidate(
    const std::vector<float>& theta,
    const std::vector<GridCell>& grid_cells,
    const io::DailyThermalGrid& thermal,
    lpdm::LpdmEngine* lpdm_engine)
{
    abm::DemographicParameters p;
    p.fecundity_F   = theta[0];
    p.q10           = theta[1];
    p.t_c           = theta[2];
    p.alpha_damage  = theta[3];
    p.beta_damage   = std::clamp(static_cast<int>(std::round(theta[4])), 1, 3);
    p.q_viab        = theta[5];
    p.q_balloon     = theta[6];
    p.k0            = theta[7];
    p.gamma1_imp    = theta[8];
    p.gamma2_borde  = theta[9];
    p.gamma3_dosel  = theta[10];

    abm::AbmEngine abm(p);
    abm.initialize_seed(grid_cells, 0.15f);

    io::AtmosphericField3D mock_atmo;
    mock_atmo.n_times = 60;
    mock_atmo.n_levels = 12;
    mock_atmo.n_lats = 81;
    mock_atmo.n_lons = 101;
    mock_atmo.pressure_levels_hpa = {1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700};
    size_t tot_pts = static_cast<size_t>(mock_atmo.n_times) * mock_atmo.n_levels * mock_atmo.n_lats * mock_atmo.n_lons;
    mock_atmo.u.assign(tot_pts, 4.0f);
    mock_atmo.v.assign(tot_pts, 2.5f);
    mock_atmo.w.assign(tot_pts, 0.02f);
    mock_atmo.t.assign(tot_pts, 295.0f);

    abm::DemographyYearSummary last_fall;

    // Simular ciclo 2014-2019
    for (int year = 2014; year <= 2019; ++year) {
        auto spring = abm.step_spring_phenology(grid_cells, thermal, year);
        lpdm_engine->initialize_particles(grid_cells, spring.ballooners_per_cell);
        auto disp = lpdm_engine->run_dispersion(mock_atmo, 60.0f, 60);
        abm.deposit_arrivals(disp.arrivals_per_cell);
        last_fall = abm.step_summer_autumn_phenology(grid_cells, thermal, year);
    }

    ModelOutputs out;
    out.total_adults = static_cast<double>(last_fall.total_adults);

    // Calcular radio del frente desde epicentro (Athens, Georgia: lat 34.25, lon -83.50)
    constexpr float epi_lat = 34.25f;
    constexpr float epi_lon = -83.50f;
    double max_dist = 0.0;

    for (const auto& [cell_id, count] : last_fall.adult_density_per_cell) {
        if (count > 5.0f) {
            // Buscar lat/lon de la celda
            for (const auto& c : grid_cells) {
                if (c.cell_id == cell_id) {
                    float dlat_km = (c.centroid_lat - epi_lat) * 111.0f;
                    float dlon_km = (c.centroid_lon - epi_lon) * 111.0f * std::cos(epi_lat * 0.0174533f);
                    double d = std::sqrt(dlat_km * dlat_km + dlon_km * dlon_km);
                    if (d > max_dist) max_dist = d;
                    break;
                }
            }
        }
    }
    out.front_radius_km = max_dist;
    return out;
}

MorrisScreeningReport run_morris_screening(
    const std::vector<GridCell>& grid_cells,
    const io::DailyThermalGrid& thermal,
    lpdm::LpdmEngine* lpdm_engine,
    int n_trajectories,
    int p_levels) 
{
    MorrisScreeningReport report;
    report.n_trajectories = n_trajectories;
    const int k = static_cast<int>(MORRIS_PARAMS.size());
    report.n_parameters = k;
    report.total_evaluations = n_trajectories * (k + 1);

    std::cout << "\n===================================================================\n";
    std::cout << "▶ Iniciando Cribado Global de Sensibilidad de Morris (1991)\n";
    std::cout << "  Espacio de parámetros : k = " << k << " dimensiones\n";
    std::cout << "  Trayectorias aleatorias: r = " << n_trajectories << "\n";
    std::cout << "  Niveles de discretización: p = " << p_levels << "\n";
    std::cout << "  Total evaluaciones acopladas: " << report.total_evaluations << "\n";
    std::cout << "===================================================================\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    // Matriz de efectos elementales por parámetro: [param_idx][traj_idx]
    std::vector<std::vector<double>> ee_pop(k, std::vector<double>(n_trajectories, 0.0));
    std::vector<std::vector<double>> ee_front(k, std::vector<double>(n_trajectories, 0.0));

    // Paso de perturbación normalizado Delta = p / (2 * (p - 1))
    const double delta = static_cast<double>(p_levels) / (2.0 * (p_levels - 1));

    std::mt19937 rng(54321);
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);

    for (int traj = 0; traj < n_trajectories; ++traj) {
        auto traj_t0 = std::chrono::high_resolution_clock::now();

        // 1. Punto base aleatorio en grilla de p niveles
        std::vector<double> x(k);
        for (int i = 0; i < k; ++i) {
            int level = std::uniform_int_distribution<int>(0, p_levels - 2)(rng);
            x[i] = static_cast<double>(level) / (p_levels - 1);
        }

        // 2. Orden aleatorio de permutación de los k parámetros
        std::vector<int> perm(k);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        // Vector para signos de paso (+delta o -delta)
        std::vector<double> s(k);
        for (int i = 0; i < k; ++i) {
            s[i] = (x[i] + delta <= 1.0) ? delta : -delta;
        }

        // Evaluar punto inicial
        auto eval_point = [&](const std::vector<double>& pt) -> ModelOutputs {
            std::vector<float> theta(k);
            for (int i = 0; i < k; ++i) {
                theta[i] = static_cast<float>(MORRIS_PARAMS[i].min_val + 
                    pt[i] * (MORRIS_PARAMS[i].max_val - MORRIS_PARAMS[i].min_val));
            }
            return evaluate_candidate(theta, grid_cells, thermal, lpdm_engine);
        };

        ModelOutputs y_current = eval_point(x);

        // Pasos a lo largo de la trayectoria
        for (int step = 0; step < k; ++step) {
            int param_idx = perm[step];
            x[param_idx] += s[param_idx];

            ModelOutputs y_next = eval_point(x);

            // Efecto elemental: Delta Y / Delta X_norm
            double d_pop = (y_next.total_adults - y_current.total_adults) / s[param_idx];
            double d_front = (y_next.front_radius_km - y_current.front_radius_km) / s[param_idx];

            ee_pop[param_idx][traj] = d_pop;
            ee_front[param_idx][traj] = d_front;

            y_current = y_next;
        }

        auto traj_t1 = std::chrono::high_resolution_clock::now();
        double traj_elapsed = std::chrono::duration<double>(traj_t1 - traj_t0).count();
        std::cout << "  Trayectoria " << std::setw(2) << (traj + 1) << " / " << n_trajectories
                  << " completada (" << (k + 1) << " evals) en "
                  << std::fixed << std::setprecision(3) << traj_elapsed << " s\n";
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    report.execution_time_s = std::chrono::duration<double>(t_end - t_start).count();

    // 3. Cálculo de mu* y sigma para cada parámetro
    report.results.resize(k);

    for (int i = 0; i < k; ++i) {
        MorrisResult res;
        res.param_name = MORRIS_PARAMS[i].name;
        res.symbol     = MORRIS_PARAMS[i].symbol;

        // mu* (media de valores absolutos)
        double sum_abs_pop = 0.0, sum_pop = 0.0;
        double sum_abs_front = 0.0, sum_front = 0.0;
        for (int traj = 0; traj < n_trajectories; ++traj) {
            sum_abs_pop += std::abs(ee_pop[i][traj]);
            sum_pop     += ee_pop[i][traj];
            sum_abs_front += std::abs(ee_front[i][traj]);
            sum_front     += ee_front[i][traj];
        }
        res.mu_star_pop   = sum_abs_pop / n_trajectories;
        res.mu_star_front = sum_abs_front / n_trajectories;

        double mean_pop   = sum_pop / n_trajectories;
        double mean_front = sum_front / n_trajectories;

        // sigma (desviación estándar)
        double sq_pop = 0.0, sq_front = 0.0;
        for (int traj = 0; traj < n_trajectories; ++traj) {
            sq_pop   += (ee_pop[i][traj] - mean_pop) * (ee_pop[i][traj] - mean_pop);
            sq_front += (ee_front[i][traj] - mean_front) * (ee_front[i][traj] - mean_front);
        }
        res.sigma_pop   = std::sqrt(sq_pop / (n_trajectories - 1));
        res.sigma_front = std::sqrt(sq_front / (n_trajectories - 1));

        // Clasificación ecofisiológica
        if (res.mu_star_pop > 50000.0 || res.mu_star_front > 15.0) {
            res.classification = "Dominante (Fase 4)";
        } else if (res.sigma_pop > 20000.0 || res.sigma_front > 8.0) {
            res.classification = "Interacción Fuerte";
        } else {
            res.classification = "Despreciable";
        }

        report.results[i] = res;
    }

    std::cout << "\n✔ Cribado de Morris completado en " << std::fixed << std::setprecision(2)
              << report.execution_time_s << " s.\n";

    return report;
}

} // namespace joro::inference
