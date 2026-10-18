#include "inference/simulated_annealing.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>

namespace joro::inference {

const std::vector<CalibrationParamDef> CALIBRATION_PARAMS = {
    {"t_c",           "T_c,s",     -5.0f,     0.0f,    -3.5f},
    {"alpha_damage",  "alpha_s",   0.01f,    0.10f,    0.04f},
    {"beta_damage",   "beta_s",     1.0f,     3.0f,     2.0f},
    {"k0",            "K0",      1000.0f, 20000.0f,  5000.0f},
    {"q_balloon",     "q_balloon", 0.10f,    0.60f,    0.35f},
    {"q_viab",        "q_viab",    0.20f,    0.90f,    0.65f},
    {"gamma1_imp",    "gamma_1",   0.10f,    1.00f,    0.45f},
    {"gamma2_borde",  "gamma_2",   0.20f,    1.20f,    0.65f},
    {"gamma3_dosel",  "gamma_3",   0.05f,    0.80f,    0.30f}
};

// Distancias empíricas observadas del frente de avance radial (km) desde Braselton, GA (2014-2019)
static const std::vector<double> OBSERVED_FRONT_KM = {42.9, 42.9, 24.1, 38.7, 56.6, 58.0};

std::vector<CalibrationObservation> load_calibration_dataset(const std::string& csv_path) {
    std::vector<CalibrationObservation> dataset;
    dataset.reserve(250000);

    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "[Simulated Annealing] ERROR: No se pudo abrir el archivo CSV de calibración: " 
                  << csv_path << "\n";
        return dataset;
    }

    std::string line;
    // Leer encabezado
    if (!std::getline(file, line)) return dataset;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        CalibrationObservation obs{};

        if (!std::getline(ss, token, ',')) continue;
        obs.cell_id = std::stoi(token);

        if (!std::getline(ss, token, ',')) continue;
        obs.year = static_cast<int16_t>(std::stoi(token));

        if (!std::getline(ss, token, ',')) continue;
        obs.y_detected = static_cast<uint8_t>(std::stoi(token));

        if (!std::getline(ss, token, ',')) continue; // joro_count (no usado directamente en detección binaria)

        if (!std::getline(ss, token, ',')) continue;
        obs.effort_araneae = std::stoi(token);

        if (!std::getline(ss, token, ',')) continue;
        obs.centroid_lon = std::stof(token);

        if (!std::getline(ss, token, ',')) continue;
        obs.centroid_lat = std::stof(token);

        if (!std::getline(ss, token, ',')) continue;
        obs.z_imp = std::stof(token);

        if (!std::getline(ss, token, ',')) continue;
        obs.z_borde = std::stof(token);

        if (!std::getline(ss, token, ',')) continue;
        obs.z_dosel = std::stof(token);

        if (!std::getline(ss, token, ',')) continue;
        obs.pop_density_h = std::stof(token);

        if (!std::getline(ss, token, ',')) continue;
        obs.road_density_r = std::stof(token);

        dataset.push_back(obs);
    }

    std::cout << "[Simulated Annealing] Dataset cargado en RAM: " << dataset.size() 
              << " registros observacionales (2014-2019).\n";
    return dataset;
}

static CostComponents evaluate_simulation_cost(
    const std::vector<float>& theta,
    const std::vector<GridCell>& grid_cells,
    const io::DailyThermalGrid& thermal,
    const std::vector<CalibrationObservation>& observations,
    const CostReferenceScales& scales,
    const CostWeights& weights = CostWeights{})
{
    // 1. Configurar parámetros ecofisiológicos y demográficos del ABM
    abm::DemographicParameters p;
    p.fecundity_F   = 450.0f; // Fijo empírico (Morris baja sensibilidad)
    p.q10           = 2.2f;   // Fijo empírico
    p.t_c           = theta[0];
    p.alpha_damage  = theta[1];
    p.beta_damage   = std::clamp(static_cast<int>(std::round(theta[2])), 1, 3);
    p.k0            = theta[3];
    p.q_balloon     = theta[4];
    p.q_viab        = theta[5];
    p.gamma1_imp    = theta[6];
    p.gamma2_borde  = theta[7];
    p.gamma3_dosel  = theta[8];

    abm::AbmEngine abm(p);
    abm.initialize_seed(grid_cells, 0.15f);

    auto lpdm = lpdm::create_cpu_baseline_engine();

    io::AtmosphericField3D mock_atmo;
    mock_atmo.n_times = 60;
    mock_atmo.n_levels = 12;
    mock_atmo.n_lats = 81;
    mock_atmo.n_lons = 101;
    mock_atmo.pressure_levels_hpa = {1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700};
    size_t tot_pts = static_cast<size_t>(mock_atmo.n_times) * mock_atmo.n_levels * mock_atmo.n_lats * mock_atmo.n_lons;
    mock_atmo.u.assign(tot_pts, 3.8f);
    mock_atmo.v.assign(tot_pts, 2.4f);
    mock_atmo.w.assign(tot_pts, 0.02f);
    mock_atmo.t.assign(tot_pts, 295.0f);

    // Estado latente de ocupación previa psi_{i,t-1}: cell_id -> psi
    std::unordered_map<int32_t, float> psi_map;
    // Condición inicial: celdas semilla con presencia
    for (const auto& c : grid_cells) {
        if (c.is_seed) {
            psi_map[c.cell_id] = 1.0f;
        }
    }

    std::vector<double> sim_front_per_year;
    sim_front_per_year.reserve(6);

    std::vector<float> pred_marginal_all;
    pred_marginal_all.reserve(observations.size());
    std::vector<uint8_t> obs_y_all;
    obs_y_all.reserve(observations.size());

    double persistence_discrepancy_sum = 0.0;
    int persistence_years_count = 0;

    // Indexar observaciones por año para acceso rápido
    // El dataset está ordenado por año: 2014..2019
    size_t obs_idx = 0;
    const size_t total_obs = observations.size();

    constexpr float epi_lat = 34.13816f;
    constexpr float epi_lon = -83.75723f;

    for (int year = 2014; year <= 2019; ++year) {
        auto spring = abm.step_spring_phenology(grid_cells, thermal, year);
        lpdm->initialize_particles(grid_cells, spring.ballooners_per_cell);
        auto disp = lpdm->run_dispersion(mock_atmo, 60.0f, 60);
        abm.deposit_arrivals(disp.arrivals_per_cell);
        auto fall = abm.step_summer_autumn_phenology(grid_cells, thermal, year);

        // 1. Radio del frente simulado en el año t (Percentil 95 de distancia al epicentro)
        std::vector<double> distances;
        distances.reserve(1024);
        for (const auto& [cell_id, count] : fall.adult_density_per_cell) {
            if (count > 2.0f) {
                // Distancia aproximada en km
                float dlat = (grid_cells[cell_id % grid_cells.size()].centroid_lat - epi_lat) * 111.0f;
                float dlon = (grid_cells[cell_id % grid_cells.size()].centroid_lon - epi_lon) * 111.0f * std::cos(epi_lat * 0.0174533f);
                distances.push_back(std::sqrt(dlat * dlat + dlon * dlon));
            }
        }
        double sim_p95_front = 0.0;
        if (!distances.empty()) {
            std::sort(distances.begin(), distances.end());
            size_t p95_idx = static_cast<size_t>(0.95 * (distances.size() - 1));
            sim_p95_front = distances[p95_idx];
        } else {
            sim_p95_front = 5.0; // Radio basal de celda semilla
        }
        sim_front_per_year.push_back(sim_p95_front);

        // 2. Evaluar observaciones y persistencia para este año
        double sim_phi_sum = 0.0;
        double obs_persist_sum = 0.0;
        int occupied_prev_count = 0;

        while (obs_idx < total_obs && observations[obs_idx].year == year) {
            const auto& obs = observations[obs_idx];
            int32_t cid = obs.cell_id;

            uint32_t arr = 0;
            auto it_arr = disp.arrivals_per_cell.find(cid);
            if (it_arr != disp.arrivals_per_cell.end()) arr = it_arr->second;

            float adult_n = 0.0f;
            auto it_n = fall.adult_density_per_cell.find(cid);
            if (it_n != fall.adult_density_per_cell.end()) adult_n = it_n->second;

            float gamma_col = compute_colonization_gamma(arr, obs.pop_density_h);
            float phi_pers = compute_persistence_phi(adult_n, 0.05f);

            float psi_prev = 0.0f;
            auto it_psi = psi_map.find(cid);
            if (it_psi != psi_map.end()) psi_prev = it_psi->second;

            float psi_curr = update_dynamic_occupancy(psi_prev, gamma_col, phi_pers);
            psi_map[cid] = psi_curr;

            float p_det = compute_detection_prob(obs.effort_araneae, obs.pop_density_h, obs.road_density_r);
            float p_marg = marginal_observation_prob(psi_curr, p_det);

            pred_marginal_all.push_back(p_marg);
            obs_y_all.push_back(obs.y_detected);

            if (psi_prev > 0.5f) {
                sim_phi_sum += phi_pers;
                obs_persist_sum += obs.y_detected;
                occupied_prev_count++;
            }

            obs_idx++;
        }

        if (occupied_prev_count > 0) {
            double sim_rate = sim_phi_sum / occupied_prev_count;
            double obs_rate = obs_persist_sum / occupied_prev_count;
            persistence_discrepancy_sum += std::abs(sim_rate - obs_rate);
            persistence_years_count++;
        }
    }

    double mean_persistence_disc = (persistence_years_count > 0) 
        ? (persistence_discrepancy_sum / persistence_years_count) : 0.05;

    return evaluate_multicriteria_cost(
        pred_marginal_all,
        obs_y_all,
        sim_front_per_year,
        OBSERVED_FRONT_KM,
        mean_persistence_disc,
        weights,
        scales);
}

SimulatedAnnealingReport run_multi_start_simulated_annealing(
    const std::vector<GridCell>& grid_cells,
    const io::DailyThermalGrid& thermal,
    const std::vector<CalibrationObservation>& observations,
    int n_chains,
    int iter_per_chain,
    double initial_temp,
    double delta_threshold,
    const std::string& checkpoint_csv_path)
{
    SimulatedAnnealingReport report;
    report.n_chains = n_chains;
    report.iter_per_chain = iter_per_chain;
    report.total_evaluations = n_chains * iter_per_chain;

    const int n_params = static_cast<int>(CALIBRATION_PARAMS.size());
    std::cout << "\n===================================================================\n";
    std::cout << "▶ Invocando Recocido Simulado Multi-inicio (Simulated Annealing)\n";
    std::cout << "  Cadenas independientes  : " << n_chains << " multi-start\n";
    std::cout << "  Iteraciones por cadena  : " << iter_per_chain << " iter\n";
    std::cout << "  Total evaluaciones 2014-2019 : " << report.total_evaluations << "\n";
    std::cout << "  Espacio de parámetros   : " << n_params << " dimensiones dominantes\n";
    std::cout << "  Temperatura inicial T0  : " << initial_temp << "\n";
    std::cout << "  Tolerancia Delta (Theta_comp) : " << delta_threshold << "\n";
    std::cout << "===================================================================\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Calcular escalas de referencia en el punto nominal (theta_default)
    std::vector<float> theta_nominal(n_params);
    for (int i = 0; i < n_params; ++i) {
        theta_nominal[i] = CALIBRATION_PARAMS[i].default_val;
    }

    std::cout << "[Simulated Annealing] Evaluando escala de referencia inicial en theta_nominal...\n";
    CostReferenceScales unit_scales{1.0, 1.0, 1.0};
    CostComponents nom_comp = evaluate_simulation_cost(
        theta_nominal, grid_cells, thermal, observations, unit_scales);

    CostReferenceScales reference_scales{
        std::max(nom_comp.j_arrival_raw, 0.0001),
        std::max(nom_comp.j_front_raw, 1.0),
        std::max(nom_comp.j_persistence_raw, 0.01)
    };

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "  Escala Brier S_llegada     : " << reference_scales.s_arrival << "\n";
    std::cout << "  Escala Frente S_frente (km): " << reference_scales.s_front << "\n";
    std::cout << "  Escala Persistencia S_pers : " << reference_scales.s_persistence << "\n";

    // Configuraciones de pesos para análisis de sensibilidad
    CostWeights weights_base{0.333333, 0.333333, 0.333334};
    CostWeights weights_config_a{0.20, 0.60, 0.20}; // Enfoque Frente/Dispersión
    CostWeights weights_config_b{0.20, 0.20, 0.60}; // Enfoque Persistencia/Resistencia Térmica

    // Contenedores por cadena
    std::vector<ChainConvergenceRecord> chain_records(n_chains);
    std::vector<std::vector<CompatibleEnsembleMember>> chain_members(n_chains);

    // Semillas para cada cadena
    const std::vector<uint32_t> chain_seeds = {1001, 2002, 3003, 4004, 5005};

    // Factor de enfriamiento geométrico T_k = T0 * alpha^k
    const double cooling_alpha = std::pow(0.001 / initial_temp, 1.0 / iter_per_chain);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < n_chains; ++c) {
        auto c_t0 = std::chrono::high_resolution_clock::now();
        uint32_t seed = chain_seeds[c % chain_seeds.size()];
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist_01(0.0, 1.0);
        std::normal_distribution<double> norm_dist(0.0, 1.0);

        ChainConvergenceRecord& rec = chain_records[c];
        rec.chain_id = c + 1;
        rec.seed = seed;
        rec.total_iterations = iter_per_chain;

        // Estado inicial de la cadena
        std::vector<float> curr_theta(n_params);
        if (c == 0) {
            curr_theta = theta_nominal;
        } else {
            // Muestreo aleatorio dentro del hipercubo de prior bounds
            for (int i = 0; i < n_params; ++i) {
                float min_v = CALIBRATION_PARAMS[i].min_val;
                float max_v = CALIBRATION_PARAMS[i].max_val;
                curr_theta[i] = min_v + static_cast<float>(dist_01(rng)) * (max_v - min_v);
            }
        }

        CostComponents curr_comp = evaluate_simulation_cost(
            curr_theta, grid_cells, thermal, observations, reference_scales, weights_base);

        rec.initial_cost = curr_comp.j_total;
        rec.min_cost = curr_comp.j_total;
        rec.best_theta = curr_theta;
        rec.best_components = curr_comp;

        double temp = initial_temp;

        for (int iter = 0; iter < iter_per_chain; ++iter) {
            // Proponer nuevo candidato mediante perturbación gaussiana escalada por temperatura
            std::vector<float> prop_theta(n_params);
            double temp_ratio = std::sqrt(temp / initial_temp);

            for (int i = 0; i < n_params; ++i) {
                float span = CALIBRATION_PARAMS[i].max_val - CALIBRATION_PARAMS[i].min_val;
                double step = norm_dist(rng) * 0.12 * span * temp_ratio;
                float val = curr_theta[i] + static_cast<float>(step);
                prop_theta[i] = std::clamp(val, CALIBRATION_PARAMS[i].min_val, CALIBRATION_PARAMS[i].max_val);
            }

            CostComponents prop_comp = evaluate_simulation_cost(
                prop_theta, grid_cells, thermal, observations, reference_scales, weights_base);

            double delta_j = prop_comp.j_total - curr_comp.j_total;
            bool accept = false;

            if (delta_j <= 0.0) {
                accept = true;
            } else {
                double p_accept = std::exp(-delta_j / temp);
                if (dist_01(rng) < p_accept) {
                    accept = true;
                }
            }

            if (accept) {
                curr_theta = prop_theta;
                curr_comp = prop_comp;
                rec.accepted_proposals++;

                if (curr_comp.j_total < rec.min_cost) {
                    rec.min_cost = curr_comp.j_total;
                    rec.best_theta = curr_theta;
                    rec.best_components = curr_comp;
                }

                // Evaluar costos bajo configuraciones alternativas A y B analíticamente a partir de componentes normalizados
                double j_comp_a = weights_config_a.w1_arrival * curr_comp.j_arrival_norm +
                                  weights_config_a.w2_front   * curr_comp.j_front_norm +
                                  weights_config_a.w3_persistence * curr_comp.j_persistence_norm;
                double j_comp_b = weights_config_b.w1_arrival * curr_comp.j_arrival_norm +
                                  weights_config_b.w2_front   * curr_comp.j_front_norm +
                                  weights_config_b.w3_persistence * curr_comp.j_persistence_norm;

                CompatibleEnsembleMember member{};
                member.chain_id = c + 1;
                member.iteration = iter;
                member.temperature = temp;
                member.j_total = curr_comp.j_total;
                member.j_arrival_norm = curr_comp.j_arrival_norm;
                member.j_front_norm = curr_comp.j_front_norm;
                member.j_persistence_norm = curr_comp.j_persistence_norm;
                member.j_config_a = j_comp_a;
                member.j_config_b = j_comp_b;
                member.theta = curr_theta;

                chain_members[c].push_back(member);
            }

            // Enfriamiento geométrico
            temp *= cooling_alpha;
        }

        auto c_t1 = std::chrono::high_resolution_clock::now();
        rec.execution_time_s = std::chrono::duration<double>(c_t1 - c_t0).count();

        #pragma omp critical
        {
            std::cout << "[Cadena " << rec.chain_id << "] Finalizada en " 
                      << std::fixed << std::setprecision(2) << rec.execution_time_s << " s | "
                      << "J0 = " << std::setprecision(4) << rec.initial_cost << " -> "
                      << "Jmin = " << rec.min_cost << " | "
                      << "Aceptadas = " << rec.accepted_proposals << "/" << iter_per_chain << " ("
                      << (rec.accepted_proposals * 100 / iter_per_chain) << "%)\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    report.total_runtime_s = std::chrono::duration<double>(t1 - t0).count();
    report.chains = chain_records;

    // 2. Determinar la mejor solución global (MAP / J_min)
    double global_min = 1e6;
    int best_chain_idx = 0;
    for (size_t c = 0; c < chain_records.size(); ++c) {
        if (chain_records[c].min_cost < global_min) {
            global_min = chain_records[c].min_cost;
            best_chain_idx = static_cast<int>(c);
        }
    }
    report.global_min_cost = global_min;
    report.map_theta = chain_records[best_chain_idx].best_theta;
    report.map_components = chain_records[best_chain_idx].best_components;

    // 3. Construir el conjunto compatible Theta_comp: { theta : J(theta) <= J_min + Delta }
    const double j_threshold = global_min + delta_threshold;
    std::vector<CompatibleEnsembleMember> all_compatible;

    for (int c = 0; c < n_chains; ++c) {
        for (const auto& mem : chain_members[c]) {
            if (mem.j_total <= j_threshold) {
                all_compatible.push_back(mem);
            }
        }
    }
    report.compatible_ensemble = all_compatible;

    // Evaluar retención frente a pesos alternativos respecto a sus mínimos específicos
    if (!all_compatible.empty()) {
        double min_j_a = 1e6;
        double min_j_b = 1e6;
        for (const auto& mem : all_compatible) {
            if (mem.j_config_a < min_j_a) min_j_a = mem.j_config_a;
            if (mem.j_config_b < min_j_b) min_j_b = mem.j_config_b;
        }

        int compatible_count_a = 0;
        int compatible_count_b = 0;
        for (const auto& mem : all_compatible) {
            if (mem.j_config_a <= min_j_a + delta_threshold) compatible_count_a++;
            if (mem.j_config_b <= min_j_b + delta_threshold) compatible_count_b++;
        }

        report.retention_rate_config_a = static_cast<double>(compatible_count_a) / all_compatible.size() * 100.0;
        report.retention_rate_config_b = static_cast<double>(compatible_count_b) / all_compatible.size() * 100.0;
    }

    // 4. Calcular percentiles P5, Mediana, P95 en Theta_comp
    report.p5_theta.resize(n_params);
    report.median_theta.resize(n_params);
    report.p95_theta.resize(n_params);

    if (!all_compatible.empty()) {
        for (int p_idx = 0; p_idx < n_params; ++p_idx) {
            std::vector<float> vals;
            vals.reserve(all_compatible.size());
            for (const auto& mem : all_compatible) {
                vals.push_back(mem.theta[p_idx]);
            }
            std::sort(vals.begin(), vals.end());
            size_t n = vals.size();
            report.p5_theta[p_idx] = vals[static_cast<size_t>(0.05 * (n - 1))];
            report.median_theta[p_idx] = vals[static_cast<size_t>(0.50 * (n - 1))];
            report.p95_theta[p_idx] = vals[static_cast<size_t>(0.95 * (n - 1))];
        }
    }

    // 5. Guardar ensamble compatible en CSV
    std::ofstream out_file(checkpoint_csv_path);
    if (out_file.is_open()) {
        out_file << "chain_id,iteration,temperature,j_total,j_arrival_norm,j_front_norm,j_persistence_norm,j_config_a,j_config_b";
        for (int i = 0; i < n_params; ++i) {
            out_file << "," << CALIBRATION_PARAMS[i].name;
        }
        out_file << "\n";

        for (const auto& mem : all_compatible) {
            out_file << mem.chain_id << "," << mem.iteration << "," << mem.temperature << ","
                     << mem.j_total << "," << mem.j_arrival_norm << "," << mem.j_front_norm << ","
                     << mem.j_persistence_norm << "," << mem.j_config_a << "," << mem.j_config_b;
            for (int i = 0; i < n_params; ++i) {
                out_file << "," << mem.theta[i];
            }
            out_file << "\n";
        }
        out_file.close();
        std::cout << "[Simulated Annealing] Ensamble Theta_comp (" << all_compatible.size() 
                  << " miembros) guardado en: " << checkpoint_csv_path << "\n";
    }

    return report;
}

} // namespace joro::inference
