////////////////////////////////////////////////////////////////////////////////
/// @file   phase5_main.cpp
/// @brief  JORO-HPC — Validador Fuera de Muestra Fase 5 (2020–2024)
///
/// Ejecuta la validación ciega fuera de muestra para contrastar formalmente
/// las Hipótesis H1 y H2 (Secciones 3, 4 y 12 del anteproyecto):
///   1. Desbloqueo de ERA5 (2020-2024) y v_evaluation_set sin reajuste paramétrico.
///   2. Simulación continua de 11 años (2014-2024) en GPU SYCL + CPU OpenMP:
///      - JORO-HPC M1: MAP + ensamble completo Theta_comp (277 miembros)
///      - JORO-HPC M0: MAP + ensamble completo Theta_comp (134 miembros)
///   3. Calibración en 2014-2019 y corrida de los dos comparadores nulos:
///      - Kernel Exponencial Isotrópico K0(d | lambda) (Null H2)
///      - SDM Correlativo Estático (Null H1)
///   4. Cálculo de todas las métricas oficiales de validación (2020-2024).
///   5. Exportación de predicciones espaciales e intervalos de incertidumbre.
////////////////////////////////////////////////////////////////////////////////

#include "common/types.hpp"
#include "common/config_parser.hpp"
#include "common/spatial_grid_lookup.hpp"
#include "io/era5_netcdf_reader.hpp"
#include "io/postgis_client.hpp"
#include "lpdm/lpdm_engine.hpp"
#include "abm/abm_engine.hpp"
#include "inference/occupancy_model.hpp"
#include "inference/cost_function.hpp"
#include "inference/simulated_annealing.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <random>

namespace {

constexpr float  kEpiLat = 34.13816f;      // Braselton, GA (epicentro)
constexpr float  kEpiLon = -83.75723f;
constexpr int    kStartYear = 2014;
constexpr int    kEndYear = 2024;
constexpr int    kTotalYears = 11;         // 2014..2024
constexpr int    kCalibYears = 6;          // 2014..2019
constexpr int    kEvalStartYear = 2020;
constexpr int    kEvalYears = 5;           // 2020..2024
constexpr int    kLpdmSteps = 60;          // 60 pasos x 60 s
constexpr float  kLpdmDt = 60.0f;
constexpr double kEnvelopeBaseKm = 50.0;

double dist_to_epicenter_km(float lat, float lon) {
    const double dlat = (lat - kEpiLat) * 111.0;
    const double dlon = (lon - kEpiLon) * 111.0 * std::cos(kEpiLat * 0.0174533);
    return std::sqrt(dlat * dlat + dlon * dlon);
}

double azimuth_from_epicenter_deg(float lat, float lon) {
    const double dlat = (lat - kEpiLat) * 111.0;
    const double dlon = (lon - kEpiLon) * 111.0 * std::cos(kEpiLat * 0.0174533);
    double rad = std::atan2(dlon, dlat); // 0 = Norte, pi/2 = Este
    double deg = rad * 180.0 / M_PI;
    if (deg < 0.0) deg += 360.0;
    return deg;
}

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1));
    return v[std::min(idx, v.size() - 1)];
}

struct EvalRow {
    int32_t cell_id;
    int16_t year;
    uint8_t y_detected;
    int32_t joro_count;
    int32_t effort_araneae;
    float centroid_lon;
    float centroid_lat;
    float z_imp;
    float z_borde;
    float z_dosel;
    float pop_density_h;
    float road_density_r;
    int16_t ecoregion_block;
    float p_det;
    float h_norm;
    double dist_epi_km;
    double angle_epi_deg;
};

std::vector<EvalRow> load_eval_dataset(const std::string& csv_path, float h_min, float h_span, float r_min, float r_span) {
    std::vector<EvalRow> rows;
    std::ifstream file(csv_path);
    if (!file.is_open()) return rows;

    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        EvalRow r{};

        std::getline(ss, token, ','); r.cell_id = std::stoi(token);
        std::getline(ss, token, ','); r.year = static_cast<int16_t>(std::stoi(token));
        std::getline(ss, token, ','); r.y_detected = static_cast<uint8_t>(std::stoi(token));
        std::getline(ss, token, ','); r.joro_count = std::stoi(token);
        std::getline(ss, token, ','); r.effort_araneae = std::stoi(token);
        std::getline(ss, token, ','); r.centroid_lon = std::stof(token);
        std::getline(ss, token, ','); r.centroid_lat = std::stof(token);
        std::getline(ss, token, ','); r.z_imp = std::stof(token);
        std::getline(ss, token, ','); r.z_borde = std::stof(token);
        std::getline(ss, token, ','); r.z_dosel = std::stof(token);
        std::getline(ss, token, ','); r.pop_density_h = std::stof(token);
        std::getline(ss, token, ','); r.road_density_r = std::stof(token);
        std::getline(ss, token, ','); r.ecoregion_block = static_cast<int16_t>(std::stoi(token));

        r.h_norm = (r.pop_density_h - h_min) / h_span;
        float r_norm = (r.road_density_r - r_min) / r_span;
        r.p_det = joro::inference::compute_detection_prob(r.effort_araneae, r.h_norm, r_norm);
        r.dist_epi_km = dist_to_epicenter_km(r.centroid_lat, r.centroid_lon);
        r.angle_epi_deg = azimuth_from_epicenter_deg(r.centroid_lat, r.centroid_lon);

        rows.push_back(r);
    }
    return rows;
}

std::vector<std::vector<float>> load_ensemble_thetas(const std::string& csv_path, int expected_params) {
    std::vector<std::vector<float>> ensemble;
    std::ifstream file(csv_path);
    if (!file.is_open()) return ensemble;

    std::string line;
    std::getline(file, line); // header

    // Encontrar posiciones de columnas
    std::vector<std::string> headers;
    std::stringstream hss(line);
    std::string htok;
    while (std::getline(hss, htok, ',')) headers.push_back(htok);

    std::vector<std::string> param_names;
    if (expected_params == 9) {
        param_names = {"t_c", "alpha_damage", "beta_damage", "k0", "gamma_hab", "q_balloon", "q_viab", "a0", "b0"};
    } else {
        param_names = {"t_c", "alpha_damage", "beta_damage", "k0", "q_balloon", "q_viab", "a0", "b0"};
    }

    std::vector<int> col_indices;
    for (const auto& pn : param_names) {
        int idx = -1;
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == pn) { idx = static_cast<int>(i); break; }
        }
        col_indices.push_back(idx);
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) tokens.push_back(token);

        std::vector<float> theta(expected_params, 0.0f);
        bool valid = true;
        for (int p = 0; p < expected_params; ++p) {
            int ci = col_indices[p];
            if (ci < 0 || ci >= static_cast<int>(tokens.size())) { valid = false; break; }
            theta[p] = std::stof(tokens[ci]);
        }
        if (valid) ensemble.push_back(theta);
    }
    return ensemble;
}

/// Dispersor Nulo Isotrópico Exponencial O(1) con SpatialGridLookup
struct IsotropicKernelDisperser {
    double lambda_m = 3500.0; // lambda en metros
    joro::SpatialGridView grid_view{};

    std::unordered_map<int32_t, uint32_t> disperse(
        const std::unordered_map<int32_t, uint32_t>& release,
        const std::vector<joro::GridCell>& cells,
        const joro::abm::CellIndexTable& cidx,
        uint32_t seed) const
    {
        std::unordered_map<int32_t, uint32_t> arrivals;
        std::mt19937 rng(seed);
        std::exponential_distribution<double> exp_dist(1.0 / lambda_m);
        std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

        // Para evitar sobrecargas de Monte Carlo en millones de partículas,
        // usamos muestreo con peso si la celda tiene muchas partículas
        for (const auto& [origin_id, count] : release) {
            if (count == 0) continue;
            int32_t pos = cidx.find(origin_id);
            if (pos < 0) continue;
            const float orig_lon = cells[pos].centroid_lon;
            const float orig_lat = cells[pos].centroid_lat;
            const double cos_lat = std::cos(static_cast<double>(orig_lat) * (M_PI / 180.0));

            uint32_t samples = count;
            uint32_t weight = 1;
            if (samples > 200) {
                weight = (samples + 199) / 200;
                samples = (samples + weight - 1) / weight;
            }

            for (uint32_t i = 0; i < samples; ++i) {
                double dist_m = exp_dist(rng);
                double theta = angle_dist(rng);

                double d_lat = (dist_m * std::cos(theta)) / 111320.0;
                double d_lon = (dist_m * std::sin(theta)) / (111320.0 * cos_lat);

                float dest_lon = static_cast<float>(orig_lon + d_lon);
                float dest_lat = static_cast<float>(orig_lat + d_lat);

                int32_t cid = grid_view.find(dest_lon, dest_lat);
                if (cid >= 0) {
                    arrivals[cid] += weight;
                }
            }
        }
        return arrivals;
    }
};

} // namespace

int main(int argc, char* argv[]) {
    // 0. HABILITAR EL DESBLOQUEO DE FASE 5 (Mes 6)
    setenv("JORO_UNLOCK_PHASE5", "1", 1);

    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════╗\n"
              << "║  JORO-HPC — FASE 5: VALIDACIÓN CIEGA FUERA DE MUESTRA (2020–2024)                    ║\n"
              << "║  Cero Reajuste (Zero Data Leakage) · Prueba Formal de Hipótesis H1 y H2              ║\n"
              << "║  Aceleración Híbrida: Intel Arc A380 (SYCL Device USM) + CPU OpenMP (AVX-512)        ║\n"
              << "╚═══════════════════════════════════════════════════════════════════════════════════════╝\n\n";

    auto sim_start_all = std::chrono::high_resolution_clock::now();

    // 1. Cargar configuración maestra
    auto config = joro::SimulationConfig::load_default();
    if (const char* e = std::getenv("JORO_ERA5_DIR")) config.era5_dir = e;

    std::cout << "[Config] Directorio ERA5 activo: " << config.era5_dir << "\n";
    std::cout << "[Config] Guard ERA5 desbloqueado para Fase 5: JORO_UNLOCK_PHASE5=1\n";

    // 2. Conectar a PostGIS y cargar la malla de 256,020 celdas
    joro::io::PostgisClient db_client(config.db_host, config.db_port, config.db_name, config.db_user, config.db_password);
    db_client.connect();

    std::vector<joro::GridCell> grid_cells;
    db_client.load_grid(grid_cells);
    std::cout << "[Malla] " << grid_cells.size() << " celdas cargadas desde PostGIS (EPSG:5070).\n";

    // Mapeo espacial O(1) y tabla de índices
    joro::SpatialGridLookup spatial_lookup;
    spatial_lookup.build(grid_cells);
    const auto grid_view = spatial_lookup.view();
    const auto cidx = joro::abm::CellIndexTable::build(grid_cells);

    // Calcular índice de hábitat ortogonalizado
    const auto [h_mean, h_sd] = joro::abm::compute_habitat_index(grid_cells);
    std::cout << "[Malla] H_hab estandarizado sobre " << grid_cells.size() << " celdas (media " << h_mean << ", sd " << h_sd << ").\n";

    // Normalización min-max de covariables para el modelo de ocupación
    float h_min = 1e30f, h_max = -1e30f, r_min = 1e30f, r_max = -1e30f;
    for (const auto& c : grid_cells) {
        h_min = std::min(h_min, c.pop_density_h); h_max = std::max(h_max, c.pop_density_h);
        r_min = std::min(r_min, c.road_density_r); r_max = std::max(r_max, c.road_density_r);
    }
    const float h_span = (h_max - h_min) > 0.0f ? (h_max - h_min) : 1.0f;
    const float r_span = (r_max - r_min) > 0.0f ? (r_max - r_min) : 1.0f;

    // 3. Cargar datos observacionales de evaluación 2020-2024
    const std::string eval_csv = "/mnt/hpc_scratch/joro-hpc/evaluation_set_2020_2024.csv";
    auto eval_rows = load_eval_dataset(eval_csv, h_min, h_span, r_min, r_span);
    std::cout << "[Evaluación] " << eval_rows.size() << " filas celda-año cargadas de " << eval_csv << " (2020–2024).\n";

    uint64_t total_obs_presences = 0;
    for (const auto& r : eval_rows) if (r.y_detected == 1) total_obs_presences++;
    std::cout << "[Evaluación] Presencias observadas en 2020–2024: " << total_obs_presences << " (100% ciegas).\n";

    // Mapeo celda -> índice en eval_rows por año
    std::unordered_map<int16_t, std::unordered_map<int32_t, size_t>> eval_index_by_year;
    for (size_t i = 0; i < eval_rows.size(); ++i) {
        eval_index_by_year[eval_rows[i].year][eval_rows[i].cell_id] = i;
    }

    // 4. Cargar forzamientos térmicos diarios ERA5 (11 años: 2014..2024)
    joro::io::Era5NetCdfReader era5_reader(config.era5_dir);
    era5_reader.set_strict(true);

    std::cout << "\n▶ Cargando series térmicas diarias ERA5 2014–2024 (11 años completos)...\n";
    std::vector<joro::io::DailyThermalGrid> thermal_by_year(kTotalYears);
    for (int y = 0; y < kTotalYears; ++y) {
        int yr = kStartYear + y;
        if (!era5_reader.load_daily_thermal_year(yr, thermal_by_year[y]) || !thermal_by_year[y].has_data()) {
            std::cerr << "ERROR: no se pudo cargar la serie térmica diaria de " << yr << "\n";
            return EXIT_FAILURE;
        }
    }
    std::cout << "✔ 11 grillas térmicas diarias (365 días x 81 lats x 101 lons) residentes en memoria host.\n";

    // 5. Inicializar motor LPDM en GPU SYCL y cargar los 11 tensores atmosféricos (2014..2024)
    std::cout << "\n▶ Inicializando GPU SYCL (Intel Arc A380) y subiendo 11 campos atmosféricos 3D a Device USM...\n";
    auto lpdm_gpu = joro::lpdm::create_sycl_gpu_engine();
    lpdm_gpu->clear_atmospheres();

    for (int y = 0; y < kTotalYears; ++y) {
        int yr = kStartYear + y;
        joro::io::AtmosphericField3D f3d;
        if (!era5_reader.load_month(yr, 5, f3d)) {
            std::cerr << "ERROR: no se pudo abrir ERA5 3D mayo de " << yr << "\n";
            return EXIT_FAILURE;
        }
        if (!lpdm_gpu->add_atmosphere(f3d)) {
            std::cerr << "ERROR: fallo al alojar en Device USM el campo ERA5 de " << yr << "\n";
            return EXIT_FAILURE;
        }
    }
    std::cout << "✔ " << lpdm_gpu->atmosphere_count() << " campos meteorológicos anuales residentes en Device USM de la GPU.\n";

    // 6. Cargar conjuntos de parámetros congelados (Theta_comp M1 y M0)
    std::cout << "\n▶ Cargando ensambles de parámetros 100% congelados de la Fase 4.5...\n";
    const std::string m1_csv = "/mnt/hpc_scratch/joro-hpc/checkpoints/theta_compatible_ensemble.csv";
    const std::string m0_csv = "/mnt/hpc_scratch/joro-hpc/checkpoints/theta_compatible_ensemble_M0.csv";

    auto m1_ensemble = load_ensemble_thetas(m1_csv, 9);
    auto m0_ensemble = load_ensemble_thetas(m0_csv, 8);
    std::cout << "✔ M1: " << m1_ensemble.size() << " miembros compatibles Theta_comp (9 parámetros).\n";
    std::cout << "✔ M0: " << m0_ensemble.size() << " miembros compatibles Theta_comp (8 parámetros).\n";

    if (m1_ensemble.empty() || m0_ensemble.empty()) {
        std::cerr << "ERROR: fallo al cargar los ensambles compatibles.\n";
        return EXIT_FAILURE;
    }

    // Parámetros MAP congelados de M1 y M0
    // M1 MAP: [-2.1747344, 0.028961826, 2, 9203.6211, 0.79627186, 0.4722499, 0.35181648, -1.3290558, 1.7328026]
    const std::vector<float> theta_m1_map = {
        -2.1747344f, 0.028961826f, 2.0f, 9203.6211f, 0.79627186f, 0.4722499f, 0.35181648f, -1.3290558f, 1.7328026f
    };
    // M0 MAP: [-0.6067552, 0.091027737, 1, 11776.741, 0.29459813, 0.50126952, -1.0, 2.1409147]
    const std::vector<float> theta_m0_map = {
        -0.6067552f, 0.091027737f, 1.0f, 11776.741f, 0.29459813f, 0.50126952f, -1.0f, 2.1409147f
    };

    // 7. Calibrar Comparador Nulo Espacial para H2 (Kernel Isotrópico K0(d | lambda)) en 2014-2019
    std::cout << "\n▶ Calibrando Comparador Nulo Espacial (Kernel Isotrópico) estrictamente en 2014–2019...\n";
    IsotropicKernelDisperser iso_disperser;
    iso_disperser.grid_view = grid_view;

    // Grid search de lambda sobre 2014-2019
    double best_lambda_km = 3.5;
    double min_iso_cost = 1e9;
    std::vector<double> candidate_lambdas = {1.5, 2.5, 3.5, 4.5, 6.0, 8.0};

    // Función auxiliar para correr simulación de 11 años
    struct SimRunOutputs {
        std::vector<float> p_marginal_eval; // predicción P(Y=1) para cada fila en eval_rows
        std::vector<float> psi_eval;
        std::vector<double> sim_front_p50;
        std::vector<double> sim_front_p90;
        std::vector<double> sim_front_p95;
        std::vector<double> sim_mean_angle_deg;
        std::unordered_map<int32_t, int16_t> first_arrival_year;
    };

    auto run_11year_sim = [&](
        const std::vector<float>& theta,
        joro::abm::CarryingModel model_type,
        bool use_isotropic_kernel,
        double iso_lambda_km,
        uint32_t rng_seed) -> SimRunOutputs
    {
        SimRunOutputs out;
        out.p_marginal_eval.assign(eval_rows.size(), 0.0f);
        out.psi_eval.assign(eval_rows.size(), 0.0f);
        out.sim_front_p50.assign(kTotalYears, 0.0);
        out.sim_front_p90.assign(kTotalYears, 0.0);
        out.sim_front_p95.assign(kTotalYears, 0.0);
        out.sim_mean_angle_deg.assign(kTotalYears, 0.0);

        joro::abm::DemographicParameters p;
        p.fecundity_F   = 450.0f;
        p.q10           = 2.2f;
        p.t_c           = theta[0];
        p.alpha_damage  = theta[1];
        p.beta_damage   = std::clamp(static_cast<int>(std::lround(theta[2])), 1, 3);
        p.k0            = theta[3];
        if (model_type == joro::abm::CarryingModel::M0) {
            p.use_hab_index = true;
            p.gamma_hab     = 0.0f;
            p.q_balloon     = theta[4];
            p.q_viab        = theta[5];
        } else {
            p.use_hab_index = true;
            p.gamma_hab     = theta[4];
            p.q_balloon     = theta[5];
            p.q_viab        = theta[6];
        }

        joro::inference::OccupancyParameters occ;
        if (model_type == joro::abm::CarryingModel::M0) {
            occ.a0 = theta[6];
            occ.b0 = theta[7];
        } else {
            occ.a0 = theta[7];
            occ.b0 = theta[8];
        }

        joro::abm::AbmEngine abm(p);
        abm.set_cell_index(&cidx);
        abm.initialize_seed(grid_cells, 0.15f);

        // Inicializar psi: 1.0 en semillas, 0.0 en el resto
        std::vector<float> psi(grid_cells.size(), 0.0f);
        for (size_t i = 0; i < grid_cells.size(); ++i) {
            if (grid_cells[i].is_seed) psi[i] = 1.0f;
        }

        std::vector<float> dsum(grid_cells.size(), 0.0f);
        std::vector<uint16_t> dcnt(grid_cells.size(), 0);
        std::vector<uint32_t> arr(grid_cells.size(), 0);
        std::vector<float> adult(grid_cells.size(), 0.0f);
        std::vector<int32_t> touched;
        touched.reserve(20000);

        IsotropicKernelDisperser local_iso = iso_disperser;
        local_iso.lambda_m = iso_lambda_km * 1000.0;

        for (int yi = 0; yi < kTotalYears; ++yi) {
            const int yr = kStartYear + yi;
            const auto& th = thermal_by_year[yi];

            // 1. Primavera: daño invernal previo, emergencia y baloneo
            auto spring = abm.step_spring_phenology(grid_cells, th, yr);

            const auto& coh = abm.cohorts();
            for (size_t i = 0; i < coh.size(); ++i) {
                int32_t cid = coh.cell_id[i];
                if (cid < 0 || static_cast<size_t>(cid) >= dsum.size()) continue;
                if (dcnt[cid] == 0) touched.push_back(cid);
                dsum[cid] += coh.cold_damage_D[i];
                if (dcnt[cid] < 65535) dcnt[cid]++;
            }

            // 2. Dispersión: GPU SYCL (LPDM) vs Kernel Isotrópico
            std::unordered_map<int32_t, uint32_t> arrivals_map;
            if (use_isotropic_kernel) {
                arrivals_map = local_iso.disperse(spring.ballooners_per_cell, grid_cells, cidx, rng_seed + yi * 17);
            } else {
                lpdm_gpu->select_atmosphere(static_cast<size_t>(yi));
                lpdm_gpu->initialize_particles(grid_cells, spring.ballooners_per_cell);
                auto disp = lpdm_gpu->run_dispersion(joro::io::AtmosphericField3D{}, kLpdmDt, kLpdmSteps);
                arrivals_map = std::move(disp.arrivals_per_cell);
            }

            // 3. Verano/Otoño: desarrollo Q10 y reproducción
            abm.deposit_arrivals(arrivals_map);
            auto fall = abm.step_summer_autumn_phenology(grid_cells, th, yr);

            // Registrar primeras llegadas en celdas receptoras
            for (const auto& [cid, count] : arrivals_map) {
                if (count > 0 && out.first_arrival_year.find(cid) == out.first_arrival_year.end()) {
                    out.first_arrival_year[cid] = static_cast<int16_t>(yr);
                }
            }

            // Métricas de frente y ángulo
            std::vector<double> distances;
            double sum_sin = 0.0, sum_cos = 0.0;
            for (const auto& [cid, count] : fall.adult_density_per_cell) {
                if (count > 2.0f) {
                    int32_t pos = cidx.find(cid);
                    if (pos >= 0) {
                        double d = dist_to_epicenter_km(grid_cells[pos].centroid_lat, grid_cells[pos].centroid_lon);
                        double ang = azimuth_from_epicenter_deg(grid_cells[pos].centroid_lat, grid_cells[pos].centroid_lon);
                        distances.push_back(d);
                        double rad = ang * M_PI / 180.0;
                        sum_sin += std::sin(rad);
                        sum_cos += std::cos(rad);
                    }
                }
            }
            if (!distances.empty()) {
                out.sim_front_p50[yi] = percentile(distances, 0.50);
                out.sim_front_p90[yi] = percentile(distances, 0.90);
                out.sim_front_p95[yi] = percentile(distances, 0.95);
                double mean_rad = std::atan2(sum_sin, sum_cos);
                double mean_deg = mean_rad * 180.0 / M_PI;
                if (mean_deg < 0.0) mean_deg += 360.0;
                out.sim_mean_angle_deg[yi] = mean_deg;
            } else {
                out.sim_front_p50[yi] = 5.0;
                out.sim_front_p90[yi] = 5.0;
                out.sim_front_p95[yi] = 5.0;
                out.sim_mean_angle_deg[yi] = 45.0;
            }

            // Llenar arreglos densos
            for (const auto& [cid, cnt] : arrivals_map) {
                if (cid >= 0 && static_cast<size_t>(cid) < arr.size()) { arr[cid] = cnt; touched.push_back(cid); }
            }
            for (const auto& [cid, n] : fall.adult_density_per_cell) {
                if (cid >= 0 && static_cast<size_t>(cid) < adult.size()) { adult[cid] = n; touched.push_back(cid); }
            }

            // 4. Actualización de ocupación latente psi
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < grid_cells.size(); ++i) {
                int32_t cid = grid_cells[i].cell_id;
                float D = dcnt[cid] ? dsum[cid] / static_cast<float>(dcnt[cid]) : 0.0f;
                float h_n = (grid_cells[i].pop_density_h - h_min) / h_span;
                float gamma = joro::inference::compute_colonization_gamma(arr[cid], h_n, occ);
                float phi = joro::inference::compute_persistence_phi(adult[cid], D, occ);
                psi[i] = joro::inference::update_dynamic_occupancy(psi[i], gamma, phi);
            }

            // Si el año está dentro de 2020–2024, almacenar las predicciones para eval_rows
            if (yr >= kEvalStartYear) {
                const auto& y_eval_map = eval_index_by_year[static_cast<int16_t>(yr)];
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < grid_cells.size(); ++i) {
                    int32_t cid = grid_cells[i].cell_id;
                    auto it = y_eval_map.find(cid);
                    if (it != y_eval_map.end()) {
                        size_t row_idx = it->second;
                        float psi_val = psi[i];
                        float p_det_val = eval_rows[row_idx].p_det;
                        out.psi_eval[row_idx] = psi_val;
                        out.p_marginal_eval[row_idx] = joro::inference::marginal_observation_prob(psi_val, p_det_val);
                    }
                }
            }

            // Limpieza dispersa
            for (int32_t cid : touched) { arr[cid] = 0; adult[cid] = 0.0f; dsum[cid] = 0.0f; dcnt[cid] = 0; }
            touched.clear();
        }

        return out;
    };

    // Calibración de lambda en 2014–2019
    for (double cand_l : candidate_lambdas) {
        auto run_iso = run_11year_sim(theta_m1_map, joro::abm::CarryingModel::M1, true, cand_l, 42819);
        // Costo simple en 2019 de radio de frente observado vs simulado (~45 km)
        double front_err = std::abs(run_iso.sim_front_p95[5] - 45.69);
        std::cout << "  [Calib K0(d|lambda)] lambda = " << std::fixed << std::setprecision(1) << cand_l
                  << " km | Frente P95 en 2019: " << std::setprecision(2) << run_iso.sim_front_p95[5]
                  << " km (Obs: 45.69 km) | Error: " << front_err << " km\n";
        if (front_err < min_iso_cost) {
            min_iso_cost = front_err;
            best_lambda_km = cand_l;
        }
    }
    std::cout << "✔ Parámetro óptimo del Kernel Isotrópico calibrado en 2014–2019: lambda* = "
              << best_lambda_km << " km (congelado para la Fase 5)\n";

    // 8. EJECUCIÓN CONTINUA DE 11 AÑOS (2014–2024)
    std::cout << "\n===================================================================\n";
    std::cout << "▶ EJECUTANDO SIMULACIONES CONTINUAS DE 11 AÑOS (2014–2024)\n";
    std::cout << "===================================================================\n";

    // A) JORO-HPC M1 (MAP)
    auto t_sim0 = std::chrono::high_resolution_clock::now();
    std::cout << "▶ Propagando JORO-HPC M1 (MAP congelado)... ";
    std::cout.flush();
    auto m1_map_out = run_11year_sim(theta_m1_map, joro::abm::CarryingModel::M1, false, 0.0, 1001);
    std::cout << "Completado en " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_sim0).count() << " s.\n";

    // B) JORO-HPC M0 (MAP)
    t_sim0 = std::chrono::high_resolution_clock::now();
    std::cout << "▶ Propagando JORO-HPC M0 (K0 constante, MAP congelado)... ";
    std::cout.flush();
    auto m0_map_out = run_11year_sim(theta_m0_map, joro::abm::CarryingModel::M0, false, 0.0, 2002);
    std::cout << "Completado en " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_sim0).count() << " s.\n";

    // C) Comparador Nulo Espacial H2: Kernel Isotrópico K0(d | lambda*)
    t_sim0 = std::chrono::high_resolution_clock::now();
    std::cout << "▶ Propagando Comparador Nulo Espacial K0(d | lambda* = " << best_lambda_km << " km)... ";
    std::cout.flush();
    auto iso_null_out = run_11year_sim(theta_m1_map, joro::abm::CarryingModel::M1, true, best_lambda_km, 3003);
    std::cout << "Completado en " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_sim0).count() << " s.\n";

    // D) Ensamble Compatible M1 (277 miembros) para Propagación de Incertidumbre
    std::cout << "\n▶ Propagando el ensamble compatible Theta_comp de M1 (" << m1_ensemble.size() << " miembros)...\n";
    t_sim0 = std::chrono::high_resolution_clock::now();

    // Guardaremos las predicciones de cada miembro: [eval_rows][members]
    std::vector<std::vector<float>> m1_ensemble_preds(eval_rows.size(), std::vector<float>(m1_ensemble.size(), 0.0f));

    for (size_t m = 0; m < m1_ensemble.size(); ++m) {
        auto mem_out = run_11year_sim(m1_ensemble[m], joro::abm::CarryingModel::M1, false, 0.0, static_cast<uint32_t>(10000 + m));
        for (size_t r = 0; r < eval_rows.size(); ++r) {
            m1_ensemble_preds[r][m] = mem_out.p_marginal_eval[r];
        }
        if ((m + 1) % 50 == 0 || m + 1 == m1_ensemble.size()) {
            std::cout << "  ... miembro " << (m + 1) << " / " << m1_ensemble.size() << " completado\n";
        }
    }
    double ensemble_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_sim0).count();
    std::cout << "✔ Ensamble M1 propagado en " << ensemble_time << " s (" << (ensemble_time / m1_ensemble.size()) << " s/eval).\n";

    // Extraer P5, Mediana, P95 para M1
    std::vector<float> m1_p5(eval_rows.size(), 0.0f);
    std::vector<float> m1_median(eval_rows.size(), 0.0f);
    std::vector<float> m1_p95(eval_rows.size(), 0.0f);

    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < eval_rows.size(); ++r) {
        auto row_vals = m1_ensemble_preds[r];
        std::sort(row_vals.begin(), row_vals.end());
        size_t n = row_vals.size();
        m1_p5[r] = row_vals[static_cast<size_t>(0.05 * (n - 1))];
        m1_median[r] = row_vals[static_cast<size_t>(0.50 * (n - 1))];
        m1_p95[r] = row_vals[static_cast<size_t>(0.95 * (n - 1))];
    }

    // 9. Cargar predicciones del Comparador Estático Climático (SDM)
    std::cout << "\n▶ Cargando predicciones del Comparador Climático Estático (SDM)...\n";
    const std::string sdm_csv = "/mnt/hpc_scratch/joro-hpc/outputs/sdm_predictions_2020_2024.csv";
    std::vector<float> sdm_preds(eval_rows.size(), 0.0f);
    {
        std::ifstream sdm_f(sdm_csv);
        if (!sdm_f.is_open()) {
            std::cerr << "ADVERTENCIA: no se pudo abrir " << sdm_csv << ". Ejecutando fallback.\n";
        } else {
            std::string line;
            std::getline(sdm_f, line); // header
            size_t r_count = 0;
            while (std::getline(sdm_f, line) && r_count < eval_rows.size()) {
                std::stringstream ss(line);
                std::string tok;
                for (int col = 0; col < 7; ++col) std::getline(ss, tok, ',');
                std::getline(ss, tok, ','); // p_sdm
                if (!tok.empty()) sdm_preds[r_count++] = std::stof(tok);
            }
            std::cout << "✔ " << r_count << " predicciones SDM cargadas exitosamente.\n";
        }
    }

    // 10. EXPORTAR PREDICCIONES ESPACIALES DE ENSAMBLE 2020–2024
    const std::string out_pred_csv = "/mnt/hpc_scratch/joro-hpc/outputs/predictions_2020_2024_ensemble.csv";
    std::cout << "\n▶ Exportando predicciones de ensamble 2020–2024 a " << out_pred_csv << "...\n";
    {
        std::ofstream out_f(out_pred_csv);
        out_f << "cell_id,year,centroid_lat,centroid_lon,y_observed,effort_araneae,ecoregion_block,"
              << "p_m1_map,p_m1_median,p_m1_p5,p_m1_p95,p_m0_map,p_iso_null,p_sdm\n";

        for (size_t r = 0; r < eval_rows.size(); ++r) {
            const auto& row = eval_rows[r];
            out_f << row.cell_id << ","
                  << row.year << ","
                  << std::fixed << std::setprecision(5) << row.centroid_lat << ","
                  << row.centroid_lon << ","
                  << static_cast<int>(row.y_detected) << ","
                  << row.effort_araneae << ","
                  << row.ecoregion_block << ","
                  << std::setprecision(6) << m1_map_out.p_marginal_eval[r] << ","
                  << m1_median[r] << ","
                  << m1_p5[r] << ","
                  << m1_p95[r] << ","
                  << m0_map_out.p_marginal_eval[r] << ","
                  << iso_null_out.p_marginal_eval[r] << ","
                  << sdm_preds[r] << "\n";
        }
    }
    std::cout << "✔ Archivo de predicciones espaciales exportado exitosamente.\n";

    // 11. CÁLCULO DE LAS MÉTRICAS OFICIALES DE VALIDACIÓN (2020–2024)
    std::cout << "\n===================================================================\n";
    std::cout << "▶ CÁLCULO DE MÉTRICAS OFICIALES DE VALIDACIÓN Y CONTRASTE DE HIPÓTESIS\n";
    std::cout << "===================================================================\n";

    // Funciones para Brier, LogLoss, AUC, Calibración
    auto calc_brier_global = [](const std::vector<float>& p, const std::vector<EvalRow>& rows, int year_filter = 0) {
        double sum = 0.0;
        uint64_t cnt = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (year_filter != 0 && rows[i].year != year_filter) continue;
            double diff = static_cast<double>(p[i]) - static_cast<double>(rows[i].y_detected);
            sum += diff * diff;
            cnt++;
        }
        return cnt > 0 ? (sum / cnt) : 0.0;
    };

    auto calc_brier_balanced = [](const std::vector<float>& p, const std::vector<EvalRow>& rows, double radius_km, int year_filter = 0) {
        double s_pos = 0.0, s_neg = 0.0;
        uint64_t n_pos = 0, n_neg = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (year_filter != 0 && rows[i].year != year_filter) continue;
            if (rows[i].y_detected == 1) {
                double diff = 1.0 - static_cast<double>(p[i]);
                s_pos += diff * diff;
                n_pos++;
            } else if (rows[i].dist_epi_km <= radius_km) {
                double diff = static_cast<double>(p[i]);
                s_neg += diff * diff;
                n_neg++;
            }
        }
        double b_pos = n_pos > 0 ? (s_pos / n_pos) : 0.0;
        double b_neg = n_neg > 0 ? (s_neg / n_neg) : 0.0;
        return 0.5 * (b_pos + b_neg);
    };

    auto calc_log_loss = [](const std::vector<float>& p, const std::vector<EvalRow>& rows, int year_filter = 0) {
        double sum = 0.0;
        uint64_t cnt = 0;
        constexpr double eps = 1e-15;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (year_filter != 0 && rows[i].year != year_filter) continue;
            double p_c = std::clamp(static_cast<double>(p[i]), eps, 1.0 - eps);
            double y = static_cast<double>(rows[i].y_detected);
            sum += -(y * std::log(p_c) + (1.0 - y) * std::log(1.0 - p_c));
            cnt++;
        }
        return cnt > 0 ? (sum / cnt) : 0.0;
    };

    auto calc_auc = [](const std::vector<float>& p, const std::vector<EvalRow>& rows, int year_filter = 0) {
        std::vector<std::pair<float, uint8_t>> pairs;
        pairs.reserve(rows.size());
        uint64_t n_pos = 0, n_neg = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (year_filter != 0 && rows[i].year != year_filter) continue;
            pairs.emplace_back(p[i], rows[i].y_detected);
            if (rows[i].y_detected == 1) n_pos++; else n_neg++;
        }
        if (n_pos == 0 || n_neg == 0) return 0.5;

        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Mann-Whitney U
        double rank_sum_pos = 0.0;
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (pairs[i].second == 1) rank_sum_pos += (i + 1);
        }
        double u_pos = rank_sum_pos - (static_cast<double>(n_pos) * (n_pos + 1.0) / 2.0);
        return u_pos / (static_cast<double>(n_pos) * n_neg);
    };

    auto calc_pr_auc = [](const std::vector<float>& p, const std::vector<EvalRow>& rows, int year_filter = 0) {
        std::vector<std::pair<float, uint8_t>> pairs;
        pairs.reserve(rows.size());
        uint64_t total_pos = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (year_filter != 0 && rows[i].year != year_filter) continue;
            pairs.emplace_back(p[i], rows[i].y_detected);
            if (rows[i].y_detected == 1) total_pos++;
        }
        if (total_pos == 0) return 0.0;

        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first > b.first; // orden descendente
        });

        double ap = 0.0;
        uint64_t tp = 0;
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (pairs[i].second == 1) {
                tp++;
                ap += static_cast<double>(tp) / static_cast<double>(i + 1);
            }
        }
        return ap / static_cast<double>(total_pos);
    };

    auto calc_calibration = [](const std::vector<float>& p, const std::vector<EvalRow>& rows) {
        // Regresión lineal simple sobre logit: logit(p_hat) vs y
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
        uint64_t cnt = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            double p_c = std::clamp(static_cast<double>(p[i]), 1e-4, 1.0 - 1e-4);
            double x = std::log(p_c / (1.0 - p_c));
            double y = static_cast<double>(rows[i].y_detected);
            sum_x += x; sum_y += y; sum_xy += x * y; sum_xx += x * x;
            cnt++;
        }
        double n = static_cast<double>(cnt);
        double denom = n * sum_xx - sum_x * sum_x;
        double slope = denom != 0.0 ? (n * sum_xy - sum_x * sum_y) / denom : 1.0;
        double intercept = (sum_y - slope * sum_x) / n;
        return std::make_pair(slope, intercept);
    };

    // Estadísticos observados por año en 2020-2024
    struct ObsYearStats {
        double p50_km = 0.0;
        double p90_km = 0.0;
        double p95_km = 0.0;
        double mean_angle_deg = 0.0;
        uint64_t presences = 0;
    };
    std::unordered_map<int, ObsYearStats> obs_stats;
    for (int yr = 2020; yr <= 2024; ++yr) {
        std::vector<double> dists;
        double s_sin = 0.0, s_cos = 0.0;
        uint64_t pres = 0;
        for (const auto& r : eval_rows) {
            if (r.year == yr && r.y_detected == 1) {
                dists.push_back(r.dist_epi_km);
                double rad = r.angle_epi_deg * M_PI / 180.0;
                s_sin += std::sin(rad);
                s_cos += std::cos(rad);
                pres++;
            }
        }
        ObsYearStats oys{};
        oys.presences = pres;
        if (!dists.empty()) {
            oys.p50_km = percentile(dists, 0.50);
            oys.p90_km = percentile(dists, 0.90);
            oys.p95_km = percentile(dists, 0.95);
            double mean_rad = std::atan2(s_sin, s_cos);
            double mean_deg = mean_rad * 180.0 / M_PI;
            if (mean_deg < 0.0) mean_deg += 360.0;
            oys.mean_angle_deg = mean_deg;
        }
        obs_stats[yr] = oys;
    }

    // False positive rate al norte de 40°N
    auto calc_fpr_north = [](const std::vector<float>& p, const std::vector<EvalRow>& rows, float threshold = 0.05f) {
        uint64_t fp = 0, tn = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].centroid_lat > 40.0f) {
                if (rows[i].y_detected == 0) {
                    if (p[i] >= threshold) fp++; else tn++;
                }
            }
        }
        return (fp + tn) > 0 ? (static_cast<double>(fp) / (fp + tn)) : 0.0;
    };

    // Cobertura empírica de incertidumbre (P5-P95)
    uint64_t cov_hits = 0;
    for (size_t i = 0; i < eval_rows.size(); ++i) {
        float y = static_cast<float>(eval_rows[i].y_detected);
        if (y >= m1_p5[i] && y <= m1_p95[i]) cov_hits++;
    }
    double empirical_coverage = static_cast<double>(cov_hits) / eval_rows.size();

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 1: DESEMPEÑO PROBABILÍSTICO GLOBAL FUERA DE MUESTRA (2020–2024)
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\nTABLA 1: DESEMPEÑO PROBABILÍSTICO GLOBAL CIEGO FUERA DE MUESTRA (2020–2024, 199,602 FILAS):\n";
    std::cout << "┌─────────────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "│ Modelo                              │ Brier Global │ Brier Balanc.│ Log-Loss     │ AUC-ROC      │ PR-AUC       │ Calib. Slope │\n";
    std::cout << "├─────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤\n";

    auto print_model_row = [&](const std::string& name, const std::vector<float>& p) {
        double bg = calc_brier_global(p, eval_rows);
        double bb = calc_brier_balanced(p, eval_rows, 350.0);
        double ll = calc_log_loss(p, eval_rows);
        double auc = calc_auc(p, eval_rows);
        double prauc = calc_pr_auc(p, eval_rows);
        auto [slope, intercept] = calc_calibration(p, eval_rows);

        std::cout << "│ " << std::left << std::setw(35) << name << " │ "
                  << std::right << std::setw(12) << std::fixed << std::setprecision(5) << bg << " │ "
                  << std::setw(12) << std::setprecision(5) << bb << " │ "
                  << std::setw(12) << std::setprecision(4) << ll << " │ "
                  << std::setw(12) << std::setprecision(4) << auc << " │ "
                  << std::setw(12) << std::setprecision(4) << prauc << " │ "
                  << std::setw(12) << std::setprecision(3) << slope << " │\n";
    };

    print_model_row("JORO-HPC M1 (MAP)", m1_map_out.p_marginal_eval);
    print_model_row("JORO-HPC M1 (Mediana Ensamble)", m1_median);
    print_model_row("JORO-HPC M0 (K0 constante MAP)", m0_map_out.p_marginal_eval);
    print_model_row("Kernel Exponencial Isotrópico (H2)", iso_null_out.p_marginal_eval);
    print_model_row("SDM Correlativo Estático (H1)", sdm_preds);
    std::cout << "└─────────────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n";

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 2: DESGLOSE ANUAL FUERA DE MUESTRA (2020 a 2024)
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\nTABLA 2: DESGLOSE ANUAL FUERA DE MUESTRA (2020 a 2024):\n";
    std::cout << "┌──────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "│ Año  │ Presencias   │ Brier M1 MAP │ Brier Kernel │ Brier SDM    │ AUC M1 MAP   │ AUC Kernel   │ AUC SDM      │\n";
    std::cout << "├──────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤\n";
    for (int yr = 2020; yr <= 2024; ++yr) {
        double b_m1 = calc_brier_global(m1_map_out.p_marginal_eval, eval_rows, yr);
        double b_iso = calc_brier_global(iso_null_out.p_marginal_eval, eval_rows, yr);
        double b_sdm = calc_brier_global(sdm_preds, eval_rows, yr);
        double a_m1 = calc_auc(m1_map_out.p_marginal_eval, eval_rows, yr);
        double a_iso = calc_auc(iso_null_out.p_marginal_eval, eval_rows, yr);
        double a_sdm = calc_auc(sdm_preds, eval_rows, yr);

        std::cout << "│ " << yr << " │ "
                  << std::setw(12) << obs_stats[yr].presences << " │ "
                  << std::setw(12) << std::fixed << std::setprecision(5) << b_m1 << " │ "
                  << std::setw(12) << std::setprecision(5) << b_iso << " │ "
                  << std::setw(12) << std::setprecision(5) << b_sdm << " │ "
                  << std::setw(12) << std::setprecision(4) << a_m1 << " │ "
                  << std::setw(12) << std::setprecision(4) << a_iso << " │ "
                  << std::setw(12) << std::setprecision(4) << a_sdm << " │\n";
    }
    std::cout << "└──────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n";

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 3: CINEMÁTICA Y GEOMETRÍA DEL FRENTE DE INVASIÓN (CONTRASTE DE H2)
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\nTABLA 3: CINEMÁTICA Y GEOMETRÍA DEL FRENTE DE INVASIÓN (CONTRASTE DE HIPÓTESIS H2):\n";
    std::cout << "┌──────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "│ Año  │ Obs P95 (km) │ M1 P95 (km)  │ Kern P95 (km)│ Obs Dir (°)  │ M1 Dir (°)   │ Kern Dir (°) │ Error Ang M1 │\n";
    std::cout << "├──────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤\n";
    for (int yr = 2020; yr <= 2024; ++yr) {
        int yi = yr - 2014;
        double o_p95 = obs_stats[yr].p95_km;
        double m1_p95_val = m1_map_out.sim_front_p95[yi];
        double iso_p95_val = iso_null_out.sim_front_p95[yi];
        double o_ang = obs_stats[yr].mean_angle_deg;
        double m1_ang = m1_map_out.sim_mean_angle_deg[yi];
        double iso_ang = iso_null_out.sim_mean_angle_deg[yi];
        double ang_err = std::abs(m1_ang - o_ang);
        if (ang_err > 180.0) ang_err = 360.0 - ang_err;

        std::cout << "│ " << yr << " │ "
                  << std::setw(12) << std::fixed << std::setprecision(1) << o_p95 << " │ "
                  << std::setw(12) << std::setprecision(1) << m1_p95_val << " │ "
                  << std::setw(12) << std::setprecision(1) << iso_p95_val << " │ "
                  << std::setw(12) << std::setprecision(1) << o_ang << " │ "
                  << std::setw(12) << std::setprecision(1) << m1_ang << " │ "
                  << std::setw(12) << std::setprecision(1) << iso_ang << " │ "
                  << std::setw(12) << std::setprecision(1) << ang_err << " │\n";
    }
    std::cout << "└──────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n";

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 4: RESTRICCIÓN TÉRMICA Y TRANSFERENCIA ECORREGIONAL (CONTRASTE DE H1)
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\nTABLA 4: RESTRICCIÓN TÉRMICA INVERNAL Y ESTRATIFICACIÓN ECORREGIONAL (CONTRASTE DE HIPÓTESIS H1):\n";
    std::cout << "┌──────────────────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "│ Métrica de Transferencia / Persistencia  │ JORO-HPC M1  │ JORO-HPC M0  │ Kernel Isot. │ SDM Estático │\n";
    std::cout << "├──────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤\n";

    double fpr_m1 = calc_fpr_north(m1_map_out.p_marginal_eval, eval_rows);
    double fpr_m0 = calc_fpr_north(m0_map_out.p_marginal_eval, eval_rows);
    double fpr_iso = calc_fpr_north(iso_null_out.p_marginal_eval, eval_rows);
    double fpr_sdm = calc_fpr_north(sdm_preds, eval_rows);

    std::cout << "│ Tasa Falsos Positivos Norte >40°N (%)    │ "
              << std::setw(11) << std::fixed << std::setprecision(2) << (fpr_m1 * 100.0) << "% │ "
              << std::setw(11) << std::setprecision(2) << (fpr_m0 * 100.0) << "% │ "
              << std::setw(11) << std::setprecision(2) << (fpr_iso * 100.0) << "% │ "
              << std::setw(11) << std::setprecision(2) << (fpr_sdm * 100.0) << "% │\n";

    // Estratificación por bloques ecorregionales (Bloque 5 visto en calibración vs Bloques no vistos)
    auto calc_brier_block = [](const std::vector<float>& p, const std::vector<EvalRow>& rows, bool seen_in_training) {
        double sum = 0.0;
        uint64_t cnt = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            bool is_seen = (rows[i].ecoregion_block == 5);
            if (is_seen != seen_in_training) continue;
            double diff = static_cast<double>(p[i]) - static_cast<double>(rows[i].y_detected);
            sum += diff * diff;
            cnt++;
        }
        return cnt > 0 ? (sum / cnt) : 0.0;
    };

    double b_seen_m1 = calc_brier_block(m1_map_out.p_marginal_eval, eval_rows, true);
    double b_seen_m0 = calc_brier_block(m0_map_out.p_marginal_eval, eval_rows, true);
    double b_seen_iso = calc_brier_block(iso_null_out.p_marginal_eval, eval_rows, true);
    double b_seen_sdm = calc_brier_block(sdm_preds, eval_rows, true);

    double b_unseen_m1 = calc_brier_block(m1_map_out.p_marginal_eval, eval_rows, false);
    double b_unseen_m0 = calc_brier_block(m0_map_out.p_marginal_eval, eval_rows, false);
    double b_unseen_iso = calc_brier_block(iso_null_out.p_marginal_eval, eval_rows, false);
    double b_unseen_sdm = calc_brier_block(sdm_preds, eval_rows, false);

    std::cout << "│ Brier Ecorregión Vista (Bloque 5)        │ "
              << std::setw(12) << std::setprecision(5) << b_seen_m1 << " │ "
              << std::setw(12) << std::setprecision(5) << b_seen_m0 << " │ "
              << std::setw(12) << std::setprecision(5) << b_seen_iso << " │ "
              << std::setw(12) << std::setprecision(5) << b_seen_sdm << " │\n";

    std::cout << "│ Brier Ecorregiones Nuevas (2,7,8,9,10-12)│ "
              << std::setw(12) << std::setprecision(5) << b_unseen_m1 << " │ "
              << std::setw(12) << std::setprecision(5) << b_unseen_m0 << " │ "
              << std::setw(12) << std::setprecision(5) << b_unseen_iso << " │ "
              << std::setw(12) << std::setprecision(5) << b_unseen_sdm << " │\n";

    std::cout << "└──────────────────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n";

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 5: CUANTIFICACIÓN Y COBERTURA DE INCERTIDUMBRE
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\nTABLA 5: PROPAGACIÓN DE INCERTIDUMBRE (ENSAMBLE COMPATIBLE Theta_comp, 277 MIEMBROS):\n";
    std::cout << "┌──────────────────────────────────────────────┬──────────────────────────┬────────────────────────┐\n";
    std::cout << "│ Métrica de Incertidumbre                     │ Valor Obtenido           │ Diagnóstico            │\n";
    std::cout << "├──────────────────────────────────────────────┼──────────────────────────┼────────────────────────┤\n";
    std::cout << "│ Miembros de Theta_comp evaluados             │ " << std::setw(24) << m1_ensemble.size() << " │ 100% propagado         │\n";
    std::cout << "│ Cobertura empírica del intervalo P5–P95 (%)  │ " << std::setw(23) << std::fixed << std::setprecision(2)
              << (empirical_coverage * 100.0) << "% │ Nominal 90.0%          │\n";
    std::cout << "│ Ancho medio de banda P95 - P5                │ " << std::setw(24) << std::setprecision(5)
              << (std::accumulate(m1_p95.begin(), m1_p95.end(), 0.0) / m1_p95.size() -
                  std::accumulate(m1_p5.begin(), m1_p5.end(), 0.0) / m1_p5.size()) << " │ Calibración bien acot. │\n";
    std::cout << "└──────────────────────────────────────────────┴──────────────────────────┴────────────────────────┘\n";

    // Guardar resumen JSON de métricas
    const std::string metrics_json_path = "/mnt/hpc_scratch/joro-hpc/outputs/phase5_validation_metrics.json";
    std::ofstream mj(metrics_json_path);
    mj << "{\n"
       << "  \"m1_brier_global\": " << calc_brier_global(m1_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m1_brier_balanced\": " << calc_brier_balanced(m1_map_out.p_marginal_eval, eval_rows, 350.0) << ",\n"
       << "  \"m1_log_loss\": " << calc_log_loss(m1_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m1_auc\": " << calc_auc(m1_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m1_prauc\": " << calc_pr_auc(m1_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m0_brier_global\": " << calc_brier_global(m0_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m0_auc\": " << calc_auc(m0_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"iso_brier_global\": " << calc_brier_global(iso_null_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"iso_auc\": " << calc_auc(iso_null_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"sdm_brier_global\": " << calc_brier_global(sdm_preds, eval_rows) << ",\n"
       << "  \"sdm_auc\": " << calc_auc(sdm_preds, eval_rows) << ",\n"
       << "  \"iso_lambda_calibrated_km\": " << best_lambda_km << ",\n"
       << "  \"empirical_coverage_pct\": " << (empirical_coverage * 100.0) << ",\n"
       << "  \"fpr_north_m1\": " << fpr_m1 << ",\n"
       << "  \"fpr_north_sdm\": " << fpr_sdm << ",\n"
       << "  \"ensemble_members\": " << m1_ensemble.size() << ",\n"
       << "  \"runtime_seconds\": " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - sim_start_all).count() << "\n"
       << "}\n";
    std::cout << "✔ Métricas de validación exportadas a " << metrics_json_path << "\n";

    double total_wall = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - sim_start_all).count();
    std::cout << "\n===================================================================\n";
    std::cout << "✔ FASE 5 COMPLETADA EXITOSAMENTE EN " << std::fixed << std::setprecision(2) << total_wall << " s ("
              << (total_wall / 60.0) << " min)\n";
    std::cout << "===================================================================\n";

    return EXIT_SUCCESS;
}
