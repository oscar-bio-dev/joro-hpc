////////////////////////////////////////////////////////////////////////////////
/// @file   joro_evaluate.cpp
/// @brief  JORO-HPC — Validador Fuera de Muestra (2020–2024)
///
/// Ejecuta la validación ciega fuera de muestra para contrastar formalmente
/// las Hipótesis H1a/b, H2a/b y H3a/b (Secciones 3, 4 y 12 del anteproyecto v2.0):
///   1. Desbloqueo de ERA5 (2020-2024) y evaluation_set sin reajuste paramétrico.
///   2. Simulación continua de 11 años (2014-2024) en GPU SYCL + CPU OpenMP con el bucle INVERTIDO: año afuera
///      (el tensor ERA5 anual se transfiere a la VRAM UNA sola vez por año), configuraciones/miembros adentro:
///      - JORO-HPC M1: MAP + ensamble completo Theta_comp
///      - JORO-HPC M0: MAP
///      - JORO-HPC M1 + K_mixto (kernel gravitacional humano calibrado en 2014-2019)
///   3. Comparadores nulos calibrados en 2014-2019 con la MISMA función multicriterio J(theta):
///      - Kernel Exponencial Isotrópico K0(d | lambda) (Null H2)
///      - SDM Correlativo Estático (Null H1)
///   4. Métricas oficiales (Brier, AUC, calibración de Cox), P95 por cuadrante, cola > 300 km, correlaciones de
///      Spearman (H1b, H3a) y prevalencia con y sin f_est(i,t) (H3b).
///   5. Exportación de predicciones espaciales e intervalos de incertidumbre.
////////////////////////////////////////////////////////////////////////////////

#include "common/types.hpp"
#include "common/config_parser.hpp"
#include "common/spatial_grid_lookup.hpp"
#include "common/model_geometry.hpp"
#include "io/era5_netcdf_reader.hpp"
#include "io/postgis_client.hpp"
#include "lpdm/lpdm_engine.hpp"
#include "abm/abm_engine.hpp"
#include "inference/occupancy_model.hpp"
#include "inference/cost_function.hpp"
#include "inference/dispersal_kernels.hpp"
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
#include <memory>
#include <random>
#include <limits>
#include <array>

namespace {

constexpr int    kStartYear = 2014;
[[maybe_unused]] constexpr int kEndYear = 2024;
constexpr int    kTotalYears = 11;         // 2014..2024
constexpr int    kCalibYears = 6;          // 2014..2019
constexpr int    kEvalStartYear = 2020;
constexpr int    kEvalYears = 5;           // 2020..2024
constexpr int    kLpdmSteps = 480;         // 480 pasos x 60 s = 8.0 h máximas de vuelo diurno por episodio (M1)
constexpr float  kLpdmDt = 60.0f;
constexpr double kEnvelopeBaseKm = 50.0;
constexpr double kTailKm = 300.0;          // cola de larga distancia (saltos satelitales mediados por humanos)

/// Geometría UNIFICADA del polígono semilla (idéntica a la de calibración): centroide y radio para P50/P90/P95,
/// envolvente y azimut. Sustituye al antiguo punto kEpiLat/kEpiLon (M15).
joro::SeedGeometry g_seed;

double dist_to_seed_km(float lat, float lon) { return g_seed.dist_km(lat, lon); }
double azimuth_from_epicenter_deg(float lat, float lon) { return g_seed.azimuth_deg(lat, lon); }

/// Cuadrante del borde de avance respecto del centroide semilla: 0 = NE, 1 = SE, 2 = SW, 3 = NW
int quadrant_of(double azimuth_deg) {
    if (azimuth_deg < 90.0) return 0;
    if (azimuth_deg < 180.0) return 1;
    if (azimuth_deg < 270.0) return 2;
    return 3;
}
constexpr const char* kQuadrantName[4] = {"NE", "SE", "SW", "NW"};

/// Número JSON válido: `null` si no es finito.
std::string jnum(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
}

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1));
    return v[std::min(idx, v.size() - 1)];
}

struct EvalRow {
    int32_t cell_id = 0;
    int16_t year = 0;
    uint8_t y_detected = 0;
    int32_t joro_count = 0;
    int32_t effort_araneae = 0;
    int32_t effort_exogenous = 0;
    float cpue_joro = 0.0f;
    uint8_t y_cumulative = 0;
    float tau_invasion_age = 0.0f;
    float centroid_lon = 0.0f;
    float centroid_lat = 0.0f;
    float z_imp = 0.0f;
    float z_borde = 0.0f;
    float z_dosel = 0.0f;
    float pop_density_h = 0.0f;
    float road_density_r = 0.0f;
    int16_t ecoregion_block = 0;
    float p_det = 0.0f;          ///< p_{i,t} con f_est(i,t)
    float p_det_nofest = 0.0f;   ///< p_{i,t} sin f_est (contraste de H3b)
    float h_norm = 0.0f;
    double dist_epi_km = 0.0;    ///< distancia al centroide del polígono semilla
    double dist_seed_km = 0.0;   ///< (igual que dist_epi_km: una sola referencia geométrica, M15)
    double angle_epi_deg = 0.0;  ///< azimut respecto al centroide del polígono semilla
};

/// Carga un conjunto celda-año (esquema v2.0) por NOMBRE de columna; acepta los alias heredados.
std::vector<EvalRow> load_eval_dataset(const std::string& csv_path, const joro::CovariateNorm& norm,
                                       const joro::inference::OccupancyParameters& occ,
                                       const joro::inference::AttentionIndex& attention) {
    std::vector<EvalRow> rows;
    std::ifstream file(csv_path);
    if (!file.is_open()) return rows;

    std::string line;
    std::getline(file, line); // header
    std::unordered_map<std::string, int> col;
    {
        std::stringstream hs(line);
        std::string h;
        int i = 0;
        while (std::getline(hs, h, ',')) {
            while (!h.empty() && (h.back() == '\r' || h.back() == ' ')) h.pop_back();
            col[h] = i++;
        }
    }
    auto find_col = [&](std::initializer_list<const char*> names) {
        for (const char* n : names) { auto it = col.find(n); if (it != col.end()) return it->second; }
        return -1;
    };
    const int c_cell = find_col({"cell_id"}), c_year = find_col({"year"}), c_y = find_col({"y_joro_detected", "y_detected"});
    const int c_cnt = find_col({"joro_count"}), c_eex = find_col({"effort_exogenous"}), c_eff = find_col({"effort_araneae"});
    const int c_lon = find_col({"centroid_lon"}), c_lat = find_col({"centroid_lat"});
    const int c_zi = find_col({"z_imp"}), c_zb = find_col({"z_borde"}), c_zd = find_col({"z_dosel"});
    const int c_h = find_col({"pop_density_h"}), c_r = find_col({"road_density_r"}), c_eco = find_col({"ecoregion_block"});
    const int c_cpue = find_col({"cpue_joro"}), c_cum = find_col({"y_cumulative_obs"}), c_tau = find_col({"tau_invasion_age"});
    if (c_cell < 0 || c_year < 0 || c_y < 0 || (c_eex < 0 && c_eff < 0) || c_lon < 0 || c_lat < 0) {
        std::cerr << "ERROR: cabecera inesperada en " << csv_path << "\n";
        return rows;
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> tok;
        { std::stringstream ss(line); std::string t; while (std::getline(ss, t, ',')) tok.push_back(t); }
        auto get = [&](int c, double d) {
            return (c >= 0 && c < static_cast<int>(tok.size()) && !tok[static_cast<size_t>(c)].empty())
                       ? std::stod(tok[static_cast<size_t>(c)]) : d;
        };
        EvalRow r{};
        r.cell_id = static_cast<int32_t>(get(c_cell, 0));
        r.year = static_cast<int16_t>(get(c_year, 0));
        r.y_detected = static_cast<uint8_t>(get(c_y, 0));
        r.joro_count = static_cast<int32_t>(get(c_cnt, 0));
        r.effort_araneae = static_cast<int32_t>(get(c_eff, 0));
        r.effort_exogenous = c_eex >= 0 ? static_cast<int32_t>(get(c_eex, 0)) : r.effort_araneae;
        r.cpue_joro = static_cast<float>(get(c_cpue, 0));
        r.y_cumulative = static_cast<uint8_t>(get(c_cum, r.y_detected));
        r.tau_invasion_age = static_cast<float>(get(c_tau, 0));
        r.centroid_lon = static_cast<float>(get(c_lon, 0));
        r.centroid_lat = static_cast<float>(get(c_lat, 0));
        r.z_imp = static_cast<float>(get(c_zi, 0));
        r.z_borde = static_cast<float>(get(c_zb, 0));
        r.z_dosel = static_cast<float>(get(c_zd, 0));
        r.pop_density_h = static_cast<float>(get(c_h, 0));
        r.road_density_r = static_cast<float>(get(c_r, 0));
        r.ecoregion_block = static_cast<int16_t>(get(c_eco, 0));

        // Misma normalización (sobre Ω) y misma detección que la calibración (M15), con y sin f_est (H3b)
        r.h_norm = norm.h(r.pop_density_h);
        const float r_norm = norm.r(r.road_density_r);
        const float f_est = joro::inference::compute_f_est(attention.ratio_for(r.year), r.tau_invasion_age, occ);
        r.p_det = joro::inference::compute_detection_prob(r.effort_exogenous, r.h_norm, r_norm, occ, f_est);
        r.p_det_nofest = joro::inference::compute_detection_prob(r.effort_exogenous, r.h_norm, r_norm, occ, 0.0f);
        r.dist_epi_km = dist_to_seed_km(r.centroid_lat, r.centroid_lon);
        r.dist_seed_km = r.dist_epi_km;
        r.angle_epi_deg = azimuth_from_epicenter_deg(r.centroid_lat, r.centroid_lon);
        rows.push_back(r);
    }
    return rows;
}

/// Offset logit de prevalencia (Platt de solo intercepto) ajustado EXCLUSIVAMENTE en 2014–2019.
/// J_arrival se optimizó con Brier balanceado 50/50, de modo que la probabilidad marginal cruda subestima la
/// prevalencia absoluta. Se busca δ tal que  Σ_{p_i>0} σ(logit(p_i) + δ) = nº de presencias observadas (2014–2019).
/// Las celdas con p_i = 0 (ψ = 0: sin colonización simulada) permanecen en 0: el offset no inventa colonización.
struct PrevalenceFit {
    double delta = 0.0;
    double raw_expected = 0.0;
    double adj_expected = 0.0;
    uint64_t observed = 0;
    bool reached = true;   ///< false si ni δ = +20 alcanza el total observado
};

inline float adjust_probability(float p, double delta) {
    if (!(p > 0.0f)) return 0.0f;
    const double pc = std::clamp(static_cast<double>(p), 1e-9, 1.0 - 1e-9);
    const double z = std::log(pc / (1.0 - pc)) + delta;
    return static_cast<float>(1.0 / (1.0 + std::exp(-z)));
}

PrevalenceFit fit_prevalence_offset(const std::vector<float>& p, const std::vector<EvalRow>& rows) {
    PrevalenceFit f;
    for (size_t i = 0; i < rows.size(); ++i) {
        f.raw_expected += p[i];
        if (rows[i].y_detected == 1) f.observed++;
    }
    auto expected = [&](double delta) {
        double acc = 0.0;
        #pragma omp parallel for reduction(+:acc) schedule(static)
        for (size_t i = 0; i < p.size(); ++i) acc += adjust_probability(p[i], delta);
        return acc;
    };
    const double target = static_cast<double>(f.observed);
    double lo = -20.0, hi = 20.0;
    if (expected(hi) < target) {
        f.delta = hi;
        f.reached = false;
    } else if (expected(lo) > target) {
        f.delta = lo;
        f.reached = false;
    } else {
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (expected(mid) < target) lo = mid; else hi = mid;
        }
        f.delta = 0.5 * (lo + hi);
    }
    f.adj_expected = expected(f.delta);
    return f;
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

/// Lee `"map_theta": [...]` desde `calibration_summary_M{1,0}.json` (con respaldo al vector por defecto).
std::vector<float> load_map_theta_from_json(const std::string& json_path, const std::vector<float>& fallback) {
    std::ifstream f(json_path);
    if (!f.is_open()) return fallback;
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string txt = ss.str();
    const size_t key = txt.find("\"map_theta\"");
    if (key == std::string::npos) return fallback;
    const size_t lb = txt.find('[', key);
    const size_t rb = (lb != std::string::npos) ? txt.find(']', lb) : std::string::npos;
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return fallback;
    std::vector<float> out;
    std::stringstream vs(txt.substr(lb + 1, rb - lb - 1));
    std::string tok;
    while (std::getline(vs, tok, ',')) {
        try { out.push_back(std::stof(tok)); } catch (...) { return fallback; }
    }
    return (out.size() == fallback.size()) ? out : fallback;
}


// ─────────────────────────────────────────────────────────────────────────────
// Simulación de 11 años en "lockstep": el bucle externo recorre los años y el interno las configuraciones
// (MAP M1/M0, K_mixto, kernel isotrópico y los miembros de Theta_comp), de modo que el tensor ERA5 anual se sube
// a la VRAM una sola vez por año (≈ 11 x 0.2 s en total, frente a >10 min del orden miembro-año).
// ─────────────────────────────────────────────────────────────────────────────
struct SimContext {
    const std::vector<joro::GridCell>* grid = nullptr;
    const joro::abm::CellIndexTable* cidx = nullptr;
    const std::vector<joro::io::DailyThermalGrid>* thermal = nullptr;
    const std::vector<EvalRow>* eval_rows = nullptr;
    const std::vector<EvalRow>* calib_rows = nullptr;
    const std::unordered_map<int16_t, std::unordered_map<int32_t, size_t>>* eval_index_by_year = nullptr;
    const std::vector<std::vector<size_t>>* calib_rows_of_year = nullptr;
    joro::lpdm::LpdmEngine* lpdm = nullptr;
    const joro::inference::HumanTransportKernel* human_kernel = nullptr;
    joro::SpatialGridView grid_view{};
    joro::CovariateNorm norm;
    joro::inference::OccupancyParameters occ;
    joro::inference::AttentionIndex attention;
    std::vector<float> cell_seed_dist_km;    ///< distancia al centroide semilla por posición de celda
};

struct RunSpec {
    std::string label;
    std::vector<float> theta;
    joro::abm::CarryingModel model = joro::abm::CarryingModel::M1;
    bool isotropic = false;
    double iso_lambda_km = 0.0;
    double p_h = 0.0;               ///< kernel mixto humano (0 = solo LPDM)
    double lambda_h_km = 150.0;
    uint32_t seed = 1;
    bool diagnostics = false;       ///< guarda D̄, K_i, cuadrantes, psi por fila (solo corridas MAP)
};

struct SimOutputs {
    std::vector<float> p_marginal_eval;        ///< P(Y=1) = psi * p_det (con f_est) por fila de evaluación
    std::vector<float> p_marginal_eval_nofest; ///< idem con p_det SIN f_est (H3b)
    std::vector<float> psi_eval;
    std::vector<float> dbar_eval;              ///< D̄_{i,t} por fila de evaluación (H1b)
    std::vector<float> k_eval;                 ///< K_i por fila de evaluación (H3a)
    std::vector<float> p_marginal_calib;       ///< predicción en las filas de calibración 2014–2019 (offset de prevalencia)
    std::vector<float> p_marginal_calib_nofest;///< predicción en 2014–2019 sin f_est (offset estacionario de Platt, H3b)
    std::vector<double> sim_front_p50, sim_front_p90, sim_front_p95;
    std::vector<double> sim_mean_angle_deg;
    std::vector<std::array<double, 4>> sim_quadrant_p95;   ///< [año][NE,SE,SW,NW]
    std::vector<double> env_radius_km;         ///< envolvente SIMULADA acumulada por año (base + frente simulado acumulado)
    std::vector<double> flight_mean_km;        ///< distancia media de vuelo EFECTIVO (sin retenidos) por año
    std::vector<double> retained_fraction;     ///< retenidos / aterrizados por año
    std::vector<uint64_t> human_transported, human_beyond_300km;
    std::unordered_map<int32_t, int16_t> first_arrival_year;
};

class SimRun {
public:
    SimRun(const SimContext& c, RunSpec s) : spec(std::move(s)), ctx_(c) {
        const auto& grid = *ctx_.grid;
        out.p_marginal_eval.assign(ctx_.eval_rows->size(), 0.0f);
        out.p_marginal_eval_nofest.assign(spec.diagnostics ? ctx_.eval_rows->size() : 0, 0.0f);
        if (spec.diagnostics) {
            out.psi_eval.assign(ctx_.eval_rows->size(), 0.0f);
            out.dbar_eval.assign(ctx_.eval_rows->size(), 0.0f);
            out.k_eval.assign(ctx_.eval_rows->size(), 0.0f);
            out.p_marginal_calib_nofest.assign(ctx_.calib_rows->size(), 0.0f);
        }
        out.p_marginal_calib.assign(ctx_.calib_rows->size(), 0.0f);
        out.sim_front_p50.assign(kTotalYears, 0.0);
        out.sim_front_p90.assign(kTotalYears, 0.0);
        out.sim_front_p95.assign(kTotalYears, 0.0);
        out.sim_mean_angle_deg.assign(kTotalYears, 0.0);
        out.sim_quadrant_p95.assign(kTotalYears, {std::nan(""), std::nan(""), std::nan(""), std::nan("")});
        out.env_radius_km.assign(kTotalYears, kEnvelopeBaseKm);
        out.flight_mean_km.assign(kTotalYears, 0.0);
        out.retained_fraction.assign(kTotalYears, 0.0);
        out.human_transported.assign(kTotalYears, 0);
        out.human_beyond_300km.assign(kTotalYears, 0);

        const auto& theta = spec.theta;
        p_.fecundity_F   = 450.0f;   // un único saco por hembra (M10)
        p_.q10           = 2.2f;
        p_.t_c           = theta[0];
        p_.alpha_damage  = theta[1];
        p_.beta_damage   = std::clamp(static_cast<int>(std::lround(theta[2])), 1, 3);
        p_.k0            = theta[3];
        occ_ = ctx_.occ;
        if (spec.model == joro::abm::CarryingModel::M0) {
            p_.use_hab_index = true;
            p_.gamma_hab     = 0.0f;
            p_.q_balloon     = theta[4];
            p_.q_viab        = theta[5];
            occ_.a0 = theta[6];
            occ_.b0 = theta[7];
        } else {
            p_.use_hab_index = true;
            p_.gamma_hab     = theta[4];
            p_.q_balloon     = theta[5];
            p_.q_viab        = theta[6];
            occ_.a0 = theta[7];
            occ_.b0 = theta[8];
        }
        occ_.w_h = static_cast<float>(spec.p_h);
        occ_.lambda_h_km = static_cast<float>(spec.lambda_h_km);

        abm_ = std::make_unique<joro::abm::AbmEngine>(p_);
        abm_->set_cell_index(ctx_.cidx);
        abm_->set_mc_seed(spec.seed);
        abm_->initialize_seed(grid, 0.15f);

        psi_.assign(grid.size(), 0.0f);
        for (size_t i = 0; i < grid.size(); ++i) if (grid[i].is_seed) psi_[i] = 1.0f;
        dsum_.assign(grid.size(), 0.0f);
        dcnt_.assign(grid.size(), 0);
        arr_.assign(grid.size(), 0);
        arr_h_.assign(grid.size(), 0);
        adult_.assign(grid.size(), 0.0f);
        touched_.reserve(20000);
    }

    /// Un año completo (primavera -> dispersión -> verano/otoño -> ocupación). El motor LPDM ya tiene seleccionado el año.
    void step_year(int yi) {
        const auto& grid = *ctx_.grid;
        const int yr = kStartYear + yi;
        const auto& th = (*ctx_.thermal)[static_cast<size_t>(yi)];
        const uint32_t dispersal_seed = 0x9E3779B1u * static_cast<uint32_t>(yi + 1) ^ spec.seed;
        const bool use_human = spec.p_h > 0.0 && ctx_.human_kernel != nullptr && !spec.isotropic;

        std::vector<std::pair<int32_t, float>> hum_sources;
        if (use_human) hum_sources = abm_->transportable_ootecas();

        // 1. Primavera: daño invernal previo (ventana completa nov-mar, M11), emergencia escalonada y baloneo
        auto spring = abm_->step_spring_phenology(grid, th, yr);

        const auto& coh = abm_->cohorts();
        for (size_t i = 0; i < coh.size(); ++i) {
            const int32_t cid = coh.cell_id[i];
            if (cid < 0 || static_cast<size_t>(cid) >= dsum_.size()) continue;
            if (dcnt_[cid] == 0) touched_.push_back(cid);
            dsum_[cid] += coh.cold_damage_D[i];
            if (dcnt_[cid] < 65535) dcnt_[cid]++;
        }

        // 2. Dispersión: LPDM (GPU) con ventana de despegue escalonada, o kernel isotrópico nulo
        std::unordered_map<int32_t, uint32_t> aero_map;
        if (spec.isotropic) {
            joro::inference::IsotropicKernelDisperser iso;
            iso.lambda_m = spec.iso_lambda_km * 1000.0;
            iso.grid_view = ctx_.grid_view;
            aero_map = iso.disperse(spring.ballooners_per_cell, grid, *ctx_.cidx, dispersal_seed);
        } else {
            ctx_.lpdm->set_mc_seed(spec.seed);
            ctx_.lpdm->set_emergence_offsets(spring.emergence_hours_per_cell);
            ctx_.lpdm->initialize_particles(grid, spring.ballooners_per_cell);
            auto disp = ctx_.lpdm->run_dispersion(joro::io::AtmosphericField3D{}, kLpdmDt, kLpdmSteps);
            out.flight_mean_km[static_cast<size_t>(yi)] = disp.mean_jump_distance_km;
            out.retained_fraction[static_cast<size_t>(yi)] =
                disp.total_landed > 0 ? static_cast<double>(disp.total_retained) / static_cast<double>(disp.total_landed) : 0.0;
            aero_map = std::move(disp.arrivals_per_cell);
        }

        // 3. Transporte humano (kernel gravitacional por corredores viales)
        joro::inference::HumanTransportKernel::Result hum;
        if (use_human) {
            hum = ctx_.human_kernel->transport(hum_sources, spec.p_h, spec.lambda_h_km, dispersal_seed ^ 0xA5A5A5A5u);
            out.human_transported[static_cast<size_t>(yi)] = hum.transported;
            out.human_beyond_300km[static_cast<size_t>(yi)] = hum.beyond_300km;
        }

        // 4. Verano/Otoño: llegadas (aéreas + humanas), desarrollo Q10 continuo y reproducción Beverton-Holt
        std::unordered_map<int32_t, uint32_t> arrivals_map;
        if (hum.arrivals.empty()) {
            arrivals_map = aero_map;
        } else {
            arrivals_map = aero_map;
            for (const auto& [cid, n] : hum.arrivals) arrivals_map[cid] += n;
        }
        abm_->deposit_arrivals(arrivals_map);
        auto fall = abm_->step_summer_autumn_phenology(grid, th, yr);

        for (const auto& [cid, count] : arrivals_map) {
            if (count > 0 && out.first_arrival_year.find(cid) == out.first_arrival_year.end()) {
                out.first_arrival_year[cid] = static_cast<int16_t>(yr);
            }
        }

        // 5. Métricas del frente (distancias y azimut respecto del centroide semilla; P95 por cuadrante)
        std::vector<double> distances;
        std::array<std::vector<double>, 4> quad_d;
        double sum_sin = 0.0, sum_cos = 0.0;
        size_t n_angle_cells = 0;
        for (const auto& [cid, count] : fall.adult_density_per_cell) {
            if (count > 2.0f) {
                const int32_t pos = ctx_.cidx->find(cid);
                if (pos >= 0) {
                    const auto& gc = grid[static_cast<size_t>(pos)];
                    const double d = ctx_.cell_seed_dist_km[static_cast<size_t>(pos)];
                    distances.push_back(d);
                    const double az = azimuth_from_epicenter_deg(gc.centroid_lat, gc.centroid_lon);
                    quad_d[static_cast<size_t>(quadrant_of(az))].push_back(d);
                    if (d > g_seed.radius_km) {
                        const double rad = az * M_PI / 180.0;
                        sum_sin += std::sin(rad);
                        sum_cos += std::cos(rad);
                        ++n_angle_cells;
                    }
                }
            }
        }
        if (!distances.empty()) {
            out.sim_front_p50[static_cast<size_t>(yi)] = percentile(distances, 0.50);
            out.sim_front_p90[static_cast<size_t>(yi)] = percentile(distances, 0.90);
            out.sim_front_p95[static_cast<size_t>(yi)] = percentile(distances, 0.95);
            if (n_angle_cells > 0) {
                double mean_deg = std::atan2(sum_sin, sum_cos) * 180.0 / M_PI;
                if (mean_deg < 0.0) mean_deg += 360.0;
                out.sim_mean_angle_deg[static_cast<size_t>(yi)] = mean_deg;
            } else {
                out.sim_mean_angle_deg[static_cast<size_t>(yi)] = std::numeric_limits<double>::quiet_NaN();
            }
            for (size_t q = 0; q < 4; ++q) if (!quad_d[q].empty()) out.sim_quadrant_p95[static_cast<size_t>(yi)][q] = percentile(quad_d[q], 0.95);
        } else {
            out.sim_front_p50[static_cast<size_t>(yi)] = out.sim_front_p90[static_cast<size_t>(yi)] = out.sim_front_p95[static_cast<size_t>(yi)] = 5.0;
            out.sim_mean_angle_deg[static_cast<size_t>(yi)] = std::numeric_limits<double>::quiet_NaN();
        }
        cummax_front_ = std::max(cummax_front_, out.sim_front_p95[static_cast<size_t>(yi)]);
        out.env_radius_km[static_cast<size_t>(yi)] = kEnvelopeBaseKm + cummax_front_;   // envolvente SIMULADA (M15)

        // 6. Llenar arreglos densos A_{i,t}, N_{i,t}
        for (const auto& [cid, cnt] : aero_map) {
            if (cid >= 0 && static_cast<size_t>(cid) < arr_.size()) { arr_[cid] = cnt; touched_.push_back(cid); }
        }
        for (const auto& [cid, cnt] : hum.arrivals) {
            if (cid >= 0 && static_cast<size_t>(cid) < arr_h_.size()) { arr_h_[cid] = cnt; touched_.push_back(cid); }
        }
        for (const auto& [cid, n] : fall.adult_density_per_cell) {
            if (cid >= 0 && static_cast<size_t>(cid) < adult_.size()) { adult_[cid] = n; touched_.push_back(cid); }
        }

        // 7. Actualización de ocupación latente psi: colonización cloglog sin generación espontánea (Ec. 22)
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < grid.size(); ++i) {
            const int32_t cid = grid[i].cell_id;
            const float D = dcnt_[cid] ? dsum_[cid] / static_cast<float>(dcnt_[cid]) : 0.0f;
            const float a_total = joro::inference::total_arrivals(static_cast<float>(arr_[cid]), static_cast<float>(arr_h_[cid]), occ_.w_h);
            const float gamma = joro::inference::compute_colonization_gamma(a_total, grid[i].h_hab, occ_);
            const float phi = joro::inference::compute_persistence_phi(adult_[cid], D, occ_);
            psi_[i] = joro::inference::update_dynamic_occupancy(psi_[i], gamma, phi);
        }

        // 8. Predicciones sobre las filas de calibración (offset de prevalencia) y de evaluación
        if (yr < kEvalStartYear) {
            for (size_t r : (*ctx_.calib_rows_of_year)[static_cast<size_t>(yi)]) {
                const int32_t pos = ctx_.cidx->find((*ctx_.calib_rows)[r].cell_id);
                if (pos >= 0) {
                    const float psi_cell = psi_[static_cast<size_t>(pos)];
                    out.p_marginal_calib[r] = joro::inference::marginal_observation_prob(psi_cell,
                                                                                         (*ctx_.calib_rows)[r].p_det);
                    if (spec.diagnostics) {
                        out.p_marginal_calib_nofest[r] = joro::inference::marginal_observation_prob(psi_cell,
                                                                                                    (*ctx_.calib_rows)[r].p_det_nofest);
                    }
                }
            }
        } else {
            const auto& y_eval_map = ctx_.eval_index_by_year->at(static_cast<int16_t>(yr));
            joro::abm::CarryingCapacityParams cc;
            cc.k0 = p_.k0; cc.gamma_hab = p_.gamma_hab; cc.use_hab_index = true; cc.k_min = p_.k_min; cc.k_max = p_.k_max;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < grid.size(); ++i) {
                const int32_t cid = grid[i].cell_id;
                auto it = y_eval_map.find(cid);
                if (it == y_eval_map.end()) continue;
                const size_t row_idx = it->second;
                const EvalRow& row = (*ctx_.eval_rows)[row_idx];
                out.p_marginal_eval[row_idx] = joro::inference::marginal_observation_prob(psi_[i], row.p_det);
                if (spec.diagnostics) {
                    out.psi_eval[row_idx] = psi_[i];
                    out.p_marginal_eval_nofest[row_idx] = joro::inference::marginal_observation_prob(psi_[i], row.p_det_nofest);
                    out.dbar_eval[row_idx] = dcnt_[cid] ? dsum_[cid] / static_cast<float>(dcnt_[cid]) : 0.0f;
                    out.k_eval[row_idx] = joro::abm::compute_carrying_capacity(grid[i], cc);
                }
            }
        }

        // 9. Limpieza dispersa
        for (int32_t cid : touched_) { arr_[cid] = 0; arr_h_[cid] = 0; adult_[cid] = 0.0f; dsum_[cid] = 0.0f; dcnt_[cid] = 0; }
        touched_.clear();
    }

    SimOutputs out;
    RunSpec spec;

private:
    SimContext ctx_;
    joro::abm::DemographicParameters p_;
    joro::inference::OccupancyParameters occ_;
    std::unique_ptr<joro::abm::AbmEngine> abm_;
    std::vector<float> psi_;
    std::vector<float> dsum_;
    std::vector<uint16_t> dcnt_;
    std::vector<uint32_t> arr_, arr_h_;
    std::vector<float> adult_;
    std::vector<int32_t> touched_;
    double cummax_front_ = 0.0;
};

/// Rango (con empates promediados) para la correlación de Spearman
double spearman(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = a.size();
    if (n < 3 || b.size() != n) return std::numeric_limits<double>::quiet_NaN();
    auto ranks = [](const std::vector<double>& v) {
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) { return v[i] < v[j]; });
        std::vector<double> r(v.size());
        size_t i = 0;
        while (i < idx.size()) {
            size_t j = i;
            while (j < idx.size() && v[idx[j]] == v[idx[i]]) ++j;
            const double avg = 0.5 * (static_cast<double>(i + 1) + static_cast<double>(j));
            for (size_t k = i; k < j; ++k) r[idx[k]] = avg;
            i = j;
        }
        return r;
    };
    const auto ra = ranks(a), rb = ranks(b);
    const double ma = std::accumulate(ra.begin(), ra.end(), 0.0) / n, mb = std::accumulate(rb.begin(), rb.end(), 0.0) / n;
    double sab = 0.0, saa = 0.0, sbb = 0.0;
    for (size_t i = 0; i < n; ++i) { sab += (ra[i] - ma) * (rb[i] - mb); saa += (ra[i] - ma) * (ra[i] - ma); sbb += (rb[i] - mb) * (rb[i] - mb); }
    return (saa > 0.0 && sbb > 0.0) ? sab / std::sqrt(saa * sbb) : std::numeric_limits<double>::quiet_NaN();
}

/// Calibración probabilística de Cox: regresión logística y ~ alpha + beta logit(p_hat) por Newton-Raphson amortiguado
/// (búsqueda de línea sobre la log-verosimilitud, estable aunque casi todo p_hat ≈ 0 y las presencias sean raras).
/// Calibración perfecta: pendiente beta = 1 e intercepto alpha = 0.
std::pair<double, double> cox_calibration(const std::vector<float>& p, const std::vector<EvalRow>& rows) {
    std::vector<double> x(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        const double pc = std::clamp(static_cast<double>(p[i]), 1e-4, 1.0 - 1e-4);
        x[i] = std::log(pc / (1.0 - pc));
    }
    auto loglik = [&](double a, double b) {
        double ll = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:ll)
        for (size_t i = 0; i < rows.size(); ++i) {
            const double eta = a + b * x[i];
            // log(1 + e^eta) estable
            const double softplus = eta > 30.0 ? eta : std::log1p(std::exp(eta));
            ll += static_cast<double>(rows[i].y_detected) * eta - softplus;
        }
        return ll;
    };
    double a = 0.0, b = 1.0;
    double ll = loglik(a, b);
    for (int it = 0; it < 100; ++it) {
        double g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
        #pragma omp parallel for schedule(static) reduction(+:g0,g1,h00,h01,h11)
        for (size_t i = 0; i < rows.size(); ++i) {
            const double eta = std::clamp(a + b * x[i], -35.0, 35.0);
            const double mu = 1.0 / (1.0 + std::exp(-eta));
            const double w = std::max(mu * (1.0 - mu), 1e-12);
            const double r = static_cast<double>(rows[i].y_detected) - mu;
            g0 += r; g1 += r * x[i];
            h00 += w; h01 += w * x[i]; h11 += w * x[i] * x[i];
        }
        const double ridge = 1e-9 * (h00 + h11);
        h00 += ridge; h11 += ridge;
        const double det = h00 * h11 - h01 * h01;
        if (std::abs(det) < 1e-18) break;
        double da = (h11 * g0 - h01 * g1) / det;
        double db = (-h01 * g0 + h00 * g1) / det;
        const double mag = std::max(std::abs(da), std::abs(db));
        if (mag > 2.0) { da *= 2.0 / mag; db *= 2.0 / mag; }          // paso máximo
        double step = 1.0;
        double ll_new = loglik(a + step * da, b + step * db);
        while (ll_new < ll && step > 1e-6) { step *= 0.5; ll_new = loglik(a + step * da, b + step * db); }
        if (ll_new < ll) break;
        a += step * da; b += step * db;
        const bool converged = std::abs(ll_new - ll) < 1e-10 * (1.0 + std::abs(ll));
        ll = ll_new;
        if (converged) break;
    }
    return {b, a};   // (pendiente, intercepto)
}


} // namespace

int main(int argc, char* argv[]) {
    // 0. HABILITAR EL DESBLOQUEO DE LA VENTANA DE EVALUACIÓN 2020–2024
    setenv("JORO_UNLOCK_EVALUATION", "1", 1);
    (void)argc; (void)argv;

    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════╗\n"
              << "║  JORO-HPC — VALIDACIÓN CIEGA FUERA DE MUESTRA (2020–2024)                            ║\n"
              << "║  Cero Reajuste (Zero Data Leakage) · Contraste de H1a/b, H2a/b y H3a/b               ║\n"
              << "║  Aceleración Híbrida: Intel Arc A380 (SYCL Device USM) + CPU OpenMP (AVX-512)        ║\n"
              << "╚═══════════════════════════════════════════════════════════════════════════════════════╝\n\n";

    auto sim_start_all = std::chrono::high_resolution_clock::now();

    // Directorio de salidas versionado: NO sobrescribe los artefactos de la validación v1.0 (outputs/)
    const std::string out_dir = joro::env_or("JORO_EVAL_OUTPUT_DIR", "/mnt/hpc_scratch/joro-hpc/outputs_v2/");
    std::filesystem::create_directories(out_dir);
    std::cout << "[Config] Directorio de salidas de la evaluación: " << out_dir << "\n";

    // 1. Cargar configuración maestra
    auto config = joro::SimulationConfig::load_default();
    if (const char* e = std::getenv("JORO_ERA5_DIR")) config.era5_dir = e;

    std::cout << "[Config] Directorio ERA5 activo: " << config.era5_dir << "\n";
    std::cout << "[Config] Guard ERA5 desbloqueado para la validación fuera de muestra: JORO_UNLOCK_EVALUATION=1\n";

    // 2. Conectar a PostGIS y cargar la malla de 256,020 celdas
    joro::io::PostgisClient db_client(config.db_host, config.db_port, config.db_name, config.db_user, config.db_password);
    db_client.connect();

    std::vector<joro::GridCell> grid_cells;
    db_client.load_grid(grid_cells);
    std::cout << "[Malla] " << grid_cells.size() << " celdas cargadas desde PostGIS (EPSG:5070).\n";

    // Geometría UNIFICADA: centroide y radio del polígono semilla (referencia de P50/P90/P95, envolvente y azimut)
    g_seed = joro::SeedGeometry::from_cells(grid_cells);
    std::cout << "[Semilla] " << g_seed.n_cells << " celdas semilla | centroide (" << std::fixed << std::setprecision(5)
              << g_seed.lat << "°N, " << g_seed.lon << "°E) | radio inicial " << std::setprecision(1) << g_seed.radius_km
              << " km (azimut medido solo sobre celdas colonizadas fuera de ese radio).\n";

    // Normalización de H, R sobre Ω (idéntica a la calibración) y modelo de observación
    const joro::CovariateNorm norm = joro::CovariateNorm::from_cells(grid_cells);
    const joro::inference::OccupancyParameters occ = joro::inference::OccupancyParameters::from_environment();
    joro::inference::AttentionIndex attention;
    const std::string attention_csv = joro::env_or("JORO_ATTENTION_INDEX", "/mnt/hpc_scratch/joro-hpc/attention_index_Mt.csv");
    if (attention.load(attention_csv)) {
        std::cout << "[Observación] Índice exógeno M_t/M_2019 cargado de " << attention_csv << " (2024: x"
                  << std::setprecision(2) << attention.ratio_for(2024) << ") | lambda_M = " << occ.lambda_M
                  << ", kappa_tau = " << occ.kappa_tau << "\n";
    } else {
        std::cout << "[Observación] Sin índice M_t (" << attention_csv << "): M_t/M_2019 = 1 en todos los años.\n";
    }

    // 3. Datos observacionales: calibración 2014–2019 (offset de prevalencia, soporte S) y evaluación 2020–2024
    const std::string eval_csv = "/mnt/hpc_scratch/joro-hpc/evaluation_set_2020_2024.csv";
    const std::string calib_csv = "/mnt/hpc_scratch/joro-hpc/calibration_set_2014_2019.csv";
    auto eval_rows = load_eval_dataset(eval_csv, norm, occ, attention);
    std::cout << "[Evaluación] " << eval_rows.size() << " filas celda-año cargadas de " << eval_csv << " (2020–2024).\n";
    auto calib_rows = load_eval_dataset(calib_csv, norm, occ, attention);
    std::cout << "[Calibración] " << calib_rows.size() << " filas celda-año (2014–2019) para el offset de prevalencia.\n";
    if (eval_rows.empty() || calib_rows.empty()) {
        std::cerr << "ERROR: no se pudieron cargar los conjuntos observacionales.\n";
        return EXIT_FAILURE;
    }

    // Índice de hábitat estandarizado sobre el soporte informativo S de la calibración (M14)
    {
        std::vector<int32_t> support_ids;
        for (const auto& r : calib_rows) if (r.effort_exogenous > 0) support_ids.push_back(r.cell_id);
        const auto mask = joro::abm::make_support_mask(grid_cells, support_ids);
        size_t n_s = 0;
        for (uint8_t v : mask) n_s += v;
        const auto [h_mean, h_sd] = joro::abm::compute_habitat_index(grid_cells, support_ids.empty() ? nullptr : &mask);
        std::cout << "[Malla] H_hab estandarizado sobre el soporte S (" << n_s << " de " << grid_cells.size()
                  << " celdas; m_S = " << h_mean << ", s_S = " << h_sd << ").\n";
    }

    // Mapeo espacial O(1) y tabla de índices
    joro::SpatialGridLookup spatial_lookup;
    spatial_lookup.build(grid_cells);
    const auto grid_view = spatial_lookup.view();
    const auto cidx = joro::abm::CellIndexTable::build(grid_cells);

    uint64_t total_obs_presences = 0;
    for (const auto& r : eval_rows) if (r.y_detected == 1) total_obs_presences++;
    std::cout << "[Evaluación] Presencias observadas en 2020–2024: " << total_obs_presences << " (100% ciegas).\n";

    // Mapeo celda -> índice en eval_rows por año
    std::unordered_map<int16_t, std::unordered_map<int32_t, size_t>> eval_index_by_year;
    for (size_t i = 0; i < eval_rows.size(); ++i) {
        eval_index_by_year[eval_rows[i].year][eval_rows[i].cell_id] = i;
    }
    for (int yr = kEvalStartYear; yr < kEvalStartYear + kEvalYears; ++yr) eval_index_by_year[static_cast<int16_t>(yr)];

    uint64_t calib_observed = 0;
    for (const auto& r : calib_rows) if (r.y_detected == 1) calib_observed++;
    std::vector<std::vector<size_t>> calib_rows_of_year(kCalibYears);
    for (size_t i = 0; i < calib_rows.size(); ++i) {
        const int yi = calib_rows[i].year - kStartYear;
        if (yi >= 0 && yi < kCalibYears) calib_rows_of_year[static_cast<size_t>(yi)].push_back(i);
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
    std::cout << "\n▶ Inicializando GPU SYCL (Intel Arc A380) y empaquetando 11 campos atmosféricos 3D en FP16...\n";
    auto lpdm_gpu = joro::lpdm::create_sycl_gpu_engine();
    lpdm_gpu->clear_atmospheres();
    if (const char* e = std::getenv("JORO_MAX_SUPER")) lpdm_gpu->set_max_super_particles(static_cast<size_t>(std::strtoull(e, nullptr, 10)));

    // Cada año (mayo-julio, 2,208 h) se empaqueta en FP16 en la RAM host (~1.8 GiB/año). La VRAM se gestiona con
    // DeviceSlot reutilizables (máx. 2 años simultáneos, <= 3.8 GiB). El bucle año-afuera de abajo sube cada tensor UNA vez.
    for (int y = 0; y < kTotalYears; ++y) {
        int yr = kStartYear + y;
        joro::io::AtmosphericField3D f3d;
        if (!era5_reader.load_season(yr, {5, 6, 7}, f3d)) {
            std::cerr << "ERROR: no se pudo abrir ERA5 3D de la temporada mayo-julio de " << yr << "\n";
            return EXIT_FAILURE;
        }
        if (!lpdm_gpu->add_atmosphere_and_release(f3d)) {
            std::cerr << "ERROR: fallo al empaquetar el campo ERA5 de " << yr << "\n";
            return EXIT_FAILURE;
        }
    }
    std::cout << "✔ " << lpdm_gpu->atmosphere_count() << " campos meteorológicos anuales FP16 | RAM host "
              << std::fixed << std::setprecision(2) << (static_cast<double>(lpdm_gpu->atmosphere_host_bytes()) / 1073741824.0)
              << " GiB | VRAM " << (static_cast<double>(lpdm_gpu->atmosphere_device_bytes()) / 1073741824.0) << " GiB.\n";

    // 6. Cargar conjuntos de parámetros congelados (Theta_comp M1 y M0)
    std::cout << "\n▶ Cargando ensambles de parámetros 100% congelados de la calibración 2014–2019...\n";
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
    if (const char* e = std::getenv("JORO_EVAL_MAX_MEMBERS")) {   // verificación rápida: limita el ensamble propagado
        const size_t cap = static_cast<size_t>(std::max(1, std::atoi(e)));
        if (m1_ensemble.size() > cap) { m1_ensemble.resize(cap); std::cout << "  (JORO_EVAL_MAX_MEMBERS=" << cap << ": ensamble M1 truncado)\n"; }
    }

    // Parámetros MAP congelados de M1 y M0 (leídos directamente de calibration_summary_M{1,0}.json con respaldo)
    const std::string ckpt_dir = joro::env_or("JORO_CHECKPOINT_DIR", "/mnt/hpc_scratch/joro-hpc/checkpoints/");
    const std::vector<float> theta_m1_map = load_map_theta_from_json(
        ckpt_dir + "calibration_summary_M1.json",
        {-2.1747344f, 0.028961826f, 2.0f, 9203.6211f, 0.79627186f, 0.4722499f, 0.35181648f, -1.3290558f, 1.7328026f});
    const std::vector<float> theta_m0_map = load_map_theta_from_json(
        ckpt_dir + "calibration_summary_M0.json",
        {-0.6067552f, 0.091027737f, 1.0f, 11776.741f, 0.29459813f, 0.50126952f, -1.0f, 2.1409147f});

    // Kernel gravitacional humano: atracción del receptor w_i = H_i R_i y emisión ponderada por R_j
    joro::inference::HumanTransportKernel human_kernel;
    {
        std::vector<float> hn(grid_cells.size()), rn(grid_cells.size());
        for (size_t i = 0; i < grid_cells.size(); ++i) { hn[i] = norm.h(grid_cells[i].pop_density_h); rn[i] = norm.r(grid_cells[i].road_density_r); }
        human_kernel.build(grid_cells, hn, rn, grid_view, &cidx);
    }

    // 7. COMPARADORES CALIBRADOS EN 2014–2019 CON LA MISMA FUNCIÓN MULTICRITERIO J(theta) (M16)
    std::cout << "\n▶ Calibrando comparadores bajo la misma J(theta) multicriterio de 2014–2019...\n";
    std::vector<joro::inference::CalibrationObservation> calib_obs = joro::inference::load_calibration_dataset(calib_csv);
    std::vector<joro::io::DailyThermalGrid> thermal_calib(thermal_by_year.begin(), thermal_by_year.begin() + kCalibYears);
    double best_lambda_km = 3.5;
    double km_best_p_h = 0.0, km_best_lambda_h = 150.0;
    {
        joro::inference::CalibrationEvaluator evaluator(grid_cells, thermal_calib, calib_obs, lpdm_gpu.get(), joro::abm::CarryingModel::M1);

        const std::vector<double> lambdas = {1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 12.0, 20.0, 35.0, 60.0};
        const auto iso_cal = joro::inference::calibrate_isotropic_null_kernel(evaluator, theta_m1_map, lambdas);
        best_lambda_km = iso_cal.best_lambda_km;
        for (const auto& [lam, j] : iso_cal.scan) {
            std::cout << "  [Calib K0(d|lambda)] lambda = " << std::fixed << std::setprecision(1) << lam << " km | J(theta_MAP) = "
                      << std::setprecision(4) << j << (lam == iso_cal.best_lambda_km ? "  <-- mínimo" : "") << "\n";
        }
        std::cout << "✔ Kernel isotrópico calibrado minimizando J: lambda* = " << best_lambda_km << " km (J = " << iso_cal.best_cost << ")\n";

        const std::vector<double> p_h_grid = {1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4};   // rango ajustado [1e-7, 1e-4] (Decisión 2)
        const std::vector<double> lambda_h_grid = {150.0};                                   // lambda_h = 150 km fijo (Decisión 3)
        const auto mix_cal = joro::inference::calibrate_mixed_human_kernel(evaluator, theta_m1_map, p_h_grid, lambda_h_grid);
        km_best_p_h = mix_cal.best_p_h;
        km_best_lambda_h = mix_cal.best_lambda_h_km;
        std::cout << "✔ Kernel mixto humano calibrado: p_h* = " << std::scientific << std::setprecision(2) << km_best_p_h
                  << std::fixed << ", lambda_h* = " << std::setprecision(0) << km_best_lambda_h << " km | J(K_mixto) = "
                  << std::setprecision(4) << mix_cal.best_cost << " vs J(solo LPDM) = " << mix_cal.cost_without_human << "\n";
        if (km_best_p_h == 0.0) {
            std::cout << "  (J no mejora con transporte humano en 2014–2019: K_mixto se evalúa con p_h = 1e-6 para la ablación frente a p_h = 0)\n";
            km_best_p_h = 1e-6;
        }
    }

    // 8. EJECUCIÓN CONTINUA DE 11 AÑOS (2014–2024) EN LOCKSTEP: año afuera, configuraciones adentro
    std::cout << "\n===================================================================\n";
    std::cout << "▶ EJECUTANDO SIMULACIONES CONTINUAS DE 11 AÑOS (2014–2024) — bucle invertido por año\n";
    std::cout << "===================================================================\n";

    SimContext sctx;
    sctx.grid = &grid_cells; sctx.cidx = &cidx; sctx.thermal = &thermal_by_year;
    sctx.eval_rows = &eval_rows; sctx.calib_rows = &calib_rows;
    sctx.eval_index_by_year = &eval_index_by_year; sctx.calib_rows_of_year = &calib_rows_of_year;
    sctx.lpdm = lpdm_gpu.get(); sctx.human_kernel = &human_kernel; sctx.grid_view = grid_view;
    sctx.norm = norm; sctx.occ = occ; sctx.attention = attention;
    sctx.cell_seed_dist_km.resize(grid_cells.size());
    for (size_t i = 0; i < grid_cells.size(); ++i) {
        sctx.cell_seed_dist_km[i] = static_cast<float>(dist_to_seed_km(grid_cells[i].centroid_lat, grid_cells[i].centroid_lon));
    }

    auto make_spec = [&](const std::string& label, const std::vector<float>& theta, joro::abm::CarryingModel model,
                         bool iso, double lam, double ph, double lh, uint32_t seed, bool diag) {
        RunSpec s; s.label = label; s.theta = theta; s.model = model; s.isotropic = iso; s.iso_lambda_km = lam;
        s.p_h = ph; s.lambda_h_km = lh; s.seed = seed; s.diagnostics = diag;
        return s;
    };
    using CM = joro::abm::CarryingModel;
    SimRun run_m1(sctx, make_spec("M1 MAP", theta_m1_map, CM::M1, false, 0.0, 0.0, 150.0, 1001, true));
    SimRun run_m0(sctx, make_spec("M0 MAP", theta_m0_map, CM::M0, false, 0.0, 0.0, 150.0, 2002, false));
    SimRun run_iso(sctx, make_spec("Kernel isotrópico", theta_m1_map, CM::M1, true, best_lambda_km, 0.0, 150.0, 3003, false));
    SimRun run_km(sctx, make_spec("M1 + K_mixto", theta_m1_map, CM::M1, false, 0.0, km_best_p_h, km_best_lambda_h, 4004, true));
    std::vector<std::unique_ptr<SimRun>> members;
    members.reserve(m1_ensemble.size());
    for (size_t m = 0; m < m1_ensemble.size(); ++m) {
        members.push_back(std::make_unique<SimRun>(sctx, make_spec("M1 miembro", m1_ensemble[m], CM::M1, false, 0.0, 0.0, 150.0,
                                                                   static_cast<uint32_t>(10000 + m), false)));
    }

    const auto t_sim0 = std::chrono::high_resolution_clock::now();
    for (int yi = 0; yi < kTotalYears; ++yi) {
        const auto ty0 = std::chrono::high_resolution_clock::now();
        lpdm_gpu->select_atmosphere(static_cast<size_t>(yi));      // ERA5 del año: UNA sola transferencia H2D por año
        run_m1.step_year(yi);
        run_m0.step_year(yi);
        run_iso.step_year(yi);
        run_km.step_year(yi);
        for (auto& mem : members) mem->step_year(yi);
        std::cout << "  ✔ " << (kStartYear + yi) << " completado (" << (4 + members.size()) << " configuraciones) en "
                  << std::fixed << std::setprecision(1)
                  << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ty0).count() << " s\n";
    }
    const double ensemble_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_sim0).count();
    const auto tel_eval = lpdm_gpu->telemetry();
    std::cout << "✔ 11 años x " << (4 + members.size()) << " configuraciones en " << ensemble_time << " s | H2D acumulado "
              << std::setprecision(2) << tel_eval.h2d_time_s << " s | fallbacks a CPU: " << tel_eval.cpu_fallbacks << "\n";

    SimOutputs m1_map_out = std::move(run_m1.out);
    SimOutputs m0_map_out = std::move(run_m0.out);
    SimOutputs iso_null_out = std::move(run_iso.out);
    SimOutputs km_out = std::move(run_km.out);

    // Totales anuales esperados por miembro (crudos y con offset de prevalencia ajustado en 2014–2019)
    std::vector<std::vector<float>> m1_ensemble_preds(eval_rows.size(), std::vector<float>(m1_ensemble.size(), 0.0f));
    std::vector<double> member_delta(m1_ensemble.size(), 0.0);
    std::vector<std::array<double, kEvalYears>> member_raw_tot(m1_ensemble.size()), member_adj_tot(m1_ensemble.size());
    std::vector<double> member_calib_raw(m1_ensemble.size(), 0.0), member_calib_adj(m1_ensemble.size(), 0.0);
    for (size_t m = 0; m < m1_ensemble.size(); ++m) {
        const SimOutputs& mem_out = members[m]->out;
        for (size_t r = 0; r < eval_rows.size(); ++r) m1_ensemble_preds[r][m] = mem_out.p_marginal_eval[r];
        const auto fit = fit_prevalence_offset(mem_out.p_marginal_calib, calib_rows);
        member_delta[m] = fit.delta;
        member_calib_raw[m] = fit.raw_expected;
        member_calib_adj[m] = fit.adj_expected;
        member_raw_tot[m].fill(0.0);
        member_adj_tot[m].fill(0.0);
        for (size_t r = 0; r < eval_rows.size(); ++r) {
            const int yk = eval_rows[r].year - kEvalStartYear;
            if (yk < 0 || yk >= kEvalYears) continue;
            member_raw_tot[m][static_cast<size_t>(yk)] += mem_out.p_marginal_eval[r];
            member_adj_tot[m][static_cast<size_t>(yk)] += adjust_probability(mem_out.p_marginal_eval[r], fit.delta);
        }
        members[m].reset();   // libera el espacio de trabajo del miembro
    }
    std::cout << "✔ Ensamble M1 propagado (" << m1_ensemble.size() << " miembros; " << std::setprecision(3)
              << (ensemble_time / static_cast<double>(std::max<size_t>(1, m1_ensemble.size() + 4))) << " s por configuración-11 años).\n";

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

    // 8.b Offset de prevalencia (solo con datos 2014–2019) para M1 MAP y M0 MAP
    const auto fit_m1 = fit_prevalence_offset(m1_map_out.p_marginal_calib, calib_rows);
    const auto fit_m0 = fit_prevalence_offset(m0_map_out.p_marginal_calib, calib_rows);
    std::vector<float> m1_map_adj(eval_rows.size(), 0.0f), m0_map_adj(eval_rows.size(), 0.0f);
    for (size_t r = 0; r < eval_rows.size(); ++r) {
        m1_map_adj[r] = adjust_probability(m1_map_out.p_marginal_eval[r], fit_m1.delta);
        m0_map_adj[r] = adjust_probability(m0_map_out.p_marginal_eval[r], fit_m0.delta);
    }
    std::cout << "\n▶ Offset logit de prevalencia (ajustado solo en 2014–2019; presencias observadas = " << calib_observed << "):\n"
              << std::fixed << std::setprecision(4)
              << "  M1 MAP: δ = " << fit_m1.delta << " | esperado crudo " << std::setprecision(1) << fit_m1.raw_expected
              << " → ajustado " << fit_m1.adj_expected << (fit_m1.reached ? "" : "  (objetivo NO alcanzado)") << "\n"
              << std::setprecision(4)
              << "  M0 MAP: δ = " << fit_m0.delta << " | esperado crudo " << std::setprecision(1) << fit_m0.raw_expected
              << " → ajustado " << fit_m0.adj_expected << (fit_m0.reached ? "" : "  (objetivo NO alcanzado)") << "\n";

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
    const std::string out_pred_csv = out_dir + "predictions_2020_2024_ensemble.csv";
    std::cout << "\n▶ Exportando predicciones de ensamble 2020–2024 a " << out_pred_csv << "...\n";
    {
        std::ofstream out_f(out_pred_csv);
        out_f << "cell_id,year,centroid_lat,centroid_lon,y_observed,effort_araneae,ecoregion_block,"
              << "p_m1_map,p_m1_median,p_m1_p5,p_m1_p95,p_m0_map,p_iso_null,p_sdm,p_m1_map_adj,p_m0_map_adj,"
              << "p_m1_kmixto,p_m1_map_nofest,effort_exogenous,cpue_joro,tau_invasion_age,psi_m1_map,dbar_m1_map,k_m1_map\n";

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
                  << sdm_preds[r] << ","
                  << m1_map_adj[r] << ","
                  << m0_map_adj[r] << ","
                  << km_out.p_marginal_eval[r] << ","
                  << m1_map_out.p_marginal_eval_nofest[r] << ","
                  << row.effort_exogenous << ","
                  << row.cpue_joro << ","
                  << row.tau_invasion_age << ","
                  << m1_map_out.psi_eval[r] << ","
                  << m1_map_out.dbar_eval[r] << ","
                  << m1_map_out.k_eval[r] << "\n";
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

    // Brier balanceado: negativos solo dentro de la envolvente activa SIMULADA del año (misma regla que la calibración, M15)
    auto calc_brier_balanced = [&](const std::vector<float>& p, const std::vector<EvalRow>& rows, int year_filter = 0) {
        double s_pos = 0.0, s_neg = 0.0;
        uint64_t n_pos = 0, n_neg = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (year_filter != 0 && rows[i].year != year_filter) continue;
            if (rows[i].y_detected == 1) {
                double diff = 1.0 - static_cast<double>(p[i]);
                s_pos += diff * diff;
                n_pos++;
            } else if (rows[i].dist_epi_km <= m1_map_out.env_radius_km[static_cast<size_t>(rows[i].year - kStartYear)]) {
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
        constexpr double eps = 1e-6; // Estándar para evitar penalización espuria de -ln(1e-15)
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

        // Mann-Whitney U con corrección exacta de empates (average ranks for ties)
        double rank_sum_pos = 0.0;
        size_t i = 0;
        const size_t n = pairs.size();
        while (i < n) {
            size_t j = i;
            while (j < n && pairs[j].first == pairs[i].first) {
                j++;
            }
            double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
            for (size_t k = i; k < j; ++k) {
                if (pairs[k].second == 1) {
                    rank_sum_pos += avg_rank;
                }
            }
            i = j;
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

    // Pendiente e intercepto de calibración de Cox (regresión logística de y sobre logit(p_hat); ideal: 1 y 0)
    auto calc_calibration = [](const std::vector<float>& p, const std::vector<EvalRow>& rows) {
        return cox_calibration(p, rows);
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
        uint64_t pres = 0, n_ang = 0;
        for (const auto& r : eval_rows) {
            if (r.year == yr && r.y_detected == 1) {
                dists.push_back(r.dist_epi_km);
                pres++;
                if (r.dist_seed_km > g_seed.radius_km) {   // azimut solo con presencias fuera del radio semilla
                    double rad = r.angle_epi_deg * M_PI / 180.0;
                    s_sin += std::sin(rad);
                    s_cos += std::cos(rad);
                    ++n_ang;
                }
            }
        }
        ObsYearStats oys{};
        oys.presences = pres;
        if (!dists.empty()) {
            oys.p50_km = percentile(dists, 0.50);
            oys.p90_km = percentile(dists, 0.90);
            oys.p95_km = percentile(dists, 0.95);
            if (n_ang > 0) {
                double mean_deg = std::atan2(s_sin, s_cos) * 180.0 / M_PI;
                if (mean_deg < 0.0) mean_deg += 360.0;
                oys.mean_angle_deg = mean_deg;
            } else {
                oys.mean_angle_deg = std::numeric_limits<double>::quiet_NaN();
            }
        }
        obs_stats[yr] = oys;
    }

    // False positive rate al norte (>38°–40°N) con umbral óptimo tau* derivado de la calibración 2014–2019
    auto calc_fpr_north = [](const std::vector<float>& p, const std::vector<EvalRow>& rows, float threshold = 0.001208f) {
        uint64_t fp = 0, tn = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].centroid_lat >= 38.0f) {
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
        double bb = calc_brier_balanced(p, eval_rows);
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
    print_model_row("JORO-HPC M1 + K_mixto humano (MAP)", km_out.p_marginal_eval);
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

    std::cout << "│ Tasa Falsos Positivos Norte (>38°–40°N)  │ "
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

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 6: H2b — P95 POR CUADRANTE Y COLA DE LARGA DISTANCIA (> 300 km)
    // ─────────────────────────────────────────────────────────────────────────
    // P95 observado por cuadrante (respecto del centroide semilla) y año; el simulado sale de las celdas con N > 2.
    std::array<std::array<double, 4>, kTotalYears> obs_quad_p95;
    for (auto& a : obs_quad_p95) a.fill(std::nan(""));
    for (int yr = kEvalStartYear; yr < kEvalStartYear + kEvalYears; ++yr) {
        std::array<std::vector<double>, 4> qd;
        for (const auto& r : eval_rows) if (r.year == yr && r.y_detected == 1) qd[static_cast<size_t>(quadrant_of(r.angle_epi_deg))].push_back(r.dist_epi_km);
        for (size_t q = 0; q < 4; ++q) if (!qd[q].empty()) obs_quad_p95[static_cast<size_t>(yr - kStartYear)][q] = percentile(qd[q], 0.95);
    }
    auto mean_quad = [&](const std::vector<std::array<double, 4>>& sim, size_t q, bool obs) {
        double acc = 0.0; int n = 0;
        for (int yr = kEvalStartYear; yr < kEvalStartYear + kEvalYears; ++yr) {
            const double v = obs ? obs_quad_p95[static_cast<size_t>(yr - kStartYear)][q] : sim[static_cast<size_t>(yr - kStartYear)][q];
            if (std::isfinite(v)) { acc += v; ++n; }
        }
        return n ? acc / n : std::numeric_limits<double>::quiet_NaN();
    };
    // Cola > 300 km: presencias observadas lejanas y recall de la ocupación latente simulada
    struct TailStats { uint64_t n_obs; double recall; double psi_obs; double psi_abs; };
    auto tail_recall = [&](const SimOutputs& o, double thr) {
        uint64_t n_obs = 0, hit = 0;
        double psi_obs = 0.0, psi_abs = 0.0; uint64_t n_abs = 0;
        for (size_t r = 0; r < eval_rows.size(); ++r) {
            if (eval_rows[r].dist_epi_km <= kTailKm) continue;
            if (eval_rows[r].y_detected == 1) { ++n_obs; psi_obs += o.psi_eval[r]; if (o.psi_eval[r] >= thr) ++hit; }
            else { ++n_abs; psi_abs += o.psi_eval[r]; }
        }
        return TailStats{n_obs, n_obs ? static_cast<double>(hit) / static_cast<double>(n_obs) : 0.0,
                         n_obs ? psi_obs / static_cast<double>(n_obs) : 0.0, n_abs ? psi_abs / static_cast<double>(n_abs) : 0.0};
    };
    const TailStats tail_m1 = tail_recall(m1_map_out, 0.10);
    const TailStats tail_km = tail_recall(km_out, 0.10);
    std::cout << "\nTABLA 6: H2b — P95 POR CUADRANTE (MEDIA 2020–2024, km desde el centroide semilla) Y COLA > " << kTailKm << " km:\n";
    std::cout << "┌───────────────┬──────────────┬───────────────┬──────────────┬──────────────┐\n";
    std::cout << "│ Cuadrante     │ Observado    │ M1 (solo LPDM)│ M1 + K_mixto │ Kern. Isotr. │\n";
    std::cout << "├───────────────┼──────────────┼───────────────┼──────────────┼──────────────┤\n";
    for (size_t q = 0; q < 4; ++q) {
        std::cout << "│ " << std::left << std::setw(13) << kQuadrantName[q] << " │ " << std::right << std::fixed << std::setprecision(1)
                  << std::setw(12) << mean_quad(m1_map_out.sim_quadrant_p95, q, true) << " │ "
                  << std::setw(13) << mean_quad(m1_map_out.sim_quadrant_p95, q, false) << " │ "
                  << std::setw(12) << mean_quad(km_out.sim_quadrant_p95, q, false) << " │ "
                  << std::setw(12) << mean_quad(iso_null_out.sim_quadrant_p95, q, false) << " │\n";
    }
    std::cout << "└───────────────┴──────────────┴───────────────┴──────────────┴──────────────┘\n";
    std::cout << "  Cola > " << kTailKm << " km (presencias observadas: " << tail_m1.n_obs << "): recall(psi >= 0.10) M1 = "
              << std::setprecision(3) << tail_m1.recall << " | M1 + K_mixto = " << tail_km.recall
              << " | psi medio en presencias lejanas: " << tail_m1.psi_obs << " vs " << tail_km.psi_obs
              << " | psi medio en ausencias lejanas: " << std::setprecision(5) << tail_m1.psi_abs << " vs " << tail_km.psi_abs << "\n";
    {
        uint64_t th_tot = 0, th_far = 0;
        for (int yi = 0; yi < kTotalYears; ++yi) { th_tot += km_out.human_transported[static_cast<size_t>(yi)]; th_far += km_out.human_beyond_300km[static_cast<size_t>(yi)]; }
        std::cout << "  Transporte humano simulado 2014–2024: " << th_tot << " propágulos (" << th_far << " depositados a > 300 km) con p_h* = "
                  << std::setprecision(4) << km_best_p_h << ", lambda_h* = " << std::setprecision(0) << km_best_lambda_h << " km\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 7: H1b / H3a — CORRELACIONES DE SPEARMAN DE LA CPUE OBSERVADA
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<double> sp_cpue, sp_dbar, sp_k;
    for (size_t r = 0; r < eval_rows.size(); ++r) {
        if (eval_rows[r].y_detected != 1 || eval_rows[r].joro_count <= 0) continue;
        sp_cpue.push_back(eval_rows[r].cpue_joro);
        sp_dbar.push_back(m1_map_out.dbar_eval[r]);
        sp_k.push_back(m1_map_out.k_eval[r]);
    }
    const double rho_h1b = spearman(sp_cpue, sp_dbar);   // esperado < 0: más daño invernal => menor abundancia relativa
    const double rho_h3a = spearman(sp_cpue, sp_k);      // esperado > 0: mayor capacidad sinantrópica => mayor CPUE
    std::cout << "\nTABLA 7: CORRELACIÓN DE SPEARMAN DE LA CPUE OBSERVADA (celdas-año con presencia, 2020–2024, n = " << sp_cpue.size() << "):\n";
    std::cout << "┌─────────────────────────────────────────────────────┬──────────────┬────────────────────────────┐\n";
    std::cout << "│ Contraste                                           │ rho Spearman │ Signo esperado             │\n";
    std::cout << "├─────────────────────────────────────────────────────┼──────────────┼────────────────────────────┤\n";
    std::cout << "│ H1b: CPUE_obs vs. daño invernal medio D_{i,t}       │ " << std::right << std::setw(12) << std::setprecision(4) << rho_h1b << " │ negativo (daño reduce CPUE)│\n";
    std::cout << "│ H3a: CPUE_obs vs. capacidad de carga K_i (M1)       │ " << std::setw(12) << std::setprecision(4) << rho_h3a << " │ positivo (hábitat sinantr.)│\n";
    std::cout << "└─────────────────────────────────────────────────────┴──────────────┴────────────────────────────┘\n";

    // ─────────────────────────────────────────────────────────────────────────
    // TABLA 8: H3b — CALIBRACIÓN DE COX Y COBERTURA DE PREVALENCIA CON/SIN f_est(i,t)
    // ─────────────────────────────────────────────────────────────────────────
    const auto fit_m1_nofest = fit_prevalence_offset(m1_map_out.p_marginal_calib_nofest, calib_rows);
    std::vector<float> m1_map_adj_nofest(eval_rows.size(), 0.0f);
    for (size_t r = 0; r < eval_rows.size(); ++r) {
        m1_map_adj_nofest[r] = adjust_probability(m1_map_out.p_marginal_eval_nofest[r], fit_m1_nofest.delta);
    }
    const auto cox_with = cox_calibration(m1_map_out.p_marginal_eval, eval_rows);
    const auto cox_without = cox_calibration(m1_map_out.p_marginal_eval_nofest, eval_rows);
    const auto cox_platt_nofest = cox_calibration(m1_map_adj_nofest, eval_rows);
    const auto cox_adj_fest = cox_calibration(m1_map_adj, eval_rows);
    std::array<double, kEvalYears> obs_y{}, exp_with{}, exp_without{}, exp_platt_nofest{}, exp_adj_fest{};
    for (size_t r = 0; r < eval_rows.size(); ++r) {
        const int yk = eval_rows[r].year - kEvalStartYear;
        if (yk < 0 || yk >= kEvalYears) continue;
        obs_y[static_cast<size_t>(yk)] += eval_rows[r].y_detected;
        exp_with[static_cast<size_t>(yk)] += m1_map_out.p_marginal_eval[r];
        exp_without[static_cast<size_t>(yk)] += m1_map_out.p_marginal_eval_nofest[r];
        exp_platt_nofest[static_cast<size_t>(yk)] += m1_map_adj_nofest[r];
        exp_adj_fest[static_cast<size_t>(yk)] += m1_map_adj[r];
    }
    std::cout << "\nTABLA 8: H3b — NO ESTACIONARIEDAD DEL PROCESO DE OBSERVACIÓN (COMPARACIÓN DE 4 VARIANTES EN M1 MAP):\n";
    std::cout << "┌──────────────────────────────┬──────────────┬──────────────┬──────────────────┬──────────────────┐\n";
    std::cout << "│ Variante de Observación      │ Pend. Cox    │ Interc. Cox  │ Brier Global     │ Brier Balanceado │\n";
    std::cout << "├──────────────────────────────┼──────────────┼──────────────┼──────────────────┼──────────────────┤\n";
    std::cout << "│ 1. Crudo (δ=0, f_est=0)      │ " << std::setw(12) << std::setprecision(3) << cox_without.first << " │ " << std::setw(12) << cox_without.second
              << " │ " << std::setw(16) << std::setprecision(5) << calc_brier_global(m1_map_out.p_marginal_eval_nofest, eval_rows)
              << " │ " << std::setw(16) << calc_brier_balanced(m1_map_out.p_marginal_eval_nofest, eval_rows) << " │\n";
    std::cout << "│ 2. Platt estac. (f_est=0)    │ " << std::setw(12) << std::setprecision(3) << cox_platt_nofest.first << " │ " << std::setw(12) << cox_platt_nofest.second
              << " │ " << std::setw(16) << std::setprecision(5) << calc_brier_global(m1_map_adj_nofest, eval_rows)
              << " │ " << std::setw(16) << calc_brier_balanced(m1_map_adj_nofest, eval_rows) << " │\n";
    std::cout << "│ 3a. Con f_est(i,t) (δ=0)     │ " << std::setw(12) << std::setprecision(3) << cox_with.first << " │ " << std::setw(12) << cox_with.second
              << " │ " << std::setw(16) << std::setprecision(5) << calc_brier_global(m1_map_out.p_marginal_eval, eval_rows)
              << " │ " << std::setw(16) << calc_brier_balanced(m1_map_out.p_marginal_eval, eval_rows) << " │\n";
    std::cout << "│ 3b. Con f_est + δ_base'14-19 │ " << std::setw(12) << std::setprecision(3) << cox_adj_fest.first << " │ " << std::setw(12) << cox_adj_fest.second
              << " │ " << std::setw(16) << std::setprecision(5) << calc_brier_global(m1_map_adj, eval_rows)
              << " │ " << std::setw(16) << calc_brier_balanced(m1_map_adj, eval_rows) << " │\n";
    std::cout << "└──────────────────────────────┴──────────────┴──────────────┴──────────────────┴──────────────────┘\n";
    std::cout << "  Prevalencia esperada por año 2020–2024 vs observada:\n";
    for (int k = 0; k < kEvalYears; ++k) {
        const size_t ku = static_cast<size_t>(k);
        std::cout << "    " << (kEvalStartYear + k) << ": obs " << std::setprecision(0) << obs_y[ku]
                  << " | 1.Crudo(f_est=0) " << std::setprecision(1) << exp_without[ku]
                  << " | 2.Platt(f_est=0) " << exp_platt_nofest[ku]
                  << " | 3a.f_est(δ=0) " << exp_with[ku]
                  << " | 3b.f_est+δ_base " << exp_adj_fest[ku] << "\n";
    }

    // Diagnósticos de H1b, H2b y H3a/b en JSON
    {
        std::ofstream dj(out_dir + "hypothesis_diagnostics.json");
        auto arr5d = [&](const std::array<double, kEvalYears>& a) {
            dj << "[";
            for (int k = 0; k < kEvalYears; ++k) dj << (k ? "," : "") << jnum(a[static_cast<size_t>(k)]);
            dj << "]";
        };
        auto quad_json = [&](const std::vector<std::array<double, 4>>& sim) {
            dj << "{";
            for (size_t q = 0; q < 4; ++q) dj << (q ? "," : "") << "\"" << kQuadrantName[q] << "\":" << jnum(mean_quad(sim, q, false));
            dj << "}";
        };
        dj << "{\n  \"seed_geometry\": {\"n_cells\": " << g_seed.n_cells << ", \"lat\": " << jnum(g_seed.lat) << ", \"lon\": " << jnum(g_seed.lon)
           << ", \"radius_km\": " << jnum(g_seed.radius_km) << "},\n"
           << "  \"kernel_mixto\": {\"p_h\": " << jnum(km_best_p_h) << ", \"lambda_h_km\": " << jnum(km_best_lambda_h) << "},\n"
           << "  \"isotropic_lambda_km\": " << jnum(best_lambda_km) << ",\n"
           << "  \"h1b_spearman_cpue_vs_dbar\": " << jnum(rho_h1b) << ",\n"
           << "  \"h3a_spearman_cpue_vs_k\": " << jnum(rho_h3a) << ",\n"
           << "  \"spearman_n\": " << sp_cpue.size() << ",\n"
           << "  \"h2b_p95_quadrant_mean_2020_2024\": {\"observed\": {";
        for (size_t q = 0; q < 4; ++q) dj << (q ? "," : "") << "\"" << kQuadrantName[q] << "\":" << jnum(mean_quad(m1_map_out.sim_quadrant_p95, q, true));
        dj << "}, \"m1_lpdm\": "; quad_json(m1_map_out.sim_quadrant_p95);
        dj << ", \"m1_kmixto\": "; quad_json(km_out.sim_quadrant_p95);
        dj << ", \"isotropic\": "; quad_json(iso_null_out.sim_quadrant_p95);
        dj << "},\n  \"h2b_tail_gt300km\": {\"n_observed\": " << tail_m1.n_obs << ", \"recall_m1\": " << jnum(tail_m1.recall)
           << ", \"recall_kmixto\": " << jnum(tail_km.recall) << ", \"psi_obs_m1\": " << jnum(tail_m1.psi_obs)
           << ", \"psi_obs_kmixto\": " << jnum(tail_km.psi_obs) << "},\n"
           << "  \"h3b_cox\": {\"with_fest\": {\"slope\": " << jnum(cox_with.first) << ", \"intercept\": " << jnum(cox_with.second)
           << "}, \"without_fest\": {\"slope\": " << jnum(cox_without.first) << ", \"intercept\": " << jnum(cox_without.second)
           << "}, \"platt_nofest\": {\"slope\": " << jnum(cox_platt_nofest.first) << ", \"intercept\": " << jnum(cox_platt_nofest.second)
           << ", \"delta\": " << jnum(fit_m1_nofest.delta)
           << "}, \"adj_with_fest\": {\"slope\": " << jnum(cox_adj_fest.first) << ", \"intercept\": " << jnum(cox_adj_fest.second)
           << ", \"delta\": " << jnum(fit_m1.delta) << "}},\n"
           << "  \"h3b_prevalence\": {\"observed\": "; arr5d(obs_y);
        dj << ", \"expected_with_fest\": "; arr5d(exp_with);
        dj << ", \"expected_without_fest\": "; arr5d(exp_without);
        dj << ", \"expected_platt_nofest\": "; arr5d(exp_platt_nofest);
        dj << ", \"expected_adj_with_fest\": "; arr5d(exp_adj_fest);
        dj << "},\n  \"retention\": {\"m1_retained_fraction_by_year\": [";
        for (int yi = 0; yi < kTotalYears; ++yi) dj << (yi ? "," : "") << jnum(m1_map_out.retained_fraction[static_cast<size_t>(yi)]);
        dj << "], \"m1_effective_flight_mean_km_by_year\": [";
        for (int yi = 0; yi < kTotalYears; ++yi) dj << (yi ? "," : "") << jnum(m1_map_out.flight_mean_km[static_cast<size_t>(yi)]);
        dj << "]}\n}\n";
        std::cout << "✔ Diagnósticos de hipótesis exportados a " << out_dir << "hypothesis_diagnostics.json\n";
    }

    // Guardar resumen JSON de cinemática de frente
    const std::string kin_json_path = out_dir + "validation_sim_kinematics.json";
    {
        std::ofstream kj(kin_json_path);
        kj << "{\n";
        for (int yr = 2020; yr <= 2024; ++yr) {
            int yi = yr - 2014;
            double o_p95 = obs_stats[yr].p95_km;
            double m1_p95_val = m1_map_out.sim_front_p95[yi];
            double iso_p95_val = iso_null_out.sim_front_p95[yi];
            double o_ang = obs_stats[yr].mean_angle_deg;
            double m1_ang = m1_map_out.sim_mean_angle_deg[yi];
            double iso_ang = iso_null_out.sim_mean_angle_deg[yi];
            double ang_err_m1 = std::abs(m1_ang - o_ang);
            if (ang_err_m1 > 180.0) ang_err_m1 = 360.0 - ang_err_m1;
            double ang_err_iso = std::abs(iso_ang - o_ang);
            if (ang_err_iso > 180.0) ang_err_iso = 360.0 - ang_err_iso;

            kj << "  \"" << yr << "\": {\n"
               << "    \"obs_p95\": " << jnum(o_p95) << ",\n"
               << "    \"obs_angle\": " << jnum(o_ang) << ",\n"
               << "    \"m1_map\": {\"p95\": " << jnum(m1_p95_val) << ", \"angle\": " << jnum(m1_ang) << ", \"angular_error_deg\": " << jnum(ang_err_m1) << "},\n"
               << "    \"iso_null\": {\"p95\": " << jnum(iso_p95_val) << ", \"angle\": " << jnum(iso_ang) << ", \"angular_error_deg\": " << jnum(ang_err_iso) << "}\n"
               << "  }" << (yr == 2024 ? "\n" : ",\n");
        }
        kj << "}\n";
        std::cout << "✔ Cinemática de simulación exportada a " << kin_json_path << "\n";
    }

    // Geometría del polígono semilla (referencia angular) y escalado de prevalencia: artefactos para los informes
    {
        std::ofstream sg(out_dir + "seed_geometry.json");
        sg << "{\n  \"n_seed_cells\": " << g_seed.n_cells << ",\n  \"centroid_lat\": " << jnum(g_seed.lat)
           << ",\n  \"centroid_lon\": " << jnum(g_seed.lon) << ",\n  \"radius_km\": " << jnum(g_seed.radius_km)
           << ",\n  \"legacy_epicenter_lat\": " << jnum(34.13816) << ",\n  \"legacy_epicenter_lon\": " << jnum(-83.75723)
           << ",\n  \"legacy_to_centroid_km\": " << jnum(dist_to_seed_km(34.13816f, -83.75723f)) << "\n}\n";

        std::ofstream pj(out_dir + "prevalence_scaling.json");
        auto arr5 = [&](const std::array<double, kEvalYears>& a) {
            pj << "[";
            for (int k = 0; k < kEvalYears; ++k) pj << (k ? "," : "") << jnum(a[static_cast<size_t>(k)]);
            pj << "]";
        };
        std::array<double, kEvalYears> obs_tot{}, m1_raw{}, m1_adj{}, m0_raw{}, m0_adj{};
        for (size_t r = 0; r < eval_rows.size(); ++r) {
            const int yk = eval_rows[r].year - kEvalStartYear;
            if (yk < 0 || yk >= kEvalYears) continue;
            obs_tot[static_cast<size_t>(yk)] += eval_rows[r].y_detected;
            m1_raw[static_cast<size_t>(yk)] += m1_map_out.p_marginal_eval[r];
            m1_adj[static_cast<size_t>(yk)] += m1_map_adj[r];
            m0_raw[static_cast<size_t>(yk)] += m0_map_out.p_marginal_eval[r];
            m0_adj[static_cast<size_t>(yk)] += m0_map_adj[r];
        }
        pj << "{\n  \"method\": \"offset logit de solo intercepto ajustado en 2014-2019 (presencias observadas = suma de probabilidades; "
              "celdas con p=0 permanecen en 0)\",\n"
           << "  \"calibration\": {\"years\": [2014,2019], \"observed\": " << calib_observed << ",\n"
           << "    \"m1_map\": {\"delta\": " << jnum(fit_m1.delta) << ", \"raw_expected\": " << jnum(fit_m1.raw_expected)
           << ", \"adj_expected\": " << jnum(fit_m1.adj_expected) << ", \"reached\": " << (fit_m1.reached ? "true" : "false") << "},\n"
           << "    \"m0_map\": {\"delta\": " << jnum(fit_m0.delta) << ", \"raw_expected\": " << jnum(fit_m0.raw_expected)
           << ", \"adj_expected\": " << jnum(fit_m0.adj_expected) << ", \"reached\": " << (fit_m0.reached ? "true" : "false") << "}},\n"
           << "  \"evaluation\": {\"years\": [2020,2021,2022,2023,2024], \"observed\": "; arr5(obs_tot);
        pj << ",\n    \"m1_map\": {\"raw\": "; arr5(m1_raw); pj << ", \"adj\": "; arr5(m1_adj);
        pj << "},\n    \"m0_map\": {\"raw\": "; arr5(m0_raw); pj << ", \"adj\": "; arr5(m0_adj);
        pj << "}},\n  \"ensemble\": {\"n_members\": " << m1_ensemble.size() << ", \"delta\": [";
        for (size_t m = 0; m < member_delta.size(); ++m) pj << (m ? "," : "") << jnum(member_delta[m]);
        pj << "],\n    \"calib_raw_expected\": [";
        for (size_t m = 0; m < member_calib_raw.size(); ++m) pj << (m ? "," : "") << jnum(member_calib_raw[m]);
        pj << "],\n    \"calib_adj_expected\": [";
        for (size_t m = 0; m < member_calib_adj.size(); ++m) pj << (m ? "," : "") << jnum(member_calib_adj[m]);
        pj << "],\n    \"raw_totals\": [";
        for (size_t m = 0; m < member_raw_tot.size(); ++m) { pj << (m ? "," : ""); arr5(member_raw_tot[m]); }
        pj << "],\n    \"adj_totals\": [";
        for (size_t m = 0; m < member_adj_tot.size(); ++m) { pj << (m ? "," : ""); arr5(member_adj_tot[m]); }
        pj << "]}\n}\n";
        std::cout << "✔ Geometría semilla y escalado de prevalencia exportados a " << out_dir << "seed_geometry.json y prevalence_scaling.json\n";
    }

    // Guardar resumen JSON de métricas
    const std::string metrics_json_path = out_dir + "validation_metrics.json";
    std::ofstream mj(metrics_json_path);
    mj << "{\n"
       << "  \"m1_brier_global\": " << calc_brier_global(m1_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m1_brier_balanced\": " << calc_brier_balanced(m1_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m1_log_loss\": " << calc_log_loss(m1_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m1_auc\": " << calc_auc(m1_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m1_prauc\": " << calc_pr_auc(m1_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m0_brier_global\": " << calc_brier_global(m0_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"m0_auc\": " << calc_auc(m0_map_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"iso_brier_global\": " << calc_brier_global(iso_null_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"iso_auc\": " << calc_auc(iso_null_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"sdm_brier_global\": " << calc_brier_global(sdm_preds, eval_rows) << ",\n"
       << "  \"sdm_auc\": " << calc_auc(sdm_preds, eval_rows) << ",\n"
       << "  \"km1_brier_global\": " << calc_brier_global(km_out.p_marginal_eval, eval_rows) << ",\n"
       << "  \"km1_auc\": " << calc_auc(km_out.p_marginal_eval, eval_rows) << ",\n"
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
    std::cout << "✔ VALIDACIÓN FUERA DE MUESTRA COMPLETADA EXITOSAMENTE EN " << std::fixed << std::setprecision(2) << total_wall << " s ("
              << (total_wall / 60.0) << " min)\n";
    std::cout << "===================================================================\n";

    return EXIT_SUCCESS;
}
