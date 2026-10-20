#include "inference/simulated_annealing.hpp"
#include "common/fast_math.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <omp.h>

namespace joro::inference {

// Vector de parámetros dominantes (Cribado de Morris). Los parámetros de baja sensibilidad
// quedan fijos en F = 450.0 y Q10 = 2.2. beta_s es DISCRETO en {1, 2, 3} (Sección 8.3 del
// anteproyecto, configs/priors_bounds.toml: beta_damage_candidates); su rango [1,3] solo delimita el
// conjunto de candidatos y el optimizador únicamente propone saltos enteros entre ellos.
//
// K_i se reparametriza con DOS parámetros ortogonales (K0, gamma_hab) sobre el índice
// compuesto H_hab, y el enlace de ocupación incorpora el intercepto de colonización a0 y el de
// persistencia b0 (acoplan la escala numérica de A_{i,t}, N_{i,t} con P_hat(Y=1) del Brier).
namespace {
const CalibrationParamDef kTc       {"t_c",          "T_c,s",       -5.0f,     0.0f,   -3.5f};
const CalibrationParamDef kAlpha    {"alpha_damage", "alpha_s",      0.01f,    0.10f,   0.04f};
const CalibrationParamDef kBeta     {"beta_damage",  "beta_s",       1.0f,     3.0f,    2.0f};
const CalibrationParamDef kK0       {"k0",           "K0",        1000.0f, 15000.0f, 5000.0f};
const CalibrationParamDef kGammaHab {"gamma_hab",    "gamma_hab",    0.0f,     2.0f,    0.5f};
const CalibrationParamDef kQBalloon {"q_balloon",    "q_balloon",    0.10f,    0.60f,   0.35f};
const CalibrationParamDef kQViab    {"q_viab",       "q_viab",       0.20f,    0.90f,   0.65f};
const CalibrationParamDef kA0       {"a0",           "a0",          -6.0f,    -1.0f,   -4.0f};
const CalibrationParamDef kB0       {"b0",           "b0",          -2.0f,     2.5f,    1.5f};
const CalibrationParamDef kG1       {"gamma1_imp",   "gamma_1",      0.10f,    1.00f,   0.45f};
const CalibrationParamDef kG2       {"gamma2_borde", "gamma_2",      0.20f,    1.20f,   0.65f};
const CalibrationParamDef kG3       {"gamma3_dosel", "gamma_3",      0.05f,    0.80f,   0.30f};
} // namespace

const std::vector<CalibrationParamDef> CALIBRATION_PARAMS = {
    kTc, kAlpha, kBeta, kK0, kGammaHab, kQBalloon, kQViab, kA0, kB0};

const std::vector<CalibrationParamDef>& calibration_params_for(abm::CarryingModel model) {
    static const std::vector<CalibrationParamDef> m0 = {kTc, kAlpha, kBeta, kK0, kQBalloon, kQViab, kA0, kB0};
    static const std::vector<CalibrationParamDef> m3 = {kTc, kAlpha, kBeta, kK0, kQBalloon, kQViab, kA0, kB0,
                                                        kG1, kG2, kG3};
    switch (model) {
        case abm::CarryingModel::M0: return m0;
        case abm::CarryingModel::M3: return m3;
        default:                     return CALIBRATION_PARAMS;
    }
}

int calibration_param_index(const std::vector<CalibrationParamDef>& defs, const std::string& name) {
    for (size_t i = 0; i < defs.size(); ++i) if (defs[i].name == name) return static_cast<int>(i);
    return -1;
}

namespace {

constexpr float  kEpiLat = 34.13816f;      // Braselton, GA (epicentro de la introducción)
constexpr float  kEpiLon = -83.75723f;
constexpr int    kFirstYear = 2014;
constexpr int    kNYears = 6;              // 2014..2019 (la ventana 2020-2024 NO se carga jamás)
constexpr int    kLpdmSteps = 168;         // 168 pasos x 60 s = 168 horas completas de ERA5
constexpr float  kLpdmDt = 60.0f;
constexpr double kEnvelopeBaseKm = 50.0;   // Alcance potencial base alrededor del frente observado

ChainTelemetry operator-(const ChainTelemetry& a, const ChainTelemetry& b) {
    ChainTelemetry d;
    d.evaluations = a.evaluations - b.evaluations;
    d.abm_time_s = a.abm_time_s - b.abm_time_s;
    d.lpdm_init_time_s = a.lpdm_init_time_s - b.lpdm_init_time_s;
    d.lpdm_h2d_time_s = a.lpdm_h2d_time_s - b.lpdm_h2d_time_s;
    d.lpdm_kernel_time_s = a.lpdm_kernel_time_s - b.lpdm_kernel_time_s;
    d.lpdm_d2h_time_s = a.lpdm_d2h_time_s - b.lpdm_d2h_time_s;
    d.lpdm_reduce_time_s = a.lpdm_reduce_time_s - b.lpdm_reduce_time_s;
    d.cost_time_s = a.cost_time_s - b.cost_time_s;
    d.particles_dispatched = a.particles_dispatched - b.particles_dispatched;
    d.cpu_fallbacks = a.cpu_fallbacks - b.cpu_fallbacks;
    d.capped_dispatches = a.capped_dispatches - b.capped_dispatches;
    return d;
}

double elapsed_s(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double dist_to_epicenter_km(float lat, float lon) {
    const double dlat = (lat - kEpiLat) * 111.0;
    const double dlon = (lon - kEpiLon) * 111.0 * std::cos(kEpiLat * 0.0174533);
    return std::sqrt(dlat * dlat + dlon * dlon);
}

double percentile95(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(0.95 * static_cast<double>(v.size() - 1))];
}

// Categorías de cada fila celda-año para el Brier estratificado
enum : uint8_t { CAT_NONE = 0, CAT_ARR_POS = 1, CAT_ARR_NEG = 2, CAT_PER_POS = 3, CAT_PER_NEG = 4 };

/// Conjunto de calibración preparado UNA vez: ordenado por año, covariables normalizadas,
/// detección p_{i,t} precalculada (no depende de theta) y partición balanceada de llegada/persistencia.
struct PreparedCalibration {
    size_t n_rows = 0;
    size_t n_slots = 0;
    std::array<size_t, kNYears + 1> year_begin{};
    std::vector<int32_t> cell_id;
    std::vector<int32_t> slot;        // posición densa de la celda para el estado latente psi
    std::vector<float>   h_norm;      // H_i normalizada a [0,1] por su máximo en el dominio
    std::vector<float>   r_norm;      // R_i normalizada a [0,1]
    std::vector<uint8_t> in_envelope; // 1 si d_km <= radius de la envolvente activa
    std::vector<float>   p_det;       // p_{i,t} (Sección 9) con el esfuerzo TGB
    std::vector<uint8_t> cat;
    std::vector<float>   psi_init;    // psi_{2013}: 1 en celdas semilla
    std::vector<double>  obs_front_km;
    CalibrationStratification strat;
};

PreparedCalibration prepare_calibration(
    const std::vector<CalibrationObservation>& raw,
    const std::vector<GridCell>& grid_cells)
{
    PreparedCalibration cal;

    // Guardia de integridad: la partición 2020-2024 está BLOQUEADA hasta la validación fuera de muestra.
    for (const auto& o : raw) {
        if (o.year < kFirstYear || o.year >= kFirstYear + kNYears) {
            throw std::runtime_error("Fila fuera de la ventana de calibración 2014-2019 (año " +
                                     std::to_string(o.year) + "): v_evaluation_set debe permanecer bloqueada.");
        }
    }

    std::vector<CalibrationObservation> obs = raw;
    std::stable_sort(obs.begin(), obs.end(), [](const auto& a, const auto& b) {
        return a.year != b.year ? a.year < b.year : a.cell_id < b.cell_id;
    });

    cal.n_rows = obs.size();
    cal.strat.n_rows = obs.size();

    // Normalización de covariables a [0,1] (H_i, R_i entran en logit(gamma) y logit(p)).
    // Las covariables de la BD están estandarizadas (media 0), así que se escala min-max.
    float h_min = 1e30f, h_max = -1e30f, r_min = 1e30f, r_max = -1e30f;
    for (const auto& o : obs) {
        h_min = std::min(h_min, o.pop_density_h); h_max = std::max(h_max, o.pop_density_h);
        r_min = std::min(r_min, o.road_density_r); r_max = std::max(r_max, o.road_density_r);
    }
    const float h_span = (h_max - h_min) > 0.0f ? (h_max - h_min) : 1.0f;
    const float r_span = (r_max - r_min) > 0.0f ? (r_max - r_min) : 1.0f;
    if (obs.empty()) { h_min = r_min = 0.0f; }

    // Esfuerzo TGB: si el esfuerzo almacenado tiene la firma del valor sintético (muy pocos valores
    // distintos: constante 238 en el fondo y cota max(n_joro, 3) en las detecciones), se imputa en las
    // filas con detección la mediana del fondo del mismo año (solo en memoria). Con esfuerzo TGB real
    // (conteos de Araneae de GBIF) no se modifica ningún valor.
    std::unordered_set<int32_t> distinct_effort;
    for (const auto& o : obs) distinct_effort.insert(o.effort_araneae);
    const bool synthetic_effort = distinct_effort.size() < 8;
    std::array<int32_t, kNYears> median_effort{};
    for (int yi = 0; yi < kNYears; ++yi) {
        std::vector<int32_t> bg;
        for (const auto& o : obs) if (o.year == kFirstYear + yi && o.y_detected == 0) bg.push_back(o.effort_araneae);
        if (!bg.empty()) {
            std::nth_element(bg.begin(), bg.begin() + bg.size() / 2, bg.end());
            median_effort[yi] = bg[bg.size() / 2];
        } else {
            median_effort[yi] = 1;
        }
    }
    cal.strat.effort_synthetic = synthetic_effort;

    // Slots densos del estado latente
    std::unordered_map<int32_t, int32_t> slot_of;
    slot_of.reserve(60000);
    for (const auto& o : obs) {
        if (slot_of.find(o.cell_id) == slot_of.end()) {
            const int32_t s = static_cast<int32_t>(slot_of.size());
            slot_of.emplace(o.cell_id, s);
        }
    }
    cal.n_slots = slot_of.size();
    cal.psi_init.assign(cal.n_slots, 0.0f);
    std::vector<uint8_t> last_y(cal.n_slots, 0);
    for (const auto& c : grid_cells) {
        if (!c.is_seed) continue;
        auto it = slot_of.find(c.cell_id);
        if (it != slot_of.end()) { cal.psi_init[it->second] = 1.0f; last_y[it->second] = 1; }
    }

    cal.cell_id.resize(cal.n_rows);
    cal.slot.resize(cal.n_rows);
    cal.h_norm.resize(cal.n_rows);
    cal.r_norm.resize(cal.n_rows);
    cal.in_envelope.resize(cal.n_rows);
    cal.p_det.resize(cal.n_rows);
    cal.cat.assign(cal.n_rows, CAT_NONE);
    cal.obs_front_km.assign(kNYears, 0.0);
    cal.strat.envelope_radius_km.assign(kNYears, 0.0);

    // Límites por año
    size_t cursor = 0;
    for (int yi = 0; yi < kNYears; ++yi) {
        cal.year_begin[yi] = cursor;
        while (cursor < obs.size() && obs[cursor].year == kFirstYear + yi) ++cursor;
    }
    cal.year_begin[kNYears] = cursor;

    double cummax_front = 0.0;       // distancia máxima de presencias observadas en años previos
    double prev_front_p95 = 5.0;     // arrastre del P95 observado si un año no tiene presencias
    for (int yi = 0; yi < kNYears; ++yi) {
        const double radius = kEnvelopeBaseKm + cummax_front;
        cal.strat.envelope_radius_km[yi] = radius;
        std::vector<double> presence_d;
        double year_max = 0.0;

        for (size_t r = cal.year_begin[yi]; r < cal.year_begin[yi + 1]; ++r) {
            const auto& o = obs[r];
            const int32_t s = slot_of[o.cell_id];
            cal.cell_id[r] = o.cell_id;
            cal.slot[r] = s;
            cal.h_norm[r] = (o.pop_density_h - h_min) / h_span;
            cal.r_norm[r] = (o.road_density_r - r_min) / r_span;
            int32_t effort = o.effort_araneae;
            if (synthetic_effort && o.y_detected == 1) { effort = median_effort[yi]; cal.strat.effort_imputed_rows++; }
            cal.p_det[r] = compute_detection_prob(effort, (o.pop_density_h - h_min) / h_span, (o.road_density_r - r_min) / r_span);

            const double d_km = dist_to_epicenter_km(o.centroid_lat, o.centroid_lon);
            const bool in_env = (d_km <= radius);
            cal.in_envelope[r] = in_env ? 1 : 0;
            const uint8_t yprev = last_y[s];
            uint8_t c = CAT_NONE;
            if (yprev == 0) {
                if (o.y_detected == 1)        { c = CAT_ARR_POS; cal.strat.arrival_pos++; }
                else if (in_env)              { c = CAT_ARR_NEG; cal.strat.arrival_neg++; }
                else                          { cal.strat.outside_envelope++; }
            } else {
                if (o.y_detected == 1)        { c = CAT_PER_POS; cal.strat.persistence_pos++; }
                else                          { c = CAT_PER_NEG; cal.strat.persistence_neg++; }
            }
            cal.cat[r] = c;
            if (o.y_detected == 1) { presence_d.push_back(d_km); year_max = std::max(year_max, d_km); }
        }

        // Actualizar historial observado para el año siguiente (después de clasificar todo el año)
        for (size_t r = cal.year_begin[yi]; r < cal.year_begin[yi + 1]; ++r) last_y[cal.slot[r]] = obs[r].y_detected;

        cal.obs_front_km[yi] = presence_d.empty() ? prev_front_p95 : percentile95(presence_d);
        prev_front_p95 = cal.obs_front_km[yi];
        cummax_front = std::max(cummax_front, year_max);
    }
    cal.strat.observed_front_km = cal.obs_front_km;
    return cal;
}

/// Contexto de evaluación reutilizable (espacio de trabajo persistente entre evaluaciones)
struct EvalContext {
    const std::vector<GridCell>* grid = nullptr;
    const abm::CellIndexTable* cidx = nullptr;
    const std::vector<io::DailyThermalGrid>* thermal = nullptr;
    const PreparedCalibration* cal = nullptr;
    lpdm::LpdmEngine* lpdm = nullptr;
    io::AtmosphericField3D atmo;           // Se ignora: el campo ERA5 reside en el motor (set_atmosphere), cargado una sola vez
    std::vector<float> epi_dist_km;        // distancia al epicentro por posición de celda

    std::vector<float> psi;
    std::vector<uint32_t> arr;             // A_{i,t} denso por cell_id
    std::vector<float> adult;              // N_{i,t} denso por cell_id
    std::vector<float> dsum;               // suma de D_{j,t} de las cohortes de la celda
    std::vector<uint16_t> dcnt;            // número de cohortes de la celda (para D_barra)
    std::vector<int32_t> touched;          // cell_ids tocados (limpieza dispersa)

    ChainTelemetry tel;                    // contadores acumulados (ABM / costo)

    // Modelo de capacidad de carga y posición de cada parámetro dentro de theta (-1 si no aplica)
    abm::CarryingModel model = abm::CarryingModel::M1;
    int i_tc = -1, i_alpha = -1, i_beta = -1, i_k0 = -1, i_gamma_hab = -1, i_q_balloon = -1, i_q_viab = -1,
        i_a0 = -1, i_b0 = -1, i_g1 = -1, i_g2 = -1, i_g3 = -1;
    size_t n_atmospheres = 0;              // campos ERA5 anuales residentes en el motor LPDM
};

void bind_param_indices(EvalContext& ctx, const std::vector<CalibrationParamDef>& defs) {
    ctx.i_tc = calibration_param_index(defs, "t_c");
    ctx.i_alpha = calibration_param_index(defs, "alpha_damage");
    ctx.i_beta = calibration_param_index(defs, "beta_damage");
    ctx.i_k0 = calibration_param_index(defs, "k0");
    ctx.i_gamma_hab = calibration_param_index(defs, "gamma_hab");
    ctx.i_q_balloon = calibration_param_index(defs, "q_balloon");
    ctx.i_q_viab = calibration_param_index(defs, "q_viab");
    ctx.i_a0 = calibration_param_index(defs, "a0");
    ctx.i_b0 = calibration_param_index(defs, "b0");
    ctx.i_g1 = calibration_param_index(defs, "gamma1_imp");
    ctx.i_g2 = calibration_param_index(defs, "gamma2_borde");
    ctx.i_g3 = calibration_param_index(defs, "gamma3_dosel");
}

CostComponents evaluate_candidate(
    EvalContext& ctx,
    const std::vector<float>& theta,
    const CostReferenceScales& scales,
    const CostWeights& weights)
{
    const auto& grid = *ctx.grid;
    const auto& cal = *ctx.cal;

    // 1. Parámetros ecofisiológicos y demográficos del ABM
    abm::DemographicParameters p;
    p.fecundity_F   = 450.0f; // Fijo empírico (Morris, baja sensibilidad)
    p.q10           = 2.2f;   // Fijo empírico
    p.t_c           = theta[ctx.i_tc];
    p.alpha_damage  = theta[ctx.i_alpha];
    p.beta_damage   = std::clamp(static_cast<int>(std::lround(theta[ctx.i_beta])), 1, 3); // entero estricto
    p.q_balloon     = theta[ctx.i_q_balloon];
    p.q_viab        = theta[ctx.i_q_viab];
    p.k0            = theta[ctx.i_k0];
    if (ctx.model == abm::CarryingModel::M3) {
        p.use_hab_index = false;
        p.gamma1_imp    = theta[ctx.i_g1];
        p.gamma2_borde  = theta[ctx.i_g2];
        p.gamma3_dosel  = theta[ctx.i_g3];
    } else {
        p.use_hab_index = true;                       // K_i = K0 exp(gamma_hab H_hab); M0 ⇒ gamma_hab = 0
        p.gamma_hab     = (ctx.model == abm::CarryingModel::M0) ? 0.0f : theta[ctx.i_gamma_hab];
    }

    // Enlace de ocupación (Sección 9): a0 (colonización) y b0 (persistencia) son calibrables
    OccupancyParameters occ;
    occ.a0 = theta[ctx.i_a0];
    occ.b0 = theta[ctx.i_b0];

    abm::AbmEngine abm(p);
    abm.set_cell_index(ctx.cidx);
    abm.initialize_seed(grid, 0.15f);

    ctx.psi = cal.psi_init;

    std::vector<double> sim_front_per_year;
    sim_front_per_year.reserve(kNYears);

    BalancedBrierAccumulator arrival_acc, persistence_acc;

    for (int yi = 0; yi < kNYears; ++yi) {
        const int year = kFirstYear + yi;
        const io::DailyThermalGrid& th = (*ctx.thermal)[static_cast<size_t>(yi)];

        // ── ABM: invierno + eclosión + baloneo (OpenMP) ──
        auto t_abm0 = std::chrono::steady_clock::now();
        auto spring = abm.step_spring_phenology(grid, th, year);

        // D_barra_{i,t}: daño invernal medio de las cohortes de cada celda (tras la mortalidad de invierno)
        const auto& coh = abm.cohorts();
        for (size_t i = 0; i < coh.size(); ++i) {
            const int32_t cid = coh.cell_id[i];
            if (cid < 0 || static_cast<size_t>(cid) >= ctx.dsum.size()) continue;
            if (ctx.dcnt[cid] == 0) ctx.touched.push_back(cid);
            ctx.dsum[cid] += coh.cold_damage_D[i];
            if (ctx.dcnt[cid] < 65535) ctx.dcnt[cid]++;
        }
        ctx.tel.abm_time_s += elapsed_s(t_abm0);

        // ── LPDM en la GPU (Device USM persistente): ERA5 REAL del año simulado (2014+yi) ──
        if (ctx.n_atmospheres > 0) ctx.lpdm->select_atmosphere(std::min(static_cast<size_t>(yi), ctx.n_atmospheres - 1));
        ctx.lpdm->initialize_particles(grid, spring.ballooners_per_cell);
        auto disp = ctx.lpdm->run_dispersion(ctx.atmo, kLpdmDt, kLpdmSteps);

        // ── ABM: llegadas, desarrollo Q10 y reproducción Beverton-Holt ──
        t_abm0 = std::chrono::steady_clock::now();
        abm.deposit_arrivals(disp.arrivals_per_cell);
        auto fall = abm.step_summer_autumn_phenology(grid, th, year);
        ctx.tel.abm_time_s += elapsed_s(t_abm0);

        // ── Frente simulado: P95 de la distancia al epicentro de las celdas con N > 2 ──
        const auto t_cost0 = std::chrono::steady_clock::now();
        std::vector<double> distances;
        distances.reserve(fall.adult_density_per_cell.size());
        for (const auto& [cell_id, count] : fall.adult_density_per_cell) {
            if (count > 2.0f) {
                const int32_t pos = ctx.cidx->find(cell_id);
                if (pos >= 0) distances.push_back(ctx.epi_dist_km[static_cast<size_t>(pos)]);
            }
        }
        sim_front_per_year.push_back(distances.empty() ? 5.0 : percentile95(distances));

        // ── Volcado disperso a arreglos densos A_{i,t}, N_{i,t} ──
        for (const auto& [cid, cnt] : disp.arrivals_per_cell) {
            if (cid >= 0 && static_cast<size_t>(cid) < ctx.arr.size()) { ctx.arr[cid] = cnt; ctx.touched.push_back(cid); }
        }
        for (const auto& [cid, n] : fall.adult_density_per_cell) {
            if (cid >= 0 && static_cast<size_t>(cid) < ctx.adult.size()) { ctx.adult[cid] = n; ctx.touched.push_back(cid); }
        }

        // ── Ocupación latente, detección TGB y Brier estratificado (OpenMP, 16 hilos) ──
        double s_ap = 0.0, s_an = 0.0, s_pp = 0.0, s_pn = 0.0;
        uint64_t n_ap = 0, n_an = 0, n_pp = 0, n_pn = 0;
        const size_t rb = cal.year_begin[yi], re = cal.year_begin[yi + 1];
        float* psi = ctx.psi.data();

        #pragma omp parallel for schedule(static) reduction(+:s_ap, s_an, s_pp, s_pn, n_ap, n_an, n_pp, n_pn)
        for (size_t r = rb; r < re; ++r) {
            const int32_t cid = cal.cell_id[r];
            const float D = ctx.dcnt[cid] ? ctx.dsum[cid] / static_cast<float>(ctx.dcnt[cid]) : 0.0f;

            // logit(gamma) = a0 + a1 log(1 + A) + a2 H ;  logit(phi) = b0 + b1 log(1 + N) - b2 D
            const float gamma = compute_colonization_gamma(ctx.arr[cid], cal.h_norm[r], occ, cal.r_norm[r], cal.in_envelope[r] != 0);
            const float phi = compute_persistence_phi(ctx.adult[cid], D, occ);
            const int32_t s = cal.slot[r];
            // psi_t = (1 - psi_{t-1}) gamma + psi_{t-1} phi   (cada slot aparece una sola vez por año)
            const float psi_cur = update_dynamic_occupancy(psi[s], gamma, phi);
            psi[s] = psi_cur;

            const uint8_t c = cal.cat[r];
            if (c == CAT_NONE) continue;
            // P_hat(Y=1) = psi * p
            const double pm = marginal_observation_prob(psi_cur, cal.p_det[r]);
            switch (c) {
                case CAT_ARR_POS: s_ap += (1.0 - pm) * (1.0 - pm); ++n_ap; break;
                case CAT_ARR_NEG: s_an += pm * pm;                 ++n_an; break;
                case CAT_PER_POS: s_pp += (1.0 - pm) * (1.0 - pm); ++n_pp; break;
                default:          s_pn += pm * pm;                 ++n_pn; break;
            }
        }
        arrival_acc.sum_pos += s_ap; arrival_acc.sum_neg += s_an; arrival_acc.n_pos += n_ap; arrival_acc.n_neg += n_an;
        persistence_acc.sum_pos += s_pp; persistence_acc.sum_neg += s_pn; persistence_acc.n_pos += n_pp; persistence_acc.n_neg += n_pn;

        // Limpieza dispersa de los arreglos densos
        for (int32_t cid : ctx.touched) { ctx.arr[cid] = 0; ctx.adult[cid] = 0.0f; ctx.dsum[cid] = 0.0f; ctx.dcnt[cid] = 0; }
        ctx.touched.clear();
        ctx.tel.cost_time_s += elapsed_s(t_cost0);
    }

    ctx.tel.evaluations++;
    return finalize_multicriteria_cost(arrival_acc.value(), sim_front_per_year, cal.obs_front_km,
                                       persistence_acc.value(), weights, scales);
}

/// Telemetría absoluta combinada (contadores del contexto + motor LPDM)
ChainTelemetry snapshot(const EvalContext& ctx) {
    ChainTelemetry t = ctx.tel;
    const auto e = ctx.lpdm->telemetry();
    t.lpdm_init_time_s = e.init_time_s;
    t.lpdm_h2d_time_s = e.h2d_time_s;
    t.lpdm_kernel_time_s = e.kernel_time_s;
    t.lpdm_d2h_time_s = e.d2h_time_s;
    t.lpdm_reduce_time_s = e.reduce_time_s;
    t.particles_dispatched = e.particles_dispatched;
    t.cpu_fallbacks = e.cpu_fallbacks;
    t.capped_dispatches = e.capped_dispatches;
    return t;
}

struct EvalRecord {
    int chain_id = 0;
    int iteration = 0;
    double temperature = 0.0;
    bool accepted = false;
    CostComponents comp;
    std::vector<float> theta;
};

} // namespace

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

SimulatedAnnealingReport run_multi_start_simulated_annealing(
    const std::vector<GridCell>& grid_cells,
    const std::vector<io::DailyThermalGrid>& thermal_by_year,
    const std::vector<CalibrationObservation>& observations,
    lpdm::LpdmEngine* lpdm_engine,
    int n_chains,
    int iter_per_chain,
    double initial_temp,
    double delta_threshold,
    const std::string& checkpoint_csv_path,
    abm::CarryingModel model)
{
    SimulatedAnnealingReport report;
    const std::vector<CalibrationParamDef>& defs = calibration_params_for(model);
    report.model = model;
    report.params = defs;
    report.n_chains = n_chains;
    report.iter_per_chain = iter_per_chain;
    report.total_evaluations = n_chains * iter_per_chain;
    report.omp_max_threads = omp_get_max_threads();

    if (lpdm_engine == nullptr) throw std::invalid_argument("Se requiere un motor LPDM (GPU SYCL).");
    if (thermal_by_year.size() < static_cast<size_t>(kNYears)) {
        throw std::invalid_argument("Se requieren 6 grillas térmicas diarias (2014-2019).");
    }
    // La GPU es obligatoria en cada evaluación; solo se admite CPU de forma explícita para depuración.
    const char* allow_cpu = std::getenv("JORO_ALLOW_CPU_FALLBACK");
    const bool cpu_ok = allow_cpu != nullptr && std::string(allow_cpu) == "1";
    report.device_description = lpdm_engine->device_description();
    if (!lpdm_engine->is_gpu_offload_active() && !cpu_ok) {
        throw std::runtime_error("El motor LPDM no está en una GPU SYCL (" + report.device_description +
                                 "). Defina JORO_ALLOW_CPU_FALLBACK=1 solo para depuración.");
    }

    const int n_params = static_cast<int>(defs.size());
    const char* env_threads = std::getenv("OMP_NUM_THREADS");
    const char* env_bind = std::getenv("OMP_PROC_BIND");
    const char* env_places = std::getenv("OMP_PLACES");

    std::cout << "\n===================================================================\n";
    std::cout << "▶ Invocando Recocido Simulado Multi-inicio (Simulated Annealing)\n";
    std::cout << "  Cadenas independientes  : " << n_chains << " multi-start (ejecución SECUENCIAL)\n";
    std::cout << "  Iteraciones por cadena  : " << iter_per_chain << " iter\n";
    std::cout << "  Total evaluaciones 2014-2019 : " << report.total_evaluations << "\n";
    std::cout << "  Espacio de parámetros   : " << n_params << " dimensiones dominantes (beta_s discreto {1,2,3}) | modelo K_i = " << abm::carrying_model_name(model) << "\n";
    std::cout << "  Temperatura inicial T0  : " << initial_temp << "\n";
    std::cout << "  Tolerancia Delta (Theta_comp) : " << delta_threshold << "\n";
    std::cout << "  Dispositivo LPDM        : " << report.device_description << "\n";
    std::cout << "  Hilos OpenMP            : " << report.omp_max_threads
              << " | OMP_NUM_THREADS=" << (env_threads ? env_threads : "(no definida)")
              << " OMP_PROC_BIND=" << (env_bind ? env_bind : "(no definida)")
              << " OMP_PLACES=" << (env_places ? env_places : "(no definida)") << "\n";
    std::cout << "===================================================================\n";

    const auto t0 = std::chrono::steady_clock::now();

    // 0. Preparación única: conjunto de calibración estratificado, índice de celdas, espacio de trabajo
    PreparedCalibration cal = prepare_calibration(observations, grid_cells);
    report.stratification = cal.strat;
    {
        const auto& s = cal.strat;
        std::cout << "[Calibración] Filas celda-año (2014-2019): " << s.n_rows << " | v_evaluation_set: 0 filas cargadas (bloqueada)\n";
        std::cout << "[Calibración] Llegada: " << s.arrival_pos << " positivos / " << s.arrival_neg
                  << " negativos en la envolvente activa (" << s.outside_envelope << " ausencias lejanas excluidas)\n";
        std::cout << "[Calibración] Persistencia: " << s.persistence_pos << " positivos / " << s.persistence_neg << " negativos\n";
        std::cout << "[Calibración] Radio de envolvente por año [km]:";
        for (double r : s.envelope_radius_km) std::cout << " " << std::fixed << std::setprecision(1) << r;
        std::cout << "\n[Calibración] Frente observado P95 por año [km]:";
        for (double f : s.observed_front_km) std::cout << " " << std::fixed << std::setprecision(1) << f;
        std::cout << "\n[Calibración] Esfuerzo TGB: " << (s.effort_synthetic ? "SINTÉTICO detectado -> imputado en " + std::to_string(s.effort_imputed_rows) + " filas con detección"
                                                                            : std::string("conteos reales de Araneae (sin imputación)")) << "\n";
        std::cout << "[Calibración] Transporte LPDM: " << (lpdm_engine->is_atmosphere_resident() ? "viento/BLH ERA5 residentes en el dispositivo"
                                                                                          : "viento uniforme heredado (ERA5 NO cargado)") << "\n";
    }

    abm::CellIndexTable cidx = abm::CellIndexTable::build(grid_cells);
    EvalContext ctx;
    ctx.grid = &grid_cells;
    ctx.cidx = &cidx;
    ctx.thermal = &thermal_by_year;
    ctx.cal = &cal;
    ctx.lpdm = lpdm_engine;
    ctx.model = model;
    bind_param_indices(ctx, defs);
    ctx.n_atmospheres = lpdm_engine->atmosphere_count();
    report.atmosphere_slots = ctx.n_atmospheres;
    ctx.atmo.n_times = kLpdmSteps;
    ctx.arr.assign(cidx.idx_of.size(), 0);
    ctx.adult.assign(cidx.idx_of.size(), 0.0f);
    ctx.dsum.assign(cidx.idx_of.size(), 0.0f);
    ctx.dcnt.assign(cidx.idx_of.size(), 0);
    ctx.epi_dist_km.resize(grid_cells.size());
    for (size_t i = 0; i < grid_cells.size(); ++i) {
        ctx.epi_dist_km[i] = static_cast<float>(dist_to_epicenter_km(grid_cells[i].centroid_lat, grid_cells[i].centroid_lon));
    }

    // 1. Escalas de referencia en el punto nominal (theta_default); sirve además de calentamiento de la GPU
    std::vector<float> theta_nominal(n_params);
    for (int i = 0; i < n_params; ++i) theta_nominal[i] = defs[i].default_val;

    std::cout << "[Simulated Annealing] Evaluando escala de referencia inicial en theta_nominal...\n";
    CostReferenceScales unit_scales{1.0, 1.0, 1.0};
    CostWeights weights_base{0.333333, 0.333333, 0.333334};
    CostComponents nom_comp = evaluate_candidate(ctx, theta_nominal, unit_scales, weights_base);

    CostReferenceScales reference_scales{
        std::max(nom_comp.j_arrival_raw, 0.0001),
        std::max(nom_comp.j_front_raw, 1.0),
        std::max(nom_comp.j_persistence_raw, 0.01)
    };
    report.reference_scales = reference_scales;
    report.nominal_components = nom_comp;

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "  Escala Brier balanceado S_llegada     : " << reference_scales.s_arrival << "\n";
    std::cout << "  Escala Frente S_frente (km)           : " << reference_scales.s_front << "\n";
    std::cout << "  Escala Brier balanceado S_persistencia: " << reference_scales.s_persistence << "\n";

    // Configuraciones de pesos para análisis de sensibilidad
    CostWeights weights_config_a{0.20, 0.60, 0.20}; // Enfoque Frente/Dispersión
    CostWeights weights_config_b{0.20, 0.20, 0.60}; // Enfoque Persistencia/Resistencia Térmica
    auto weighted = [](const CostWeights& w, const CostComponents& c) {
        return w.w1_arrival * c.j_arrival_norm + w.w2_front * c.j_front_norm + w.w3_persistence * c.j_persistence_norm;
    };

    // Registro de TODAS las evaluaciones (para Theta_comp y sensibilidad de pesos)
    std::vector<EvalRecord> all_evals;
    all_evals.reserve(static_cast<size_t>(n_chains) * (iter_per_chain + 1));
    report.chains.resize(n_chains);

    const std::vector<uint32_t> chain_seeds = {1001, 2002, 3003, 4004, 5005};

    // Factor de enfriamiento geométrico T_k = T0 * alpha^k
    const double cooling_alpha = std::pow(0.001 / initial_temp, 1.0 / iter_per_chain);

    // 2. Cadenas SECUENCIALES: cada evaluación dispone del 100 % de la GPU y de los 16 hilos del host
    for (int c = 0; c < n_chains; ++c) {
        const auto c_t0 = std::chrono::steady_clock::now();
        const ChainTelemetry tel_start = snapshot(ctx);

        const uint32_t seed = chain_seeds[static_cast<size_t>(c) % chain_seeds.size()];
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist_01(0.0, 1.0);
        std::normal_distribution<double> norm_dist(0.0, 1.0);

        ChainConvergenceRecord& rec = report.chains[c];
        rec.chain_id = c + 1;
        rec.seed = seed;
        rec.total_iterations = iter_per_chain;
        rec.trace.reserve(static_cast<size_t>(iter_per_chain) + 1);

        // Estado inicial: cadena 1 en theta_nominal; las demás, muestreo uniforme en el hipercubo
        std::vector<float> curr_theta(n_params);
        if (c == 0) {
            curr_theta = theta_nominal;
        } else {
            for (int i = 0; i < n_params; ++i) {
                const float min_v = defs[i].min_val;
                const float max_v = defs[i].max_val;
                if (i == kBetaDamageIndex) {
                    curr_theta[i] = static_cast<float>(1 + std::min(2, static_cast<int>(dist_01(rng) * 3.0))); // {1,2,3}
                } else {
                    curr_theta[i] = min_v + static_cast<float>(dist_01(rng)) * (max_v - min_v);
                }
            }
        }

        CostComponents curr_comp = evaluate_candidate(ctx, curr_theta, reference_scales, weights_base);
        rec.initial_cost = curr_comp.j_total;
        rec.initial_components = curr_comp;
        rec.min_cost = curr_comp.j_total;
        rec.best_theta = curr_theta;
        rec.best_components = curr_comp;
        all_evals.push_back({c + 1, 0, initial_temp, true, curr_comp, curr_theta});
        rec.trace.push_back({0.0, initial_temp, curr_comp.j_total, rec.min_cost});

        double temp = initial_temp;

        for (int iter = 0; iter < iter_per_chain; ++iter) {
            // Propuesta: perturbación gaussiana escalada por temperatura en los parámetros continuos;
            // salto discreto entre {1,2,3} para beta_s.
            std::vector<float> prop_theta(n_params);
            const double temp_ratio = std::sqrt(temp / initial_temp);

            for (int i = 0; i < n_params; ++i) {
                if (i == kBetaDamageIndex) {
                    const int cur_b = std::clamp(static_cast<int>(std::lround(curr_theta[i])), 1, 3);
                    int new_b = cur_b;
                    if (dist_01(rng) < 0.30) {
                        // Salta a uno de los otros dos valores con igual probabilidad
                        new_b = ((cur_b - 1) + (dist_01(rng) < 0.5 ? 1 : 2)) % 3 + 1;
                    }
                    prop_theta[i] = static_cast<float>(new_b);
                    continue;
                }
                const float span = defs[i].max_val - defs[i].min_val;
                const double step = norm_dist(rng) * 0.12 * span * temp_ratio;
                const float val = curr_theta[i] + static_cast<float>(step);
                prop_theta[i] = std::clamp(val, defs[i].min_val, defs[i].max_val);
            }

            CostComponents prop_comp = evaluate_candidate(ctx, prop_theta, reference_scales, weights_base);

            const double delta_j = prop_comp.j_total - curr_comp.j_total;
            bool accept = (delta_j <= 0.0) || (dist_01(rng) < std::exp(-delta_j / temp));

            all_evals.push_back({c + 1, iter + 1, temp, accept, prop_comp, prop_theta});

            if (accept) {
                curr_theta = prop_theta;
                curr_comp = prop_comp;
                rec.accepted_proposals++;
                if (curr_comp.j_total < rec.min_cost) {
                    rec.min_cost = curr_comp.j_total;
                    rec.best_theta = curr_theta;
                    rec.best_components = curr_comp;
                }
            }

            temp *= cooling_alpha;
            rec.trace.push_back({static_cast<double>(iter + 1), temp, curr_comp.j_total, rec.min_cost});

            if ((iter + 1) % 100 == 0) {
                const double el = elapsed_s(c_t0);
                std::cout << "  [Cadena " << (c + 1) << "] iter " << std::setw(3) << (iter + 1) << "/" << iter_per_chain
                          << " | T = " << std::setprecision(4) << temp
                          << " | J = " << curr_comp.j_total << " | J* = " << rec.min_cost
                          << " | " << std::setprecision(2) << el << " s (" << (static_cast<double>(iter + 2) / el) << " eval/s)\n"
                          << std::flush;
            }
        }

        rec.execution_time_s = elapsed_s(c_t0);
        rec.telemetry = snapshot(ctx) - tel_start;

        std::cout << "[Cadena " << rec.chain_id << "] Finalizada en "
                  << std::fixed << std::setprecision(2) << rec.execution_time_s << " s | "
                  << "J0 = " << std::setprecision(4) << rec.initial_cost << " -> "
                  << "Jmin = " << rec.min_cost << " | "
                  << "Aceptadas = " << rec.accepted_proposals << "/" << iter_per_chain << " ("
                  << (rec.accepted_proposals * 100 / iter_per_chain) << "%)\n";
    }

    report.total_runtime_s = elapsed_s(t0);
    report.telemetry_total = snapshot(ctx);

    // 3. Mejor solución global (MAP / J_min)
    double global_min = 1e6;
    int best_chain_idx = 0;
    for (size_t c = 0; c < report.chains.size(); ++c) {
        if (report.chains[c].min_cost < global_min) {
            global_min = report.chains[c].min_cost;
            best_chain_idx = static_cast<int>(c);
        }
    }
    report.global_min_cost = global_min;
    report.map_theta = report.chains[best_chain_idx].best_theta;
    report.map_components = report.chains[best_chain_idx].best_components;

    // 4. Conjunto compatible Theta_comp = { theta evaluado : J(theta) <= J_min + Delta }
    //    sobre TODAS las evaluaciones (aceptadas o no).
    const double j_threshold = global_min + delta_threshold;
    std::vector<size_t> comp_idx;
    for (size_t k = 0; k < all_evals.size(); ++k) {
        if (all_evals[k].comp.j_total <= j_threshold) comp_idx.push_back(k);
    }
    for (size_t k : comp_idx) {
        const auto& e = all_evals[k];
        CompatibleEnsembleMember m;
        m.chain_id = e.chain_id;
        m.iteration = e.iteration;
        m.temperature = e.temperature;
        m.j_total = e.comp.j_total;
        m.j_arrival_norm = e.comp.j_arrival_norm;
        m.j_front_norm = e.comp.j_front_norm;
        m.j_persistence_norm = e.comp.j_persistence_norm;
        m.j_config_a = weighted(weights_config_a, e.comp);
        m.j_config_b = weighted(weights_config_b, e.comp);
        m.theta = e.theta;
        report.compatible_ensemble.push_back(std::move(m));
    }

    // 5. Sensibilidad a pesos: Theta_comp^A, Theta_comp^B se reconstruyen sobre todas las evaluaciones
    //    con su propio J_min; la retención es la fracción de Theta_comp (base) que permanece en ellos.
    if (!comp_idx.empty()) {
        double min_a = 1e9, min_b = 1e9;
        for (const auto& e : all_evals) {
            min_a = std::min(min_a, weighted(weights_config_a, e.comp));
            min_b = std::min(min_b, weighted(weights_config_b, e.comp));
        }
        report.j_min_config_a = min_a;
        report.j_min_config_b = min_b;
        size_t keep_a = 0, keep_b = 0;
        for (size_t k : comp_idx) {
            if (weighted(weights_config_a, all_evals[k].comp) <= min_a + delta_threshold) ++keep_a;
            if (weighted(weights_config_b, all_evals[k].comp) <= min_b + delta_threshold) ++keep_b;
        }
        report.retention_rate_config_a = 100.0 * static_cast<double>(keep_a) / static_cast<double>(comp_idx.size());
        report.retention_rate_config_b = 100.0 * static_cast<double>(keep_b) / static_cast<double>(comp_idx.size());
    }

    // 6. Percentiles P5, Mediana, P95 en Theta_comp
    report.p5_theta.assign(n_params, 0.0f);
    report.median_theta.assign(n_params, 0.0f);
    report.p95_theta.assign(n_params, 0.0f);
    if (!report.compatible_ensemble.empty()) {
        for (int p_idx = 0; p_idx < n_params; ++p_idx) {
            std::vector<float> vals;
            vals.reserve(report.compatible_ensemble.size());
            for (const auto& mem : report.compatible_ensemble) vals.push_back(mem.theta[p_idx]);
            std::sort(vals.begin(), vals.end());
            const size_t n = vals.size();
            report.p5_theta[p_idx] = vals[static_cast<size_t>(0.05 * (n - 1))];
            report.median_theta[p_idx] = vals[static_cast<size_t>(0.50 * (n - 1))];
            report.p95_theta[p_idx] = vals[static_cast<size_t>(0.95 * (n - 1))];
        }
    }

    // 7. Persistencia en disco: ensamble compatible y traza de convergencia
    abm::CarryingCapacityParams cc_base{};
    cc_base.k_min = 100.0f;
    cc_base.k_max = 50000.0f;
    const GridCell* seed_cell = nullptr;
    for (const auto& gc : grid_cells) if (gc.is_seed) { seed_cell = &gc; break; }

    std::filesystem::path ckpt_path(checkpoint_csv_path);
    std::error_code ec;
    std::filesystem::create_directories(ckpt_path.parent_path(), ec);

    std::ofstream out_file(checkpoint_csv_path);
    if (out_file.is_open()) {
        out_file << "chain_id,iteration,temperature,j_total,j_arrival_norm,j_front_norm,j_persistence_norm,j_config_a,j_config_b";
        for (int i = 0; i < n_params; ++i) out_file << "," << defs[i].name;
        out_file << ",k_eff_seed\n";
        out_file << std::setprecision(7);
        for (const auto& mem : report.compatible_ensemble) {
            out_file << mem.chain_id << "," << mem.iteration << "," << mem.temperature << ","
                     << mem.j_total << "," << mem.j_arrival_norm << "," << mem.j_front_norm << ","
                     << mem.j_persistence_norm << "," << mem.j_config_a << "," << mem.j_config_b;
            for (int i = 0; i < n_params; ++i) out_file << "," << mem.theta[i];
            // Capacidad efectiva K_i de la celda epicentral bajo este theta
            float k_eff = 0.0f;
            if (seed_cell != nullptr) {
                abm::CarryingCapacityParams cc = cc_base;
                cc.k0 = mem.theta[static_cast<size_t>(ctx.i_k0)];
                if (model == abm::CarryingModel::M3) {
                    cc.gamma1_imp = mem.theta[static_cast<size_t>(ctx.i_g1)];
                    cc.gamma2_borde = mem.theta[static_cast<size_t>(ctx.i_g2)];
                    cc.gamma3_dosel = mem.theta[static_cast<size_t>(ctx.i_g3)];
                } else {
                    cc.use_hab_index = true;
                    cc.gamma_hab = (model == abm::CarryingModel::M0) ? 0.0f : mem.theta[static_cast<size_t>(ctx.i_gamma_hab)];
                }
                k_eff = abm::compute_carrying_capacity(*seed_cell, cc);
            }
            out_file << "," << k_eff << "\n";
        }
        out_file.close();
        std::cout << "[Simulated Annealing] Ensamble Theta_comp (" << report.compatible_ensemble.size()
                  << " miembros) guardado en: " << checkpoint_csv_path << "\n";
    }

    const std::filesystem::path trace_path = ckpt_path.parent_path() / "sa_convergence_trace.csv";
    std::ofstream trace_file(trace_path);
    if (trace_file.is_open()) {
        trace_file << "chain_id,iteration,temperature,j_current,j_best\n" << std::setprecision(7);
        for (const auto& ch : report.chains) {
            for (const auto& row : ch.trace) {
                trace_file << ch.chain_id << "," << static_cast<int>(row[0]) << "," << row[1] << "," << row[2] << "," << row[3] << "\n";
            }
        }
        std::cout << "[Simulated Annealing] Curva de convergencia guardada en: " << trace_path.string() << "\n";
    }

    return report;
}

} // namespace joro::inference
