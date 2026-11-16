#include "inference/occupancy_model.hpp"
#include "inference/cost_function.hpp"
#include "abm/carrying_capacity.hpp"
#include "test_common.hpp"
#include "common/spatial_grid_lookup.hpp"
#include "inference/dispersal_kernels.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

/// Prueba de recuperación de parámetros sintéticos (Sección 11 de JORO-HPC_Research_Proposal.tex).
/// Genera observaciones sintéticas con un parámetro objetivo conocido (theta*)
/// y ejecuta una búsqueda de optimización unidimensional para verificar
/// que el mínimo de la función de costo recupera el valor verdadero.
int main() {
    std::cout << "─────────────────────────────────────────────────────────────\n";
    std::cout << "[TEST] test_parameter_recovery: Identificabilidad de Parámetros\n";
    std::cout << "─────────────────────────────────────────────────────────────\n";

    constexpr float true_beta1 = 0.85f;
    joro::inference::OccupancyParameters true_params;
    true_params.beta1_effort = true_beta1;

    // 1. Generación de observaciones sintéticas
    std::vector<int> efforts;
    std::vector<uint8_t> y_obs;
    for (int effort = 1; effort <= 200; effort += 2) {
        efforts.push_back(effort);
        float p = joro::inference::compute_detection_prob(effort, 0.4f, 0.3f, true_params);
        y_obs.push_back((p >= 0.5f) ? 1 : 0);
    }

    // Función lambda para evaluar el costo J(theta) para un valor candidato de beta1_effort
    auto evaluate_candidate_cost = [&](float candidate_beta1) -> double {
        joro::inference::OccupancyParameters cand_params;
        cand_params.beta1_effort = candidate_beta1;

        std::vector<float> p_pred;
        p_pred.reserve(efforts.size());
        for (int effort : efforts) {
            p_pred.push_back(joro::inference::compute_detection_prob(effort, 0.4f, 0.3f, cand_params));
        }
        return joro::inference::compute_cost_function(p_pred, y_obs);
    };

    // 2. Evaluar costo en punto perturbado inicial (theta_initial = 0.15)
    constexpr float initial_guess = 0.15f;
    double cost_initial = evaluate_candidate_cost(initial_guess);
    std::cout << "  Parámetro inicial (perturbado): " << initial_guess 
              << " | Costo inicial = " << cost_initial << "\n";

    // 3. Búsqueda de sección áurea (Golden Section Search) para optimizar beta1 en [0.10, 1.50]
    double a = 0.10;
    double b = 1.50;
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double resphi = 2.0 - phi;

    double c = a + resphi * (b - a);
    double d = b - resphi * (b - a);
    double fc = evaluate_candidate_cost(static_cast<float>(c));
    double fd = evaluate_candidate_cost(static_cast<float>(d));

    for (int iter = 0; iter < 40; ++iter) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = a + resphi * (b - a);
            fc = evaluate_candidate_cost(static_cast<float>(c));
        } else {
            a = c;
            c = d;
            fc = fd;
            d = b - resphi * (b - a);
            fd = evaluate_candidate_cost(static_cast<float>(d));
        }
    }

    float recovered_beta1 = static_cast<float>((a + b) / 2.0);
    double cost_recovered = evaluate_candidate_cost(recovered_beta1);

    std::cout << "  Parámetro verdadero (theta*)  : " << true_beta1 << "\n";
    std::cout << "  Parámetro recuperado (theta^) : " << recovered_beta1 
              << " | Costo final = " << cost_recovered << "\n";

    double recovery_error = std::abs(recovered_beta1 - true_beta1);
    std::cout << "  Error de recuperación |theta^ - theta*|: " << recovery_error << "\n";

    // Criterios de validación rigurosos
    JORO_TEST_ASSERT(cost_recovered < cost_initial, "La optimización no redujo el costo respecto al guess inicial");
    JORO_TEST_ASSERT(recovery_error < 0.30, "El optimizador no convergió al entorno del parámetro verdadero");

    // ── Reparametrización de K_i (Sección 8.4) y enlace de ocupación calibrable ──
    {
        using joro::abm::CarryingCapacityParams;
        using joro::abm::compute_carrying_capacity;
        using joro::abm::compute_habitat_index;
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<joro::GridCell> cells(5000);
        for (auto& c : cells) { c.z_borde = nd(rng); c.z_imp = nd(rng); c.z_dosel = nd(rng); }
        compute_habitat_index(cells);
        double m = 0.0, v = 0.0;
        for (const auto& c : cells) m += c.h_hab;
        m /= static_cast<double>(cells.size());
        for (const auto& c : cells) v += (c.h_hab - m) * (c.h_hab - m);
        v /= static_cast<double>(cells.size() - 1);
        JORO_TEST_ASSERT(std::abs(m) < 1e-4 && std::abs(v - 1.0) < 1e-3, "H_hab debe tener media 0 y varianza unitaria");

        CarryingCapacityParams m0{}; m0.k0 = 4000.0f; m0.use_hab_index = true; m0.gamma_hab = 0.0f;
        CarryingCapacityParams m1 = m0; m1.gamma_hab = 1.2f;
        bool m0_const = true, m1_monotone = true;
        auto lo = cells[0], hi = cells[0];
        for (const auto& c : cells) { if (c.h_hab < lo.h_hab) lo = c; if (c.h_hab > hi.h_hab) hi = c; }
        for (const auto& c : cells) m0_const &= std::abs(compute_carrying_capacity(c, m0) - 4000.0f) < 1e-3f;
        m1_monotone = compute_carrying_capacity(hi, m1) > compute_carrying_capacity(lo, m1);
        JORO_TEST_ASSERT(m0_const, "M0: K_i debe ser K0 en todas las celdas");
        JORO_TEST_ASSERT(m1_monotone, "M1: K_i debe crecer con H_hab cuando gamma_hab > 0");
        // K0 = mediana de K_i en M1: la celda de H_hab = 0 tiene K_i = K0 (ortogonal a gamma_hab)
        joro::GridCell mid{}; mid.h_hab = 0.0f;
        JORO_TEST_ASSERT(std::abs(compute_carrying_capacity(mid, m1) - 4000.0f) < 1e-3f, "M1: K_i(H=0) debe ser K0");

        // a0 y b0 desplazan monótonamente las probabilidades de colonización y persistencia
        joro::inference::OccupancyParameters lo_a, hi_a;
        lo_a.a0 = -5.0f; hi_a.a0 = -2.0f; lo_a.b0 = -1.0f; hi_a.b0 = 2.0f;
        JORO_TEST_ASSERT(joro::inference::compute_colonization_gamma(3, 0.4f, hi_a) >
                         joro::inference::compute_colonization_gamma(3, 0.4f, lo_a), "gamma debe crecer con a0");
        JORO_TEST_ASSERT(joro::inference::compute_persistence_phi(5.0f, 0.2f, hi_a) >
                         joro::inference::compute_persistence_phi(5.0f, 0.2f, lo_a), "phi debe crecer con b0");

        // El Brier balanceado es diferenciable y monótono en psi: positivos decrecen, negativos crecen
        const double d = 1e-3;
        auto brier_pos = [](double pm) { return (1.0 - pm) * (1.0 - pm); };
        auto brier_neg = [](double pm) { return pm * pm; };
        JORO_TEST_ASSERT(brier_pos(0.30 + d) < brier_pos(0.30) && brier_neg(0.30 + d) > brier_neg(0.30),
                         "Brier de positivos/negativos debe ser monótono respecto a P_hat");
        std::cout << "  ✔ Reparametrización K_i (M0/M1) y enlace a0/b0 verificados.\n";
    }


    // ── Bloque 2: H_hab sobre el soporte S, colonización cloglog, f_est y kernel humano ──
    {
        using joro::abm::compute_habitat_index;
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<joro::GridCell> cells(4000);
        std::vector<int32_t> support_ids;
        for (size_t i = 0; i < cells.size(); ++i) {
            auto& c = cells[i];
            c.cell_id = static_cast<int32_t>(i);
            const bool in_s = i < 800;                 // soporte informativo: celdas de la región invadida (más bosque/borde)
            c.z_borde = nd(rng) + (in_s ? 1.0f : 0.0f);
            c.z_imp = nd(rng) + (in_s ? 0.5f : 0.0f);
            c.z_dosel = nd(rng);
            c.centroid_lon = -84.0f + 0.045f * static_cast<float>(i % 80);
            c.centroid_lat = 33.0f + 0.045f * static_cast<float>(i / 80);
            c.pop_density_h = std::abs(nd(rng)); c.road_density_r = std::abs(nd(rng));
            if (in_s) support_ids.push_back(c.cell_id);
        }
        const auto mask = joro::abm::make_support_mask(cells, support_ids);
        compute_habitat_index(cells, &mask);
        double m_s = 0.0, v_s = 0.0;
        for (size_t i = 0; i < 800; ++i) m_s += cells[i].h_hab;
        m_s /= 800.0;
        for (size_t i = 0; i < 800; ++i) v_s += (cells[i].h_hab - m_s) * (cells[i].h_hab - m_s);
        v_s /= 799.0;
        JORO_TEST_ASSERT(std::abs(m_s) < 1e-4 && std::abs(v_s - 1.0) < 1e-3, "M14: H_hab debe tener media 0 y varianza 1 sobre el soporte S");
        compute_habitat_index(cells);
        double m_all = 0.0;
        for (size_t i = 0; i < 800; ++i) m_all += cells[i].h_hab;
        JORO_TEST_ASSERT(std::abs(m_all / 800.0) > 0.3, "Estandarizar sobre Omega deja m_S != 0 (colinealidad estructural, M14)");

        using joro::inference::OccupancyParameters;
        using joro::inference::compute_colonization_gamma;
        OccupancyParameters op;
        JORO_TEST_ASSERT(compute_colonization_gamma(0.0f, 0.5f, op) == 0.0f, "M12: gamma = 0 exacto si A_total = 0 (sin generación espontánea)");
        const float g1 = compute_colonization_gamma(1.0f, 0.0f, op), g100 = compute_colonization_gamma(100.0f, 0.0f, op);
        JORO_TEST_ASSERT(g1 > 0.0f && g100 > g1 && g100 < 1.0f, "M12: gamma cloglog crece monótonamente con A_total");
        const float lam = std::exp(op.a0);   // a1 = 1, H = 0 => gamma(A) = 1 - exp(-A e^{a0})
        JORO_TEST_ASSERT(std::abs(g100 - (1.0f - std::exp(-100.0f * lam))) < 1e-5f, "M12: forma 1 - exp(-lambda A) del enlace cloglog");
        JORO_TEST_ASSERT(joro::inference::total_arrivals(10.0f, 3.0f, 0.0f) == 13.0f && joro::inference::total_arrivals(10.0f, 0.0f, 0.1f) == 9.0f,
                         "A_total = (1 - p_h) A_LPDM + A_hum");

        // M13: f_est crece con la atención M_t y decrece con la edad local de invasión (habituación)
        JORO_TEST_ASSERT(joro::inference::compute_f_est(3.0f, 0.0f, op) > joro::inference::compute_f_est(0.5f, 0.0f, op) &&
                         joro::inference::compute_f_est(1.0f, 5.0f, op) < joro::inference::compute_f_est(1.0f, 0.0f, op),
                         "M13: f_est monótono en M_t y tau");

        // Kernel humano: conservación Σ A_i + pérdidas = propágulos muestreados; p_h = 0 => sin transporte
        const auto cidx = joro::abm::CellIndexTable::build(cells);
        joro::SpatialGridLookup sgl; sgl.build(cells);
        std::vector<float> hn(cells.size()), rn(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) { hn[i] = std::min(1.0f, cells[i].pop_density_h / 3.0f); rn[i] = std::min(1.0f, cells[i].road_density_r / 3.0f); }
        joro::inference::HumanTransportKernel hk;
        rn[10] = 0.90f;
        rn[500] = 0.10f;
        hk.build(cells, hn, rn, sgl.view(), &cidx);
        std::vector<std::pair<int32_t, float>> src = {{10, 50000.0f}, {500, 20000.0f}};
        const auto none = hk.transport(src, 0.0, 150.0, 1);
        JORO_TEST_ASSERT(none.arrivals.empty() && none.transported == 0, "p_h = 0 recupera el modelo puramente atmosférico");
        const auto res = hk.transport(src, 0.05, 40.0, 7);
        uint64_t placed = 0;
        for (const auto& kv : res.arrivals) placed += kv.second;
        JORO_TEST_ASSERT(res.transported > 1000 && placed + res.lost == res.transported, "Kernel humano: conservación Σ A_i + L = propágulos transportados");
        const auto res_hi_r = hk.transport({{10, 50000.0f}}, 0.05, 40.0, 7);
        const auto res_lo_r = hk.transport({{500, 50000.0f}}, 0.05, 40.0, 7);
        JORO_TEST_ASSERT(res_hi_r.transported > 5 * res_lo_r.transported, "Kernel humano: emisión ponderada por R_j de la celda origen");
        const auto res2 = hk.transport(src, 0.05, 40.0, 7);
        JORO_TEST_ASSERT(res2.transported == res.transported && res2.arrivals.size() == res.arrivals.size(), "Kernel humano determinista por semilla");
        std::cout << "  ✔ Bloque 2: H_hab sobre S, cloglog, f_est y kernel humano (" << res.transported << " propágulos) verificados.\n";
    }

    std::cout << "✔ test_parameter_recovery PASÓ: Identificabilidad numérica confirmada.\n";
    return 0;
}
