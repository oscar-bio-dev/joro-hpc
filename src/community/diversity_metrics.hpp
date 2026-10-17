#pragma once

#include <vector>
#include <cmath>
#include <cstdint>

namespace joro::community {

/// Estimador no paramétrico de riqueza Chao2 corregido para muestras pequeñas (Chao 1987)
/// S_Chao2 = S_obs + ( (m - 1) / m ) * ( Q1 * (Q1 - 1) ) / ( 2 * (Q2 + 1) )
[[nodiscard]] inline double compute_chao2(
    uint32_t s_obs,
    uint32_t q1_singletons,
    uint32_t q2_doubletons,
    uint32_t m_samples) noexcept
{
    if (m_samples == 0 || s_obs == 0) return 0.0;
    if (q2_doubletons == 0) {
        // Corrección de sesgo si no hay doubletons
        double correction = (static_cast<double>(m_samples - 1) / m_samples) *
                            (static_cast<double>(q1_singletons) * (q1_singletons - 1)) / 2.0;
        return static_cast<double>(s_obs) + correction;
    }
    double factor = static_cast<double>(m_samples - 1) / m_samples;
    double num = static_cast<double>(q1_singletons) * (q1_singletons - 1);
    double den = 2.0 * (q2_doubletons + 1);
    return static_cast<double>(s_obs) + factor * (num / den);
}

/// Números de Hill (Hill 1973): q=0 (Riqueza), q=1 (Exponencial de Shannon), q=2 (Inverso de Simpson)
[[nodiscard]] inline double compute_hill_number(
    const std::vector<double>& relative_abundances,
    int q_order) noexcept
{
    if (relative_abundances.empty()) return 0.0;

    if (q_order == 0) {
        // Riqueza de especies observada
        double count = 0.0;
        for (double p : relative_abundances) {
            if (p > 1e-9) count += 1.0;
        }
        return count;
    } else if (q_order == 1) {
        // Exponencial de la entropía de Shannon: exp(- sum p_i * ln(p_i))
        double h_shannon = 0.0;
        for (double p : relative_abundances) {
            if (p > 1e-9) {
                h_shannon -= p * std::log(p);
            }
        }
        return std::exp(h_shannon);
    } else if (q_order == 2) {
        // Inverso del índice de Simpson: 1 / sum p_i^2
        double simpson = 0.0;
        for (double p : relative_abundances) {
            simpson += p * p;
        }
        return (simpson > 1e-9) ? (1.0 / simpson) : 0.0;
    }
    return 0.0;
}

/// Índice de similitud de Jaccard entre dos comunidades (presencia/ausencia de especies)
[[nodiscard]] inline double compute_jaccard_similarity(
    const std::vector<uint8_t>& presence_a,
    const std::vector<uint8_t>& presence_b) noexcept
{
    if (presence_a.empty() || presence_a.size() != presence_b.size()) return 0.0;
    size_t intersection_count = 0;
    size_t union_count = 0;
    for (size_t i = 0; i < presence_a.size(); ++i) {
        bool a = (presence_a[i] != 0);
        bool b = (presence_b[i] != 0);
        if (a && b) intersection_count++;
        if (a || b) union_count++;
    }
    return (union_count > 0) ? (static_cast<double>(intersection_count) / union_count) : 1.0;
}

} // namespace joro::community
