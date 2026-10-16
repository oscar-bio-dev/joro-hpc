#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

namespace joro::abm {

/// Estructura de Arreglos (Structure of Arrays - SoA) para cohortes demográficas.
/// Diseñada con alineación de 64 bytes para procesamiento vectorial continuo
/// mediante instrucciones AVX-512 (zmm0..zmm31, 16 floats FP32 por ciclo) en i9-11900K.
struct alignas(64) CohortSoA {
    std::vector<int32_t> cell_id;           ///< ID de la celda de la cuadrícula
    std::vector<uint8_t> stage;             ///< Estadio: 0 = ooteca, 1 = juv I-IV, 2 = subadulto, 3 = adulto
    std::vector<float>   weight;            ///< Ponderación / número de individuos reales (super-individuo)
    std::vector<float>   degree_days;       ///< Grados-día fisiológicos acumulados
    std::vector<float>   cold_damage_D;     ///< Daño térmico invernal acumulado D_{j,t}
    std::vector<uint8_t> is_alive;          ///< 1 = viva, 0 = muerta

    [[nodiscard]] size_t size() const noexcept {
        return cell_id.size();
    }

    void reserve(size_t n) {
        cell_id.reserve(n);
        stage.reserve(n);
        weight.reserve(n);
        degree_days.reserve(n);
        cold_damage_D.reserve(n);
        is_alive.reserve(n);
    }

    void add_cohort(int32_t c_id, uint8_t s, float w) {
        cell_id.push_back(c_id);
        stage.push_back(s);
        weight.push_back(w);
        degree_days.push_back(0.0f);
        cold_damage_D.push_back(0.0f);
        is_alive.push_back(1);
    }

    void clear() {
        cell_id.clear();
        stage.clear();
        weight.clear();
        degree_days.clear();
        cold_damage_D.clear();
        is_alive.clear();
    }
};

} // namespace joro::abm
