#pragma once

#include "common/types.hpp"
#include "abm/abm_state_soa.hpp"
#include "abm/carrying_capacity.hpp"
#include <unordered_map>
#include <vector>

namespace joro::abm {

struct DemographicParameters {
    float fecundity_F = 450.0f;    ///< Fecundidad por saco [huevos/saco] (Tabla 8.3: 400-500)
    float q10 = 2.2f;
    float t_c = -3.5f;             ///< Temperatura crítica de daño por frío [°C] (Tabla 8.3: [-5, 0])
    float alpha_damage = 0.04f;    ///< Tasa de acumulación de daño
    int   beta_damage = 2;         ///< Exponente de estrés térmico
    float q_viab = 0.65f;          ///< Viabilidad de ootecas
    float q_balloon = 0.35f;       ///< Fracción de baloneo aéreo
    float kappa_b = 1.8f;          ///< Dispersión NegBin
};

struct DemographyYearSummary {
    uint64_t total_adults = 0;
    uint64_t total_offspring = 0;
    uint64_t total_ballooners = 0;
    
    /// Número de propágulos aéreos listos para el LPDM por celda: cell_id -> P^{aereos}_{i,t}
    std::unordered_map<int32_t, uint32_t> ballooners_per_cell;
    
    /// Población adulta superviviente N_{i,t} por celda
    std::unordered_map<int32_t, float> adult_density_per_cell;
};

class AbmEngine {
public:
    explicit AbmEngine(DemographicParameters params = DemographicParameters{});

    /// Inicializa las cohortes a partir de la condición de semilla en NE Georgia
    void initialize_seed(const std::vector<GridCell>& cells, float rho0 = 0.15f);

    /// Incorpora las llegadas del LPDM como nuevas cohortes juveniles
    void deposit_arrivals(const std::unordered_map<int32_t, uint32_t>& arrivals);

    /// Ejecuta el ciclo demográfico anual completo (desarrollo, daño invernal, reproducción)
    DemographyYearSummary step_annual_cycle(
        const std::vector<GridCell>& cells,
        const std::vector<float>& mean_winter_temp_c,
        const std::vector<float>& mean_summer_temp_c);

private:
    DemographicParameters params_;
    CohortSoA cohorts_;
};

} // namespace joro::abm
