#pragma once

#include "common/types.hpp"
#include "abm/abm_state_soa.hpp"
#include "abm/carrying_capacity.hpp"
#include "io/era5_netcdf_reader.hpp"
#include <unordered_map>
#include <vector>

namespace joro::abm {

/// Parámetros ecofisiológicos y demográficos (Secciones 8.3 y 8.4)
struct DemographicParameters {
    float fecundity_F = 450.0f;    ///< Fecundidad por saco [huevos/saco] (Tabla 8.3: 400-500)
    float q10 = 2.2f;              ///< Coeficiente Q10 de desarrollo térmico (Tabla 8.3: 2.0-3.0)
    float t_c = -4.0f;             ///< Temperatura crítica de daño por frío [°C] (Sección 8 / Tarea 2.3)
    float alpha_damage = 0.04f;    ///< Tasa de daño por frío [días^-1 °C^-beta] (Tabla 8.3: 0.01-0.10)
    int   beta_damage = 2;         ///< Exponente de estrés térmico (1, 2 o 3)
    float q_viab = 0.65f;          ///< Viabilidad de ootecas (Tabla 8.3: 0.20-0.90)
    float q_balloon = 0.35f;       ///< Fracción de baloneo aéreo (Tabla 8.3: 0.10-0.60)
    float kappa_b = 1.8f;          ///< Dispersión NegBin

    // Capacidad de carga K_i (Sección 8.4)
    float k0 = 5000.0f;            ///< Capacidad de carga base en hábitat promedio
    float gamma1_imp = 0.45f;      ///< Sensibilidad a impermeabilidad urbana Z_imp
    float gamma2_borde = 0.65f;    ///< Sensibilidad a densidad de borde Z_borde
    float gamma3_dosel = 0.30f;    ///< Sensibilidad a dosel arbóreo Z_dosel
    float k_min = 100.0f;          ///< Cota inferior
    float k_max = 50000.0f;        ///< Cota superior
    float gamma_hab = 0.0f;        ///< Pendiente global sobre H_hab (M1; M0 ⇒ 0)
    bool  use_hab_index = false;   ///< true: K_i = K0 exp(gamma_hab H_hab) (M0/M1); false: M3 (3 gammas)

    // Umbrales fenológicos de desarrollo
    float t_min_dev = 13.0f;       ///< Umbral mínimo de desarrollo (°C)
    float t_max_dev = 35.0f;       ///< Umbral máximo de desarrollo (°C)
    float t_ref = 20.0f;           ///< Temperatura de referencia (°C)
    float r_ref = 0.010f;          ///< Tasa diaria de desarrollo a T_ref (~100 días a 20°C)
};

/// Tabla densa cell_id -> posición en el vector de celdas (-1 si no existe).
/// Se construye UNA vez por malla y se reutiliza en todas las evaluaciones (acceso O(1) sin hashing).
struct CellIndexTable {
    std::vector<int32_t> idx_of;

    [[nodiscard]] static CellIndexTable build(const std::vector<GridCell>& cells);

    [[nodiscard]] inline int32_t find(int32_t cell_id) const noexcept {
        return (cell_id >= 0 && static_cast<size_t>(cell_id) < idx_of.size())
                   ? idx_of[static_cast<size_t>(cell_id)] : -1;
    }
};

struct DemographyYearSummary {
    uint64_t total_adults = 0;
    uint64_t total_offspring = 0;
    uint64_t total_ballooners = 0;
    
    /// Número de propágulos aéreos listos para el LPDM por celda: cell_id -> P^{aereos}_{i,t}
    std::unordered_map<int32_t, uint32_t> ballooners_per_cell;
    
    /// Población adulta superviviente N_{i,t} por celda
    std::unordered_map<int32_t, float> adult_density_per_cell;

    /// Ratio medio de saturación N_i / K_i en celdas ocupadas
    float mean_saturation = 0.0f;
    /// Supervivencia invernal observada en ootecas (%)
    float winter_survival_pct = 100.0f;
};

class AbmEngine {
public:
    explicit AbmEngine(DemographicParameters params = DemographicParameters{});

    /// Inicializa las cohortes a partir de la condición de semilla en NE Georgia
    void initialize_seed(const std::vector<GridCell>& cells, float rho0 = 0.15f);

    /// Inyecta una cohorte de prueba para evaluación de barrera térmica (ej. Zona C > 41°N)
    void inject_test_cohort(int32_t cell_id, float count);

    /// Incorpora las llegadas del LPDM como nuevas cohortes juveniles
    void deposit_arrivals(const std::unordered_map<int32_t, uint32_t>& arrivals);

    /// Fase 1 Primaveral: Días 1 a 120 (Enero a Mayo)
    /// Aplica mortalidad por frío D_{j,t} acumulada en invierno, eclosiona juveniles
    /// y genera propágulos para baloneo en Mayo (Día 121)
    DemographyYearSummary step_spring_phenology(
        const std::vector<GridCell>& cells,
        const io::DailyThermalGrid& thermal,
        int year);

    /// Fase 2 Estival/Otoñal: Días 121 a 365 (Mayo a Diciembre)
    /// Acumula grados-día Q10, maduración a adultos, reproducción densodependiente Ki,
    /// e inicio de daño invernal en las nuevas ootecas
    DemographyYearSummary step_summer_autumn_phenology(
        const std::vector<GridCell>& cells,
        const io::DailyThermalGrid& thermal,
        int year);

    /// Ciclo anual completo diario (365 días) con forzamiento térmico diario ERA5
    DemographyYearSummary step_annual_cycle_daily(
        const std::vector<GridCell>& cells,
        const io::DailyThermalGrid& thermal,
        int year);

    /// Ejecuta el ciclo demográfico anual completo (compatibilidad con baseline previo)
    DemographyYearSummary step_annual_cycle(
        const std::vector<GridCell>& cells,
        const std::vector<float>& mean_winter_temp_c,
        const std::vector<float>& mean_summer_temp_c);

    const CohortSoA& cohorts() const noexcept { return cohorts_; }
    const DemographicParameters& params() const noexcept { return params_; }
    void set_params(const DemographicParameters& p) noexcept { params_ = p; }

    /// Registra una tabla de índices externa (no propietaria) que debe sobrevivir al motor.
    /// Si no se registra, cada paso anual construye una tabla local equivalente.
    void set_cell_index(const CellIndexTable* table) noexcept { cell_index_ext_ = table; }

private:
    DemographicParameters params_;
    CohortSoA cohorts_;
    const CellIndexTable* cell_index_ext_ = nullptr;
};

} // namespace joro::abm
