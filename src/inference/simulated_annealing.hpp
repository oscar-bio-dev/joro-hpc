#pragma once

#include "common/types.hpp"
#include "abm/abm_engine.hpp"
#include "lpdm/lpdm_engine.hpp"
#include "io/era5_netcdf_reader.hpp"
#include "inference/cost_function.hpp"
#include "inference/occupancy_model.hpp"
#include <array>
#include <string>
#include <vector>

namespace joro::inference {

struct CalibrationParamDef {
    std::string name;
    std::string symbol;
    float min_val;
    float max_val;
    float default_val;
};

/// Conjunto de parámetros calibrables según el modelo de capacidad de carga (Sección 8.4):
///   M1 (principal, 9 libres): T_c, alpha_s, beta_s, K0, gamma_hab, q_balloon, q_viab, a0, b0
///   M0 (K_i = K0, 8 libres) : igual que M1 sin gamma_hab
///   M3 (completo, 11 libres): M0 + gamma_1, gamma_2, gamma_3 (sobreparametrizado en 2014-2019)
[[nodiscard]] const std::vector<CalibrationParamDef>& calibration_params_for(abm::CarryingModel model);

/// Alias del conjunto del modelo principal M1 (9 parámetros libres)
extern const std::vector<CalibrationParamDef> CALIBRATION_PARAMS;

/// Índice de beta_s dentro del vector theta (variable DISCRETA en {1, 2, 3}); es el mismo en M0/M1/M3
constexpr int kBetaDamageIndex = 2;

/// Índice de un parámetro por nombre (-1 si el modelo no lo incluye)
[[nodiscard]] int calibration_param_index(const std::vector<CalibrationParamDef>& defs, const std::string& name);

/// Telemetría de un tramo de cómputo (una cadena o el total de la calibración)
struct ChainTelemetry {
    uint64_t evaluations = 0;
    double abm_time_s = 0.0;           ///< ABM (OpenMP + AVX-512): invierno, desarrollo Q10, reproducción
    double lpdm_init_time_s = 0.0;     ///< Inicialización de super-partículas (host)
    double lpdm_h2d_time_s = 0.0;      ///< Host -> Device USM
    double lpdm_kernel_time_s = 0.0;   ///< Kernel SYCL en la GPU
    double lpdm_d2h_time_s = 0.0;      ///< Device USM -> host
    double lpdm_reduce_time_s = 0.0;   ///< Reducción de aterrizajes (host)
    double cost_time_s = 0.0;          ///< Ocupación latente + detección + reducción del costo (OpenMP)
    uint64_t particles_dispatched = 0; ///< Super-partículas integradas en la GPU
    uint64_t cpu_fallbacks = 0;        ///< Despachos que NO se ejecutaron en la GPU (debe ser 0)
    uint64_t capped_dispatches = 0;    ///< Despachos anuales con tope adaptativo de super-partículas aplicado

    [[nodiscard]] double lpdm_total_s() const noexcept {
        return lpdm_init_time_s + lpdm_h2d_time_s + lpdm_kernel_time_s + lpdm_d2h_time_s + lpdm_reduce_time_s;
    }
};

struct ChainConvergenceRecord {
    int chain_id = 0;
    uint32_t seed = 0;
    int total_iterations = 400;
    int accepted_proposals = 0;
    double initial_cost = 0.0;
    double min_cost = 0.0;
    double execution_time_s = 0.0;
    std::vector<float> best_theta;
    CostComponents best_components;
    CostComponents initial_components;
    ChainTelemetry telemetry;
    /// Traza de convergencia: (iteración, temperatura, costo actual, costo mínimo hasta la fecha)
    std::vector<std::array<double, 4>> trace;
};

struct CompatibleEnsembleMember {
    int chain_id = 0;
    int iteration = 0;
    double temperature = 0.0;
    double j_total = 0.0;
    double j_arrival_norm = 0.0;
    double j_front_norm = 0.0;
    double j_persistence_norm = 0.0;
    double j_config_a = 0.0; // w = (0.20, 0.60, 0.20)
    double j_config_b = 0.0; // w = (0.20, 0.20, 0.60)
    std::vector<float> theta;
};

/// Resumen de la partición balanceada del conjunto de calibración (para el informe)
struct CalibrationStratification {
    uint64_t n_rows = 0;
    uint64_t arrival_pos = 0;      ///< Y_t=1, Y_{t-1}=0  (colonizaciones observadas)
    uint64_t arrival_neg = 0;      ///< Y_t=0, Y_{t-1}=0 dentro de la envolvente activa
    uint64_t persistence_pos = 0;  ///< Y_t=1, Y_{t-1}=1
    uint64_t persistence_neg = 0;  ///< Y_t=0, Y_{t-1}=1
    uint64_t outside_envelope = 0; ///< Ausencias lejanas excluidas del Brier de llegada
    std::vector<double> envelope_radius_km;   ///< Radio de la envolvente por año (2014..2019)
    std::vector<double> observed_front_km;    ///< P95 observado de distancia al epicentro por año
    uint64_t effort_imputed_rows = 0;         ///< Filas con Y=1 cuyo esfuerzo se imputó (ver informe)
    bool effort_synthetic = false;            ///< True si el esfuerzo TGB tiene la firma sintética (pocos valores distintos)
};

struct SimulatedAnnealingReport {
    int n_chains = 5;
    int iter_per_chain = 400;
    int total_evaluations = 2000;
    double total_runtime_s = 0.0;
    abm::CarryingModel model = abm::CarryingModel::M1;   ///< Modelo de K_i evaluado
    std::vector<CalibrationParamDef> params;             ///< Definición de theta para este modelo
    size_t atmosphere_slots = 0;                          ///< Campos ERA5 anuales residentes (2014..2019 ⇒ 6)
    double global_min_cost = 1e6;
    std::vector<float> map_theta;
    CostComponents map_components;
    std::vector<ChainConvergenceRecord> chains;
    std::vector<CompatibleEnsembleMember> compatible_ensemble;

    // Estadísticas de incertidumbre P5, Mediana, P95
    std::vector<float> p5_theta;
    std::vector<float> median_theta;
    std::vector<float> p95_theta;

    // Sensibilidad a pesos alternativos
    double retention_rate_config_a = 0.0;
    double retention_rate_config_b = 0.0;
    double j_min_config_a = 0.0;
    double j_min_config_b = 0.0;

    // Escalas de referencia y costo nominal
    CostReferenceScales reference_scales;
    CostComponents nominal_components;
    CalibrationStratification stratification;

    // Telemetría híbrida CPU + GPU
    ChainTelemetry telemetry_total;
    std::string device_description;
    int omp_max_threads = 1;
};

/// Carga el conjunto de calibración observacional (2014-2019)
struct CalibrationObservation {
    int32_t cell_id;
    int16_t year;
    uint8_t y_detected;
    int32_t effort_araneae;
    float centroid_lon;
    float centroid_lat;
    float z_imp;
    float z_borde;
    float z_dosel;
    float pop_density_h;
    float road_density_r;
};

std::vector<CalibrationObservation> load_calibration_dataset(const std::string& csv_path);

/// Ejecuta el algoritmo de Recocido Simulado Multi-inicio (5 cadenas INDEPENDIENTES ejecutadas
/// de forma SECUENCIAL). Cada evaluación del simulador acoplado usa el 100 % del hardware:
/// LPDM en la GPU (kernel SYCL sobre Device USM persistente) y ABM/ocupación/costo con todos
/// los hilos OpenMP del host. `thermal_by_year[i]` es el forzamiento diario ERA5 del año 2014+i.
/// `lpdm_engine` debe ser un motor GPU SYCL (se aborta si no hay offload a GPU).
SimulatedAnnealingReport run_multi_start_simulated_annealing(
    const std::vector<GridCell>& grid_cells,
    const std::vector<io::DailyThermalGrid>& thermal_by_year,
    const std::vector<CalibrationObservation>& observations,
    lpdm::LpdmEngine* lpdm_engine,
    int n_chains = 5,
    int iter_per_chain = 400,
    double initial_temp = 1.0,
    double delta_threshold = 0.05,
    const std::string& checkpoint_csv_path = "/mnt/hpc_scratch/joro-hpc/checkpoints/theta_compatible_ensemble.csv",
    abm::CarryingModel model = abm::CarryingModel::M1);

} // namespace joro::inference
