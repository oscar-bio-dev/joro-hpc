#include "inference/cost_function.hpp"
#include <iostream>
#include <vector>
#include <random>

namespace joro::inference {

/// Optimización global multi-start por recocido simulado (Kirkpatrick et al. 1983)
/// para encontrar el conjunto de parámetros compatibles Theta_comp.
void run_simulated_annealing(int max_iterations = 1000, double initial_temp = 100.0) {
    std::cout << "[Simulated Annealing] Calibración multi-start iniciada (T0 = "
              << initial_temp << ", max_iter = " << max_iterations << ")...\n";
    // TODO (Mes 5): Ejecutar cadenas de Markov independientes y registrar Theta_comp
    std::cout << "[Simulated Annealing] Calibración multi-start finalizada.\n";
}

} // namespace joro::inference
