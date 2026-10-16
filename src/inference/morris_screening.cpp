#include "inference/cost_function.hpp"
#include <iostream>
#include <vector>
#include <random>

namespace joro::inference {

/// Cribado de sensibilidad global por el método de efectos elementales (Morris 1991)
/// Evalúa qué parámetros de Theta dominan la función de costo J(theta).
void run_morris_screening(int n_trajectories = 20, int n_parameters = 8) {
    std::cout << "[Morris Screening] Iniciando cribado de sensibilidad con "
              << n_trajectories << " trayectorias en " << n_parameters << " dimensiones...\n";
    // TODO (Mes 4): Generar grilla p-dimensional y calcular medias mu* y varianzas sigma de efectos elementales
    std::cout << "[Morris Screening] Stub completado.\n";
}

} // namespace joro::inference
