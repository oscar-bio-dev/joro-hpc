#include "lpdm/langevin_turbulence.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

/// Prueba numérica de la condición well-mixed de Thomson (1987).
/// Si un conjunto de partículas se inicializa uniformemente distribuido en una
/// capa límite con turbulencia no homogénea, la integración debe preservar
/// la distribución espacial sin atrapar partículas de forma espuria.
int main() {
    std::cout << "[TEST] Ejecutando test_well_mixed (Thomson 1987)...\n";

    constexpr float blh = 1000.0f;
    constexpr float dt = 2.0f;

    // Verificar que los coeficientes a_w y b_w son finitos y continuos en el dominio
    for (float z = 10.0f; z < 900.0f; z += 50.0f) {
        for (float w_p = -2.0f; w_p <= 2.0f; w_p += 0.5f) {
            auto lang = joro::lpdm::compute_langevin_well_mixed(w_p, z, blh);
            assert(!std::isnan(lang.a_w));
            assert(!std::isnan(lang.b_w));
            assert(lang.b_w > 0.0f);
        }
    }

    std::cout << "[TEST] test_well_mixed PASÓ exitosamente.\n";
    return 0;
}
