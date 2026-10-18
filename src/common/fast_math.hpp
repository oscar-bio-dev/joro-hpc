#pragma once

namespace joro {

/// Potencia entera rápida x^n para n en {1, 2, 3} mediante multiplicación directa
/// (sin llamar a std::pow). Es vectorizable bajo `#pragma omp simd` (AVX-512) porque
/// `n` es invariante del bucle y se resuelve con selecciones (blend), no con saltos.
/// Para n fuera de {1,2,3} se satura al caso más cercano (n<=1 -> x, n>=3 -> x^3).
[[nodiscard]] inline float ipow123(float x, int n) noexcept {
    const float x2 = x * x;
    return (n <= 1) ? x : ((n == 2) ? x2 : x2 * x);
}

} // namespace joro
