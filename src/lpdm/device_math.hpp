#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

#if defined(JORO_HAS_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace joro::lpdm {

/// Funciones matemáticas portables host/dispositivo: en SYCL resuelven a los builtins del dispositivo,
/// en host a <cmath>. Todo el código de física del LPDM (trazador, Langevin, sedimentación) las usa.
#if defined(JORO_HAS_SYCL)
namespace jm {
using sycl::cbrt;
using sycl::cos;
using sycl::exp;
using sycl::log;
using sycl::pow;
using sycl::sin;
using sycl::sqrt;
} // namespace jm
#else
namespace jm {
using std::cbrt;
using std::cos;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
} // namespace jm
#endif

namespace jm {
[[nodiscard]] inline float clampf(float x, float lo, float hi) noexcept { return x < lo ? lo : (x > hi ? hi : x); }
[[nodiscard]] inline float maxf(float a, float b) noexcept { return a > b ? a : b; }
[[nodiscard]] inline float minf(float a, float b) noexcept { return a < b ? a : b; }
/// Piso portátil en `float`: en el dispositivo SYCL evita `sycl::floor` (`OpExtInst floor`), cuya
/// legalización vectorial falla con ICE (`SplitVectorResult`) en el runtime OpenCL CPU de oneAPI 2023.0
/// sobre CPUs AMD EPYC Zen4 con AVX-512 (runners de GitHub Actions).
[[nodiscard]] inline float floor(float x) noexcept
{
#if defined(__SYCL_DEVICE_ONLY__)
    const float t = sycl::trunc(x);
    return (x < t) ? (t - 1.0f) : t;
#else
    return std::floor(x);
#endif
}
/// Módulo para argumentos no negativos: a - b floor(a / b). Evita sycl::fmod, cuya implementación iterativa
/// en el compilador del dispositivo es divergente por hilo y no es robusta dentro del bucle de integración.
[[nodiscard]] inline float fmod(float a, float b) noexcept { return a - b * floor(a / b); }
} // namespace jm

/// IEEE 754 binary16 almacenado como uint16_t: conversión por aritmética de bits (idéntica en host y en
/// el kernel SYCL, sin depender de sycl::half ni de _Float16 en el compilador anfitrión).
[[nodiscard]] inline float half_to_float(uint16_t h) noexcept
{
#if defined(__SYCL_DEVICE_ONLY__)
    return static_cast<float>(sycl::bit_cast<sycl::half>(h));   // instrucción nativa de conversión del dispositivo
#else
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    const uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;
    if (exp == 0u) {
        // cero o subnormal: valor = mant * 2^-24 (sin bucles: seguro con divergencia SIMD en el kernel)
        const float mag = static_cast<float>(mant) * (1.0f / 16777216.0f);
        return std::bit_cast<float>(std::bit_cast<uint32_t>(mag) | sign);
    }
    const uint32_t bits = (exp == 31u) ? (sign | 0x7F800000u | (mant << 13))
                                       : (sign | ((exp + 112u) << 23) | (mant << 13));
    return std::bit_cast<float>(bits);
#endif
}

/// float -> binary16 con redondeo al par más cercano; satura a ±inf por encima de 65520.
[[nodiscard]] inline uint16_t float_to_half(float f) noexcept
{
    const uint32_t x = std::bit_cast<uint32_t>(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t a = x & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | (a > 0x7F800000u ? 0x200u : 0u));
    if (a >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);
    if (a < 0x33000001u) return static_cast<uint16_t>(sign);
    if (a < 0x38800000u) {             // subnormal en binary16
        const int e = static_cast<int>(a >> 23);
        const uint32_t mant = (a & 0x7FFFFFu) | 0x800000u;
        const int shift = 126 - e;
        uint32_t m = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1u))) ++m;
        return static_cast<uint16_t>(sign | m);
    }
    uint32_t r = a - 0x38000000u;       // re-sesgo del exponente (127 -> 15)
    r += 0xFFFu + ((r >> 13) & 1u);
    return static_cast<uint16_t>(sign | (r >> 13));
}

} // namespace joro::lpdm
