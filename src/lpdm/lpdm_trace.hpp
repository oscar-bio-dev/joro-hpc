#pragma once

#include "common/spatial_grid_lookup.hpp"
#include "lpdm/era5_view.hpp"
#include <cstdint>

#if defined(JORO_HAS_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace joro::lpdm {

#if defined(JORO_HAS_SYCL)
namespace jm {
using sycl::cos;
using sycl::log;
using sycl::sin;
using sycl::sqrt;
} // namespace jm
#else
namespace jm {
using std::cos;
using std::log;
using std::sin;
using std::sqrt;
} // namespace jm
#endif

/// Lote de super-partículas liberado por UNA celda de origen. Es el único dato de partículas que
/// viaja al dispositivo (24 B por celda activa): no se materializa ningún vector de partículas.
struct EmissionJob {
    int32_t cell_id = 0;
    float lon = 0.0f;
    float lat = 0.0f;
    uint32_t total = 0;   ///< Propágulos reales a representar (conservación 1:1 con el ABM)
    uint32_t n_sim = 0;   ///< Super-partículas simuladas (tope adaptativo ya aplicado)
    uint32_t offset = 0;  ///< Índice global de la primera super-partícula de este lote
};

/// Histograma de distancias de aterrizaje (ponderado por propágulos)
constexpr int kDistHistBins = 2048;
constexpr float kDistHistBinKm = 0.25f;

/// Parámetros físicos y punteros de solo lectura que consume el trazador de trayectorias
struct TraceParams {
    float dt = 60.0f;
    float sqrt_dt = 7.745967f;
    int steps = 60;
    float vs_aero = 0.2947f;
    // Modo heredado (sin ERA5): viento uniforme y techo fijo
    float fixed_u = 4.0f, fixed_v = 1.5f, fixed_w = 0.02f, fixed_blh = 1200.0f;
    // Modo ERA5: campo residente + límites de la BLH
    Era5View atm;
    float blh_min = 250.0f, blh_max = 3000.0f;
    SpatialGridView grid;
};

struct TraceResult {
    uint32_t weight = 0;
    int32_t dest_cell = -1; ///< -1 si no aterrizó o aterrizó fuera de la malla
    bool landed = false;
    float dist_km = 0.0f;
};

/// Philox-4x32-10 (Salmon et al. 2011): generador basado en contador, sin estado compartido.
inline void philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                          uint32_t k0, uint32_t k1, uint32_t out[4]) noexcept
{
    constexpr uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    constexpr uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    for (int r = 0; r < 10; ++r) {
        const uint64_t p0 = static_cast<uint64_t>(M0) * c0;
        const uint64_t p1 = static_cast<uint64_t>(M1) * c2;
        const uint32_t hi0 = static_cast<uint32_t>(p0 >> 32), lo0 = static_cast<uint32_t>(p0);
        const uint32_t hi1 = static_cast<uint32_t>(p1 >> 32), lo1 = static_cast<uint32_t>(p1);
        const uint32_t n0 = hi1 ^ c1 ^ k0, n1 = lo1, n2 = hi0 ^ c3 ^ k1, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += W0; k1 += W1;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

[[nodiscard]] inline float u01(uint32_t x) noexcept {
    return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
}

/// Integra la trayectoria completa de UNA super-partícula en registros y devuelve su destino.
/// Misma física que el baseline CPU (Langevin vertical, cinemática esférica, techo reflectante,
/// suelo absorbente); en modo ERA5 el viento (u, v, w) y la BLH se interpolan del campo residente.
/// Determinista: depende solo de (cell_id, índice local), no del orden de ejecución.
[[nodiscard]] inline TraceResult trace_super_particle(const TraceParams& tp, const EmissionJob& job,
                                                      uint32_t local_idx) noexcept
{
    constexpr float kPi = 3.14159265f;
    constexpr float deg_to_rad = kPi / 180.0f;
    constexpr float rad_to_deg = 180.0f / kPi;
    constexpr float R_earth = 6371000.0f;
    constexpr float tau_L = 100.0f;
    constexpr float sigma_w = 0.5f;

    TraceResult res;
    const uint32_t base_w = job.total / job.n_sim;
    const uint32_t rem = job.total % job.n_sim;
    res.weight = base_w + (local_idx < rem ? 1u : 0u);

    const uint32_t key0 = static_cast<uint32_t>(job.cell_id);
    const uint32_t key1 = 0x4A0B1C2Du;
    uint32_t r4[4];

    // Despegue: altura y velocidad vertical convectiva iniciales (como en el baseline)
    philox4x32_10(local_idx, 0u, 0u, 0u, key0, key1, r4);
    float x = job.lon, y = job.lat;
    float z = 30.0f + 220.0f * u01(r4[0]);
    float w_prime = 0.35f + 0.15f * u01(r4[1]);
    // Hora de liberación dentro de la ventana ERA5 (los 7 días disponibles)
    float t0_h = 0.0f;
    if (tp.atm.valid) {
        const float flight_h = static_cast<float>(tp.steps) * tp.dt / 3600.0f;
        const float span = static_cast<float>(tp.atm.n_t - 1) - flight_h;
        t0_h = span > 0.0f ? span * u01(r4[2]) : 0.0f;
    }

    for (int s = 0; s < tp.steps; ++s) {
        const float sim_time_s = static_cast<float>(s) * tp.dt;
        const float diurnal_phase = jm::sin(2.0f * kPi * (sim_time_s / 86400.0f));
        const float w_convective = (diurnal_phase > 0.0f) ? (0.20f * diurnal_phase) : 0.0f;

        philox4x32_10(local_idx, static_cast<uint32_t>(s) + 1u, 0u, 0u, key0, key1, r4);
        const float u1 = u01(r4[0]);
        const float u2 = u01(r4[1]);
        const float rr = jm::sqrt(-2.0f * jm::log(u1 > 1e-6f ? u1 : 1e-6f));
        const float dW = (rr * jm::cos(6.2831853f * u2)) * tp.sqrt_dt;

        // Viento advectivo y techo de la capa límite
        float u_wind = tp.fixed_u, v_wind = tp.fixed_v, w_mean = tp.fixed_w, blh = tp.fixed_blh;
        if (tp.atm.valid) {
            float b = 0.0f;
            tp.atm.sample(x, y, z, t0_h + sim_time_s / 3600.0f, u_wind, v_wind, w_mean, b);
            blh = b < tp.blh_min ? tp.blh_min : (b > tp.blh_max ? tp.blh_max : b);
        }

        // Turbulencia vertical de Langevin (Well-Mixed)
        const float a_w = -w_prime / tau_L;
        const float b_w = jm::sqrt(2.0f * sigma_w * sigma_w / tau_L);
        w_prime += a_w * tp.dt + b_w * dW;

        // Cinemática esférica
        const float cos_lat = jm::cos(y * deg_to_rad);
        const float cl = cos_lat > 0.01f ? cos_lat : 0.01f;
        x += (u_wind * tp.dt / (R_earth * cl)) * rad_to_deg;
        y += (v_wind * tp.dt / R_earth) * rad_to_deg;
        z += (w_mean + w_convective + w_prime - tp.vs_aero) * tp.dt;

        if (z > blh) {            // techo reflectante
            z = 2.0f * blh - z;
            w_prime = -w_prime;
        }
        if (z <= 0.0f) {          // suelo absorbente: aterrizaje
            res.landed = true;
            break;
        }
    }

    if (res.landed) {
        const float mid_lat = 0.5f * (job.lat + y) * deg_to_rad;
        const float dx = (x - job.lon) * deg_to_rad * R_earth * jm::cos(mid_lat);
        const float dy = (y - job.lat) * deg_to_rad * R_earth;
        res.dist_km = jm::sqrt(dx * dx + dy * dy) * 0.001f;
        res.dest_cell = tp.grid.find(x, y);
    }
    return res;
}

} // namespace joro::lpdm
