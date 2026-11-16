#pragma once

#include "common/spatial_grid_lookup.hpp"
#include "lpdm/device_math.hpp"
#include "lpdm/era5_view.hpp"
#include "lpdm/langevin_turbulence.hpp"
#include "lpdm/settling_velocity.hpp"
#include <cstdint>

namespace joro::lpdm {

/// Lote de super-partículas liberado por UNA celda de origen. Es el único dato de partículas que
/// viaja al dispositivo (24 B por celda activa): no se materializa ningún vector de partículas.
struct EmissionJob {
    int32_t cell_id = 0;
    float lon = 0.0f;
    float lat = 0.0f;
    uint32_t total = 0;   ///< Propágulos reales a representar (conservación 1:1 con el ABM)
    uint32_t n_sim = 0;   ///< Super-partículas simuladas (tope adaptativo ya aplicado)
    uint32_t offset = 0;  ///< Índice global de la primera super-partícula de este lote
    float start_h = 0.0f; ///< Inicio de la ventana de despegue [h desde el 1 de mayo 00 UTC] (emergencia escalonada por celda)
};

/// Histograma de distancias de aterrizaje (ponderado por propágulos)
constexpr int kDistHistBins = 2048;
constexpr float kDistHistBinKm = 0.25f;

/// Umbral biomecánico de despegue (ballooning) sobre el viento a 10 m [m/s] (M3)
constexpr float kUCritTakeoffDefault = 3.0f;
/// Ventana convectiva diurna de despegue en hora solar local [h]: [09:00, 16:00)
constexpr float kTakeoffLocalHourMin = 9.0f;
constexpr float kTakeoffLocalHourSpan = 7.0f;
/// Número de ventanas diurnas candidatas antes de abortar el vuelo distante
constexpr int kTakeoffMaxAttempts = 4;

/// Parámetros físicos y punteros de solo lectura que consume el trazador de trayectorias
struct TraceParams {
    float dt = 60.0f;
    float sqrt_dt = 7.745967f;
    int steps = 60;
    float vs_aero = 0.2947f;  ///< Velocidad de sedimentación nominal del escenario [m/s]
    float vs_ref = 0.2947f;   ///< Velocidad nominal de referencia (M=0.6 mg, Ls=1.2 m, chi=0.85)
    // Modo heredado (sin ERA5): viento uniforme y techo fijo
    float fixed_u = 4.0f, fixed_v = 1.5f, fixed_w = 0.02f, fixed_blh = 1200.0f;
    // Modo ERA5: campo residente + límites de la BLH
    Era5View atm;
    float blh_min = 250.0f, blh_max = 3000.0f;
    SpatialGridView grid;
    // Semillas dinámicas (M8): índice de año y réplica Monte Carlo entran en la clave de Philox
    uint32_t year_idx = 0;
    uint32_t mc_seed = 0;
    float u_crit = kUCritTakeoffDefault;
};

struct TraceResult {
    uint32_t weight = 0;
    int32_t dest_cell = -1; ///< -1 si no aterrizó o aterrizó fuera de la malla
    bool landed = false;
    bool retained = false;  ///< Vuelo distante abortado por viento >= u_crit: retención local (dest_cell = celda de origen)
    float dist_km = 0.0f;
};

/// Producto 32x32 -> 64 bit separado en (hi, lo). En el dispositivo se usa `sycl::mul_hi` + multiplicación de 32 bit:
/// el compilador del dispositivo (Arc A380 / Level-Zero) NO trunca correctamente `uint32_t(uint64_t(a) * b)` dentro
/// de rondas encadenadas con divergencia SIMD (devolvía palabras de >32 bit y rompía la equivalencia CPU/GPU).
inline void mulhilo32(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) noexcept
{
#if defined(JORO_HAS_SYCL)
    hi = static_cast<uint32_t>(sycl::mul_hi(a, b));
    lo = a * b;
#else
    const uint64_t p = static_cast<uint64_t>(a) * b;
    hi = static_cast<uint32_t>(p >> 32);
    lo = static_cast<uint32_t>(p);
#endif
}

/// Philox-4x32-10 (Salmon et al. 2011): generador basado en contador, sin estado compartido.
inline void philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                          uint32_t k0, uint32_t k1, uint32_t out[4]) noexcept
{
    constexpr uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    constexpr uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    for (int r = 0; r < 10; ++r) {
        uint32_t hi0, lo0, hi1, lo1;
        mulhilo32(M0, c0, hi0, lo0);
        mulhilo32(M1, c2, hi1, lo1);
        const uint32_t n0 = hi1 ^ c1 ^ k0, n1 = lo1, n2 = hi0 ^ c3 ^ k1, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += W0; k1 += W1;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

[[nodiscard]] inline float u01(uint32_t x) noexcept {
    return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Funciones físicas compartidas por el kernel SYCL (Philox) y el baseline CPU (mt19937)
// ─────────────────────────────────────────────────────────────────────────────

/// Hora solar local [0, 24) a partir de la hora UTC acumulada de la serie (00 UTC del primer día) y la longitud
[[nodiscard]] inline float local_solar_hour(float t_hours, float lon_deg) noexcept
{
    return jm::fmod(t_hours + lon_deg / 15.0f + 2400.0f, 24.0f);
}

/// Fase convectiva diurna (M2): sin(pi (h - 8) / 10), positiva únicamente entre las 08:00 y las 18:00 locales
[[nodiscard]] inline float diurnal_phase(float local_hour) noexcept
{
    constexpr float kPi = 3.14159265f;
    return (local_hour > 8.0f && local_hour < 18.0f) ? jm::sin(kPi * (local_hour - 8.0f) / 10.0f) : 0.0f;
}

/// Corriente ascendente media de las térmicas acotada dentro de la capa límite (M2): 0 fuera de [0, BLH] o de noche
[[nodiscard]] inline float convective_updraft(float phase, float z_m, float blh_m) noexcept
{
    if (phase <= 0.0f || z_m > blh_m) return 0.0f;
    const float z_rel = jm::clampf(z_m / jm::maxf(blh_m, 50.0f), 0.0f, 1.0f);
    return 0.20f * phase * 4.0f * z_rel * (1.0f - z_rel);
}

/// Escala convectiva w* [m/s] (≈2 m/s al mediodía con BLH = 1.5 km) y velocidad de fricción u* [m/s]
[[nodiscard]] inline float convective_velocity_scale(float phase, float blh_m) noexcept
{
    return phase > 0.0f ? 2.0f * phase * jm::cbrt(jm::maxf(blh_m, 50.0f) / 1500.0f) : 0.0f;
}
[[nodiscard]] inline float friction_velocity(float wind_speed_ms) noexcept
{
    return jm::clampf(0.06f * wind_speed_ms, 0.05f, 0.6f);
}

/// Velocidad terminal individual (M6): masa juvenil M_j en [0.4, 0.9] mg y longitud de seda L_s,j en [0.8, 1.6] m,
/// escalada alrededor de la velocidad nominal del escenario vs_aero.
[[nodiscard]] inline float sample_sedimentation_velocity(float u_mass, float u_silk, float vs_aero, float vs_ref) noexcept
{
    const float mass_mg = 0.4f + 0.5f * u_mass;
    const float silk_m = 0.8f + 0.8f * u_silk;
    return vs_aero * compute_settling_velocity(mass_mg, silk_m, 0.85f) / vs_ref;
}

/// Hora de despegue candidata [h desde el inicio de la serie] (M3): día de vuelo dentro de la ventana estacional y
/// hora solar local dentro de la ventana convectiva diurna [09:00, 16:00). `day_offset` desplaza el día base: si el
/// viento supera u_crit la ninfa espera en la vegetación y reintenta en DÍAS SUBSECUENTES DISTINTOS (+1 a +10 días).
/// Devuelve un valor negativo si el día candidato cae fuera de la temporada cargada (intento no disponible).
/// Con ventanas cortas (< 24 h de margen) con ERA5 el despegue es uniforme en el margen disponible (pruebas de humo).
[[nodiscard]] inline float candidate_takeoff_hour(float lon_deg, float span_h, bool atm_valid,
                                                  float u_day, float u_hour, int day_offset, float start_h) noexcept
{
    if (atm_valid && span_h < 24.0f) return day_offset > 0 ? -1.0f : jm::maxf(span_h, 0.0f) * u_day;
    const float local_target = kTakeoffLocalHourMin + kTakeoffLocalHourSpan * u_hour;
    int n_days = 1;
    if (atm_valid) n_days = static_cast<int>(span_h / 24.0f);
    // Emergencia escalonada: el día base se muestrea entre el día de emergencia de la celda y el final de la temporada
    int d_start = atm_valid ? static_cast<int>(jm::maxf(start_h, 0.0f) / 24.0f) : 0;
    d_start = d_start >= n_days ? n_days - 1 : d_start;
    const int avail = n_days - d_start;
    int d = d_start + static_cast<int>(u_day * static_cast<float>(avail));
    d = d >= n_days ? n_days - 1 : d;
    d += day_offset;
    if (!atm_valid) d = 0;
    if (d >= n_days) return -1.0f;
    const float t = static_cast<float>(d) * 24.0f + jm::fmod(local_target - lon_deg / 15.0f + 48.0f, 24.0f);
    return atm_valid ? jm::minf(t, jm::maxf(span_h, 0.0f)) : t;
}

/// Incremento de días entre reintentos sucesivos (1 a 3 días): con 3 reintentos el desplazamiento máximo es +9 d
[[nodiscard]] inline int takeoff_retry_day_step(float u) noexcept { return 1 + static_cast<int>(u * 3.0f); }

/// Integra la trayectoria completa de UNA super-partícula en registros y devuelve su destino.
/// Física: despegue diurno con compuerta u10 < u_crit (M3), convección acotada en la BLH con hora solar real (M2),
/// Langevin inhomogéneo con deriva de Thomson (M4), w sinóptica ERA5 (M5), sedimentación alométrica individual (M6),
/// cinemática esférica, techo reflectante y suelo absorbente.
/// Determinista: depende solo de (cell_id, índice local, year_idx, mc_seed), no del orden de ejecución.
[[nodiscard]] inline TraceResult trace_super_particle(const TraceParams& tp, const EmissionJob& job,
                                                      uint32_t local_idx) noexcept
{
    constexpr float kPi = 3.14159265f;
    constexpr float deg_to_rad = kPi / 180.0f;
    constexpr float rad_to_deg = 180.0f / kPi;
    constexpr float R_earth = 6371000.0f;

    TraceResult res;
    const uint32_t base_w = job.total / job.n_sim;
    const uint32_t rem = job.total % job.n_sim;
    res.weight = base_w + (local_idx < rem ? 1u : 0u);

    const uint32_t key0 = static_cast<uint32_t>(job.cell_id);
    const uint32_t key1 = 0x4A0B1C2Du ^ (tp.year_idx * 0x9E3779B9u) ^ tp.mc_seed;
    uint32_t r4[4];

    // Despegue: altura y velocidad vertical convectiva iniciales
    philox4x32_10(local_idx, 0u, 0u, 0u, key0, key1, r4);
    float x = job.lon, y = job.lat;
    float z = 30.0f + 220.0f * u01(r4[0]);
    float w_prime = 0.35f + 0.15f * u01(r4[1]);

    // Velocidad terminal individual (M6)
    philox4x32_10(local_idx, 0u, 100u, 0u, key0, key1, r4);
    const float v_sed = sample_sedimentation_velocity(u01(r4[0]), u01(r4[1]), tp.vs_aero, tp.vs_ref);

    // Hora y día de despegue con compuerta biomecánica de viento superficial (M3): hasta 4 días candidatos distintos
    float t0_h = 0.0f;
    const float flight_h = static_cast<float>(tp.steps) * tp.dt / 3600.0f;
    const float span_h = tp.atm.valid ? static_cast<float>(tp.atm.n_hours - 1) - flight_h : 0.0f;
    bool takeoff_ok = !tp.atm.valid;   // sin ERA5 no hay compuerta
    float u_day0 = 0.0f;
    int day_offset = 0;
    for (int a = 0; a < kTakeoffMaxAttempts && !takeoff_ok; ++a) {
        philox4x32_10(local_idx, 0u, 1u + static_cast<uint32_t>(a), 0u, key0, key1, r4);
        if (a == 0) u_day0 = u01(r4[0]); else day_offset += takeoff_retry_day_step(u01(r4[2]));
        const float cand = candidate_takeoff_hour(x, span_h, true, u_day0, u01(r4[1]), day_offset, job.start_h);
        if (cand < 0.0f) continue;                       // el día candidato queda fuera de la temporada cargada
        float ws10 = 0.0f, blh0 = 0.0f;
        tp.atm.sample_surface(job.lon, job.lat, cand, ws10, blh0);
        if (ws10 < tp.u_crit) { t0_h = cand; takeoff_ok = true; }
    }
    if (!tp.atm.valid) {
        philox4x32_10(local_idx, 0u, 1u, 0u, key0, key1, r4);
        t0_h = candidate_takeoff_hour(x, span_h, false, u01(r4[0]), u01(r4[1]), 0, 0.0f);
    }
    if (!takeoff_ok) {
        // Viento fuerte en todos los días candidatos: no hay vuelo distante; retención local en la celda de origen
        res.landed = true;
        res.retained = true;
        res.dist_km = 0.0f;
        res.dest_cell = job.cell_id;
        return res;
    }

    for (int s = 0; s < tp.steps; ++s) {
        const float sim_time_s = static_cast<float>(s) * tp.dt;
        const float t_h = t0_h + sim_time_s / 3600.0f;
        const float phase = diurnal_phase(local_solar_hour(t_h, x));

        philox4x32_10(local_idx, static_cast<uint32_t>(s) + 1u, 0u, 0u, key0, key1, r4);
        const float u1 = u01(r4[0]);
        const float u2 = u01(r4[1]);
        const float rr = jm::sqrt(-2.0f * jm::log(u1 > 1e-6f ? u1 : 1e-6f));
        const float xi = rr * jm::cos(6.2831853f * u2);

        // Viento advectivo y techo de la capa límite
        float u_wind = tp.fixed_u, v_wind = tp.fixed_v, w_mean = tp.fixed_w, blh = tp.fixed_blh;
        if (tp.atm.valid) {
            float b = 0.0f;
            tp.atm.sample(x, y, z, t_h, u_wind, v_wind, w_mean, b);
            blh = jm::clampf(b, tp.blh_min, tp.blh_max);
        }

        // Turbulencia vertical inhomogénea de Langevin (Thomson 1987)
        const float speed = jm::sqrt(u_wind * u_wind + v_wind * v_wind);
        w_prime = langevin_advance(w_prime, z, blh, friction_velocity(speed),
                                   convective_velocity_scale(phase, blh), tp.dt, xi);
        const float w_conv = convective_updraft(phase, z, blh);

        // Cinemática esférica
        const float cos_lat = jm::cos(y * deg_to_rad);
        const float cl = cos_lat > 0.01f ? cos_lat : 0.01f;
        x += (u_wind * tp.dt / (R_earth * cl)) * rad_to_deg;
        y += (v_wind * tp.dt / R_earth) * rad_to_deg;
        z += (w_mean + w_conv + w_prime - v_sed) * tp.dt;

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
