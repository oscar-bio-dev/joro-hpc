#pragma once

#include "io/era5_netcdf_reader.hpp"
#include "lpdm/device_math.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace joro::lpdm {

/// Viento (u, v, w) en binary16 empaquetado en UNA palabra de 64 bit (bits 0-15 u, 16-31 v, 32-47 w, 48-63 relleno):
/// cada esquina del interpolador es una sola carga alineada de 8 B (el tensor 3D anual de 2,208 h ocupa 1.61 GiB,
/// la mitad que FP32 en 16 B). Se usa una palabra de 64 bit en lugar de tres campos uint16_t para que el
/// compilador del dispositivo emita cargas naturalmente alineadas.
struct alignas(8) Wind4Half {
    uint64_t bits = 0;

    [[nodiscard]] static Wind4Half make(uint16_t u, uint16_t v, uint16_t w) noexcept {
        Wind4Half r;
        r.bits = static_cast<uint64_t>(u) | (static_cast<uint64_t>(v) << 16) | (static_cast<uint64_t>(w) << 32);
        return r;
    }
};
static_assert(sizeof(Wind4Half) == 8, "Wind4Half debe ocupar exactamente 8 bytes");

/// Vista de solo lectura del campo ERA5 empaquetado (punteros a memoria host o Device USM).
/// `sample` es aritmética pura: se compila igual en host y en el kernel SYCL.
struct Era5View {
    bool valid = false;
    int n_t = 0, n_lev = 0, n_lat = 0, n_lon = 0;   ///< n_t = pasos temporales ALMACENADOS (cada `stride` horas)
    int n_hours = 0;                ///< Horas físicas de la serie original (ventana de muestreo del despegue)
    float inv_stride = 1.0f;        ///< 1 / (horas entre pasos almacenados): interpolación lineal temporal en el kernel
    float lat_max = 45.0f, lon_min = -95.0f, inv_dlat = 4.0f, inv_dlon = 4.0f;
    float zlev[16] = {};              ///< Altura estándar de cada nivel isobárico [m] (ascendente)
    const Wind4Half* wind = nullptr;  ///< [t][nivel][lat][lon] en FP16
    const float* blh = nullptr;       ///< [t][lat][lon] altura de capa límite [m]
    const float* zground = nullptr;   ///< [t][lat][lon] altura estándar de la presión superficial [m]
    const float* ws10 = nullptr;      ///< [t][lat][lon] rapidez del viento a 10 m hypot(u10, v10) [m/s]

    /// Viento (u, v, w), BLH y altura del terreno interpolados en (lon, lat, z sobre el suelo, hora).
    /// Los valores FP16 se desempaquetan a float ANTES de la interpolación bilineal horizontal +
    /// lineal en tiempo + lineal en altura sobre el suelo.
    inline void sample(float lon, float lat, float z_agl, float t_h,
                       float& u_out, float& v_out, float& w_out, float& blh_out) const noexcept
    {
        const Stencil s = horizontal(lon, lat);
        const TimeBlend tb = temporal(t_h);
        const size_t hp = static_cast<size_t>(n_lat) * static_cast<size_t>(n_lon);

        // Terreno y BLH: ambos interpolados linealmente en el tiempo entre t0 y t1 (peso at)
        const float zg = blend2d(zground, tb, hp, s);
        blh_out = blend2d(blh, tb, hp, s);

        // Niveles que rodean a z_agl (alturas ascendentes; los niveles bajo el terreno se descartan)
        int k = 0;
        while (k < n_lev - 1 && (zlev[k + 1] - zg) <= z_agl) ++k;
        int kb = k;           // nivel inferior
        float a = 0.0f;       // peso del nivel superior
        bool single = true;
        if (k + 1 < n_lev && (zlev[k] - zg) > 0.0f) {
            const float zlo = zlev[k] - zg, zhi = zlev[k + 1] - zg;
            a = (z_agl - zlo) / (zhi - zlo);
            a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
            single = false;
        } else if (k + 1 < n_lev && (zlev[k] - zg) <= 0.0f) {
            kb = k + 1;       // el nivel k está bajo el suelo: se usa el primero sobre el terreno
        }

        float uu = 0.0f, vv = 0.0f, ww = 0.0f;
        accumulate_level(kb, tb, s, 1.0f - a, uu, vv, ww, hp);
        if (!single) accumulate_level(kb + 1, tb, s, a, uu, vv, ww, hp);
        u_out = uu;
        v_out = vv;
        w_out = ww;
    }

    /// Variables superficiales en (lon, lat, hora): rapidez del viento a 10 m (compuerta de despegue) y BLH.
    inline void sample_surface(float lon, float lat, float t_h, float& ws10_out, float& blh_out) const noexcept
    {
        const Stencil s = horizontal(lon, lat);
        const TimeBlend tb = temporal(t_h);
        const size_t hp = static_cast<size_t>(n_lat) * static_cast<size_t>(n_lon);
        ws10_out = ws10 ? blend2d(ws10, tb, hp, s) : 0.0f;
        blh_out = blend2d(blh, tb, hp, s);
    }

private:
    struct Stencil {
        size_t p00, p01, p10, p11;
        float w00, w01, w10, w11;
    };
    struct TimeBlend {
        int t0, t1;
        float at;
    };

    inline Stencil horizontal(float lon, float lat) const noexcept
    {
        const float fmax_i = static_cast<float>(n_lat) - 1.001f;
        const float fmax_j = static_cast<float>(n_lon) - 1.001f;
        float fi = (lat_max - lat) * inv_dlat;
        float fj = (lon - lon_min) * inv_dlon;
        fi = fi < 0.0f ? 0.0f : (fi > fmax_i ? fmax_i : fi);
        fj = fj < 0.0f ? 0.0f : (fj > fmax_j ? fmax_j : fj);
        const int i0 = static_cast<int>(fi);
        const int j0 = static_cast<int>(fj);
        const float ai = fi - static_cast<float>(i0);
        const float aj = fj - static_cast<float>(j0);
        Stencil s;
        s.w00 = (1.0f - ai) * (1.0f - aj);
        s.w01 = (1.0f - ai) * aj;
        s.w10 = ai * (1.0f - aj);
        s.w11 = ai * aj;
        s.p00 = static_cast<size_t>(i0) * static_cast<size_t>(n_lon) + static_cast<size_t>(j0);
        s.p01 = s.p00 + 1;
        s.p10 = s.p00 + static_cast<size_t>(n_lon);
        s.p11 = s.p10 + 1;
        return s;
    }

    inline TimeBlend temporal(float t_h) const noexcept
    {
        const float fmax_t = static_cast<float>(n_t) - 1.001f;
        const float t_idx = t_h * inv_stride;
        float ft = t_idx < 0.0f ? 0.0f : (t_idx > fmax_t ? fmax_t : t_idx);
        if (fmax_t < 0.0f) ft = 0.0f;
        TimeBlend tb;
        tb.t0 = static_cast<int>(ft);
        tb.t1 = (tb.t0 + 1 < n_t) ? tb.t0 + 1 : tb.t0;
        tb.at = ft - static_cast<float>(tb.t0);
        return tb;
    }

    static inline float hu(uint64_t c) noexcept { return half_to_float(static_cast<uint16_t>(c & 0xFFFFu)); }
    static inline float hv(uint64_t c) noexcept { return half_to_float(static_cast<uint16_t>((c >> 16) & 0xFFFFu)); }
    static inline float hw(uint64_t c) noexcept { return half_to_float(static_cast<uint16_t>((c >> 32) & 0xFFFFu)); }

    static inline float bilinear(const float* f, const Stencil& s) noexcept
    {
        return s.w00 * f[s.p00] + s.w01 * f[s.p01] + s.w10 * f[s.p10] + s.w11 * f[s.p11];
    }

    /// Campo 2D [t][lat][lon]: bilineal en horizontal y lineal en el tiempo
    static inline float blend2d(const float* base, const TimeBlend& tb, size_t hp, const Stencil& s) noexcept
    {
        const float a = bilinear(base + static_cast<size_t>(tb.t0) * hp, s);
        const float b = bilinear(base + static_cast<size_t>(tb.t1) * hp, s);
        return (1.0f - tb.at) * a + tb.at * b;
    }

    inline void accumulate_level(int lev, const TimeBlend& tb, const Stencil& s, float weight,
                                 float& u, float& v, float& w, size_t hp) const noexcept
    {
        const Wind4Half* a = wind + (static_cast<size_t>(tb.t0) * static_cast<size_t>(n_lev) + static_cast<size_t>(lev)) * hp;
        const Wind4Half* b = wind + (static_cast<size_t>(tb.t1) * static_cast<size_t>(n_lev) + static_cast<size_t>(lev)) * hp;
        const uint64_t a00 = a[s.p00].bits, a01 = a[s.p01].bits, a10 = a[s.p10].bits, a11 = a[s.p11].bits;
        const uint64_t b00 = b[s.p00].bits, b01 = b[s.p01].bits, b10 = b[s.p10].bits, b11 = b[s.p11].bits;
        const float ua = s.w00 * hu(a00) + s.w01 * hu(a01) + s.w10 * hu(a10) + s.w11 * hu(a11);
        const float va = s.w00 * hv(a00) + s.w01 * hv(a01) + s.w10 * hv(a10) + s.w11 * hv(a11);
        const float wa = s.w00 * hw(a00) + s.w01 * hw(a01) + s.w10 * hw(a10) + s.w11 * hw(a11);
        const float ub = s.w00 * hu(b00) + s.w01 * hu(b01) + s.w10 * hu(b10) + s.w11 * hu(b11);
        const float vb = s.w00 * hv(b00) + s.w01 * hv(b01) + s.w10 * hv(b10) + s.w11 * hv(b11);
        const float wb = s.w00 * hw(b00) + s.w01 * hw(b01) + s.w10 * hw(b10) + s.w11 * hw(b11);
        u += weight * ((1.0f - tb.at) * ua + tb.at * ub);
        v += weight * ((1.0f - tb.at) * va + tb.at * vb);
        w += weight * ((1.0f - tb.at) * wa + tb.at * wb);
    }
};

/// Campo ERA5 empaquetado en memoria host (propietario de los buffers).
struct Era5Pack {
    Era5View view;                 ///< Punteros apuntan a los vectores de esta estructura
    std::vector<Wind4Half> wind;
    std::vector<float> blh;
    std::vector<float> zground;
    std::vector<float> ws10;

    /// Altura de la atmósfera estándar [m] para una presión [hPa]
    static float standard_height_m(float p_hpa) noexcept {
        return 44330.77f * (1.0f - std::pow(p_hpa / 1013.25f, 0.190263f));
    }

    /// Límite físico de la velocidad vertical sinóptica empaquetada [m/s] (convección profunda / ondas orográficas)
    static constexpr float kWindWClip = 5.0f;

    /// Empaqueta (u, v, w) en FP16 y las variables superficiales. Devuelve false si el campo no es utilizable.
    /// `time_stride` >= 1: se almacena un paso temporal cada `time_stride` horas (el kernel interpola linealmente entre
    /// ellos). Con 3 h, 6 años de 2,208 h ocupan 3.64 GiB y caben residentes en la VRAM de la Arc A380 (0 transferencias
    /// H2D por evaluación); con 1 h (por defecto) es la serie horaria completa.
    bool build(const io::AtmosphericField3D& f, int time_stride = 1) {
        const size_t stride = static_cast<size_t>(std::max(1, time_stride));
        const size_t n_t_full = static_cast<size_t>(f.n_times);
        const size_t n_t_store = (n_t_full + stride - 1) / stride + ((n_t_full > 1 && (n_t_full - 1) % stride != 0) ? 1 : 0);
        const size_t n_t = stride == 1 ? n_t_full : n_t_store;
        const size_t n_lev = static_cast<size_t>(f.n_levels);
        const size_t n_lat = static_cast<size_t>(f.n_lats), n_lon = static_cast<size_t>(f.n_lons);
        const size_t hp = n_lat * n_lon;
        const size_t total_full = n_t_full * n_lev * hp;
        if (total_full == 0 || n_lev > 16 || n_lev < 2 || n_lat < 2 || n_lon < 2 ||
            f.u.size() != total_full || f.v.size() != total_full || f.w.size() != total_full ||
            f.pressure_levels_hpa.size() != n_lev) {
            return false;
        }
        // Índice temporal de origen de cada paso almacenado (el último paso físico siempre se conserva)
        auto src_t = [&](size_t s) { return std::min(s * stride, n_t_full - 1); };
        const size_t vol = n_lev * hp;
        wind.resize(n_t * vol);
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_t * vol); ++i) {
            const size_t k = static_cast<size_t>(i);
            const size_t src = src_t(k / vol) * vol + (k % vol);
            const float w = f.w[src] < -kWindWClip ? -kWindWClip : (f.w[src] > kWindWClip ? kWindWClip : f.w[src]);
            wind[k] = Wind4Half::make(float_to_half(f.u[src]), float_to_half(f.v[src]), float_to_half(w));
        }
        auto surf = [&](const std::vector<float>& full, std::vector<float>& dst) {
            dst.resize(n_t * hp);
            for (size_t s = 0; s < n_t; ++s) std::copy_n(full.begin() + static_cast<std::ptrdiff_t>(src_t(s) * hp), hp, dst.begin() + static_cast<std::ptrdiff_t>(s * hp));
        };
        blh.assign(n_t * hp, 1200.0f);
        if (f.blh.size() == n_t_full * hp) surf(f.blh, blh);
        zground.assign(n_t * hp, 0.0f);
        if (f.sp.size() == n_t_full * hp) {
            std::vector<float> sp_sub;
            surf(f.sp, sp_sub);
            for (size_t i = 0; i < zground.size(); ++i) zground[i] = standard_height_m(sp_sub[i] * 0.01f);
        }
        ws10.assign(n_t * hp, 0.0f);
        if (f.u10.size() == n_t_full * hp && f.v10.size() == n_t_full * hp) {
            std::vector<float> u10s, v10s;
            surf(f.u10, u10s); surf(f.v10, v10s);
            for (size_t i = 0; i < ws10.size(); ++i) ws10[i] = std::hypot(u10s[i], v10s[i]);
        }

        view = Era5View{};
        view.valid = true;
        view.n_t = static_cast<int>(n_t);
        view.n_hours = static_cast<int>(n_t_full);
        view.inv_stride = 1.0f / static_cast<float>(stride);
        view.n_lev = static_cast<int>(n_lev);
        view.n_lat = static_cast<int>(n_lat);
        view.n_lon = static_cast<int>(n_lon);
        // Geometría: ERA5 latitud descendente, longitud ascendente (se usa lo leído del NetCDF si existe)
        float lat_max = 45.0f, lon_min = -95.0f, dlat = 0.25f, dlon = 0.25f;
        if (f.lats.size() == n_lat && f.lons.size() == n_lon) {
            lat_max = std::max(f.lats.front(), f.lats.back());
            lon_min = std::min(f.lons.front(), f.lons.back());
            dlat = std::fabs(f.lats[1] - f.lats[0]);
            dlon = std::fabs(f.lons[1] - f.lons[0]);
        }
        view.lat_max = lat_max;
        view.lon_min = lon_min;
        view.inv_dlat = 1.0f / dlat;
        view.inv_dlon = 1.0f / dlon;
        for (size_t k = 0; k < n_lev; ++k) view.zlev[k] = standard_height_m(f.pressure_levels_hpa[k]);
        refresh_view();
        return true;
    }

    /// Reapunta la vista a los buffers propios (tras mover/copiar la estructura)
    void refresh_view() {
        view.wind = wind.data();
        view.blh = blh.data();
        view.zground = zground.data();
        view.ws10 = ws10.data();
    }

    [[nodiscard]] size_t wind_bytes() const noexcept { return wind.size() * sizeof(Wind4Half); }
    [[nodiscard]] size_t surface_bytes() const noexcept {
        return (blh.size() + zground.size() + ws10.size()) * sizeof(float);
    }
    [[nodiscard]] size_t bytes() const noexcept { return wind_bytes() + surface_bytes(); }
};

} // namespace joro::lpdm
