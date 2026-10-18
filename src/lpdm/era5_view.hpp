#pragma once

#include "io/era5_netcdf_reader.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace joro::lpdm {

/// Viento (u, v, w) empaquetado en 16 B para que cada esquina del interpolador sea UNA sola carga.
struct alignas(16) Wind4 {
    float u = 0.0f;
    float v = 0.0f;
    float w = 0.0f;
    float pad = 0.0f;
};

/// Vista de solo lectura del campo ERA5 empaquetado (punteros a memoria host o Device USM).
/// `sample` es aritmética pura: se compila igual en host y en el kernel SYCL.
struct Era5View {
    bool valid = false;
    int n_t = 0, n_lev = 0, n_lat = 0, n_lon = 0;
    float lat_max = 45.0f, lon_min = -95.0f, inv_dlat = 4.0f, inv_dlon = 4.0f;
    float zlev[16] = {};            ///< Altura estándar de cada nivel isobárico [m] (ascendente)
    const Wind4* wind = nullptr;    ///< [t][nivel][lat][lon]
    const float* blh = nullptr;     ///< [t][lat][lon] altura de capa límite [m]
    const float* zground = nullptr; ///< [t][lat][lon] altura estándar de la presión superficial [m]

    /// Viento (u, v, w), BLH y altura del terreno interpolados en (lon, lat, z sobre el suelo, hora).
    /// Interpolación bilineal horizontal + lineal en tiempo + lineal en altura sobre el suelo.
    inline void sample(float lon, float lat, float z_agl, float t_h,
                       float& u_out, float& v_out, float& w_out, float& blh_out) const noexcept
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
        const float w00 = (1.0f - ai) * (1.0f - aj), w01 = (1.0f - ai) * aj;
        const float w10 = ai * (1.0f - aj),          w11 = ai * aj;

        const float fmax_t = static_cast<float>(n_t) - 1.001f;
        float ft = t_h < 0.0f ? 0.0f : (t_h > fmax_t ? fmax_t : t_h);
        if (fmax_t < 0.0f) ft = 0.0f;
        const int t0 = static_cast<int>(ft);
        const int t1 = (t0 + 1 < n_t) ? t0 + 1 : t0;
        const float at = ft - static_cast<float>(t0);

        const size_t hp = static_cast<size_t>(n_lat) * static_cast<size_t>(n_lon);
        const size_t p00 = static_cast<size_t>(i0) * static_cast<size_t>(n_lon) + static_cast<size_t>(j0);
        const size_t p01 = p00 + 1, p10 = p00 + static_cast<size_t>(n_lon), p11 = p10 + 1;

        // Terreno (hora t0) y BLH (lineal en tiempo)
        const float* zg0 = zground + static_cast<size_t>(t0) * hp;
        const float zg = w00 * zg0[p00] + w01 * zg0[p01] + w10 * zg0[p10] + w11 * zg0[p11];
        const float* b0 = blh + static_cast<size_t>(t0) * hp;
        const float* b1 = blh + static_cast<size_t>(t1) * hp;
        const float blh_a = w00 * b0[p00] + w01 * b0[p01] + w10 * b0[p10] + w11 * b0[p11];
        const float blh_b = w00 * b1[p00] + w01 * b1[p01] + w10 * b1[p10] + w11 * b1[p11];
        blh_out = (1.0f - at) * blh_a + at * blh_b;

        // Niveles que rodean a z_agl (alturas ascendentes; los niveles bajo el terreno se descartan)
        int k = 0;
        while (k < n_lev - 1 && (zlev[k + 1] - zg) <= z_agl) ++k;
        int kb = k;           // nivel inferior
        float a = 0.0f;       // peso del nivel superior
        bool single = true;
        if (k + 1 < n_lev && (zlev[k] - zg) > 0.0f) {
            // z_agl entre dos niveles sobre el terreno
            const float zlo = zlev[k] - zg, zhi = zlev[k + 1] - zg;
            a = (z_agl - zlo) / (zhi - zlo);
            a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
            single = false;
        } else if (k + 1 < n_lev && (zlev[k] - zg) <= 0.0f) {
            kb = k + 1;       // el nivel k está bajo el suelo: se usa el primero sobre el terreno
        }

        float uu = 0.0f, vv = 0.0f, ww = 0.0f;
        accumulate_level(kb, t0, t1, at, p00, p01, p10, p11, w00, w01, w10, w11, 1.0f - a, uu, vv, ww, hp);
        if (!single) {
            accumulate_level(kb + 1, t0, t1, at, p00, p01, p10, p11, w00, w01, w10, w11, a, uu, vv, ww, hp);
        }
        u_out = uu;
        v_out = vv;
        w_out = ww;
    }

private:
    inline void accumulate_level(int lev, int t0, int t1, float at,
                                 size_t p00, size_t p01, size_t p10, size_t p11,
                                 float w00, float w01, float w10, float w11, float weight,
                                 float& u, float& v, float& w, size_t hp) const noexcept
    {
        const Wind4* a = wind + (static_cast<size_t>(t0) * static_cast<size_t>(n_lev) + static_cast<size_t>(lev)) * hp;
        const Wind4* b = wind + (static_cast<size_t>(t1) * static_cast<size_t>(n_lev) + static_cast<size_t>(lev)) * hp;
        const float ua = w00 * a[p00].u + w01 * a[p01].u + w10 * a[p10].u + w11 * a[p11].u;
        const float va = w00 * a[p00].v + w01 * a[p01].v + w10 * a[p10].v + w11 * a[p11].v;
        const float wa = w00 * a[p00].w + w01 * a[p01].w + w10 * a[p10].w + w11 * a[p11].w;
        const float ub = w00 * b[p00].u + w01 * b[p01].u + w10 * b[p10].u + w11 * b[p11].u;
        const float vb = w00 * b[p00].v + w01 * b[p01].v + w10 * b[p10].v + w11 * b[p11].v;
        const float wb = w00 * b[p00].w + w01 * b[p01].w + w10 * b[p10].w + w11 * b[p11].w;
        u += weight * ((1.0f - at) * ua + at * ub);
        v += weight * ((1.0f - at) * va + at * vb);
        w += weight * ((1.0f - at) * wa + at * wb);
    }
};

/// Campo ERA5 empaquetado en memoria host (propietario de los buffers).
struct Era5Pack {
    Era5View view;                 ///< Punteros apuntan a los vectores de esta estructura
    std::vector<Wind4> wind;
    std::vector<float> blh;
    std::vector<float> zground;

    /// Altura de la atmósfera estándar [m] para una presión [hPa]
    static float standard_height_m(float p_hpa) noexcept {
        return 44330.77f * (1.0f - std::pow(p_hpa / 1013.25f, 0.190263f));
    }

    /// Empaqueta (u, v, w) y las variables superficiales. Devuelve false si el campo no es utilizable.
    bool build(const io::AtmosphericField3D& f) {
        const size_t n_t = static_cast<size_t>(f.n_times), n_lev = static_cast<size_t>(f.n_levels);
        const size_t n_lat = static_cast<size_t>(f.n_lats), n_lon = static_cast<size_t>(f.n_lons);
        const size_t hp = n_lat * n_lon;
        const size_t total = n_t * n_lev * hp;
        if (total == 0 || n_lev > 16 || n_lev < 2 || n_lat < 2 || n_lon < 2 ||
            f.u.size() != total || f.v.size() != total || f.w.size() != total ||
            f.pressure_levels_hpa.size() != n_lev) {
            return false;
        }
        wind.resize(total);
        for (size_t i = 0; i < total; ++i) {
            const float w = f.w[i] < -1.0f ? -1.0f : (f.w[i] > 1.0f ? 1.0f : f.w[i]);
            wind[i] = Wind4{f.u[i], f.v[i], w, 0.0f};
        }
        blh.assign(n_t * hp, 1200.0f);
        if (f.blh.size() == n_t * hp) blh = f.blh;
        zground.assign(n_t * hp, 0.0f);
        if (f.sp.size() == n_t * hp) {
            for (size_t i = 0; i < zground.size(); ++i) zground[i] = standard_height_m(f.sp[i] * 0.01f);
        }

        view = Era5View{};
        view.valid = true;
        view.n_t = static_cast<int>(n_t);
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
    }

    [[nodiscard]] size_t bytes() const noexcept {
        return wind.size() * sizeof(Wind4) + (blh.size() + zground.size()) * sizeof(float);
    }
};

} // namespace joro::lpdm
