#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace joro {

/// Constantes físicas fundamentales (unidades SI)
namespace constants {
    constexpr double g = 9.80665;             ///< Aceleración gravitacional [m/s^2]
    constexpr double R_d = 287.058;           ///< Constante de los gases para aire seco [J/(kg*K)]
    constexpr double rho_air_sfc = 1.225;     ///< Densidad del aire a nivel del mar [kg/m^3]
    constexpr double mu_air = 1.81e-5;        ///< Viscosidad dinámica del aire [Pa*s]
    constexpr double von_karman = 0.40;       ///< Constante de von Kármán (adimensional)
}

/// Límites del dominio geográfico (MVP: Este de EE. UU.)
struct BoundingBox {
    double lon_min = -95.0;
    double lon_max = -70.0;
    double lat_min =  25.0;
    double lat_max =  45.0;

    [[nodiscard]] constexpr bool contains(double lon, double lat) const noexcept {
        return (lon >= lon_min && lon <= lon_max && lat >= lat_min && lat <= lat_max);
    }
};

/// Estado cinemático 3D de una partícula en el LPDM
struct Particle3D {
    float x = 0.0f;           ///< Longitud [grados decimales WGS84] o coordenadas proyectadas [m]
    float y = 0.0f;           ///< Latitud [grados decimales WGS84] o coordenadas proyectadas [m]
    float z = 10.0f;          ///< Altura sobre el suelo [m] o geopotencial
    float w_prime = 0.0f;     ///< Fluctuación turbulenta vertical [m/s]
    float mass_mg = 1.0f;     ///< Masa de la araña [mg]
    float silk_length_m = 0.5f; ///< Longitud total de hilo de seda de baloneo [m]
    float posture_chi = 0.8f; ///< Factor aerodinámico de postura (0.5 = compacta, 1.0 = paracaídas)
    uint32_t origin_cell = 0; ///< ID de la celda de origen en grid_5km
    bool is_active = true;    ///< True si está en suspensión, False si aterrizó o salió
};

/// Registro de una celda de la cuadrícula de 5 km x 5 km
struct GridCell {
    int32_t cell_id = 0;
    float centroid_lon = 0.0f;
    float centroid_lat = 0.0f;
    float z_imp = 0.0f;
    float z_borde = 0.0f;
    float z_dosel = 0.0f;
    float pop_density_h = 0.0f;
    float road_density_r = 0.0f;
    int16_t ecoregion_block = 0;
    bool is_seed = false;
};

/// Observación anual por celda
struct AnnualObservation {
    int32_t cell_id = 0;
    int16_t year = 0;
    uint8_t y_detected = 0;   ///< 0 = no detectada, 1 = detectada
    int32_t joro_count = 0;
    int32_t effort_araneae = 0; ///< Esfuerzo Target-Group Background
    int32_t unique_inat_obs = 0;
};

} // namespace joro
