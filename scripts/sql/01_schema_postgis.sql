--------------------------------------------------------------------------------
-- JORO-HPC  ·  01_schema_postgis.sql
-- Esquema relacional y geoespacial para la malla demográfica y el esfuerzo TGB
--
-- Requisitos:
--   - PostgreSQL ≥ 14 con extensión PostGIS ≥ 3.3
--
-- Uso:
--   psql -U joro_admin -d joro_db -f scripts/sql/01_schema_postgis.sql
--------------------------------------------------------------------------------

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Extensiones
-- ─────────────────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Tabla: grid_5km
--    Malla demográfica receptora de 5 km x 5 km.
--    Almacenada preferentemente en EPSG:5070 (NAD83 / Conus Albers Equal Area)
--    para evitar la distorsión de área con la latitud, con transformación a 4326.
--    Alberga las covariables estáticas de NLCD, TIGER/Line y demografía humana.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS grid_5km (
    cell_id         INTEGER         PRIMARY KEY,
    geom            GEOMETRY(Polygon, 5070) NOT NULL,        -- Polígono igual-área 25 km²
    geom_wgs84      GEOMETRY(Polygon, 4326),                 -- Vista en coordenadas geográficas
    centroid_lon    REAL            NOT NULL,                -- Centroide en grados [-95, -70]
    centroid_lat    REAL            NOT NULL,                -- Centroide en grados [25, 45]
    
    -- Covariables ambientales NLCD estandarizadas (media 0, sd 1)
    z_imp           REAL            DEFAULT 0.0,             -- Cobertura impermeable (desarrollo urbano)
    z_borde         REAL            DEFAULT 0.0,             -- Densidad de bordes de bosque
    z_dosel         REAL            DEFAULT 0.0,             -- Cobertura de dosel arbóreo
    
    -- Covariables antropogénicas
    pop_density_h   REAL            DEFAULT 0.0,             -- Densidad poblacional humana H_i (habitantes/km²)
    road_density_r  REAL            DEFAULT 0.0,             -- Densidad vial TIGER/Line R_i (km de vía/km²)
    
    -- Partición espacial y condición inicial
    ecoregion_block SMALLINT,                                -- Bloque EPA Nivel III para validación cruzada
    is_seed_2014_15 BOOLEAN         DEFAULT FALSE,           -- Celdas semilla en NE Georgia (2014-2015)
    created_at      TIMESTAMPTZ     DEFAULT now()
);

-- Índices espaciales y de partición
CREATE INDEX IF NOT EXISTS idx_grid_5km_geom ON grid_5km USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_grid_5km_geom_wgs84 ON grid_5km USING GIST (geom_wgs84);
CREATE INDEX IF NOT EXISTS idx_grid_5km_ecoregion ON grid_5km (ecoregion_block);

-- Trigger para sincronizar geom_wgs84 automáticamente
CREATE OR REPLACE FUNCTION fn_sync_grid_wgs84()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geom_wgs84 := ST_Transform(NEW.geom, 4326);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sync_grid_wgs84 ON grid_5km;
CREATE TRIGGER trg_sync_grid_wgs84
    BEFORE INSERT OR UPDATE ON grid_5km
    FOR EACH ROW EXECUTE FUNCTION fn_sync_grid_wgs84();


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Tabla: cell_annual_observations
--    Matriz anual de ocupación y esfuerzo observacional Target-Group Background
--    (Araneae) según Secciones 9.2 y 13.1 de JORO-HPC.tex.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cell_annual_observations (
    cell_id         INTEGER         REFERENCES grid_5km(cell_id) ON DELETE CASCADE,
    year            SMALLINT        NOT NULL CHECK (year BETWEEN 2010 AND 2030),
    y_joro_detected SMALLINT        NOT NULL DEFAULT 0 CHECK (y_joro_detected IN (0, 1)),
    joro_count      INTEGER         DEFAULT 0,
    effort_araneae  INTEGER         DEFAULT 0,               -- E^{Araneae}_{i,t} (Target-Group Background)
    unique_inat_obs INTEGER         DEFAULT 0,               -- Observadores únicos validados vía API iNaturalist
    PRIMARY KEY (cell_id, year)
);

CREATE INDEX IF NOT EXISTS idx_cell_obs_year ON cell_annual_observations (year);
CREATE INDEX IF NOT EXISTS idx_cell_obs_detected ON cell_annual_observations (y_joro_detected, year);


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Tabla: occurrences_raw
--    Registros de presencia descargados desde GBIF e iNaturalist para auditoría.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS occurrences_raw (
    id              BIGSERIAL       PRIMARY KEY,
    gbif_id         BIGINT          UNIQUE,
    species         TEXT            NOT NULL DEFAULT 'Trichonephila clavata',
    event_date      DATE,
    year            SMALLINT,
    month           SMALLINT,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    geom            GEOMETRY(Point, 4326),
    coord_uncertainty_m DOUBLE PRECISION,
    basis_of_record TEXT,
    cell_id_5km     INTEGER         REFERENCES grid_5km(cell_id),
    created_at      TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_occurrences_raw_geom ON occurrences_raw USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_occurrences_raw_date ON occurrences_raw (year, month);


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. Tabla: env_rasters
--    Catálogo de metadatos de rasters climáticos y geomorfológicos en disco.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS env_rasters (
    id              SERIAL          PRIMARY KEY,
    variable_name   TEXT            NOT NULL,
    source          TEXT            NOT NULL,                -- 'NLCD', 'WorldClim', 'CHELSA'
    resolution_m    REAL,                                    -- 30m, 1000m, etc.
    file_path       TEXT            NOT NULL,
    srid            INTEGER         DEFAULT 5070,
    bbox            GEOMETRY(Polygon, 5070),
    ingested_at     TIMESTAMPTZ     DEFAULT now()
);
