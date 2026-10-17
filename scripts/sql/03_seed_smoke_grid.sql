--------------------------------------------------------------------------------
-- JORO-HPC  ·  03_seed_smoke_grid.sql
-- Población sintética ultrarrápida de la malla demográfica EPSG:5070 para Smoke Test
--
-- Genera una cuadrícula de 5 km x 5 km sobre el Bounding Box de Norteamérica
-- (Lon: [-95.0, -70.0], Lat: [25.0, 45.0]) con covariables dummy (0.5)
-- y activa al menos 10 celdas semilla en NE Georgia (~34.0-34.5° N, 83.2-83.8° W).
--
-- Uso:
--   psql -h localhost -p 5433 -U joro_admin -d joro_db -f scripts/sql/03_seed_smoke_grid.sql
--------------------------------------------------------------------------------

BEGIN;

-- 1. Desactivar temporalmente el trigger para máxima velocidad de ingesta masiva
ALTER TABLE grid_5km DISABLE TRIGGER trg_sync_grid_wgs84;

-- 2. Limpiar registros anteriores si existieran
TRUNCATE TABLE grid_5km CASCADE;

-- 3. Generar la cuadrícula sintética en EPSG:5070 usando ST_SquareGrid de PostGIS
WITH raw_grid AS (
    SELECT 
        (row_number() OVER ())::INTEGER AS cell_id,
        geom
    FROM ST_SquareGrid(
        5000, -- Resolución 5000 m (5 km x 5 km)
        ST_Transform(ST_MakeEnvelope(-95.0, 25.0, -70.0, 45.0, 4326), 5070)
    )
)
INSERT INTO grid_5km (
    cell_id,
    geom,
    geom_wgs84,
    centroid_lon,
    centroid_lat,
    z_imp,
    z_borde,
    z_dosel,
    pop_density_h,
    road_density_r,
    ecoregion_block,
    is_seed_2014_15
)
SELECT 
    g.cell_id,
    g.geom,
    ST_Transform(g.geom, 4326) AS geom_wgs84,
    ST_X(ST_Centroid(ST_Transform(g.geom, 4326)))::REAL AS centroid_lon,
    ST_Y(ST_Centroid(ST_Transform(g.geom, 4326)))::REAL AS centroid_lat,
    0.5::REAL AS z_imp,            -- Covariable dummy estandarizada
    0.5::REAL AS z_borde,          -- Covariable dummy estandarizada
    0.5::REAL AS z_dosel,          -- Covariable dummy estandarizada
    15.0::REAL AS pop_density_h,   -- Densidad poblacional humana base
    1.2::REAL AS road_density_r,   -- Densidad vial TIGER/Line base
    1::SMALLINT AS ecoregion_block,-- Partición espacial base para CV
    (
        -- Foco de invasión histórico 2014-2015 en el Noreste de Georgia (condados Braselton / Hoschton)
        ST_X(ST_Centroid(ST_Transform(g.geom, 4326))) BETWEEN -83.8 AND -83.2
        AND ST_Y(ST_Centroid(ST_Transform(g.geom, 4326))) BETWEEN 34.0 AND 34.5
    ) AS is_seed_2014_15
FROM raw_grid g;

-- 4. Reactivar el trigger para sincronización en inserciones futuras
ALTER TABLE grid_5km ENABLE TRIGGER trg_sync_grid_wgs84;

COMMIT;

-- 5. Actualizar estadísticas del optimizador de PostgreSQL
VACUUM ANALYZE grid_5km;
