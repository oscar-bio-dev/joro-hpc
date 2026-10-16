--------------------------------------------------------------------------------
-- JORO-HPC  ·  02_spatial_blocks.sql
-- Partición de la malla 5x5 km en bloques espaciales para validación cruzada
-- según la Sección 12 del anteproyecto JORO-HPC.tex
--------------------------------------------------------------------------------

-- Vista de partición para la fase de calibración (2014-2019)
CREATE OR REPLACE VIEW v_calibration_set AS
SELECT 
    g.cell_id,
    g.geom,
    g.centroid_lon,
    g.centroid_lat,
    g.z_imp,
    g.z_borde,
    g.z_dosel,
    g.pop_density_h,
    g.road_density_r,
    g.ecoregion_block,
    obs.year,
    obs.y_joro_detected,
    obs.joro_count,
    obs.effort_araneae
FROM grid_5km g
JOIN cell_annual_observations obs ON g.cell_id = obs.cell_id
WHERE obs.year BETWEEN 2014 AND 2019;

-- Vista de partición para la prueba fuera de muestra (2020-2024 congelada)
CREATE OR REPLACE VIEW v_evaluation_set AS
SELECT 
    g.cell_id,
    g.geom,
    g.centroid_lon,
    g.centroid_lat,
    g.z_imp,
    g.z_borde,
    g.z_dosel,
    g.pop_density_h,
    g.road_density_r,
    g.ecoregion_block,
    obs.year,
    obs.y_joro_detected,
    obs.joro_count,
    obs.effort_araneae
FROM grid_5km g
JOIN cell_annual_observations obs ON g.cell_id = obs.cell_id
WHERE obs.year BETWEEN 2020 AND 2024;
