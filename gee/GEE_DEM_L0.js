// ===== Tehran: mean elevation of every L0 cell (paper, branch paper_TC; check A5 of docs/paper/plan_q1_fa.md).
// Why: the north of the city is ~700 m higher than the south. A column over high ground holds less air, and
// Rey-Pommier et al. (2025, ESSD) report artefacts in the TROPOMI NOx of Tehran near Tochal. The paper checks how much
// of the hotspot map elevation alone explains.
// Paste in the GEE Code Editor, press Run, then RUN the task in the "Tasks" tab. Put the file in the repository:
//   dem_L0.tif -> data/raw/spatial/dem_L0.tif      then (laptop or server): python main.py manifest --write
// Source: SRTM 1 arc-second (USGS/SRTMGL1_003), averaged over each 1/120 degree cell (about 900 SRTM pixels).
var region = ee.Geometry.Rectangle([50.85, 35.35, 51.90, 35.95], 'EPSG:4326', false);
var crsTransform = [1 / 120, 0, 50.85, 0, -1 / 120, 35.95];
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 4096})
  .reproject({crs: 'EPSG:4326', crsTransform: crsTransform})
  .rename('elevation_m');
Export.image.toDrive({image: dem.toFloat(), description: 'dem_L0', folder: 'Tehran_paper', fileNamePrefix: 'dem_L0',
  crs: 'EPSG:4326', crsTransform: crsTransform, dimensions: '126x72', maxPixels: 1e8});
Map.centerObject(region, 10);
Map.addLayer(dem, {min: 900, max: 2500, palette: ['black', 'white']}, 'elevation (m)');
