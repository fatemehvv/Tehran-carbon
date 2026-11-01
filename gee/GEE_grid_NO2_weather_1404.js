// ===== Tehran 1404 (2025-03-21 .. 2026-03-20): the held-out test year of the confirmatory analysis.
// Paste in the GEE Code Editor, press Run, then RUN both tasks in the "Tasks" tab. Put the files in the repository:
//   Tehran_NO2_grid_1404.tif         -> data/raw/satellite/
//   Tehran_ERA5_daily_2023_2026.csv  -> data/raw/weather/
//
// 1. Monthly TROPOMI NO2 on the L0 grid (same recipe as GEE_grid_NO2_1402_1403.js): 13 bands m_2025_03 ... m_2026_03
//    plus the VIIRS mean of 2025-03 .. 2026-03 (viirs_2025).
// 2. Daily ERA5-Land weather over the city, 2023-03-21 .. 2026-03-20. 1402 and 1403 are included on purpose: they are
//    compared with the weather file already in use, so the new 1404 values are on the same scale.
var region = ee.Geometry.Rectangle([50.85, 35.35, 51.90, 35.95], 'EPSG:4326', false);
var city = ee.Geometry.Rectangle([51.10, 35.55, 51.62, 35.83], 'EPSG:4326', false);   // the 22 districts

var col = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2').filterBounds(region)
  .select('tropospheric_NO2_column_number_density')
  .map(function(im){return im.updateMask(im.gte(-0.001));});
var first = ee.Date('2025-03-01');
var bands = ee.List.sequence(0, 12).map(function(i){
  var a = first.advance(i, 'month'), b = a.advance(1, 'month');
  return col.filterDate(a, b).mean().rename(a.format('YYYY_MM'));
});
var img = ee.ImageCollection.fromImages(bands).toBands();
img = img.rename(img.bandNames().map(function(n){return ee.String('m_').cat(ee.String(n).slice(-7));}));
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').select('avg_rad');
img = img.addBands(viirs.filterDate('2025-03-01', '2026-03-01').mean().rename('viirs_2025'));
print('bands', img.bandNames());
Export.image.toDrive({image: img.toFloat(), description: 'Tehran_NO2_grid_1404', folder: 'Tehran_1404',
  fileNamePrefix: 'Tehran_NO2_grid_1404', crs: 'EPSG:4326',
  crsTransform: [1/120, 0, 50.85, 0, -1/120, 35.95], dimensions: '126x72', maxPixels: 1e8});

var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate('2023-03-21', '2026-03-21')
  .select(['temperature_2m', 'temperature_2m_min', 'temperature_2m_max',
           'u_component_of_wind_10m', 'v_component_of_wind_10m']);
var table = era.map(function(im){
  var m = im.reduceRegion({reducer: ee.Reducer.mean(), geometry: city, scale: 11132});
  var u = ee.Number(m.get('u_component_of_wind_10m')), v = ee.Number(m.get('v_component_of_wind_10m'));
  return ee.Feature(null, {
    date: im.date().format('YYYY-MM-dd'),
    t2m_mean_C: ee.Number(m.get('temperature_2m')).subtract(273.15),
    t2m_min_C: ee.Number(m.get('temperature_2m_min')).subtract(273.15),
    t2m_max_C: ee.Number(m.get('temperature_2m_max')).subtract(273.15),
    wind10_ms: u.hypot(v)});
});
Export.table.toDrive({collection: table, description: 'Tehran_ERA5_daily_2023_2026', folder: 'Tehran_1404',
  fileNamePrefix: 'Tehran_ERA5_daily_2023_2026', fileFormat: 'CSV',
  selectors: ['date', 't2m_mean_C', 't2m_min_C', 't2m_max_C', 'wind10_ms']});
Map.centerObject(region, 10);
Map.addLayer(img.select(9), {min: 0, max: 0.0003, palette: ['black', 'purple', 'orange', 'yellow']}, 'NO2 2025-12');
