// ===== Tehran 1405 (from 2026-03-21): the second, untouched test year of the paper (docs/paper/plan_q1_fa.md, stage D).
// Only months 1-7 (Farvardin .. Mehr 1405 = 2026-03-21 .. 2026-10-22) are complete. Nothing in the code reads this
// file before stage D; the decision rules are registered first (docs/paper/preregistration_paper_v2.md).
// Paste in the GEE Code Editor, press Run, then RUN both tasks in the "Tasks" tab. Put the files in the repository:
//   Tehran_NO2_grid_1405.tif               -> data/raw/satellite/
//   Tehran_ERA5_daily_2023_2026b.csv       -> data/raw/weather/
// then harmonise the weather (same mapping as 1404; the 1402-1404 rows stay exactly as they are):
//   python gee/harmonise_weather_1404.py data/raw/weather/weather_daily_era5_1402_1404.csv \
//       data/raw/weather/Tehran_ERA5_daily_2023_2026b.csv data/raw/weather/weather_daily_era5_1402_1405.csv
//   python main.py manifest --write
//
// 1. Monthly TROPOMI NO2 on the L0 grid (same recipe as the 1402-1404 scripts): bands m_2026_03 ... m_2026_10. The
//    Jalali months 1-7 read the Gregorian month of largest overlap (April ... October). Later months are absent; the
//    loader leaves them empty (data.partial_years).
// 2. viirs_2026: VIIRS night lights, mean of the 12 most recent monthly composites available (the newest months reach
//    GEE with a delay of a few months).
// 3. Daily ERA5-Land weather over the city, 2023-03-21 .. 2026-10-31 (overlap with the old file for the mapping).
var region = ee.Geometry.Rectangle([50.85, 35.35, 51.90, 35.95], 'EPSG:4326', false);
var city = ee.Geometry.Rectangle([51.10, 35.55, 51.62, 35.83], 'EPSG:4326', false);   // the 22 districts

var col = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2').filterBounds(region)
  .select('tropospheric_NO2_column_number_density')
  .map(function(im){return im.updateMask(im.gte(-0.001));});
var first = ee.Date('2026-03-01');
var bands = ee.List.sequence(0, 7).map(function(i){
  var a = first.advance(i, 'month'), b = a.advance(1, 'month');
  return col.filterDate(a, b).mean().rename(a.format('YYYY_MM'));
});
var img = ee.ImageCollection.fromImages(bands).toBands();
img = img.rename(img.bandNames().map(function(n){return ee.String('m_').cat(ee.String(n).slice(-7));}));
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').select('avg_rad')
  .filterDate('2025-01-01', '2026-11-01').sort('system:time_start', false).limit(12);
print('VIIRS months used for viirs_2026', viirs.aggregate_array('system:index'));
img = img.addBands(viirs.mean().rename('viirs_2026'));
print('bands', img.bandNames());
Export.image.toDrive({image: img.toFloat(), description: 'Tehran_NO2_grid_1405', folder: 'Tehran_1405',
  fileNamePrefix: 'Tehran_NO2_grid_1405', crs: 'EPSG:4326',
  crsTransform: [1/120, 0, 50.85, 0, -1/120, 35.95], dimensions: '126x72', maxPixels: 1e8});

var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate('2023-03-21', '2026-11-01')
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
Export.table.toDrive({collection: table, description: 'Tehran_ERA5_daily_2023_2026b', folder: 'Tehran_1405',
  fileNamePrefix: 'Tehran_ERA5_daily_2023_2026b', fileFormat: 'CSV',
  selectors: ['date', 't2m_mean_C', 't2m_min_C', 't2m_max_C', 'wind10_ms']});
