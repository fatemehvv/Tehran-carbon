// DAILY TROPOMI NO2 on the L0 grid + wind at the satellite overpass (input of the daily RQ2b experiments).
// Why: monthly means are very smooth (~4 independent patches per month). Daily fields with the day's wind let a
// graph model learn how emissions are carried downwind - the case where message passing should matter most.
//
// Run in the GEE Code Editor; it creates 4 export tasks (Tasks tab -> RUN each). Put the files in the repository:
//   Tehran_NO2_daily_1402.tif, Tehran_NO2_daily_1403.tif   -> data/raw/satellite/daily/
//   Tehran_wind10UTC_1402.csv, Tehran_wind10UTC_1403.csv   -> data/raw/weather/
// then run (laptop):  python main.py manifest --write   (or ask Claude to regenerate MANIFEST.csv)
//
// Cloudy pixels (cloud fraction >= 0.3) and days without an overpass are written as -9999 (not 0), so they are
// read as missing values; the grid (126 x 72 cells of 1/120 degree, NW corner 50.85E 35.95N) must match
// configs/config.yaml -> grid.
var region = ee.Geometry.Rectangle([50.85, 35.35, 51.90, 35.95], null, false);
var crsTransform = [1 / 120, 0, 50.85, 0, -1 / 120, 35.95];
var NODATA = -9999;

function dailyNO2(start, nDays) {
  var s5p = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
    .filterBounds(region)
    .select(['tropospheric_NO2_column_number_density', 'cloud_fraction']);
  var days = ee.List.sequence(0, nDays - 1);
  var imgs = days.map(function (d) {
    var a = ee.Date(start).advance(d, 'day'), b = a.advance(1, 'day');
    var c = s5p.filterDate(a, b).map(function (im) {
      return im.select(0).updateMask(im.select('cloud_fraction').lt(0.3));
    });
    var img = ee.Image(ee.Algorithms.If(c.size().gt(0), c.mean(), ee.Image.constant(0).updateMask(0)));
    return img.rename(a.format('YYYY_MM_dd')).toFloat();
  });
  return ee.ImageCollection.fromImages(imgs).toBands()
    .reproject({crs: 'EPSG:4326', crsTransform: crsTransform})
    .unmask(NODATA);
}

function exportNO2(start, nDays, name) {
  Export.image.toDrive({image: dailyNO2(start, nDays), description: name, fileNamePrefix: name, region: region,
    crs: 'EPSG:4326', crsTransform: crsTransform, dimensions: '126x72', maxPixels: 1e9});
}

// Jalali 1402 = 2023-03-21 .. 2024-03-19 (365 days); 1403 = 2024-03-20 .. 2025-03-20 (366 days)
exportNO2('2023-03-21', 365, 'Tehran_NO2_daily_1402');
exportNO2('2024-03-20', 366, 'Tehran_NO2_daily_1403');

// Wind at 10 UTC (= 13:30 Tehran, TROPOMI overpass), city-box mean, ERA5-Land hourly
function windTable(start, end) {
  var era = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate(start, end)
    .filter(ee.Filter.calendarRange(10, 10, 'hour'))
    .select(['u_component_of_wind_10m', 'v_component_of_wind_10m', 'temperature_2m']);
  return era.map(function (im) {
    var m = im.reduceRegion({reducer: ee.Reducer.mean(), geometry: region, scale: 11132});
    return ee.Feature(null, m).set('date', im.date().format('YYYY-MM-dd'));
  });
}
Export.table.toDrive({collection: windTable('2023-03-21', '2024-03-20'), description: 'Tehran_wind10UTC_1402',
  fileNamePrefix: 'Tehran_wind10UTC_1402', fileFormat: 'CSV'});
Export.table.toDrive({collection: windTable('2024-03-20', '2025-03-21'), description: 'Tehran_wind10UTC_1403',
  fileNamePrefix: 'Tehran_wind10UTC_1403', fileFormat: 'CSV'});
