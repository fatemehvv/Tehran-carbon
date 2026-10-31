// ===== Tehran: monthly Sentinel-5P NO2 on the 1 km study grid (1/120 deg, same as ODIAC and L0), 1402 + 1403.
// Paste in GEE Code Editor, press Run, then RUN the single task in the "Tasks" tab.
// Output: Google Drive folder Tehran_1402, file Tehran_NO2_grid_2023_2025.tif (25 bands = months 2023-03 ... 2025-03)
var region = ee.Geometry.Rectangle([50.85, 35.35, 51.90, 35.95], 'EPSG:4326', false);
var col = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2').filterBounds(region)
  .select('tropospheric_NO2_column_number_density')
  .map(function(im){return im.updateMask(im.gte(-0.001));});
var first = ee.Date('2023-03-01');
var bands = ee.List.sequence(0, 24).map(function(i){
  var a = first.advance(i, 'month'), b = a.advance(1, 'month');
  return col.filterDate(a, b).mean().rename(a.format('YYYY_MM'));
});
var img = ee.ImageCollection.fromImages(bands).toBands();
img = img.rename(img.bandNames().map(function(n){return ee.String('m_').cat(ee.String(n).slice(-7));}));
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').select('avg_rad');
img = img.addBands(viirs.filterDate('2023-03-01','2024-03-01').mean().rename('viirs_2023'))
         .addBands(viirs.filterDate('2024-03-01','2025-03-01').mean().rename('viirs_2024'));
print('bands', img.bandNames());
Export.image.toDrive({image: img.toFloat(), description: 'Tehran_NO2_grid_2023_2025', folder: 'Tehran_1402',
  fileNamePrefix: 'Tehran_NO2_grid_2023_2025', crs: 'EPSG:4326',
  crsTransform: [1/120, 0, 50.85, 0, -1/120, 35.95], dimensions: '126x72', maxPixels: 1e8});
Map.centerObject(region, 10);
Map.addLayer(img.select(10), {min: 0, max: 0.0003, palette: ['black', 'purple', 'orange', 'yellow']}, 'NO2 2024-01');
