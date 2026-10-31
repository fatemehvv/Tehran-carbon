// Pretrained satellite embeddings (Google DeepMind AlphaEarth Foundations) averaged on the L0 grid.
// Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL  (64 bands A00..A63, 10 m, one image per year and tile)
// Run in the GEE Code Editor -> Tasks tab -> RUN both tasks. Put the two GeoTIFFs in data/raw/embeddings/:
//     alphaearth_L0_2023.tif  (Jalali 1402)      alphaearth_L0_2024.tif  (Jalali 1403)
// Then set data.inputs.embeddings.enabled: true (experiment rq1_gated_emb_v1 already does) and run
//     python main.py manifest --write
// The grid must match configs/config.yaml -> grid (126 x 72 cells of 1/120 degree, NW corner 50.85E 35.95N).

var region = ee.Geometry.Rectangle([50.85, 35.35, 51.90, 35.95], null, false);
var crsTransform = [1 / 120, 0, 50.85, 0, -1 / 120, 35.95];

function exportYear(year) {
  var img = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    .filterDate(year + '-01-01', (year + 1) + '-01-01')
    .filterBounds(region)
    .mosaic()
    .setDefaultProjection({crs: 'EPSG:4326', scale: 10})
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 65535})   // mean of the 10 m pixels in each cell
    .reproject({crs: 'EPSG:4326', crsTransform: crsTransform})
    .toFloat();
  Export.image.toDrive({
    image: img,
    description: 'alphaearth_L0_' + year,
    fileNamePrefix: 'alphaearth_L0_' + year,
    region: region,
    crs: 'EPSG:4326',
    crsTransform: crsTransform,
    maxPixels: 1e9
  });
}

exportYear(2023);
exportYear(2024);
