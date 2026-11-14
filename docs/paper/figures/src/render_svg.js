// Render an SVG figure to PDF (vector) and PNG (2x) with the preinstalled Chromium.
// Usage: node render_svg.js <figure.svg>   -> <figure>.pdf and <figure>.png next to it
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const svgPath = path.resolve(process.argv[2]);
  const svg = fs.readFileSync(svgPath, 'utf8');
  const m = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
  const [w, h] = [Math.ceil(+m[1]), Math.ceil(+m[2])];
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;background:#fff}
svg{display:block}</style></head><body>${svg}</body></html>`;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
  await page.setContent(html);
  await page.evaluate(() => document.fonts.ready);
  const base = svgPath.replace(/\.svg$/, '');
  await page.screenshot({ path: base + '.png', clip: { x: 0, y: 0, width: w, height: h } });
  await page.pdf({ path: base + '.pdf', width: w + 'px', height: h + 'px', printBackground: true, pageRanges: '1' });
  await browser.close();
  console.log(base + '.pdf', base + '.png');
})();
