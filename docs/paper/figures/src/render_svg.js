// Render an SVG figure to PDF (vector, fonts embedded) and PNG (2x) with the preinstalled Chromium.
// Usage: node render_svg.js <figure.svg> [font-folder]
//   font-folder: optional folder with Vazirmatn-Regular.ttf and Vazirmatn-Bold.ttf (for the Persian figures;
//   e.g. from `npm pack vazirmatn`, OFL licence). The fonts are inlined only for rendering, not written to the SVG.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const svgPath = path.resolve(process.argv[2]);
  const fontDir = process.argv[3];
  const svg = fs.readFileSync(svgPath, 'utf8');
  const m = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
  const [w, h] = [Math.ceil(+m[1]), Math.ceil(+m[2])];
  let faces = '';
  if (fontDir) {
    for (const [file, weight] of [['Vazirmatn-Regular.ttf', 400], ['Vazirmatn-Bold.ttf', 700]]) {
      const b64 = fs.readFileSync(path.join(fontDir, file)).toString('base64');
      faces += `@font-face{font-family:Vazirmatn;font-weight:${weight};src:url(data:font/ttf;base64,${b64}) format("truetype")}`;
    }
  }
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>${faces}html,body{margin:0;padding:0;background:#fff}
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
