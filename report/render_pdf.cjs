// Renders report/HyperBot_Project_Report.html to PDF with headless Chromium.
// Usage: python3 report/build_report.py && node report/render_pdf.cjs
const path = require('path');
const { execSync } = require('child_process');
let playwright;
try { playwright = require('playwright'); } catch {
  playwright = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));
}

(async () => {
  const src = path.join(__dirname, 'HyperBot_Project_Report.html');
  const out = path.join(__dirname, 'HyperBot_Project_Report.pdf');
  const browser = await playwright.chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + src, { waitUntil: 'load' });
  await page.pdf({
    path: out, format: 'A4', printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="width:100%;font-size:8px;color:#888;font-family:Inter,sans-serif;padding:0 18mm;display:flex;justify-content:space-between">' +
      '<span>HyperBot: Voice-Enabled Supercar &amp; Hypercar Chatbot</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',
  });
  await browser.close();
  console.log('wrote', out);
})();
