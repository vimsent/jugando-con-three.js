// Headless capture / automation driver: starts Vite, opens the lab in Chromium with WebGPU on the
// real GPU (Vulkan), runs a list of steps and writes screenshots + the HUD text + console logs.
//
// Usage:
//   node scripts/capture.mjs [--out results/shots] [--steps '<json>' | --steps-file f.json] [--headed] [--bench [json opts]]
// Steps are objects such as:
//   {"method":"flat"} {"bookmark":2} {"view":"indirecta"} {"wait":120} {"shot":"name"}
//   {"eval":"app.lights.setState({hour:9})"}
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const argv = process.argv.slice(2);
const arg = (name, def) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : def;
};
const has = (name) => argv.includes(name);
const outDir = arg('--out', 'results/shots');
mkdirSync(outDir, { recursive: true });

const defaultSteps = [{ wait: 60 }, { shot: 'final' }];
const stepsFile = arg('--steps-file', null);
const steps = stepsFile ? JSON.parse(readFileSync(stepsFile, 'utf8')) : JSON.parse(arg('--steps', JSON.stringify(defaultSteps)));
const timeoutMs = +arg('--timeout', '600000');

const server = await createServer({ server: { port: +arg('--port', '5199'), strictPort: true }, logLevel: 'error' });
await server.listen();
const url = server.resolvedUrls.local[0] + (arg('--query', '') ? '?' + arg('--query', '') : '');

// Persistent profile so reference images cached in IndexedDB survive between runs.
const profileDir = arg('--profile', 'results/.pw-profile');
const browser = await chromium.launchPersistentContext(profileDir, {
  headless: !has('--headed'),
  viewport: { width: 1280, height: 720 },
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--ignore-gpu-blocklist', '--enable-gpu'],
});
const page = browser.pages()[0] ?? (await browser.newPage());
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

let exitCode = 0;
try {
  await page.goto(url);
  await page.waitForFunction(() => window.__lab && (window.__lab.ready || window.__lab.error), null, { timeout: timeoutMs });
  const err = await page.evaluate(() => window.__lab.error);
  if (err) throw new Error('App init failed: ' + err);

  const waitFrames = (n) =>
    page.evaluate((n) => new Promise((res) => {
      const app = window.__lab.app;
      const start = app.frame;
      const poll = () => (app.frame - start >= n ? res() : requestAnimationFrame(poll));
      poll();
    }), n);

  if (has('--bench')) {
    const opts = JSON.parse(arg('--bench', '{}') || '{}');
    const result = await page.evaluate((o) => window.__lab.app.runBenchmark(o), opts);
    writeFileSync(join(outDir, 'benchmark.json'), JSON.stringify(result, null, 2));
    console.log('benchmark written to', join(outDir, 'benchmark.json'));
  }

  for (const step of steps) {
    if (step.method !== undefined) await page.evaluate((k) => window.__lab.app.setMethod(k), step.method);
    if (step.compare !== undefined) await page.evaluate((k) => window.__lab.app.setCompare(k), step.compare);
    if (step.bookmark !== undefined) await page.evaluate((i) => window.__lab.app.gotoBookmark(i), step.bookmark);
    if (step.view !== undefined) await page.evaluate((v) => (window.__lab.app.settings.view = v), step.view);
    if (step.eval !== undefined) {
      const r = await page.evaluate((code) => {
        const app = window.__lab.app;
        return Promise.resolve(new Function('app', 'return (' + code + ')')(app)).then((v) => JSON.stringify(v ?? null));
      }, step.eval);
      console.log('eval:', step.eval.slice(0, 80), '=>', r?.slice(0, 2000));
    }
    if (step.wait !== undefined) await waitFrames(step.wait);
    if (step.shot !== undefined) {
      const file = join(outDir, step.shot + '.png');
      await page.locator('#view').screenshot({ path: file });
      const hud = await page.locator('#hud').innerText();
      writeFileSync(join(outDir, step.shot + '.hud.txt'), hud);
      console.log('shot:', file);
      console.log(hud);
    }
  }
} catch (e) {
  console.error('FAILED:', e.message);
  exitCode = 1;
} finally {
  const relevant = logs.filter((l) => !l.includes('[vite]') && !l.includes('Download the React DevTools'));
  writeFileSync(join(outDir, 'console.log'), relevant.join('\n'));
  if (relevant.length) console.log('--- console (' + relevant.length + ' lines, first 60) ---\n' + relevant.slice(0, 60).join('\n'));
  await browser.close();
  await server.close();
  process.exit(exitCode);
}
