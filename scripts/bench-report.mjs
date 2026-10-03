// Turns results/benchmark.json into CSV files, an SVG chart of the dynamic-response curves and the
// comparison table in README.md (between <!-- BENCH:START --> and <!-- BENCH:END -->).
//
// Usage: node scripts/bench-report.mjs [results/benchmark.json]
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

const file = process.argv[2] ?? 'results/benchmark.json';
const outDir = dirname(file);
const d = JSON.parse(readFileSync(file, 'utf8'));

const MiB = (b) => (b / 2 ** 20).toFixed(1);
const f = (x, n = 2) => (x === null || x === undefined || Number.isNaN(x) ? '–' : x.toFixed(n));
const pct = (x) => (x === null || x === undefined || Number.isNaN(x) ? '–' : (x * 100).toFixed(1) + ' %');
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const rays = (r) => (!r || Number.isNaN(r) ? '0' : r >= 1e6 ? (r / 1e6).toFixed(2) + ' M' : (r / 1e3).toFixed(0) + ' k');

// ---- CSV: one row per method x bookmark, and one per pass
const rows = [['method', 'bookmark', 'gi_gpu_ms_mean', 'gi_gpu_ms_median', 'gi_gpu_ms_p95', 'frame_gpu_ms_mean', 'rays_per_frame', 'memory_bytes', 'shared_bytes', 'memory_measured_delta', 'rmse_mean', 'rel_mean', 'ref_spp']];
const passRows = [['method', 'bookmark', 'pass', 'mean_ms', 'median_ms', 'p95_ms', 'frames']];
for (const m of d.methods) {
  for (const b of m.bookmarks) {
    rows.push([m.key, b.bookmark, b.giGpuMs.mean, b.giGpuMs.median, b.giGpuMs.p95, b.frameGpuMs.mean, b.raysPerFrame.mean || 0, m.memory.declared, m.memory.shared, m.memory.measuredDelta, b.error?.rmse.mean ?? '', b.error?.rel.mean ?? '', b.error?.refSpp ?? '']);
    for (const [p, s] of Object.entries(b.passes)) passRows.push([m.key, b.bookmark, p, s.mean, s.median, s.p95, s.n]);
  }
}
const toCsv = (r) => r.map((x) => x.join(',')).join('\n') + '\n';
writeFileSync(join(outDir, 'benchmark.csv'), toCsv(rows));
writeFileSync(join(outDir, 'passes.csv'), toCsv(passRows));

// ---- Summary table (averaged over bookmarks)
const summary = d.methods.map((m) => {
  const bs = m.bookmarks;
  const errs = bs.filter((b) => b.error);
  return {
    m,
    gi: mean(bs.map((b) => b.giGpuMs.mean)),
    giP95: Math.max(...bs.map((b) => b.giGpuMs.p95)),
    frame: mean(bs.map((b) => b.frameGpuMs.mean)),
    rays: mean(bs.map((b) => b.raysPerFrame.mean || 0)),
    rmse: errs.length ? mean(errs.map((b) => b.error.rmse.mean)) : NaN,
    rel: errs.length ? mean(errs.map((b) => b.error.rel.mean)) : NaN,
  };
});

let md = '';
md += `Medido el ${d.date.slice(0, 10)} · ${d.env.adapter.vendor ?? '?'} ${d.env.adapter.architecture ?? ''} · three r${d.env.threeRevision} · ${d.env.resolution.join('×')} · `;
md += `${d.options.warmup} frames de calentamiento + ${d.options.measure} medidos por método y cámara · referencia ${d.references[0]?.spp ?? '?'} spp`;
md += d.env.timestampQuery ? ' · tiempos de GPU por timestamp queries.\n\n' : ' · **sin timestamp-query: tiempos de CPU**.\n\n';
md += '| # | Método | GI GPU ms (media) | GI GPU ms (p95 peor cámara) | Frame GPU ms | Rayos/frame | Memoria propia | RMSE | Error rel. |\n';
md += '|---|---|---:|---:|---:|---:|---:|---:|---:|\n';
for (const s of summary) {
  const mem = `${MiB(s.m.memory.declared)} MiB${s.m.memory.shared ? ` (+${MiB(s.m.memory.shared)} BVH)` : ''}`;
  md += `| ${s.m.hotkey} | ${s.m.label} | ${f(s.gi, 3)} | ${f(s.giP95, 3)} | ${f(s.frame, 3)} | ${rays(s.rays)} | ${mem} | ${s.rmse.toExponential ? s.rmse.toExponential(2) : '–'} | ${pct(s.rel)} |\n`;
}
if (d.referenceTiming) {
  md += `| 0 | Referencia (1 spp/frame, para escala) | ${f(d.referenceTiming.giGpuMs.mean, 2)} | ${f(d.referenceTiming.giGpuMs.p95, 2)} | ${f(d.referenceTiming.frameGpuMs.mean, 2)} | ${rays(d.referenceTiming.raysPerFrame.mean)} | – | – | – |\n`;
}
md += '\n"GI GPU ms" suma solo los pases del método (excluye G-buffer + sombras, composición, display y la métrica). ';
md += '"Memoria propia" es lo que declara cada método (texturas + buffers); el BVH (+geometría) se comparte entre los métodos con rayos.\n\n';

// Per-bookmark relative error
md += '**Error relativo por cámara** (L1 de luminancia, lineal, antes del tonemapping):\n\n';
md += '| Método | ' + d.bookmarks.map((n, i) => `${i + 1}. ${n}`).join(' | ') + ' |\n';
md += '|---|' + d.bookmarks.map(() => '---:').join('|') + '|\n';
for (const m of d.methods) {
  md += `| ${m.label} | ` + d.bookmarks.map((_, i) => pct(m.bookmarks.find((b) => b.bookmark === i)?.error?.rel.mean)).join(' | ') + ' |\n';
}
md += '\n';

// Memory: declared vs measured
md += '**Memoria**: declarada por el método vs. delta medido en `renderer.info.memory.total` al crearlo (el BVH ya existía):\n\n';
md += '| Método | Declarada | Delta medido | Desglose |\n|---|---:|---:|---|\n';
for (const m of d.methods) {
  const br = Object.entries(m.memory.breakdown).map(([k, v]) => `${k}: ${MiB(v)}`).join('; ');
  md += `| ${m.label} | ${MiB(m.memory.declared)} MiB | ${MiB(m.memory.measuredDelta)} MiB | ${br} |\n`;
}
md += '\n';

// Dynamic response
const conv = d.methods.filter((m) => m.convergence);
if (conv.length) {
  const c0 = conv[0].convergence;
  md += `**Respuesta dinámica** (cámara ${c0.bookmark + 1}, sol ${c0.fromHour} h → ${c0.toHour} h, error contra la referencia del nuevo estado):\n\n`;
  md += `| Método | Error justo después | Error estable | t90 (frames) | Frames hasta error < ${(d.convergenceThreshold * 100).toFixed(0)} % |\n|---|---:|---:|---:|---:|\n`;
  for (const m of conv) {
    const c = m.convergence;
    md += `| ${m.label} | ${pct(c.initialRel)} | ${pct(c.steadyRel)} | ${c.t90 ?? 'no llega'} | ${c.framesToAbsThreshold ?? 'no llega'} |\n`;
  }
  md += '\n![Curvas de error tras el cambio de sol](results/convergence.svg)\n\nLas curvas que no se ven quedan exactamente debajo de otra (p. ej. 1 y 2, con el mismo error). La tabla de arriba tiene los valores.\n\n';
  writeFileSync(join(outDir, 'convergence.svg'), convergenceSvg(conv, d.convergenceThreshold));
}

if (d.noiseFloor) {
  md += `**Piso de ruido de la referencia** (${d.noiseFloor.spp} spp, estimado con una segunda referencia independiente): RMSE ≈ ${d.noiseFloor.rmse.toExponential(2)}, error relativo ≈ ${pct(d.noiseFloor.rel)}. `;
  md += 'Diferencias entre métodos por debajo de ese nivel no son significativas.\n\n';
}
if (d.calibration?.values) {
  const v = d.calibration.values;
  md += `**Parámetros calibrados contra la referencia** (búsqueda en grilla, mínimo error relativo medio en las 5 cámaras): ambiente de "Sin GI" = ${v.flat}, ambiente de GTAO = ${v.gtao}, SSGI: ambiente = ${v.ssgiAmbient}, escala GI = ${v.ssgiGi}. `;
  md += 'Es un ajuste con oráculo que favorece a esos métodos; los métodos con rayos no se calibraron.\n';
}

const readme = readFileSync('README.md', 'utf8');
const start = '<!-- BENCH:START -->';
const end = '<!-- BENCH:END -->';
if (readme.includes(start) && readme.includes(end)) {
  const out = readme.slice(0, readme.indexOf(start) + start.length) + '\n' + md + readme.slice(readme.indexOf(end));
  writeFileSync('README.md', out);
  console.log('README.md updated');
} else {
  console.log(md);
}
console.log('wrote', join(outDir, 'benchmark.csv'), join(outDir, 'passes.csv'));

// ---- SVG line chart (static, light surface; categorical slots in fixed order)
function convergenceSvg(methods, threshold) {
  const series = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
  const W = 760, H = 380, L = 56, R = 190, T = 36, B = 44;
  const pw = W - L - R, ph = H - T - B;
  const maxF = Math.max(...methods.map((m) => m.convergence.frames));
  const maxE = Math.min(1.5, Math.max(...methods.flatMap((m) => m.convergence.curve.map((p) => p[1])))) * 1.05;
  const x = (fr) => L + (fr / maxF) * pw;
  const y = (e) => T + ph - (Math.min(e, maxE) / maxE) * ph;
  const ticks = (max, n) => Array.from({ length: n + 1 }, (_, i) => (max * i) / n);
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="system-ui, sans-serif" font-size="12">\n`;
  s += `<rect width="${W}" height="${H}" fill="#fcfcfb"/>\n`;
  s += `<text x="${L}" y="20" fill="#0b0b0b" font-size="14" font-weight="600">Error relativo tras cambiar el sol (frames desde el cambio)</text>\n`;
  for (const t of ticks(maxE, 5)) s += `<line x1="${L}" x2="${L + pw}" y1="${y(t)}" y2="${y(t)}" stroke="#e6e5e1"/><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end" fill="#52514e">${(t * 100).toFixed(0)}%</text>\n`;
  for (const t of ticks(maxF, 6)) s += `<text x="${x(t)}" y="${T + ph + 18}" text-anchor="middle" fill="#52514e">${Math.round(t)}</text>\n`;
  s += `<line x1="${L}" x2="${L + pw}" y1="${T + ph}" y2="${T + ph}" stroke="#a3a29b"/>\n`;
  s += `<line x1="${L}" x2="${L + pw}" y1="${y(threshold)}" y2="${y(threshold)}" stroke="#52514e" stroke-dasharray="4 4"/><text x="${L + pw - 4}" y="${y(threshold) - 5}" text-anchor="end" fill="#52514e">umbral ${(threshold * 100).toFixed(0)}%</text>\n`;
  methods.forEach((m, i) => {
    const c = series[i % series.length];
    const pts = m.convergence.curve.map(([fr, e]) => `${x(fr).toFixed(1)},${y(e).toFixed(1)}`).join(' ');
    s += `<polyline fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" points="${pts}"><title>${m.label}</title></polyline>\n`;
    const ly = T + 10 + i * 20;
    s += `<line x1="${L + pw + 16}" x2="${L + pw + 36}" y1="${ly}" y2="${ly}" stroke="${c}" stroke-width="3" stroke-linecap="round"/><text x="${L + pw + 42}" y="${ly + 4}" fill="#0b0b0b">${m.hotkey}. ${m.label}</text>\n`;
  });
  s += `<text x="${L + pw / 2}" y="${H - 8}" text-anchor="middle" fill="#52514e">frames</text>\n</svg>\n`;
  return s;
}
