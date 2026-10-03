import type { App } from '../app';
import { BOOKMARKS } from '../scene/bookmarks';
import { BENCH_MEASURE_FRAMES, BENCH_WARMUP_FRAMES, CONVERGENCE_REL_ERROR, RENDER_HEIGHT, RENDER_WIDTH } from '../config';
import { METHODS } from '../gi/registry';
import type { ErrorResult } from '../metrics/error';
import { ReferenceCache } from '../metrics/referenceCache';
import type { ReferencePathTracer } from '../gi/reference';
import { REVISION } from 'three';

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface BenchmarkOptions {
  methods?: string[];
  bookmarks?: number[];
  warmup?: number;
  measure?: number;
  refSpp?: number;
  calibrate?: boolean;
  convergence?: { bookmark: number; fromHour: number; toHour: number; frames: number } | null;
  noiseFloor?: boolean;
  referenceTiming?: boolean;
}

interface Stat {
  mean: number;
  median: number;
  p95: number;
  n: number;
}

function stat(values: number[]): Stat {
  if (values.length === 0) return { mean: NaN, median: NaN, p95: NaN, n: 0 };
  const s = [...values].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
  return { mean: s.reduce((a, b) => a + b, 0) / s.length, median: q(0.5), p95: q(0.95), n: s.length };
}

const COMMON_PASSES = new Set(['gbuffer+sombras', 'composición', 'display', 'métrica (no GI)']);

/**
 * Benchmark driver (runs in the page; launched by `npm run bench` through Playwright or from the
 * GUI). Everything is measured with the same fixed light state and internal resolution.
 */
export class Benchmark {
  private log: (s: string) => void;
  constructor(private readonly app: App) {
    this.log = (s) => {
      console.info('[bench]', s);
      app.setStatus(`Benchmark: ${s}`);
    };
  }

  async run(opts: BenchmarkOptions = {}): Promise<any> {
    const app = this.app;
    const o = {
      methods: opts.methods ?? METHODS.filter((m) => m.key !== 'reference').map((m) => m.key),
      bookmarks: opts.bookmarks ?? BOOKMARKS.map((_, i) => i),
      warmup: opts.warmup ?? BENCH_WARMUP_FRAMES,
      measure: opts.measure ?? BENCH_MEASURE_FRAMES,
      refSpp: opts.refSpp ?? 2048,
      calibrate: opts.calibrate ?? true,
      convergence: opts.convergence === undefined ? { bookmark: 0, fromHour: 13.5, toHour: 9.5, frames: 360 } : opts.convergence,
      noiseFloor: opts.noiseFloor ?? true,
      referenceTiming: opts.referenceTiming ?? true,
    };
    const t0 = performance.now();
    app.settings.paused = true;
    app.settings.compare = 'ninguno';
    await app.setCompare('ninguno');
    const baseLights = { hour: o.convergence?.fromHour ?? 13.5, animateSun: false, animatePoint: false, pointPhase: 0 };
    app.lights.setState(baseLights);

    // 1. References (cached in IndexedDB across runs).
    this.log('referencias');
    const refInfo = await this.ensureReferences(o.bookmarks, o.refSpp);
    if (o.convergence) {
      app.lights.setState({ ...baseLights, hour: o.convergence.toHour });
      refInfo.push(...(await this.ensureReferences([o.convergence.bookmark], o.refSpp)));
      app.lights.setState(baseLights);
    }

    // 2. Reference noise floor: a second, independent reference for bookmark 0.
    let noiseFloor: any = null;
    if (o.noiseFloor) {
      this.log('piso de ruido de la referencia');
      noiseFloor = await this.measureNoiseFloor(o.bookmarks[0], o.refSpp);
    }

    // 3. Make sure the shared BVH exists so it does not count towards a method's memory delta.
    await app.getRTScene();

    // 4. Calibration of free parameters of the screen-space family.
    let calibration: any = null;
    if (o.calibrate) calibration = await this.calibrate(o.bookmarks);

    // 5. Per method x bookmark measurements.
    const methods: any[] = [];
    for (const key of o.methods) {
      const entry = METHODS.find((m) => m.key === key)!;
      this.log(`${entry.label}: memoria`);
      await app.setMethod('flat');
      await app.waitFrames(3);
      const m0 = app.renderer.info.memory.total;
      await app.setMethod(key);
      await app.waitFrames(5);
      const m1 = app.renderer.info.memory.total;
      const st = app.methodA!.method.stats();
      const result: any = {
        key,
        label: entry.label,
        hotkey: entry.hotkey,
        params: this.methodParams(),
        memory: { declared: st.memoryBytes, shared: st.sharedBytes ?? 0, measuredDelta: m1 - m0, breakdown: st.memoryBreakdown ?? {} },
        bookmarks: [],
      };
      for (const b of o.bookmarks) {
        this.log(`${entry.label}: cámara ${b + 1}`);
        result.bookmarks.push(await this.measureBookmark(b, o.warmup, o.measure));
      }
      if (o.convergence) {
        this.log(`${entry.label}: respuesta dinámica`);
        result.convergence = await this.measureConvergence(o.convergence, o.warmup);
        app.lights.setState(baseLights);
      }
      methods.push(result);
    }

    // 6. Reference path tracer cost (1 spp per frame), for scale.
    let referenceTiming: any = null;
    if (o.referenceTiming) {
      this.log('costo de la referencia');
      await app.setMethod('reference');
      const ref = app.methodA!.method as ReferencePathTracer;
      ref.params.sppPerFrame = 1;
      ref.params.targetSpp = 1 << 30;
      app.gotoBookmark(o.bookmarks[0]);
      referenceTiming = await this.measureBookmark(o.bookmarks[0], 30, 120, false);
    }

    await app.setMethod('flat');
    app.setStatus('');
    const adapterInfo = (app.renderer.backend as any).device?.adapterInfo ?? {};
    return {
      date: new Date().toISOString(),
      durationSeconds: (performance.now() - t0) / 1000,
      env: {
        userAgent: navigator.userAgent,
        adapter: { vendor: adapterInfo.vendor, architecture: adapterInfo.architecture, description: adapterInfo.description },
        threeRevision: REVISION,
        resolution: [RENDER_WIDTH, RENDER_HEIGHT],
        timestampQuery: app.timer.supported,
      },
      options: o,
      lights: app.lights.getState(),
      references: refInfo,
      noiseFloor,
      calibration,
      methods,
      referenceTiming,
      bookmarks: BOOKMARKS.map((b) => b.name),
      convergenceThreshold: CONVERGENCE_REL_ERROR,
    };
  }

  private async ensureReferences(bookmarks: number[], spp: number): Promise<any[]> {
    const app = this.app;
    const out: any[] = [];
    for (const b of bookmarks) {
      const key = ReferenceCache.key(b, app.lights.stateKey);
      app.refCache.get(key); // trigger IndexedDB load
      await app.waitFrames(3);
      const e = app.refCache.get(key);
      if (e && e.spp >= spp) {
        out.push({ key, spp: e.spp, cached: true });
        continue;
      }
      const [r] = await app.generateReferences([b], spp, 8);
      out.push({ ...r, cached: false });
    }
    return out;
  }

  private async measureNoiseFloor(bookmark: number, spp: number): Promise<any> {
    const app = this.app;
    const key = ReferenceCache.key(bookmark, app.lights.stateKey);
    const a = app.refCache.get(key)!;
    const [r] = await app.generateReferences([bookmark], spp, 8, '#indep');
    const b = app.refCache.get(r.key)!;
    const err = await app.errorMetric.measure(app.renderer, b.final, a.final, app.frame);
    // Two independent estimates: the error of one of them against the truth is ≈ err / √2.
    return { spp, rmse: err.rmse / Math.SQRT2, rel: err.rel / Math.SQRT2, rawBetweenTwo: err };
  }

  /** Average relative error over the given bookmarks after `settle` frames each. */
  private async meanError(bookmarks: number[], settle: number): Promise<number> {
    const app = this.app;
    let sum = 0;
    for (const b of bookmarks) {
      app.gotoBookmark(b);
      await app.waitFrames(settle);
      const ref = app.currentReference()!;
      const e = await app.errorMetric.measure(app.renderer, app.compositeA.target.texture, ref.final, app.frame);
      sum += e.rel;
    }
    return sum / bookmarks.length;
  }

  /**
   * Picks, by grid search on the mean relative error over the bookmarks, the ambient intensity of
   * methods 1–3 and SSGI's GI scale. This is an oracle tuning that favours those methods; it is
   * reported as such.
   */
  private async calibrate(bookmarks: number[]): Promise<any> {
    const app = this.app;
    const res: any = {};
    const search = async (method: string, apply: (v: number) => void, values: number[], settle: number) => {
      await app.setMethod(method);
      let best = { value: values[0], rel: Infinity };
      const curve: [number, number][] = [];
      for (const v of values) {
        this.log(`calibrando ${method}: ${v}`);
        apply(v);
        app.methodA!.method.reset?.();
        const rel = await this.meanError(bookmarks, settle);
        curve.push([v, rel]);
        if (rel < best.rel) best = { value: v, rel };
      }
      return { ...best, curve };
    };
    const amb = (m: any) => (v: number) => (m().ambient.params.intensity = v);
    const cur = () => app.methodA!.method as any;
    const ambientValues = [0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.03, 0.045, 0.07];
    res.flat = await search('flat', amb(cur), ambientValues, 3);
    res.gtao = await search('gtao', amb(cur), ambientValues, 40);
    const ssgiAmb = await search('ssgi', amb(cur), ambientValues, 60);
    const ambBest = ssgiAmb.value;
    const ssgiGi = await search('ssgi', (v) => {
      cur().ambient.params.intensity = ambBest;
      cur().params.giIntensity = v;
    }, [0, 1, 2, 4, 8, 16, 32, 64], 60);
    // Second coordinate-descent step: re-fit the ambient with the chosen GI scale.
    const ssgiAmb2 = await search('ssgi', (v) => {
      cur().ambient.params.intensity = v;
      cur().params.giIntensity = ssgiGi.value;
    }, ambientValues, 60);
    const ambFinal = ssgiAmb2.value;
    res.ssgi = { ambient: ssgiAmb, giIntensity: ssgiGi, ambient2: ssgiAmb2 };
    // Keep the calibrated values for every later instance of these methods (benchmark + gallery).
    const v = { flat: res.flat.value, gtao: res.gtao.value, ssgiAmbient: ambFinal, ssgiGi: ssgiGi.value };
    app.methodOverrides.flat = (m: any) => (m.ambient.params.intensity = v.flat);
    app.methodOverrides.gtao = (m: any) => (m.ambient.params.intensity = v.gtao);
    app.methodOverrides.ssgi = (m: any) => {
      m.ambient.params.intensity = v.ssgiAmbient;
      m.params.giIntensity = v.ssgiGi;
    };
    res.values = v;
    return res;
  }

  private methodParams(): any {
    const m = this.app.methodA!.method as any;
    const out: any = {};
    if (m.params) out.params = { ...m.params };
    if (m.ambient) out.ambient = { ...m.ambient.params };
    if (m.volume) out.ddgi = { ...m.volume.params };
    if (m.ssdo) out.ssdo = { ...m.ssdo };
    if (m.atrous) out.atrous = { ...m.atrous.params };
    return out;
  }

  private async measureBookmark(b: number, warmup: number, measure: number, withError = true): Promise<any> {
    const app = this.app;
    app.gotoBookmark(b);
    await app.waitFrames(warmup);
    const frames: Map<string, number>[] = [];
    const startFrame = app.renderer.info.frame;
    app.timer.onFrame = (f, passes) => {
      if (f >= startFrame) frames.push(passes);
    };
    const errors: ErrorResult[] = [];
    const rays: number[] = [];
    const prevOnResult = app.errorMetric.onResult;
    app.errorMetric.onResult = (r) => errors.push(r);
    const t0 = performance.now();
    const f0 = app.frame;
    while (app.frame - f0 < measure) {
      await app.waitFrames(1);
      rays.push(app.methodA!.method.stats().raysPerFrame);
    }
    const wall = performance.now() - t0;
    await app.waitFrames(4); // let the last timestamp resolves arrive
    app.timer.onFrame = null;
    app.errorMetric.onResult = prevOnResult;

    const passNames = new Set<string>();
    frames.forEach((m) => m.forEach((_, k) => passNames.add(k)));
    const passes: Record<string, Stat> = {};
    for (const name of passNames) passes[name] = stat(frames.map((m) => m.get(name) ?? 0));
    const giTotals = frames.map((m) => [...m].filter(([k]) => !COMMON_PASSES.has(k)).reduce((s, [, v]) => s + v, 0));
    const frameTotals = frames.map((m) => [...m].filter(([k]) => k !== 'métrica (no GI)').reduce((s, [, v]) => s + v, 0));
    const ref = app.currentReference();
    const last = errors[errors.length - 1];
    return {
      bookmark: b,
      name: BOOKMARKS[b].name,
      framesTimed: frames.length,
      passes,
      giGpuMs: stat(giTotals),
      frameGpuMs: stat(frameTotals),
      wallFps: (measure / wall) * 1000,
      raysPerFrame: stat(rays.filter((r) => r > 0)),
      error: withError && ref
        ? { refSpp: ref.spp, rmse: stat(errors.map((e) => e.rmse)), rel: stat(errors.map((e) => e.rel)), last: last ?? null }
        : null,
    };
  }

  private async measureConvergence(c: { bookmark: number; fromHour: number; toHour: number; frames: number }, warmup: number): Promise<any> {
    const app = this.app;
    app.lights.setState({ hour: c.fromHour, animateSun: false, animatePoint: false, pointPhase: 0 });
    app.gotoBookmark(c.bookmark);
    await app.waitFrames(warmup);
    const curve: [number, number, number][] = [];
    const prev = app.errorMetric.onResult;
    let switchFrame = 0;
    app.errorMetric.onResult = (r) => {
      if (r.frame > switchFrame) curve.push([r.frame - switchFrame, r.rel, r.rmse]);
    };
    switchFrame = app.frame;
    app.lights.setState({ hour: c.toHour });
    await app.waitFrames(c.frames);
    await app.waitFrames(3);
    app.errorMetric.onResult = prev;
    const tail = curve.filter(([f]) => f > c.frames - 60).map(([, rel]) => rel);
    const steady = tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : NaN;
    const initial = curve.length ? curve[0][1] : NaN;
    const firstBelow = (thr: number) => {
      // first frame from which the error stays below thr for the next 10 samples
      for (let i = 0; i < curve.length; i++) {
        if (curve.slice(i, i + 10).every(([, rel]) => rel <= thr)) return curve[i][0];
      }
      return null;
    };
    return {
      ...c,
      initialRel: initial,
      steadyRel: steady,
      // frames until the error first stays below the absolute threshold
      framesToAbsThreshold: firstBelow(CONVERGENCE_REL_ERROR),
      // t90: frames to cover 90 % of the way from the post-change error to the steady-state error
      t90: initial - steady > 0.01 * steady ? firstBelow(steady + 0.1 * (initial - steady)) : 0,
      curve,
    };
  }
}
