import { Matrix4, PerspectiveCamera, Scene, Texture } from 'three';
import { WebGPURenderer } from 'three/webgpu';
import GUI from 'lil-gui';

import { CONVERGENCE_REL_ERROR, RENDER_HEIGHT, RENDER_WIDTH, SPONZA_URL } from './config';
import { loadSponza, type SponzaScene } from './scene/sponza';
import { SceneLights } from './scene/lights';
import { BOOKMARKS, applyBookmark } from './scene/bookmarks';
import { SpectatorControls } from './scene/spectator';
import { GBuffer } from './render/gbuffer';
import { Composite } from './render/composite';
import { Display, VIEW_MODES, type ViewMode } from './render/display';
import { SkyUniforms } from './render/sky';
import { GpuTimer } from './metrics/gpuTimer';
import { Hud } from './ui/hud';
import { METHODS, findMethod, type MethodEntry } from './gi/registry';
import type { FrameInfo, GIContext, GIMethod } from './gi/types';
import { RTScene } from './rt/rtScene';
import { ErrorMetric, type ErrorResult } from './metrics/error';
import { ReferenceCache, type ReferenceEntry } from './metrics/referenceCache';
import type { ReferencePathTracer } from './gi/reference';
import { Benchmark, type BenchmarkOptions } from './bench/benchmark';

interface ActiveMethod {
  entry: MethodEntry;
  method: GIMethod;
  folder: GUI | null;
}

export class App {
  renderer!: WebGPURenderer;
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(60, RENDER_WIDTH / RENDER_HEIGHT, 0.05, 120);
  controls!: SpectatorControls;
  lights!: SceneLights;
  sponza!: SponzaScene;
  gbuffer!: GBuffer;
  sky = new SkyUniforms();
  compositeA!: Composite;
  compositeB!: Composite;
  display!: Display;
  timer!: GpuTimer;
  hud: Hud;
  gui!: GUI;

  errorMetric!: ErrorMetric;
  refCache!: ReferenceCache;
  lastError: ErrorResult | null = null;
  private lightChangeFrame = 0;
  private convergedAfter: number | null = null;
  methodA: ActiveMethod | null = null;
  methodB: ActiveMethod | null = null;
  readonly settings = {
    method: 'flat',
    compare: 'ninguno',
    view: 'final' as ViewMode,
    split: 0.5,
    exposure: 4.0,
    bookmark: 0,
    paused: false,
  };

  frame = 0;
  private lastTime = performance.now();
  private fpsAvg = 60;
  private lastCamMatrix = new Matrix4();
  private seenLightVersion = -1;
  private rtScenePromise: Promise<RTScene> | null = null;
  private status = '';
  private switching = false;

  constructor(private readonly canvas: HTMLCanvasElement, hudEl: HTMLElement) {
    this.hud = new Hud(hudEl);
  }

  async init(onStatus: (s: string) => void): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU no disponible en este navegador (navigator.gpu es undefined).');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No se obtuvo un adaptador WebGPU.');
    const L = adapter.limits;
    this.renderer = new WebGPURenderer({
      canvas: this.canvas,
      antialias: false,
      trackTimestamp: true,
      powerPreference: 'high-performance',
      requiredLimits: {
        maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxBufferSize: L.maxBufferSize,
        maxStorageTexturesPerShaderStage: L.maxStorageTexturesPerShaderStage,
        maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
        maxColorAttachmentBytesPerSample: L.maxColorAttachmentBytesPerSample,
      },
    } as ConstructorParameters<typeof WebGPURenderer>[0]);
    await this.renderer.init();
    if (!(this.renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend) {
      throw new Error('three.js cayó al backend WebGL2: este laboratorio requiere WebGPU.');
    }
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(RENDER_WIDTH, RENDER_HEIGHT, false);
    this.renderer.shadowMap.enabled = true;
    this.timer = new GpuTimer(this.renderer);

    onStatus('Cargando Sponza…');
    this.sponza = await loadSponza(SPONZA_URL, (f) => onStatus(`Cargando Sponza… ${(f * 100).toFixed(0)} %`));
    this.scene.add(this.sponza.root);
    this.lights = new SceneLights(this.scene);

    this.controls = new SpectatorControls(this.camera, this.canvas);
    this.gotoBookmark(0);

    this.gbuffer = new GBuffer(RENDER_WIDTH, RENDER_HEIGHT);
    this.compositeA = new Composite(this.gbuffer, this.camera, this.sky);
    this.compositeB = new Composite(this.gbuffer, this.camera, this.sky);
    this.display = new Display(this.gbuffer);
    this.errorMetric = new ErrorMetric(this.gbuffer);
    this.refCache = new ReferenceCache(RENDER_WIDTH, RENDER_HEIGHT);

    this.buildGui();
    this.bindKeys();
    onStatus('Compilando shaders…');
    await this.setMethod(this.settings.method);
    onStatus('');
    this.renderer.setAnimationLoop(() => this.tick());
  }

  get context(): GIContext {
    return {
      renderer: this.renderer,
      scene: this.scene,
      camera: this.camera,
      lights: this.lights,
      gbuffer: this.gbuffer,
      width: RENDER_WIDTH,
      height: RENDER_HEIGHT,
      timer: this.timer,
      getRTScene: () => this.getRTScene(),
    };
  }

  getRTScene(): Promise<RTScene> {
    if (!this.rtScenePromise) {
      this.rtScenePromise = (async () => {
        this.setStatus('Construyendo BVH…');
        await new Promise((r) => setTimeout(r, 0));
        const rt = new RTScene(this.sponza);
        console.info(`BVH: ${rt.triangleCount} triángulos, ${(rt.buildMs / 1000).toFixed(2)} s, ${(rt.memoryBytes / 2 ** 20).toFixed(1)} MiB`);
        this.setStatus('');
        return rt;
      })();
    }
    return this.rtScenePromise;
  }

  // ---------------------------------------------------------------- methods

  async setMethod(key: string): Promise<void> {
    const entry = findMethod(key);
    if (!entry || this.switching) return;
    this.switching = true;
    try {
      this.disposeActive(this.methodA);
      this.methodA = await this.createActive(entry, 'Método A');
      this.settings.method = key;
      this.timer.resetAverages();
      this.gui?.controllersRecursive().forEach((c) => c.updateDisplay());
    } finally {
      this.switching = false;
    }
  }

  async setCompare(key: string): Promise<void> {
    this.disposeActive(this.methodB);
    this.methodB = null;
    this.settings.compare = key;
    const entry = findMethod(key);
    if (entry) this.methodB = await this.createActive(entry, 'Método B');
    this.timer.resetAverages();
  }

  /** Parameter overrides applied to methods right after creation (set by the benchmark calibration). */
  readonly methodOverrides: Record<string, (m: GIMethod) => void> = {};

  private async createActive(entry: MethodEntry, title: string): Promise<ActiveMethod> {
    const method = await entry.create();
    await method.init(this.context);
    this.methodOverrides[entry.key]?.(method);
    const folder = this.gui ? this.gui.addFolder(`${title}: ${entry.label}`) : null;
    if (folder) method.buildGui?.(folder);
    return { entry, method, folder };
  }

  private disposeActive(a: ActiveMethod | null): void {
    if (!a) return;
    a.method.dispose();
    a.folder?.destroy();
  }

  // ---------------------------------------------------------------- frame

  private tick(): void {
    const now = performance.now();
    const dt = Math.min((now - this.lastTime) / 1000, 0.1);
    this.lastTime = now;
    this.fpsAvg = this.fpsAvg * 0.95 + (1 / Math.max(dt, 1e-4)) * 0.05;
    this.controls.update(dt);
    if (this.switching || !this.methodA) return;
    this.renderFrame(this.settings.paused ? 0 : dt);
  }

  /** Renders one frame. Exposed for deterministic stepping from the benchmark. */
  renderFrame(dt: number): void {
    this.frame++;
    this.camera.updateMatrixWorld();
    const cameraMoved = !this.camera.matrixWorld.equals(this.lastCamMatrix);
    this.lastCamMatrix.copy(this.camera.matrixWorld);
    const lightsChanged = this.lights.update(dt) || this.lights.version !== this.seenLightVersion;
    this.seenLightVersion = this.lights.version;
    if (lightsChanged) {
      this.lightChangeFrame = this.frame;
      this.convergedAfter = null;
    }
    this.sky.update(this.lights);
    const info: FrameInfo = { frame: this.frame, time: now(), cameraMoved, lightsChanged };

    const { renderer, timer } = this;
    this.methodA!.method.update(dt, info);
    this.methodB?.method.update(dt, info);

    timer.begin('gbuffer+sombras');
    this.gbuffer.render(renderer, this.scene, this.camera);
    timer.end('gbuffer+sombras');

    const indirectA = this.methodA!.method.run(this.gbuffer, { prevFinal: this.compositeA.target.texture });
    timer.begin('composición');
    this.compositeA.render(renderer, indirectA);
    timer.end('composición');

    let indirectB: Texture | null = null;
    if (this.methodB) {
      indirectB = this.methodB.method.run(this.gbuffer, { prevFinal: this.compositeB.target.texture });
      this.compositeB.render(renderer, indirectB);
    }

    // Reference image for this camera + light state, if one was generated.
    const ref = this.currentReference();
    if (ref && this.methodA!.entry.key !== 'reference') {
      timer.begin('métrica (no GI)');
      this.errorMetric.run(renderer, this.compositeA.target.texture, ref.final, this.frame);
      timer.end('métrica (no GI)');
    }
    this.lastError = ref ? this.errorMetric.last : null;
    const le = this.lastError;
    if (le && this.convergedAfter === null && le.frame > this.lightChangeFrame && le.rel < CONVERGENCE_REL_ERROR) {
      this.convergedAfter = le.frame - this.lightChangeFrame;
    }

    const compareRef = this.settings.compare === 'ref-img' && ref;
    this.display.split.value = this.methodB || compareRef ? this.settings.split : 1;
    this.display.exposure.value = this.settings.exposure;
    this.display.setModeByName(this.settings.view);
    timer.begin('display');
    this.display.render(renderer, {
      finalA: this.compositeA.target.texture,
      indirectA,
      finalB: this.methodB ? this.compositeB.target.texture : compareRef ? ref.final : null,
      indirectB: this.methodB ? indirectB : compareRef ? ref.indirect : null,
      reference: ref ? ref.final : null,
    });
    timer.end('display');
    timer.resolve();
    this.updateHud();
  }

  private updateHud(): void {
    const a = this.methodA!;
    const stats = a.method.stats();
    const passes = [...this.timer.avg.entries()];
    const total = passes.filter(([k]) => !k.includes('no GI')).reduce((s, [, v]) => s + v, 0);
    const ref = this.currentReference();
    this.hud.update({
      method: `${a.entry.hotkey}. ${a.entry.label}`,
      compare: this.methodB ? this.methodB.entry.label : null,
      bookmark: `${this.settings.bookmark + 1}. ${BOOKMARKS[this.settings.bookmark].name}`,
      view: this.settings.view,
      fps: this.fpsAvg,
      passes,
      gpuTotal: total,
      timerSupported: this.timer.supported,
      rays: stats.raysPerFrame,
      memory: stats.memoryBytes,
      shared: stats.sharedBytes ?? 0,
      error: ref && this.lastError ? { rmse: this.lastError.rmse, rel: this.lastError.rel, refSamples: ref.spp } : null,
      converge: ref
        ? this.convergedAfter !== null
          ? `${this.convergedAfter} frames tras el último cambio de luz (umbral ${CONVERGENCE_REL_ERROR * 100} %)`
          : `aún no bajo ${CONVERGENCE_REL_ERROR * 100} % (${this.frame - this.lightChangeFrame} frames desde el cambio de luz)`
        : null,
      extra: stats.extra ?? {},
      status: this.status,
    });
  }

  // ---------------------------------------------------------------- references

  get referenceKey(): string {
    return ReferenceCache.key(this.settings.bookmark, this.lights.stateKey);
  }

  currentReference(): ReferenceEntry | undefined {
    return this.refCache?.get(this.referenceKey);
  }

  /** Resolves after `n` more frames have been rendered. */
  waitFrames(n: number): Promise<void> {
    const target = this.frame + n;
    return new Promise((resolve) => {
      const poll = () => (this.frame >= target ? resolve() : requestAnimationFrame(poll));
      poll();
    });
  }

  /**
   * Renders the ground truth for the current light state at the given bookmark(s) with the
   * reference path tracer and stores it in the cache. Animations are paused meanwhile.
   */
  async generateReferences(bookmarks: number[] = [this.settings.bookmark], spp = 4096, sppPerFrame = 8, keySuffix = ''): Promise<{ key: string; spp: number; seconds: number }[]> {
    const prevMethod = this.settings.method;
    const prevBookmark = this.settings.bookmark;
    const prevPaused = this.settings.paused;
    this.settings.paused = true;
    const out: { key: string; spp: number; seconds: number }[] = [];
    try {
      await this.setMethod('reference');
      const ref = this.methodA!.method as ReferencePathTracer;
      ref.params.targetSpp = spp;
      ref.params.sppPerFrame = sppPerFrame;
      for (const b of bookmarks) {
        this.gotoBookmark(b);
        const t0 = performance.now();
        await this.waitFrames(1);
        ref.reset();
        while (ref.sampleCount < spp) {
          this.setStatus(`Referencia cámara ${b + 1}: ${ref.sampleCount} / ${spp} spp`);
          await this.waitFrames(1);
        }
        await this.waitFrames(1);
        const key = this.referenceKey + keySuffix;
        await this.refCache.store(
          this.renderer,
          { key, bookmark: b, lightKey: this.lights.stateKey, spp: ref.sampleCount },
          this.compositeA.target.texture,
          ref.outputTexture,
        );
        out.push({ key, spp: ref.sampleCount, seconds: (performance.now() - t0) / 1000 });
      }
    } finally {
      this.setStatus('');
      this.settings.paused = prevPaused;
      await this.setMethod(prevMethod);
      this.gotoBookmark(prevBookmark);
    }
    return out;
  }

  /** Runs the full benchmark (see src/bench/benchmark.ts) and returns the JSON result. */
  async runBenchmark(opts: BenchmarkOptions = {}): Promise<unknown> {
    return new Benchmark(this).run(opts);
  }

  // ---------------------------------------------------------------- camera / UI

  gotoBookmark(i: number): void {
    this.settings.bookmark = i;
    applyBookmark(this.camera, BOOKMARKS[i]);
    this.methodA?.method.onCameraCut?.();
    this.methodB?.method.onCameraCut?.();
  }

  private buildGui(): void {
    const gui = (this.gui = new GUI({ title: 'Sponza GI Lab' }));
    const g = gui.addFolder('General');
    const methodOptions = Object.fromEntries(METHODS.map((m) => [`${m.hotkey}. ${m.label}`, m.key]));
    g.add(this.settings, 'method', methodOptions).name('método A').onChange((k: string) => this.setMethod(k));
    g.add(this.settings, 'compare', { ninguno: 'ninguno', 'imagen de referencia': 'ref-img', ...methodOptions }).name('comparar con B').onChange((k: string) => this.setCompare(k));
    g.add(this.settings, 'split', 0, 1, 0.001).name('split A|B');
    g.add(this.settings, 'view', [...VIEW_MODES]).name('vista');
    g.add(this.settings, 'exposure', 0.05, 8, 0.01).name('exposición');
    g.add(this.settings, 'bookmark', Object.fromEntries(BOOKMARKS.map((b, i) => [`${i + 1}. ${b.name}`, i]))).name('cámara').onChange((i: number) => this.gotoBookmark(i));
    g.add(this.settings, 'paused').name('pausar animación');
    g.add(this.display.diffGain, 'value', 0.1, 50, 0.1).name('ganancia diferencia');

    const r = gui.addFolder('Referencia');
    const refOpts = { spp: 4096 };
    r.add(refOpts, 'spp', 256, 65536, 256).name('spp');
    r.add({ go: () => void this.generateReferences([this.settings.bookmark], refOpts.spp) }, 'go').name('generar (cámara actual)');
    r.add({ go: () => void this.generateReferences(BOOKMARKS.map((_, i) => i), refOpts.spp) }, 'go').name('generar (5 cámaras)');
    r.add({ go: () => void this.refCache.clear() }, 'go').name('borrar referencias guardadas');

    const b = gui.addFolder('Benchmark');
    const benchOpts = { warmup: 300, measure: 600, refSpp: 2048, calibrate: true };
    b.add(benchOpts, 'warmup', 0, 1000, 10).name('frames calentamiento');
    b.add(benchOpts, 'measure', 10, 2000, 10).name('frames medidos');
    b.add(benchOpts, 'refSpp', 256, 16384, 256).name('spp referencia');
    b.add(benchOpts, 'calibrate').name('calibrar parámetros libres');
    b.add({
      go: async () => {
        const result = await this.runBenchmark(benchOpts);
        const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'benchmark.json';
        a.click();
      },
    }, 'go').name('ejecutar y descargar JSON');
    b.close();

    const l = gui.addFolder('Luces');
    const p = this.lights.params;
    const apply = () => this.lights.apply();
    l.add(p, 'hour', 6, 18, 0.01).name('hora del día').onChange(apply).listen();
    l.add(p, 'animateSun').name('animar sol');
    l.add(p, 'sunSpeed', 0, 3, 0.01).name('velocidad sol (h/s)');
    l.add(p, 'sunIntensity', 0, 20, 0.1).name('intensidad sol').onChange(apply);
    l.add(p, 'skyIntensity', 0, 4, 0.01).name('intensidad cielo').onChange(apply);
    l.add(p, 'pointIntensity', 0, 100, 0.1).name('intensidad puntuales (cd)').onChange(apply);
    l.add(p, 'animatePoint').name('mover luz puntual');
    l.add(p, 'pointSpeed', 0, 3, 0.01).name('velocidad luz puntual');
    l.add(p, 'pointShadows').name('sombras puntuales').onChange(apply);
    l.close();
  }

  private bindKeys(): void {
    window.addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      const digit = /^Digit(\d)$/.exec(e.code);
      if (digit) {
        const n = +digit[1];
        if (e.shiftKey) {
          if (n >= 1 && n <= BOOKMARKS.length) this.gotoBookmark(n - 1);
        } else {
          const m = METHODS.find((x) => x.hotkey === n);
          if (m) void this.setMethod(m.key);
        }
        this.gui.controllersRecursive().forEach((c) => c.updateDisplay());
        return;
      }
      switch (e.code) {
        case 'KeyV': {
          const i = VIEW_MODES.indexOf(this.settings.view);
          this.settings.view = VIEW_MODES[(i + 1) % VIEW_MODES.length];
          break;
        }
        case 'KeyX':
          this.settings.split = this.settings.split < 1 ? 1 : 0.5;
          break;
        case 'KeyR':
          this.methodA?.method.reset?.();
          this.methodB?.method.reset?.();
          break;
        case 'KeyH':
          this.hud.toggle();
          break;
        default:
          return;
      }
      this.gui.controllersRecursive().forEach((c) => c.updateDisplay());
    });
  }

  setStatus(s: string): void {
    this.status = s;
  }
}

function now(): number {
  return performance.now() / 1000;
}
