import type { WebGPURenderer } from 'three/webgpu';

interface PassRange {
  name: string;
  frame: number;
  render: [number, number]; // (begin, end] in renderer.info.render.frameCalls
  compute: [number, number]; // (begin, end] in renderer.info.compute.frameCalls
}

interface PoolLike {
  timestamps: Map<string, number>;
}

/**
 * Per-pass GPU timing on top of three.js' own timestamp queries.
 *
 * three.js (r186) writes a begin/end timestamp pair for every render pass and compute dispatch
 * group and labels it `r:<frameCall>:<contextId>:f<frame>` / `c:<frameCall>:...`, where
 * `frameCall` is the value of `info.render|compute.frameCalls` after it was incremented for that
 * call. `begin()`/`end()` record which calls belong to a named pass; after
 * `renderer.resolveTimestampsAsync()` the per-uid durations in the backend's query pools are summed
 * per pass. Nested passes (e.g. shadow maps inside the G-buffer pass) are included in the outer
 * pass. Without the `timestamp-query` feature, the timer falls back to CPU wall time per pass
 * (which, without GPU synchronization, measures only command recording — this is reported).
 */
export class GpuTimer {
  readonly supported: boolean;
  private open = new Map<string, PassRange>();
  private pending: PassRange[] = [];
  private resolving = false;
  private cpuStart = new Map<string, number>();
  private minFrame = 0;
  // Latest per-pass times (ms) and exponential moving averages.
  readonly last = new Map<string, number>();
  readonly avg = new Map<string, number>();
  // Optional sink for raw per-frame samples (benchmark mode).
  onFrame: ((frame: number, passes: Map<string, number>) => void) | null = null;

  constructor(private readonly renderer: WebGPURenderer) {
    this.supported = renderer.hasFeature('timestamp-query');
  }

  begin(name: string): void {
    const info = this.renderer.info;
    if (!this.supported) {
      this.cpuStart.set(name, performance.now());
      return;
    }
    this.open.set(name, {
      name,
      frame: info.frame,
      render: [info.render.frameCalls, info.render.frameCalls],
      compute: [info.compute.frameCalls, info.compute.frameCalls],
    });
  }

  end(name: string): void {
    const info = this.renderer.info;
    if (!this.supported) {
      const t = performance.now() - (this.cpuStart.get(name) ?? performance.now());
      this.record(name, t);
      return;
    }
    const r = this.open.get(name);
    if (!r) return;
    this.open.delete(name);
    r.render[1] = info.render.frameCalls;
    r.compute[1] = info.compute.frameCalls;
    this.pending.push(r);
  }

  /** Call once per frame after all passes were submitted. Non-blocking. */
  resolve(): void {
    if (!this.supported || this.resolving || this.pending.length === 0) return;
    const ranges = this.pending;
    this.pending = [];
    this.resolving = true;
    const backend = (this.renderer as unknown as { backend: { timestampQueryPool: Record<string, PoolLike | null> } }).backend;
    Promise.all([this.renderer.resolveTimestampsAsync('render'), this.renderer.resolveTimestampsAsync('compute')])
      .then(() => {
        const parsed: { type: 'r' | 'c'; call: number; frame: number; ms: number }[] = [];
        for (const key of ['render', 'compute']) {
          const pool = backend.timestampQueryPool[key];
          if (!pool) continue;
          for (const [uid, ms] of pool.timestamps) {
            const m = /^([rc]):(\d+):.*:f(\d+)$/.exec(uid);
            if (m) parsed.push({ type: m[1] as 'r' | 'c', call: +m[2], frame: +m[3], ms });
          }
        }
        const perFrame = new Map<number, Map<string, number>>();
        for (const r of ranges) {
          if (r.frame < this.minFrame) continue;
          let sum = 0;
          let found = false;
          for (const p of parsed) {
            if (p.frame !== r.frame) continue;
            const [a, b] = p.type === 'r' ? r.render : r.compute;
            if (p.call > a && p.call <= b) {
              sum += p.ms;
              found = true;
            }
          }
          if (!found) continue;
          this.record(r.name, sum);
          let fm = perFrame.get(r.frame);
          if (!fm) perFrame.set(r.frame, (fm = new Map()));
          fm.set(r.name, (fm.get(r.name) ?? 0) + sum);
        }
        if (this.onFrame) for (const [f, m] of perFrame) this.onFrame(f, m);
      })
      .finally(() => {
        this.resolving = false;
      });
  }

  private record(name: string, ms: number): void {
    this.last.set(name, ms);
    const prev = this.avg.get(name);
    this.avg.set(name, prev === undefined ? ms : prev * 0.95 + ms * 0.05);
  }

  resetAverages(): void {
    this.avg.clear();
    this.last.clear();
    // Ignore timings of frames submitted before the reset that resolve later.
    this.minFrame = this.renderer.info.frame + 1;
  }
}
