import { DataTexture, DataUtils, HalfFloatType, RGBAFormat, type Texture } from 'three';
import type GUI from 'lil-gui';
import type { GIContext, GIMethod, GIStats } from './types';
import { AmbientTerm } from './ambient';

/** Method 1: no GI, a constant ambient term (classic "flat ambient"). */
export class FlatAmbient implements GIMethod {
  readonly key = 'flat';
  readonly label = 'Sin GI (ambiente plano)';
  private readonly tex = new DataTexture(new Uint16Array(4), 1, 1, RGBAFormat, HalfFloatType);
  private readonly ambient = new AmbientTerm();
  private ctx!: GIContext;

  async init(ctx: GIContext): Promise<void> {
    this.ctx = ctx;
    this.update();
  }

  update(): void {
    const c = this.ambient.update(this.ctx.lights);
    const d = this.tex.image.data as Uint16Array;
    const h = [c.r, c.g, c.b, 1].map((v) => DataUtils.toHalfFloat(v));
    if (h.some((v, i) => v !== d[i])) {
      d.set(h);
      this.tex.needsUpdate = true;
    }
  }

  run(): Texture {
    return this.tex;
  }

  stats(): GIStats {
    return { raysPerFrame: 0, memoryBytes: 8 };
  }

  buildGui(folder: GUI): void {
    this.ambient.buildGui(folder);
  }

  dispose(): void {
    this.tex.dispose();
  }
}
