import { Color, DataTexture, FloatType, RGBAFormat, type Texture } from 'three';
import type GUI from 'lil-gui';
import type { GIContext, GIMethod, GIStats } from './types';

/** Method 1: no GI, a constant ambient term (classic "flat ambient"). */
export class FlatAmbient implements GIMethod {
  readonly key = 'flat';
  readonly label = 'Sin GI (ambiente plano)';
  private readonly tex = new DataTexture(new Float32Array(4), 1, 1, RGBAFormat, FloatType);
  private readonly params = { color: '#9fb4d9', intensity: 0.25, followSky: true };
  private ctx!: GIContext;

  async init(ctx: GIContext): Promise<void> {
    this.ctx = ctx;
    this.refresh();
  }

  update(): void {
    this.refresh();
  }

  private refresh(): void {
    const c = new Color(this.params.color).multiplyScalar(this.params.intensity);
    if (this.params.followSky) c.multiplyScalar(this.ctx.lights.skyScale);
    const d = this.tex.image.data as Float32Array;
    if (d[0] !== c.r || d[1] !== c.g || d[2] !== c.b) {
      d.set([c.r, c.g, c.b, 1]);
      this.tex.needsUpdate = true;
    }
  }

  run(): Texture {
    return this.tex;
  }

  stats(): GIStats {
    return { raysPerFrame: 0, memoryBytes: 16 };
  }

  buildGui(folder: GUI): void {
    folder.addColor(this.params, 'color').name('color');
    folder.add(this.params, 'intensity', 0, 2, 0.01).name('intensidad');
    folder.add(this.params, 'followSky').name('sigue al cielo');
  }

  dispose(): void {
    this.tex.dispose();
  }
}
