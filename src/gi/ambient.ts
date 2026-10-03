import { Color } from 'three';
import { uniform } from 'three/tsl';
import type GUI from 'lil-gui';
import type { SceneLights } from '../scene/lights';

/**
 * Constant ambient term (E/π) used by the screen-space family (methods 1–3, and as a fallback in
 * the hybrids): a sky-tinted color, optionally scaled with the sky brightness of the time of day.
 */
export class AmbientTerm {
  // Default intensity = value calibrated against the reference by the benchmark (see README).
  readonly params = { color: '#9fb4d9', intensity: 0.004, followSky: true };
  readonly uniform = uniform(new Color());

  update(lights: SceneLights): Color {
    const c = this.uniform.value.set(this.params.color).multiplyScalar(this.params.intensity);
    if (this.params.followSky) c.multiplyScalar(lights.skyScale);
    return c;
  }

  buildGui(folder: GUI): void {
    folder.addColor(this.params, 'color').name('ambiente: color');
    folder.add(this.params, 'intensity', 0, 2, 0.01).name('ambiente: intensidad');
    folder.add(this.params, 'followSky').name('ambiente sigue al cielo');
  }
}
