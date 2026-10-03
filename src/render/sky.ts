import { Color } from 'three';
import { Fn, mix, uniform, vec3, max } from 'three/tsl';
import type { Node } from 'three/webgpu';
import type { SceneLights } from '../scene/lights';

/**
 * Analytic sky radiance (no sun disk; the sun is handled as a direct light). It is seen by
 * background pixels and by every indirect ray that escapes the scene. The WGSL twin used by the
 * ray tracers lives in rt/wgsl.ts and reads the same values from the lights storage buffer.
 */
export class SkyUniforms {
  readonly zenith = uniform(new Color());
  readonly horizon = uniform(new Color());
  readonly scale = uniform(1);

  update(lights: SceneLights): void {
    this.zenith.value.copy(lights.skyZenith);
    this.horizon.value.copy(lights.skyHorizon);
    this.scale.value = lights.skyScale;
  }

  /** TSL: radiance for a normalized world-space direction. */
  readonly radiance = Fn(([dir]: [Node<'vec3'>]) => {
    const t = max(dir.y, 0).pow(0.5);
    return mix(this.horizon, this.zenith, t).mul(this.scale).mul(vec3(1));
  });
}
