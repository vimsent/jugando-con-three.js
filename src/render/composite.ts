import { HalfFloatType, LinearFilter, PerspectiveCamera, Texture } from 'three';
import { NodeMaterial, QuadMesh, RenderTarget, type WebGPURenderer } from 'three/webgpu';
import { Fn, getViewPosition, normalize, select, texture, uniform, uv, vec4, float } from 'three/tsl';
import type { GBuffer } from './gbuffer';
import type { SkyUniforms } from './sky';

/**
 * Common composition pass: final = direct + albedo * indirect (linear HDR, before tonemapping).
 * Background pixels (depth == 1) get the analytic sky. The output feeds both the display and the
 * error metrics.
 */
export class Composite {
  readonly target: RenderTarget;
  private readonly quad: QuadMesh;
  private readonly material: NodeMaterial;
  private readonly indirectNode = texture(new Texture());
  private currentIndirect: Texture | null = null;

  constructor(gbuffer: GBuffer, camera: PerspectiveCamera, sky: SkyUniforms) {
    this.target = new RenderTarget(gbuffer.width, gbuffer.height, { type: HalfFloatType, depthBuffer: false });
    this.target.texture.minFilter = LinearFilter;
    this.target.texture.generateMipmaps = false;

    const projInv = uniform(camera.projectionMatrixInverse);
    const camWorld = uniform(camera.matrixWorld);
    const material = (this.material = new NodeMaterial());
    material.fragmentNode = Fn(() => {
      const st = uv();
      const depth = texture(gbuffer.depth, st).r;
      const direct = texture(gbuffer.direct, st).rgb;
      const albedo = texture(gbuffer.albedo, st).rgb;
      const indirect = this.indirectNode.sample(st).rgb;
      const viewPos = getViewPosition(st, float(1), projInv);
      const dir = normalize(camWorld.mul(vec4(viewPos, 0)).xyz);
      const lit = direct.add(albedo.mul(indirect));
      return vec4(select(depth.greaterThanEqual(1), sky.radiance(dir), lit), 1);
    })();
    this.quad = new QuadMesh(material);
  }

  render(renderer: WebGPURenderer, indirect: Texture, target: RenderTarget = this.target): void {
    if (indirect !== this.currentIndirect) {
      this.indirectNode.value = indirect;
      this.currentIndirect = indirect;
      this.material.needsUpdate = true;
    }
    renderer.setRenderTarget(target);
    this.quad.render(renderer);
    renderer.setRenderTarget(null);
  }

  dispose(): void {
    this.target.dispose();
    this.material.dispose();
  }
}
