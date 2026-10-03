import { DepthTexture, FloatType, HalfFloatType, NearestFilter, PerspectiveCamera, Scene, Texture } from 'three';
import { RenderTarget, type WebGPURenderer } from 'three/webgpu';
import { diffuseColor, mrt, normalView, output, vec4 } from 'three/tsl';
import { textureBytes } from '../metrics/memory';

/**
 * Minimal deferred G-buffer rendered with MRT:
 *   0 `output`  rgba16f  direct diffuse radiance (sun + point lights with shadow maps, no ambient)
 *   1 `albedo`  rgba16f  linear base color (diffuse reflectance)
 *   2 `normal`  rgba16f  view-space shading normal (normal-mapped), unpacked
 *   depth       float32  hardware depth; positions are reconstructed from it
 *
 * Note: three.js binds shadow sampling and light loops to materials, so the direct term is
 * evaluated while filling the G-buffer instead of in a separate full-screen lighting pass. It is the
 * same for every GI method, which is what the comparison needs.
 */
export class GBuffer {
  readonly target: RenderTarget;
  private readonly mrtNode = mrt({
    output,
    albedo: vec4(diffuseColor.rgb, 1),
    normal: vec4(normalView, 1),
  });

  constructor(readonly width: number, readonly height: number) {
    const depthTexture = new DepthTexture(width, height, FloatType);
    this.target = new RenderTarget(width, height, { count: 3, type: HalfFloatType, depthTexture });
    const names = ['output', 'albedo', 'normal'];
    this.target.textures.forEach((t, i) => {
      t.name = names[i];
      t.minFilter = NearestFilter;
      t.magFilter = NearestFilter;
      t.generateMipmaps = false;
    });
  }

  get direct(): Texture { return this.target.textures[0]; }
  get albedo(): Texture { return this.target.textures[1]; }
  get normal(): Texture { return this.target.textures[2]; }
  get depth(): DepthTexture { return this.target.depthTexture!; }

  render(renderer: WebGPURenderer, scene: Scene, camera: PerspectiveCamera): void {
    renderer.setMRT(this.mrtNode);
    renderer.setRenderTarget(this.target);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.setMRT(null);
    renderer.setRenderTarget(null);
  }

  get memoryBytes(): number {
    return 3 * textureBytes(this.width, this.height, 8) + textureBytes(this.width, this.height, 4);
  }

  dispose(): void {
    this.target.dispose();
  }
}
