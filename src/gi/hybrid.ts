import { HalfFloatType, LinearFilter, Texture } from 'three';
import { NodeMaterial, QuadMesh, RenderTarget } from 'three/webgpu';
import { Fn, max, texture, uniform, uv, vec3, vec4 } from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import type GUI from 'lil-gui';
import { DDGIMethod } from './ddgi';
import type { GIContext, GIStats } from './types';
import type { GBuffer } from '../render/gbuffer';
import { TemporalFilter } from '../render/temporal';
import { renderTargetBytes } from '../metrics/memory';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Method 5: DDGI probes + screen-space directional occlusion. The probe volume resolves
 * large-scale occlusion through its visibility test but has no contact detail below the probe
 * spacing; a short-radius GTAO (three.js GTAONode) supplies it. AO is temporally filtered on its
 * own (so DDGI changes are not lagged twice) and applied with Jimenez et al.'s multi-bounce fit,
 * which brightens occlusion on light albedos:
 *     indirect = ddgi · mb(ao, albedo)
 */
export class DDGISSDOMethod extends DDGIMethod {
  override readonly key: string = 'ddgi-ssdo';
  override readonly label: string = 'DDGI + SSDO';
  readonly ssdo = { radius: 0.35, thickness: 0.5, samples: 16, multiBounce: true, strength: 1.0 };
  private aoNode: any;
  private aoTarget!: RenderTarget;
  private aoQuad!: QuadMesh;
  private aoMaterial!: NodeMaterial;
  private aoTemporal!: TemporalFilter;
  private combineTarget!: RenderTarget;
  private combineQuad!: QuadMesh;
  private combineMaterial!: NodeMaterial;
  private readonly aoFilteredNode = texture(new Texture());
  private readonly ddgiNode = texture(new Texture());
  private readonly multiBounceU = uniform(1);
  private readonly strengthU = uniform(1);

  override async init(ctx: GIContext): Promise<void> {
    await super.init(ctx);
    const { width, height, gbuffer, camera } = ctx;
    const mkRT = () => {
      const rt = new RenderTarget(width, height, { type: HalfFloatType, depthBuffer: false });
      rt.texture.minFilter = LinearFilter;
      return rt;
    };
    this.aoTarget = mkRT();
    this.combineTarget = mkRT();
    this.aoNode = ao(texture(gbuffer.depth), texture(gbuffer.normal), camera);
    this.aoNode.useTemporalFiltering = true;
    this.aoMaterial = new NodeMaterial();
    this.aoMaterial.fragmentNode = vec4(this.aoNode.r, this.aoNode.r, this.aoNode.r, 1);
    this.aoQuad = new QuadMesh(this.aoMaterial);
    this.aoTemporal = new TemporalFilter(ctx.renderer, gbuffer, camera, 'ssdo');

    const albedoNode = texture(gbuffer.albedo);
    this.combineMaterial = new NodeMaterial();
    this.combineMaterial.fragmentNode = Fn(() => {
      const st = uv();
      const x = this.aoFilteredNode.sample(st).r.toVar();
      const a = albedoNode.sample(st).rgb;
      // Jimenez et al. 2016, "Practical Realtime Strategies for Accurate Indirect Occlusion".
      const ca = a.mul(2.0404).sub(0.3324);
      const cb = a.mul(-4.7951).add(0.6417);
      const cc = a.mul(2.7552).add(0.6903);
      const mb = max(vec3(x), x.mul(ca).add(cb).mul(x).add(cc).mul(x));
      const occl = this.multiBounceU.greaterThan(0.5).select(mb, vec3(x));
      const strength = occl.sub(1).mul(this.strengthU).add(1);
      return vec4(this.ddgiNode.sample(st).rgb.mul(strength), 1);
    })();
    this.combineQuad = new QuadMesh(this.combineMaterial);
  }

  override reset(): void {
    super.reset();
    this.aoTemporal.reset();
  }

  override run(gbuffer: GBuffer): Texture {
    const ddgi = this.runVolumeAndGather();
    const { renderer, timer, camera } = this.ctx;
    const n = this.aoNode;
    n.radius.value = this.ssdo.radius;
    n.thickness.value = this.ssdo.thickness;
    n.samples.value = this.ssdo.samples;
    void gbuffer;
    timer.begin('ssdo.ao');
    renderer.setRenderTarget(this.aoTarget);
    this.aoQuad.render(renderer);
    renderer.setRenderTarget(null);
    timer.end('ssdo.ao');
    timer.begin('ssdo.temporal');
    this.aoFilteredNode.value = this.aoTemporal.run(this.aoTarget.texture, camera);
    timer.end('ssdo.temporal');
    this.ddgiNode.value = ddgi;
    this.multiBounceU.value = this.ssdo.multiBounce ? 1 : 0;
    this.strengthU.value = this.ssdo.strength;
    timer.begin('ssdo.combine');
    renderer.setRenderTarget(this.combineTarget);
    this.combineQuad.render(renderer);
    renderer.setRenderTarget(null);
    timer.end('ssdo.combine');
    return this.combineTarget.texture;
  }

  override stats(): GIStats {
    const s = super.stats();
    const extra = {
      'SSDO: AO (rgba16f)': renderTargetBytes(this.aoTarget as any),
      'SSDO: GTAONode RT interno': renderTargetBytes(this.aoNode._aoRenderTarget),
      'SSDO: filtro temporal AO': this.aoTemporal.memoryBytes,
      'SSDO: combinación (rgba16f)': renderTargetBytes(this.combineTarget as any),
    };
    const add = Object.values(extra).reduce((a, b) => a + b, 0);
    return { ...s, memoryBytes: s.memoryBytes + add, memoryBreakdown: { ...s.memoryBreakdown, ...extra } };
  }

  override buildGui(folder: GUI): void {
    super.buildGui(folder);
    folder.add(this.ssdo, 'radius', 0.05, 2, 0.01).name('SSDO radio (m)');
    folder.add(this.ssdo, 'thickness', 0.05, 4, 0.01).name('SSDO grosor');
    folder.add(this.ssdo, 'samples', 4, 32, 1).name('SSDO muestras');
    folder.add(this.ssdo, 'strength', 0, 1, 0.01).name('SSDO intensidad');
    folder.add(this.ssdo, 'multiBounce').name('SSDO multi-rebote');
  }

  override dispose(): void {
    super.dispose();
    this.aoTarget.dispose();
    this.combineTarget.dispose();
    this.aoNode?.dispose();
    this.aoMaterial.dispose();
    this.combineMaterial.dispose();
    this.aoTemporal.dispose();
  }
}
