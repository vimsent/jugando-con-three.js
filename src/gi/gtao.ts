import { HalfFloatType, LinearFilter, type Texture } from 'three';
import { NodeMaterial, QuadMesh, RenderTarget } from 'three/webgpu';
import { texture, vec4 } from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import type GUI from 'lil-gui';
import type { GIContext, GIMethod, GIStats } from './types';
import type { GBuffer } from '../render/gbuffer';
import { AmbientTerm } from './ambient';
import { TemporalFilter } from '../render/temporal';
import { MemoryLedger, renderTargetBytes } from '../metrics/memory';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Method 2: GTAO only — the constant ambient term modulated by ground-truth ambient occlusion
 * (three.js GTAONode, horizon-based, reads the G-buffer depth and view normals), followed by the
 * shared temporal filter. No light transport between surfaces: occlusion only.
 */
export class GTAOMethod implements GIMethod {
  readonly key = 'gtao';
  readonly label = 'GTAO (AO × ambiente)';
  private ctx!: GIContext;
  private readonly ambient = new AmbientTerm();
  private aoNode: any = null;
  private quad: QuadMesh | null = null;
  private material: NodeMaterial | null = null;
  private target!: RenderTarget;
  private temporal: TemporalFilter | null = null;
  private gbufferRef: GBuffer | null = null;
  readonly params = { radius: 0.5, thickness: 1.0, distanceExponent: 1.0, scale: 1.0, samples: 16 };

  async init(ctx: GIContext): Promise<void> {
    this.ctx = ctx;
    this.target = new RenderTarget(ctx.width, ctx.height, { type: HalfFloatType, depthBuffer: false });
    this.target.texture.minFilter = LinearFilter;
    this.build(ctx.gbuffer);
  }

  private build(gbuffer: GBuffer): void {
    this.gbufferRef = gbuffer;
    this.aoNode = ao(texture(gbuffer.depth), texture(gbuffer.normal), this.ctx.camera);
    this.aoNode.useTemporalFiltering = true;
    this.material = new NodeMaterial();
    this.material.fragmentNode = vec4(this.ambient.uniform.mul(this.aoNode.r), 1);
    this.quad = new QuadMesh(this.material);
    this.temporal = new TemporalFilter(this.ctx.renderer, gbuffer, this.ctx.camera, 'gtao');
  }

  update(): void {
    this.ambient.update(this.ctx.lights);
  }

  reset(): void {
    this.temporal?.reset();
  }

  run(gbuffer: GBuffer): Texture {
    if (this.gbufferRef !== gbuffer) this.build(gbuffer);
    const { renderer, timer, camera } = this.ctx;
    const n = this.aoNode;
    n.radius.value = this.params.radius;
    n.thickness.value = this.params.thickness;
    n.distanceExponent.value = this.params.distanceExponent;
    n.scale.value = this.params.scale;
    n.samples.value = this.params.samples;
    timer.begin('gtao.ao');
    renderer.setRenderTarget(this.target);
    this.quad!.render(renderer);
    renderer.setRenderTarget(null);
    timer.end('gtao.ao');
    timer.begin('gtao.temporal');
    const out = this.temporal!.run(this.target.texture, camera);
    timer.end('gtao.temporal');
    return out;
  }

  stats(): GIStats {
    const ledger = new MemoryLedger();
    ledger.add('salida ambiente×AO (rgba16f)', renderTargetBytes(this.target as any));
    if (this.aoNode) ledger.add('GTAONode RT interno', renderTargetBytes(this.aoNode._aoRenderTarget));
    if (this.temporal) ledger.add('filtro temporal', this.temporal.memoryBytes);
    return { raysPerFrame: 0, memoryBytes: ledger.total, memoryBreakdown: ledger.breakdown() };
  }

  buildGui(folder: GUI): void {
    this.ambient.buildGui(folder);
    folder.add(this.params, 'radius', 0.05, 3, 0.01).name('radio (m)');
    folder.add(this.params, 'thickness', 0.1, 4, 0.01).name('grosor');
    folder.add(this.params, 'distanceExponent', 1, 4, 0.01).name('exponente distancia');
    folder.add(this.params, 'scale', 0, 2, 0.01).name('escala AO');
    folder.add(this.params, 'samples', 4, 32, 1).name('muestras');
    folder.add(this.temporal!.params, 'enabled').name('filtro temporal');
    folder.add(this.temporal!.params, 'maxHistory', 1, 64, 1).name('historia máx.');
  }

  dispose(): void {
    this.target.dispose();
    this.aoNode?.dispose();
    this.material?.dispose();
    this.temporal?.dispose();
  }
}
