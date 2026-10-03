import { HalfFloatType, LinearFilter, Texture } from 'three';
import { NodeMaterial, QuadMesh, RenderTarget } from 'three/webgpu';
import { texture, uniform, vec4 } from 'three/tsl';
import { ssgi } from 'three/addons/tsl/display/SSGINode.js';
import type GUI from 'lil-gui';
import type { GIContext, GIMethod, GIStats } from './types';
import type { GBuffer } from '../render/gbuffer';
import { AmbientTerm } from './ambient';
import { TemporalFilter } from '../render/temporal';
import { MemoryLedger, renderTargetBytes } from '../metrics/memory';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Method 3: screen-space GI with three.js SSGINode (SSILVB: horizon-based visibility bitmask,
 * gathers radiance from on-screen pixels). Light source: the previous frame's final composite of
 * this method (multi-bounce feedback), or only the current direct light. Off-screen light and the
 * sky are approximated by the ambient term weighted by SSGI's AO:
 *     indirect = giScale · gi + ambient · ao
 * `giScale` is SSGINode's `giIntensity`: SSILVB is not radiometrically normalized, so this factor is
 * a free parameter (see README for how its default was chosen).
 */
export class SSGIMethod implements GIMethod {
  readonly key = 'ssgi';
  readonly label = 'SSGI (espacio de pantalla)';
  private ctx!: GIContext;
  private readonly ambient = new AmbientTerm();
  private node: any = null;
  private quad: QuadMesh | null = null;
  private material: NodeMaterial | null = null;
  private target!: RenderTarget;
  private temporal!: TemporalFilter;
  private readonly beautyNode = texture(new Texture());
  private readonly giScale = uniform(1.0);
  readonly params = { source: 'final previo', sliceCount: 2, stepCount: 8, radius: 12, giIntensity: 8, aoIntensity: 1, thickness: 0.5, backfaceLighting: 0 };

  async init(ctx: GIContext): Promise<void> {
    this.ctx = ctx;
    this.target = new RenderTarget(ctx.width, ctx.height, { type: HalfFloatType, depthBuffer: false });
    this.target.texture.minFilter = LinearFilter;
    this.temporal = new TemporalFilter(ctx.renderer, ctx.gbuffer, ctx.camera, 'ssgi');
    this.build(ctx.gbuffer);
  }

  private build(gbuffer: GBuffer): void {
    this.node?.dispose();
    this.material?.dispose();
    this.node = ssgi(this.beautyNode, texture(gbuffer.depth), texture(gbuffer.normal), this.ctx.camera);
    this.node.useTemporalFiltering = true;
    const gi = this.node.getGINode().rgb;
    const aoV = this.node.getAONode().r;
    this.material = new NodeMaterial();
    this.material.fragmentNode = vec4(gi.mul(this.giScale).add(this.ambient.uniform.mul(aoV)), 1);
    this.quad = new QuadMesh(this.material);
  }

  update(): void {
    this.ambient.update(this.ctx.lights);
  }

  reset(): void {
    this.temporal.reset();
  }

  onCameraCut(): void {
    this.reset();
  }

  run(gbuffer: GBuffer, history: { prevFinal: Texture }): Texture {
    const { renderer, timer, camera } = this.ctx;
    this.beautyNode.value = this.params.source === 'directa' ? gbuffer.direct : history.prevFinal;
    const n = this.node;
    const p = this.params;
    n.sliceCount.value = p.sliceCount;
    n.stepCount.value = p.stepCount;
    n.radius.value = p.radius;
    n.giIntensity.value = 1;
    n.aoIntensity.value = p.aoIntensity;
    n.thickness.value = p.thickness;
    n.backfaceLighting.value = p.backfaceLighting;
    this.giScale.value = p.giIntensity;
    timer.begin('ssgi.trace');
    renderer.setRenderTarget(this.target);
    this.quad!.render(renderer);
    renderer.setRenderTarget(null);
    timer.end('ssgi.trace');
    timer.begin('ssgi.temporal');
    const out = this.temporal.run(this.target.texture, camera);
    timer.end('ssgi.temporal');
    return out;
  }

  stats(): GIStats {
    const ledger = new MemoryLedger();
    ledger.add('salida (rgba16f)', renderTargetBytes(this.target as any));
    if (this.node) ledger.add('SSGINode RT interno (R8 + RG11B10)', renderTargetBytes(this.node._ssgiRenderTarget));
    ledger.add('filtro temporal', this.temporal.memoryBytes);
    const p = this.params;
    return {
      raysPerFrame: 0,
      memoryBytes: ledger.total,
      memoryBreakdown: ledger.breakdown(),
      extra: { 'Muestras de pantalla/píxel': p.sliceCount * p.stepCount * 2 },
    };
  }

  buildGui(folder: GUI): void {
    this.ambient.buildGui(folder);
    folder.add(this.params, 'source', ['final previo', 'directa']).name('fuente de luz');
    folder.add(this.params, 'sliceCount', 1, 4, 1).name('slices');
    folder.add(this.params, 'stepCount', 1, 32, 1).name('pasos');
    folder.add(this.params, 'radius', 1, 50, 0.5).name('radio');
    folder.add(this.params, 'giIntensity', 0, 20, 0.01).name('escala GI');
    folder.add(this.params, 'aoIntensity', 0, 4, 0.01).name('intensidad AO');
    folder.add(this.params, 'thickness', 0.01, 5, 0.01).name('grosor');
    folder.add(this.params, 'backfaceLighting', 0, 1, 0.01).name('luz por detrás');
    folder.add(this.temporal.params, 'enabled').name('filtro temporal');
    folder.add(this.temporal.params, 'maxHistory', 1, 64, 1).name('historia máx.');
  }

  dispose(): void {
    this.target.dispose();
    this.node?.dispose();
    this.material?.dispose();
    this.temporal.dispose();
  }
}
