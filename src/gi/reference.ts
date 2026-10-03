import type { Texture } from 'three';
import type { StorageTexture } from 'three/webgpu';
import { instancedArray, uniform } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import type GUI from 'lil-gui';
import type { FrameInfo, GIContext, GIMethod, GIStats } from './types';
import type { RTScene } from '../rt/rtScene';
import type { GBuffer } from '../render/gbuffer';
import { GBufferAccess, createOutputTexture, dispatch2D, disposeKernel, disposeStorage, makeKernel, storeNode } from '../rt/kernels';
import { MemoryLedger, textureBytes } from '../metrics/memory';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Method 0: reference. A progressive path tracer that starts at the G-buffer surface of each pixel
 * and estimates indirect diffuse light (E_ind/π) with many bounces, accumulating samples while the
 * camera and lights stay still. It shares the BVH and WGSL library with the other ray-traced
 * methods, so differences against it are due to the GI approximation, not to scene mismatch.
 */
export class ReferencePathTracer implements GIMethod {
  readonly key = 'reference';
  readonly label = 'Referencia (path tracing)';
  readonly params = { sppPerFrame: 4, maxBounces: 8, targetSpp: 4096, accumulate: true };
  private ctx!: GIContext;
  private rt!: RTScene;
  private gb!: GBufferAccess;
  private output!: StorageTexture;
  private accum: any;
  private kernel: any;
  private readonly frameU = uniform(0, 'uint');
  private readonly resetU = uniform(1, 'uint');
  private readonly sppU = uniform(4, 'uint');
  private readonly bouncesU = uniform(8, 'uint');
  private samples = 0;
  private frameCounter = 0;
  private readonly ledger = new MemoryLedger();
  private gbufferRef: GBuffer | null = null;

  async init(ctx: GIContext): Promise<void> {
    this.ctx = ctx;
    this.rt = await ctx.getRTScene();
    const { width, height } = ctx;
    this.output = createOutputTexture(width, height);
    this.accum = instancedArray(width * height, 'vec4').setName('ref_accum');
    this.ledger.add('acumulación (vec4f32)', width * height * 16);
    this.ledger.add('salida (rgba16f)', textureBytes(width, height, 8));
  }

  get outputTexture(): StorageTexture {
    return this.output;
  }

  /** Samples accumulated per pixel since the last reset. */
  get sampleCount(): number {
    return this.samples;
  }

  private build(gbuffer: GBuffer): void {
    this.gbufferRef = gbuffer;
    this.gb = new GBufferAccess(gbuffer, this.ctx.camera);
    const { fns } = this.rt;
    const out = storeNode(this.output);
    const accum = this.accum;
    const W = this.ctx.width;
    const H = this.ctx.height;
    const fn = wgslTagFn/* wgsl */ `
      fn referenceKernel( globalId: vec3u ) -> void {
        let px = globalId.xy;
        if ( px.x >= ${W}u || px.y >= ${H}u ) {
          return;
        }
        let i = px.y * ${W}u + px.x;
        var acc = vec4f( 0.0 );
        if ( ${this.resetU} == 0u ) {
          acc = ${accum}[ i ];
        }
        var p: vec3f;
        var n: vec3f;
        if ( ! ${this.gb.surface}( px, vec2u( ${W}u, ${H}u ), &p, &n ) ) {
          ${accum}[ i ] = vec4f( 0.0 );
          textureStore( ${out}, px, vec4f( 0.0, 0.0, 0.0, 1.0 ) );
          return;
        }
        var state = ${fns.seed}( px, ${this.frameU}, 0x68e31da4u );
        var rays = 0u;
        var sum = vec3f( 0.0 );
        for ( var s = 0u; s < ${this.sppU}; s = s + 1u ) {
          sum += ${fns.pathTrace}( p, n, n, &state, ${this.bouncesU}, &rays );
        }
        acc += vec4f( sum, f32( ${this.sppU} ) );
        ${accum}[ i ] = acc;
        textureStore( ${out}, px, vec4f( acc.xyz / max( acc.w, 1.0 ), 1.0 ) );
        ${fns.countRays}( px, rays );
      }
    `;
    this.kernel = makeKernel(fn);
  }

  update(_dt: number, info: FrameInfo): void {
    if (info.cameraMoved || info.lightsChanged || !this.params.accumulate) this.reset();
  }

  reset(): void {
    this.samples = 0;
  }

  run(gbuffer: GBuffer): Texture {
    if (this.gbufferRef !== gbuffer) this.build(gbuffer);
    const { renderer, timer, width, height } = this.ctx;
    if (this.samples >= this.params.targetSpp) return this.output;
    this.rt.writeLights(this.ctx.lights);
    this.resetU.value = this.samples === 0 ? 1 : 0;
    this.sppU.value = this.params.sppPerFrame;
    this.bouncesU.value = this.params.maxBounces;
    this.frameU.value = this.frameCounter++;
    timer.begin('ref.pathtrace');
    dispatch2D(renderer, this.kernel, width, height);
    timer.end('ref.pathtrace');
    this.samples += this.params.sppPerFrame;
    this.rt.sampleRayCounter(renderer);
    return this.output;
  }

  stats(): GIStats {
    const done = this.samples >= this.params.targetSpp;
    return {
      raysPerFrame: done ? 0 : this.rt.rayCountLast,
      memoryBytes: this.ledger.total + this.rt.memoryBytes,
      memoryBreakdown: { ...this.ledger.breakdown(), 'BVH + geometría': this.rt.memoryBytes },
      extra: { Muestras: `${this.samples} / ${this.params.targetSpp} spp${done ? ' (completa)' : ''}` },
    };
  }

  buildGui(folder: GUI): void {
    folder.add(this.params, 'sppPerFrame', 1, 32, 1).name('spp por frame');
    folder.add(this.params, 'maxBounces', 1, 16, 1).name('rebotes máx.').onChange(() => this.reset());
    folder.add(this.params, 'targetSpp', 64, 65536, 64).name('spp objetivo');
    folder.add(this.params, 'accumulate').name('acumular');
    folder.add({ reset: () => this.reset() }, 'reset').name('reiniciar');
  }

  dispose(): void {
    this.output.dispose();
    disposeKernel(this.kernel);
    disposeStorage(this.ctx.renderer, this.accum);
  }
}
