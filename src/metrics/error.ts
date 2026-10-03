import { Texture } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import { instancedArray, texture } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { dispatch2D, makeKernel } from '../rt/kernels';
import type { GBuffer } from '../render/gbuffer';

/* eslint-disable @typescript-eslint/no-explicit-any */

const BLOCK = 16;

export interface ErrorResult {
  frame: number;
  rmse: number; // sqrt(mean over pixels and RGB channels of (x - ref)^2), linear HDR
  rel: number; // sum |Y - Yref| / sum Yref (relative L1 error on luminance)
  pixels: number;
}

/**
 * Image error of a composite (final, linear, pre-tonemap) against the reference composite for the
 * same camera bookmark and light state. Background pixels are excluded. A compute pass reduces
 * 16x16 blocks into partial sums; the partials are read back asynchronously and summed on the CPU.
 */
export class ErrorMetric {
  private readonly bw: number;
  private readonly bh: number;
  private readonly partials: any;
  private readonly testNode = texture(new Texture());
  private readonly refNode = texture(new Texture());
  private readonly kernel: any;
  private busy = false;
  last: ErrorResult | null = null;
  onResult: ((r: ErrorResult) => void) | null = null;

  constructor(gbuffer: GBuffer) {
    const { width: W, height: H } = gbuffer;
    this.bw = Math.ceil(W / BLOCK);
    this.bh = Math.ceil(H / BLOCK);
    this.partials = instancedArray(this.bw * this.bh, 'vec4').setName('err_partials');
    const depth = texture(gbuffer.depth);
    const fn = wgslTagFn/* wgsl */ `
      fn errorKernel( globalId: vec3u ) -> void {
        let b = globalId.xy;
        if ( b.x >= ${this.bw}u || b.y >= ${this.bh}u ) {
          return;
        }
        var acc = vec4f( 0.0 );
        let lw = vec3f( 0.2126, 0.7152, 0.0722 );
        for ( var y = 0u; y < ${BLOCK}u; y = y + 1u ) {
          for ( var x = 0u; x < ${BLOCK}u; x = x + 1u ) {
            let px = b * ${BLOCK}u + vec2u( x, y );
            if ( px.x < ${W}u && px.y < ${H}u && textureLoad( ${depth}, px, 0 ) < 1.0 ) {
              let a = textureLoad( ${this.testNode}, px, 0 ).xyz;
              let r = textureLoad( ${this.refNode}, px, 0 ).xyz;
              let d = a - r;
              acc += vec4f( dot( d, d ), abs( dot( d, lw ) ), dot( r, lw ), 1.0 );
            }
          }
        }
        ${this.partials}[ b.y * ${this.bw}u + b.x ] = acc;
      }
    `;
    this.kernel = makeKernel(fn);
  }

  /** Records the reduction pass for `test` vs `ref` (both full-resolution rgba16f textures). */
  run(renderer: WebGPURenderer, test: Texture, ref: Texture, frame: number): void {
    if (this.busy) return;
    // Texture bindings follow `.value` changes without rebuilding the pipeline.
    this.testNode.value = test;
    this.refNode.value = ref;
    dispatch2D(renderer, this.kernel, this.bw, this.bh);
    this.busy = true;
    renderer
      .getArrayBufferAsync(this.partials.value)
      .then((buf: ArrayBuffer) => {
        const f = new Float32Array(buf);
        let sq = 0, abs = 0, ref = 0, n = 0;
        for (let i = 0; i < f.length; i += 4) {
          sq += f[i];
          abs += f[i + 1];
          ref += f[i + 2];
          n += f[i + 3];
        }
        const r: ErrorResult = { frame, rmse: Math.sqrt(sq / Math.max(3 * n, 1)), rel: abs / Math.max(ref, 1e-12), pixels: n };
        this.last = r;
        this.onResult?.(r);
      })
      .finally(() => (this.busy = false));
  }

  /** Synchronous variant used by the benchmark: waits for the readback. */
  async measure(renderer: WebGPURenderer, test: Texture, ref: Texture, frame: number): Promise<ErrorResult> {
    while (this.busy) await new Promise((r) => setTimeout(r, 0));
    return new Promise((resolve) => {
      const prev = this.onResult;
      this.onResult = (r) => {
        this.onResult = prev;
        resolve(r);
      };
      this.run(renderer, test, ref, frame);
    });
  }
}
