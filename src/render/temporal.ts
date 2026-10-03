import { Matrix4, Texture, Vector3, type PerspectiveCamera } from 'three';
import type { StorageTexture, WebGPURenderer } from 'three/webgpu';
import { instancedArray, texture, uniform } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { GBufferAccess, createOutputTexture, dispatch2D, disposeKernel, disposeStorage, makeKernel, storeNode } from '../rt/kernels';
import type { GBuffer } from './gbuffer';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Generic temporal accumulation of a noisy full-resolution GI signal: reprojects the history with
 * the previous frame's view-projection, rejects it when the stored distance to the camera does not
 * match (disocclusion), and blends with weight max(1/n, 1/maxHistory). History lives in two
 * ping-pong storage buffers (rgb + sample count packed as 4 halves, and distance to the camera).
 */
export class TemporalFilter {
  readonly output: StorageTexture;
  readonly params = { enabled: true, maxHistory: 16 };
  private readonly inputNode = texture(new Texture());
  private readonly prevViewProj = uniform(new Matrix4());
  private readonly prevCamPos = uniform(new Vector3());
  private readonly resetU = uniform(1, 'uint');
  private readonly maxHistU = uniform(16);
  private readonly hist: any[];
  private readonly dist: any[];
  private readonly kernels: any[];
  private ping = 0;
  private needsReset = true;
  private readonly savedViewProj = new Matrix4();
  private readonly savedCamPos = new Vector3();

  constructor(private readonly renderer: WebGPURenderer, gbuffer: GBuffer, camera: PerspectiveCamera, name: string) {
    const W = gbuffer.width;
    const H = gbuffer.height;
    this.output = createOutputTexture(W, H);
    this.hist = [0, 1].map((i) => instancedArray(W * H, 'uvec2').setName(`${name}_hist${i}`));
    this.dist = [0, 1].map((i) => instancedArray(W * H, 'float').setName(`${name}_dist${i}`));
    const gb = new GBufferAccess(gbuffer, camera);
    const out = storeNode(this.output);
    this.kernels = [0, 1].map((k) => {
      const hIn = this.hist[k];
      const hOut = this.hist[1 - k];
      const dIn = this.dist[k];
      const dOut = this.dist[1 - k];
      return makeKernel(wgslTagFn/* wgsl */ `
        fn temporalKernel( globalId: vec3u ) -> void {
          let px = globalId.xy;
          if ( px.x >= ${W}u || px.y >= ${H}u ) {
            return;
          }
          let i = px.y * ${W}u + px.x;
          var p: vec3f;
          var n: vec3f;
          if ( ! ${gb.surface}( px, vec2u( ${W}u, ${H}u ), &p, &n ) ) {
            ${hOut}[ i ] = vec2u( 0u );
            ${dOut}[ i ] = -1.0;
            textureStore( ${out}, px, vec4f( 0.0, 0.0, 0.0, 1.0 ) );
            return;
          }
          let cur = textureLoad( ${this.inputNode}, px, 0 ).xyz;
          var h = vec4f( 0.0 );
          if ( ${this.resetU} == 0u ) {
            let clip = ${this.prevViewProj} * vec4f( p, 1.0 );
            if ( clip.w > 0.0 ) {
              let ndc = clip.xyz / clip.w;
              let uvp = vec2f( ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5 );
              let q = vec2i( floor( uvp * vec2f( f32( ${W} ), f32( ${H} ) ) ) );
              if ( q.x >= 0 && q.y >= 0 && q.x < ${W} && q.y < ${H} ) {
                let j = u32( q.y ) * ${W}u + u32( q.x );
                let expected = distance( p, ${this.prevCamPos} );
                let stored = ${dIn}[ j ];
                if ( stored > 0.0 && abs( stored - expected ) < 0.02 * expected + 0.03 ) {
                  let packed = ${hIn}[ j ];
                  h = vec4f( unpack2x16float( packed.x ), unpack2x16float( packed.y ) );
                }
              }
            }
          }
          let count = min( h.w + 1.0, ${this.maxHistU} );
          let res = mix( h.xyz, cur, 1.0 / count );
          ${hOut}[ i ] = vec2u( pack2x16float( res.xy ), pack2x16float( vec2f( res.z, count ) ) );
          ${dOut}[ i ] = distance( p, ${gb.camPos} );
          textureStore( ${out}, px, vec4f( res, 1.0 ) );
        }
      `);
    });
  }

  get memoryBytes(): number {
    const { width, height } = this.output.image as { width: number; height: number };
    return width * height * (2 * 8 + 2 * 4 + 8);
  }

  reset(): void {
    this.needsReset = true;
  }

  /** Filters `input` (full resolution) and returns the filtered texture (or the input if disabled). */
  run(input: Texture, camera: PerspectiveCamera): Texture {
    if (!this.params.enabled) {
      this.needsReset = true;
      return input;
    }
    this.inputNode.value = input;
    this.resetU.value = this.needsReset ? 1 : 0;
    this.maxHistU.value = this.params.maxHistory;
    this.prevViewProj.value.copy(this.savedViewProj);
    this.prevCamPos.value.copy(this.savedCamPos);
    const { width, height } = this.output.image as { width: number; height: number };
    dispatch2D(this.renderer, this.kernels[this.ping], width, height);
    this.ping = 1 - this.ping;
    this.needsReset = false;
    this.savedViewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.savedCamPos.copy(camera.position);
    return this.output;
  }

  dispose(): void {
    this.output.dispose();
    this.kernels.forEach(disposeKernel);
    [...this.hist, ...this.dist].forEach((n) => disposeStorage(this.renderer, n));
  }
}
