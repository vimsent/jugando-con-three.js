import { Texture } from 'three';
import type { StorageTexture, WebGPURenderer } from 'three/webgpu';
import { texture, uniform } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { GBufferAccess, createOutputTexture, dispatch2D, disposeKernel, makeKernel, storeNode } from '../rt/kernels';
import type { GBuffer } from './gbuffer';
import type { PerspectiveCamera } from 'three';
import { textureBytes } from '../metrics/memory';

/* eslint-disable @typescript-eslint/no-explicit-any */

const MAX_ITERATIONS = 5;

/**
 * Edge-avoiding à-trous wavelet filter (Dammertz et al. 2010) with a 5x5 B3-spline kernel and
 * step sizes 1, 2, 4, 8, 16. Edge stopping uses world-space normals and the distance of the tap
 * to the center pixel's tangent plane (no luminance term: the input is a noisy irradiance signal).
 * Each iteration is its own kernel (the step size is baked in) and ping-pongs between two
 * rgba16f storage textures.
 */
export class AtrousFilter {
  readonly params = { iterations: 4, normalPower: 64, planeSigma: 0.05 };
  private readonly tex: StorageTexture[];
  private readonly inputNode = texture(new Texture());
  private readonly kernels: any[] = [];
  private readonly normalPowerU = uniform(64);
  private readonly planeSigmaU = uniform(0.05);

  constructor(private readonly renderer: WebGPURenderer, gbuffer: GBuffer, camera: PerspectiveCamera) {
    const W = gbuffer.width;
    const H = gbuffer.height;
    this.tex = [createOutputTexture(W, H), createOutputTexture(W, H)];
    const gb = new GBufferAccess(gbuffer, camera);
    for (let it = 0; it < MAX_ITERATIONS; it++) {
      const step = 1 << it;
      // iteration 0 reads the external input; afterwards ping-pong: odd iterations write tex[1].
      const src = it === 0 ? this.inputNode : texture(this.tex[(it + 1) % 2]);
      const dst = storeNode(this.tex[it % 2]);
      this.kernels.push(makeKernel(wgslTagFn/* wgsl */ `
        fn atrousKernel( globalId: vec3u ) -> void {
          let px = globalId.xy;
          if ( px.x >= ${W}u || px.y >= ${H}u ) {
            return;
          }
          var p: vec3f;
          var n: vec3f;
          if ( ! ${gb.surface}( px, vec2u( ${W}u, ${H}u ), &p, &n ) ) {
            textureStore( ${dst}, px, vec4f( 0.0, 0.0, 0.0, 1.0 ) );
            return;
          }
          let scale = max( distance( p, ${gb.camPos} ), 0.1 );
          let kw = array<f32, 3>( 0.375, 0.25, 0.0625 );
          var sum = vec3f( 0.0 );
          var wsum = 0.0;
          for ( var dy = -2; dy <= 2; dy = dy + 1 ) {
            for ( var dx = -2; dx <= 2; dx = dx + 1 ) {
              let q = vec2i( px ) + vec2i( dx, dy ) * ${step};
              if ( q.x < 0 || q.y < 0 || q.x >= ${W} || q.y >= ${H} ) {
                continue;
              }
              var qp: vec3f;
              var qn: vec3f;
              if ( ! ${gb.surface}( vec2u( q ), vec2u( ${W}u, ${H}u ), &qp, &qn ) ) {
                continue;
              }
              let wn = pow( max( dot( n, qn ), 0.0 ), ${this.normalPowerU} );
              let plane = abs( dot( qp - p, n ) ) / scale;
              let wp = exp( -plane / ${this.planeSigmaU} );
              let w = kw[ abs( dx ) ] * kw[ abs( dy ) ] * wn * wp;
              sum += w * textureLoad( ${src}, vec2u( q ), 0 ).xyz;
              wsum += w;
            }
          }
          textureStore( ${dst}, px, vec4f( select( vec3f( 0.0 ), sum / wsum, wsum > 0.0 ), 1.0 ) );
        }
      `));
    }
  }

  get memoryBytes(): number {
    const { width, height } = this.tex[0].image as { width: number; height: number };
    return 2 * textureBytes(width, height, 8);
  }

  run(input: Texture): Texture {
    const n = Math.min(this.params.iterations, MAX_ITERATIONS);
    if (n <= 0) return input;
    this.inputNode.value = input;
    this.normalPowerU.value = this.params.normalPower;
    this.planeSigmaU.value = this.params.planeSigma;
    const { width, height } = this.tex[0].image as { width: number; height: number };
    for (let i = 0; i < n; i++) dispatch2D(this.renderer, this.kernels[i], width, height);
    return this.tex[(n - 1) % 2];
  }

  dispose(): void {
    this.tex.forEach((t) => t.dispose());
    this.kernels.forEach(disposeKernel);
  }
}
