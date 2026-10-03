import { Matrix4, Vector3, type Texture } from 'three';
import type { StorageTexture } from 'three/webgpu';
import { instancedArray, uniform } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import type GUI from 'lil-gui';
import type { FrameInfo, GIContext, GIMethod, GIStats } from './types';
import type { RTScene } from '../rt/rtScene';
import type { GBuffer } from '../render/gbuffer';
import { GBufferAccess, createOutputTexture, dispatch2D, disposeKernel, disposeStorage, makeKernel, storeNode } from '../rt/kernels';
import { AtrousFilter } from '../render/atrous';
import { MemoryLedger, textureBytes } from '../metrics/memory';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Method 7 (optional): simplified ReSTIR GI (Ouyang et al. 2021), 1 sample per pixel.
 *
 * A sample is a secondary surface point x1 seen from the pixel's surface x0 with its outgoing
 * radiance Lo(x1) (direct light + one more cosine bounce with NEE, sky on escape). Reservoirs hold
 * one sample with (w_sum, M, W); the target function is p̂ = lum(Lo) · max(0, n0·ω).
 *  1. Initial + temporal: a fresh sample (source pdf cos/π) is merged with the reprojected
 *     previous reservoir (M clamped to `temporalMaxM`; rejected on depth/normal mismatch).
 *  2. Spatial: `spatialSamples` neighbours within `spatialRadius` px with similar depth/normal are
 *     merged with the reconnection-shift Jacobian; the chosen sample gets one visibility ray from
 *     x0 (biased otherwise: neighbours' visibility is not re-tested per candidate).
 *  3. Shade: indirect (E/π) = Lo · max(0, n0·ω)/π · W, followed by the shared à-trous filter.
 * Reservoir layout: 3 × vec4f per pixel: (x1, W), (n1, M), (Lo, w_sum).
 */
export class ReSTIRGIMethod implements GIMethod {
  readonly key = 'restir';
  readonly label = 'ReSTIR GI (simplificado)';
  readonly params = { temporal: true, temporalMaxM: 20, spatial: true, spatialSamples: 4, spatialRadius: 24, visibility: true };
  private ctx!: GIContext;
  private rt!: RTScene;
  private gb!: GBufferAccess;
  private output!: StorageTexture;
  private atrous!: AtrousFilter;
  private res: any[] = []; // [history, current] ping-pong
  private dist: any[] = [];
  private kernels: any[] = [];
  private ping = 0;
  private frameCounter = 1;
  private needsReset = true;
  private readonly frameU = uniform(1, 'uint');
  private readonly resetU = uniform(1, 'uint');
  private readonly temporalU = uniform(1, 'uint');
  private readonly maxMU = uniform(20);
  private readonly spatialU = uniform(1, 'uint');
  private readonly kU = uniform(4, 'uint');
  private readonly radiusU = uniform(24);
  private readonly visU = uniform(1, 'uint');
  private readonly prevViewProj = uniform(new Matrix4());
  private readonly prevCamPos = uniform(new Vector3());
  private readonly savedViewProj = new Matrix4();
  private readonly savedCamPos = new Vector3();
  private readonly ledger = new MemoryLedger();

  async init(ctx: GIContext): Promise<void> {
    this.ctx = ctx;
    this.rt = await ctx.getRTScene();
    this.gb = new GBufferAccess(ctx.gbuffer, ctx.camera);
    this.output = createOutputTexture(ctx.width, ctx.height);
    this.atrous = new AtrousFilter(ctx.renderer, ctx.gbuffer, ctx.camera);
    const W = ctx.width, H = ctx.height, N = W * H;
    this.res = [0, 1].map((i) => instancedArray(N * 3, 'vec4').setName(`restir_res${i}`));
    this.dist = [0, 1].map((i) => instancedArray(N, 'float').setName(`restir_dist${i}`));
    this.ledger.add('reservorios temporales (2 × 48 B/px)', 2 * N * 48);
    this.ledger.add('distancias para reproyección (2 × f32)', 2 * N * 4);
    this.ledger.add('salida (rgba16f)', textureBytes(W, H, 8));
    this.ledger.add('à-trous (2×rgba16f)', this.atrous.memoryBytes);
    this.kernels = [0, 1].map((k) => this.buildKernels(k));
  }

  private buildKernels(k: number): { temporal: any; spatial: any } {
    const { fns } = this.rt;
    const gb = this.gb;
    const W = this.ctx.width, H = this.ctx.height;
    const hist = this.res[k];
    const cur = this.res[1 - k];
    const dHist = this.dist[k];
    const dCur = this.dist[1 - k];
    const out = storeNode(this.output);

    const lum = wgslTagFn/* wgsl */ `
      fn restir_lum( c: vec3f ) -> f32 {
        return dot( c, vec3f( 0.2126, 0.7152, 0.0722 ) );
      }
    `;

    const temporal = makeKernel(wgslTagFn/* wgsl */ `
      fn restirTemporal( globalId: vec3u ) -> void {
        let px = globalId.xy;
        if ( px.x >= ${W}u || px.y >= ${H}u ) {
          return;
        }
        let i = px.y * ${W}u + px.x;
        var p: vec3f;
        var n: vec3f;
        if ( ! ${gb.surface}( px, vec2u( ${W}u, ${H}u ), &p, &n ) ) {
          ${cur}[ i * 3u ] = vec4f( 0.0 );
          ${cur}[ i * 3u + 1u ] = vec4f( 0.0 );
          ${cur}[ i * 3u + 2u ] = vec4f( 0.0 );
          ${dCur}[ i ] = -1.0;
          return;
        }
        var state = ${fns.seed}( px, ${this.frameU}, 0x7f4a7c15u );
        var rays = 1u;

        // Fresh candidate.
        let dir = ${fns.cosineDir}( n, vec2f( ${fns.rand}( &state ), ${fns.rand}( &state ) ) );
        let h = ${fns.trace}( p + n * 0.004, dir, 0.0 );
        var x1 = p + dir * 1e4;
        var n1 = -dir;
        var Lo = ${fns.sky}( dir );
        if ( h.hit != 0u ) {
          x1 = h.position;
          n1 = h.normal;
          Lo = h.albedo * ${fns.directSampled}( h.position, h.normal, h.geomNormal, &state, &rays ) * 0.3183098862;
          var d2 = ${fns.cosineDir}( h.normal, vec2f( ${fns.rand}( &state ), ${fns.rand}( &state ) ) );
          if ( dot( d2, h.geomNormal ) <= 0.0 ) {
            d2 = reflect( d2, h.geomNormal );
          }
          rays += 1u;
          let h2 = ${fns.trace}( h.position + h.geomNormal * 0.002, d2, 0.0 );
          var L2 = ${fns.sky}( d2 );
          if ( h2.hit != 0u ) {
            L2 = h2.albedo * ${fns.directSampled}( h2.position, h2.normal, h2.geomNormal, &state, &rays ) * 0.3183098862;
          }
          Lo += h.albedo * L2;
        }
        let cosT = max( dot( n, dir ), 0.0 );
        let pdf = max( cosT * 0.3183098862, 1e-6 );
        let pHat = ${lum}( Lo ) * cosT;
        var rX = x1;
        var rN = n1;
        var rL = Lo;
        var wSum = pHat / pdf;
        var M = 1.0;

        // Temporal reuse.
        if ( ${this.temporalU} != 0u && ${this.resetU} == 0u ) {
          let clip = ${this.prevViewProj} * vec4f( p, 1.0 );
          if ( clip.w > 0.0 ) {
            let ndc = clip.xyz / clip.w;
            let q = vec2i( floor( vec2f( ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5 ) * vec2f( ${W}.0, ${H}.0 ) ) );
            if ( q.x >= 0 && q.y >= 0 && q.x < ${W} && q.y < ${H} ) {
              let j = u32( q.y ) * ${W}u + u32( q.x );
              let expected = distance( p, ${this.prevCamPos} );
              let stored = ${dHist}[ j ];
              let a = ${hist}[ j * 3u ];
              let b = ${hist}[ j * 3u + 1u ];
              let c = ${hist}[ j * 3u + 2u ];
              if ( stored > 0.0 && abs( stored - expected ) < 0.02 * expected + 0.03 && b.w > 0.0 ) {
                let hM = min( b.w, ${this.maxMU} );
                let dirH = normalize( a.xyz - p );
                let pHatH = ${lum}( c.xyz ) * max( dot( n, dirH ), 0.0 );
                let wH = pHatH * a.w * hM;
                wSum += wH;
                M += hM;
                if ( ${fns.rand}( &state ) * wSum < wH ) {
                  rX = a.xyz;
                  rN = b.xyz;
                  rL = c.xyz;
                }
              }
            }
          }
        }
        let pHatR = ${lum}( rL ) * max( dot( n, normalize( rX - p ) ), 0.0 );
        let Wr = select( 0.0, wSum / ( M * pHatR ), pHatR > 0.0 );
        ${cur}[ i * 3u ] = vec4f( rX, Wr );
        ${cur}[ i * 3u + 1u ] = vec4f( rN, M );
        ${cur}[ i * 3u + 2u ] = vec4f( rL, wSum );
        ${dCur}[ i ] = distance( p, ${gb.camPos} );
        ${fns.countRays}( px, rays );
      }
    `);

    const spatial = makeKernel(wgslTagFn/* wgsl */ `
      fn restirSpatial( globalId: vec3u ) -> void {
        let px = globalId.xy;
        if ( px.x >= ${W}u || px.y >= ${H}u ) {
          return;
        }
        let i = px.y * ${W}u + px.x;
        var p: vec3f;
        var n: vec3f;
        if ( ! ${gb.surface}( px, vec2u( ${W}u, ${H}u ), &p, &n ) ) {
          textureStore( ${out}, px, vec4f( 0.0, 0.0, 0.0, 1.0 ) );
          return;
        }
        var state = ${fns.seed}( px, ${this.frameU}, 0x3c6ef372u );
        var rays = 0u;
        let a0 = ${cur}[ i * 3u ];
        let b0 = ${cur}[ i * 3u + 1u ];
        let c0 = ${cur}[ i * 3u + 2u ];
        var rX = a0.xyz;
        var rN = b0.xyz;
        var rL = c0.xyz;
        let pHat0 = ${lum}( rL ) * max( dot( n, normalize( rX - p ) ), 0.0 );
        var wSum = pHat0 * a0.w * b0.w;
        var M = b0.w;
        let depth0 = distance( p, ${gb.camPos} );
        if ( ${this.spatialU} != 0u ) {
          for ( var s = 0u; s < ${this.kU}; s = s + 1u ) {
            let ang = 6.283185307 * ${fns.rand}( &state );
            let rad = ${this.radiusU} * sqrt( ${fns.rand}( &state ) );
            let q = vec2i( vec2f( px ) + vec2f( cos( ang ), sin( ang ) ) * rad );
            if ( q.x < 0 || q.y < 0 || q.x >= ${W} || q.y >= ${H} ) {
              continue;
            }
            var qp: vec3f;
            var qn: vec3f;
            if ( ! ${gb.surface}( vec2u( q ), vec2u( ${W}u, ${H}u ), &qp, &qn ) ) {
              continue;
            }
            if ( dot( qn, n ) < 0.9 || abs( distance( qp, ${gb.camPos} ) - depth0 ) > 0.1 * depth0 ) {
              continue;
            }
            let j = u32( q.y ) * ${W}u + u32( q.x );
            let a = ${cur}[ j * 3u ];
            let b = ${cur}[ j * 3u + 1u ];
            let c = ${cur}[ j * 3u + 2u ];
            if ( b.w <= 0.0 ) {
              continue;
            }
            // Reconnection-shift Jacobian from the neighbour's surface to ours.
            let vq = qp - a.xyz;
            let vr = p - a.xyz;
            let dq2 = max( dot( vq, vq ), 1e-6 );
            let dr2 = max( dot( vr, vr ), 1e-6 );
            let cq = abs( dot( b.xyz, vq ) ) / sqrt( dq2 );
            let cr = abs( dot( b.xyz, vr ) ) / sqrt( dr2 );
            var jac = select( 0.0, ( cr / max( cq, 1e-4 ) ) * ( dq2 / dr2 ), cq > 1e-4 );
            jac = min( jac, 10.0 );
            let dirN = normalize( a.xyz - p );
            let pHatN = ${lum}( c.xyz ) * max( dot( n, dirN ), 0.0 );
            let wN = pHatN * a.w * b.w * jac;
            wSum += wN;
            M += b.w;
            if ( ${fns.rand}( &state ) * wSum < wN ) {
              rX = a.xyz;
              rN = b.xyz;
              rL = c.xyz;
            }
          }
        }
        let dir = normalize( rX - p );
        let cosT = max( dot( n, dir ), 0.0 );
        let pHatR = ${lum}( rL ) * cosT;
        var Wr = select( 0.0, wSum / ( M * pHatR ), pHatR > 0.0 );
        if ( ${this.visU} != 0u && Wr > 0.0 ) {
          let toX = rX - p;
          let dist = length( toX );
          rays += 1u;
          let maxD = select( 0.0, dist - 0.01, dist < 9e3 );
          if ( ${fns.occluded}( p + n * 0.004, toX / dist, maxD ) ) {
            Wr = 0.0;
          }
        }
        textureStore( ${out}, px, vec4f( rL * cosT * 0.3183098862 * Wr, 1.0 ) );
        ${fns.countRays}( px, rays );
      }
    `);
    return { temporal, spatial };
  }

  // Reservoirs keep the radiance Lo computed when the sample was created; after a light change the
  // stale values fade out as fresh samples arrive (bounded by the temporal M clamp).
  update(_dt: number, _info: FrameInfo): void {}

  reset(): void {
    this.needsReset = true;
  }

  onCameraCut(): void {
    this.needsReset = true;
  }

  run(_gbuffer: GBuffer): Texture {
    const { renderer, timer, width, height, camera } = this.ctx;
    const p = this.params;
    this.rt.writeLights(this.ctx.lights);
    this.frameU.value = this.frameCounter++;
    this.resetU.value = this.needsReset ? 1 : 0;
    this.temporalU.value = p.temporal ? 1 : 0;
    this.maxMU.value = p.temporalMaxM;
    this.spatialU.value = p.spatial ? 1 : 0;
    this.kU.value = p.spatialSamples;
    this.radiusU.value = p.spatialRadius;
    this.visU.value = p.visibility ? 1 : 0;
    this.prevViewProj.value.copy(this.savedViewProj);
    this.prevCamPos.value.copy(this.savedCamPos);
    const k = this.kernels[this.ping];
    timer.begin('restir.inicial+temporal');
    dispatch2D(renderer, k.temporal, width, height);
    timer.end('restir.inicial+temporal');
    timer.begin('restir.espacial+shade');
    dispatch2D(renderer, k.spatial, width, height);
    timer.end('restir.espacial+shade');
    timer.begin('restir.atrous');
    const out = this.atrous.run(this.output);
    timer.end('restir.atrous');
    this.ping = 1 - this.ping;
    this.needsReset = false;
    this.savedViewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.savedCamPos.copy(camera.position);
    this.rt.sampleRayCounter(renderer);
    return out;
  }

  stats(): GIStats {
    return {
      raysPerFrame: this.rt.rayCountLast,
      memoryBytes: this.ledger.total,
      sharedBytes: this.rt.memoryBytes,
      memoryBreakdown: this.ledger.breakdown(),
    };
  }

  buildGui(folder: GUI): void {
    const p = this.params;
    folder.add(p, 'temporal').name('reuso temporal');
    folder.add(p, 'temporalMaxM', 1, 100, 1).name('M máx. temporal');
    folder.add(p, 'spatial').name('reuso espacial');
    folder.add(p, 'spatialSamples', 1, 16, 1).name('vecinos espaciales');
    folder.add(p, 'spatialRadius', 2, 64, 1).name('radio espacial (px)');
    folder.add(p, 'visibility').name('rayo de visibilidad final');
    folder.add(this.atrous.params, 'iterations', 0, 5, 1).name('à-trous iteraciones');
  }

  dispose(): void {
    this.kernels.forEach((k) => {
      disposeKernel(k.temporal);
      disposeKernel(k.spatial);
    });
    [...this.res, ...this.dist].forEach((n) => disposeStorage(this.ctx.renderer, n));
    this.output.dispose();
    this.atrous.dispose();
  }
}
