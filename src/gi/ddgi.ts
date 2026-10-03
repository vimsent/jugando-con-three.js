import { Box3, HalfFloatType, LinearFilter, Matrix4, Quaternion, RGBAFormat, Vector3, type Texture } from 'three';
import { StorageTexture, type WebGPURenderer } from 'three/webgpu';
import { instancedArray, sampler, texture, uniform } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import type GUI from 'lil-gui';
import type { FrameInfo, GIContext, GIMethod, GIStats } from './types';
import type { RTScene } from '../rt/rtScene';
import type { GBuffer } from '../render/gbuffer';
import { GBufferAccess, createOutputTexture, dispatch1D, dispatch2D, disposeKernel, disposeStorage, makeKernel, storeNode } from '../rt/kernels';
import { MemoryLedger, textureBytes } from '../metrics/memory';

/* eslint-disable @typescript-eslint/no-explicit-any */
type FnNode = any;

const IRR_N = 6; // interior texels of an irradiance tile (octahedral)
const DEP_N = 14; // interior texels of a depth tile
const ATLAS_COLS = 64; // probes per atlas row

export interface DDGIParams {
  countX: number;
  countY: number;
  countZ: number;
  raysPerProbe: number; // multiple of 64
  updateFraction: number; // fraction of probes traced per frame (round-robin)
  hysteresis: number;
  normalBias: number;
  viewBias: number;
  depthSharpness: number;
  backfaceThreshold: number; // fraction of back-face hits above which a probe is disabled
}

/**
 * DDGI probe volume (Majercik et al. 2019) on the shared BVH:
 * - a regular grid of probes over the scene bounds;
 * - each frame a round-robin subset of probes traces `raysPerProbe` rays (spherical Fibonacci,
 *   randomly rotated every frame); hits are shaded with ray-traced direct light plus the previous
 *   frame's volume (infinite bounces);
 * - irradiance (E/π, 6x6 octahedral) and mean/mean² distance (14x14) are blended into atlases with
 *   hysteresis; borders are written in the same pass for bilinear filtering;
 * - probes with too many back-face hits (inside walls) are disabled;
 * - sampling uses trilinear weights, a wrap-shading backface term and the Chebyshev visibility test.
 * Atlas state is kept in storage buffers (read-modify-write) and mirrored to rgba16f storage
 * textures that are sampled with hardware filtering.
 */
export class DDGIVolume {
  readonly params: DDGIParams = {
    countX: 16, countY: 8, countZ: 16, raysPerProbe: 128, updateFraction: 0.25,
    hysteresis: 0.97, normalBias: 0.25, viewBias: 0.25, depthSharpness: 50, backfaceThreshold: 0.25,
  };
  // Built resources (rebuilt when the grid or ray count changes).
  irradianceTex!: StorageTexture;
  depthTex!: StorageTexture;
  sampleFn!: FnNode; // fn ddgi_irradiance( p: vec3f, n: vec3f, v: vec3f ) -> vec3f
  private rays: any;
  private irrState: any;
  private depState: any;
  private probeState: any;
  private kernels: Record<string, any> = {};
  private builtKey = '';
  private probeOffset = 0;
  private needsClear = true;
  probeCount = 0;
  probesThisFrame = 0;
  readonly ledger = new MemoryLedger();

  // Uniforms
  private readonly originU = uniform(new Vector3());
  private readonly spacingU = uniform(new Vector3());
  private readonly rotationU = uniform(new Matrix4());
  private readonly offsetU = uniform(0, 'uint');
  private readonly countU = uniform(0, 'uint');
  private readonly hysteresisU = uniform(0.97);
  private readonly normalBiasU = uniform(0.25);
  private readonly viewBiasU = uniform(0.25);
  private readonly maxDistU = uniform(4.0);
  private readonly sharpnessU = uniform(50);
  private readonly backfaceU = uniform(0.25);
  private readonly q = new Quaternion();

  constructor(private readonly renderer: WebGPURenderer, private readonly rt: RTScene, private readonly bounds: Box3) {}

  get key(): string {
    const p = this.params;
    return `${p.countX}x${p.countY}x${p.countZ}/${p.raysPerProbe}`;
  }

  /** (Re)creates buffers, textures and kernels if the grid or ray count changed. */
  ensureBuilt(): void {
    if (this.builtKey === this.key) return;
    this.disposeResources();
    this.builtKey = this.key;
    const p = this.params;
    const NX = p.countX, NY = p.countY, NZ = p.countZ;
    const N = NX * NY * NZ;
    const RAYS = Math.max(64, Math.round(p.raysPerProbe / 64) * 64);
    const rows = Math.ceil(N / ATLAS_COLS);
    const IRR_T = IRR_N + 2, DEP_T = DEP_N + 2;
    const irrW = ATLAS_COLS * IRR_T, irrH = rows * IRR_T;
    const depW = ATLAS_COLS * DEP_T, depH = rows * DEP_T;
    this.probeCount = N;

    const size = this.bounds.getSize(new Vector3());
    this.spacingU.value.set(size.x / NX, size.y / NY, size.z / NZ);
    this.originU.value.copy(this.bounds.min);
    this.maxDistU.value = 1.5 * this.spacingU.value.length();

    const mkTex = (w: number, h: number) => {
      const t = new StorageTexture(w, h);
      t.type = HalfFloatType;
      t.format = RGBAFormat;
      t.minFilter = LinearFilter;
      t.magFilter = LinearFilter;
      t.generateMipmaps = false;
      return t;
    };
    this.irradianceTex = mkTex(irrW, irrH);
    this.depthTex = mkTex(depW, depH);
    this.rays = instancedArray(N * RAYS, 'vec4').setName('ddgi_rays');
    this.irrState = instancedArray(irrW * irrH, 'vec4').setName('ddgi_irrState');
    this.depState = instancedArray(depW * depH, 'vec2').setName('ddgi_depState');
    this.probeState = instancedArray(N, 'uvec2').setName('ddgi_probeState');

    this.ledger.clear();
    this.ledger.add('atlas irradiancia (rgba16f)', textureBytes(irrW, irrH, 8));
    this.ledger.add('atlas distancia (rgba16f)', textureBytes(depW, depH, 8));
    this.ledger.add('estado irradiancia (vec4f32)', irrW * irrH * 16);
    this.ledger.add('estado distancia (vec2f32)', depW * depH * 8);
    this.ledger.add('buffer de rayos (vec4f32)', N * RAYS * 16);
    this.ledger.add('estado de probes', N * 8);

    const { fns } = this.rt;
    const irrTexNode = texture(this.irradianceTex);
    const depTexNode = texture(this.depthTex);
    const irrSampler = sampler(irrTexNode);
    const depSampler = sampler(depTexNode);
    const irrStore = storeNode(this.irradianceTex);
    const depStore = storeNode(this.depthTex);
    const { originU: O, spacingU: S, rotationU: R, offsetU, countU } = this;

    const octEncode = wgslTagFn/* wgsl */ `
      fn ddgi_octEncode( n: vec3f ) -> vec2f {
        var p = n.xy * ( 1.0 / ( abs( n.x ) + abs( n.y ) + abs( n.z ) ) );
        if ( n.z < 0.0 ) {
          let s = select( vec2f( -1.0 ), vec2f( 1.0 ), p >= vec2f( 0.0 ) );
          p = ( 1.0 - abs( p.yx ) ) * s;
        }
        return p;
      }
    `;
    const octDecode = wgslTagFn/* wgsl */ `
      fn ddgi_octDecode( o: vec2f ) -> vec3f {
        var v = vec3f( o, 1.0 - abs( o.x ) - abs( o.y ) );
        if ( v.z < 0.0 ) {
          let s = select( vec2f( -1.0 ), vec2f( 1.0 ), v.xy >= vec2f( 0.0 ) );
          v = vec3f( ( 1.0 - abs( v.yx ) ) * s, v.z );
        }
        return normalize( v );
      }
    `;
    const fib = wgslTagFn/* wgsl */ `
      fn ddgi_rayDir( i: u32 ) -> vec3f {
        let fi = f32( i );
        let phi = 6.283185307 * fract( fi * 0.6180339887 );
        let cosT = 1.0 - ( 2.0 * fi + 1.0 ) / ${RAYS}.0;
        let sinT = sqrt( clamp( 1.0 - cosT * cosT, 0.0, 1.0 ) );
        return normalize( ( ${R} * vec4f( cos( phi ) * sinT, sin( phi ) * sinT, cosT, 0.0 ) ).xyz );
      }
    `;
    const probeCoord = wgslTagFn/* wgsl */ `
      fn ddgi_probeCoord( probe: u32 ) -> vec3i {
        return vec3i( i32( probe % ${NX}u ), i32( ( probe / ${NX}u ) % ${NY}u ), i32( probe / ${NX * NY}u ) );
      }
    `;
    const probePos = wgslTagFn/* wgsl */ `
      fn ddgi_probePos( c: vec3i ) -> vec3f {
        return ${O} + ( vec3f( c ) + 0.5 ) * ${S};
      }
    `;
    const border = wgslTagFn/* wgsl */ `
      fn ddgi_border( t: vec2u, n: u32 ) -> vec2u {
        let last = n + 1u;
        if ( ( t.x == 0u || t.x == last ) && ( t.y == 0u || t.y == last ) ) {
          return vec2u( select( 1u, n, t.x == 0u ), select( 1u, n, t.y == 0u ) );
        }
        if ( t.y == 0u ) {
          return vec2u( last - t.x, 1u );
        }
        if ( t.y == last ) {
          return vec2u( last - t.x, n );
        }
        if ( t.x == 0u ) {
          return vec2u( 1u, last - t.y );
        }
        if ( t.x == last ) {
          return vec2u( n, last - t.y );
        }
        return t;
      }
    `;

    this.sampleFn = wgslTagFn/* wgsl */ `
      fn ddgi_irradiance( p: vec3f, n: vec3f, v: vec3f ) -> vec3f {
        let biased = p + n * ${this.normalBiasU} + v * ${this.viewBiasU};
        let g = ( biased - ${O} ) / ${S} - 0.5;
        let base = clamp( vec3i( floor( g ) ), vec3i( 0 ), vec3i( ${NX - 2}, ${NY - 2}, ${NZ - 2} ) );
        let alpha = clamp( g - vec3f( base ), vec3f( 0.0 ), vec3f( 1.0 ) );
        var sum = vec3f( 0.0 );
        var wsum = 0.0;
        for ( var i = 0u; i < 8u; i = i + 1u ) {
          let off = vec3i( i32( i & 1u ), i32( ( i >> 1u ) & 1u ), i32( ( i >> 2u ) & 1u ) );
          let c = base + off;
          let probe = u32( c.x ) + u32( c.y ) * ${NX}u + u32( c.z ) * ${NX * NY}u;
          if ( ${this.probeState}[ probe ].x == 0u ) {
            continue;
          }
          let pp = ${probePos}( c );
          let tile = vec2f( f32( probe % ${ATLAS_COLS}u ), f32( probe / ${ATLAS_COLS}u ) );
          let tri = mix( 1.0 - alpha, alpha, vec3f( off ) );
          var w = 1.0;
          let wrap = ( dot( normalize( pp - p ), n ) + 1.0 ) * 0.5;
          w *= wrap * wrap + 0.2;
          let toPoint = biased - pp;
          let dist = length( toPoint );
          let dir = toPoint / max( dist, 1e-4 );
          let duv = ( tile * ${DEP_T}.0 + 1.0 + ( ${octEncode}( dir ) * 0.5 + 0.5 ) * ${DEP_N}.0 ) / vec2f( ${depW}.0, ${depH}.0 );
          let m = textureSampleLevel( ${depTexNode}, ${depSampler}, duv, 0.0 ).xy;
          if ( dist > m.x ) {
            let variance = abs( m.x * m.x - m.y );
            let dd = dist - m.x;
            var cheb = variance / ( variance + dd * dd );
            cheb = max( cheb * cheb * cheb, 0.0 );
            w *= max( 0.05, cheb );
          }
          w = max( 1e-6, w );
          if ( w < 0.2 ) {
            w *= w * w / 0.04;
          }
          w *= tri.x * tri.y * tri.z;
          let iuv = ( tile * ${IRR_T}.0 + 1.0 + ( ${octEncode}( n ) * 0.5 + 0.5 ) * ${IRR_N}.0 ) / vec2f( ${irrW}.0, ${irrH}.0 );
          sum += w * textureSampleLevel( ${irrTexNode}, ${irrSampler}, iuv, 0.0 ).xyz;
          wsum += w;
        }
        return select( vec3f( 0.0 ), sum / wsum, wsum > 0.0 );
      }
    `;

    // 1. Trace: one thread per ray of the probes updated this frame.
    this.kernels.trace = makeKernel(wgslTagFn/* wgsl */ `
      fn ddgiTrace( globalId: vec3u ) -> void {
        let r = globalId.x;
        if ( r >= ${countU} * ${RAYS}u ) {
          return;
        }
        let local = r / ${RAYS}u;
        let ri = r % ${RAYS}u;
        let probe = ( ${offsetU} + local ) % ${N}u;
        let origin = ${probePos}( ${probeCoord}( probe ) );
        let dir = ${fib}( ri );
        var rays = 1u;
        let h = ${fns.trace}( origin, dir, 0.0 );
        var out = vec4f( ${fns.sky}( dir ), 1e4 );
        if ( h.hit != 0u ) {
          if ( ( h.flags & 3u ) == 0u ) {
            out = vec4f( 0.0, 0.0, 0.0, -h.t );
          } else {
            let E = ${fns.directIrradiance}( h.position, h.normal, h.geomNormal, &rays );
            let ind = ${this.sampleFn}( h.position, h.normal, -dir );
            out = vec4f( h.albedo * ( E * 0.3183098862 + ind ), h.t );
          }
        }
        ${this.rays}[ r ] = out;
        ${fns.countRays}( vec2u( r, r >> 7u ), rays );
      }
    `, [64, 1, 1]);

    // 2. Blend irradiance: one thread per atlas texel (incl. border) of each updated probe.
    this.kernels.blendIrr = makeKernel(wgslTagFn/* wgsl */ `
      fn ddgiBlendIrr( globalId: vec3u ) -> void {
        let t = globalId.x;
        let local = t / ${IRR_T * IRR_T}u;
        if ( local >= ${countU} ) {
          return;
        }
        let probe = ( ${offsetU} + local ) % ${N}u;
        let k = t % ${IRR_T * IRR_T}u;
        let tc = vec2u( k % ${IRR_T}u, k / ${IRR_T}u );
        let ic = ${border}( tc, ${IRR_N}u );
        let d = ${octDecode}( ( ( vec2f( ic ) - 0.5 ) / ${IRR_N}.0 ) * 2.0 - 1.0 );
        var sum = vec3f( 0.0 );
        var wsum = 0.0;
        for ( var ri = 0u; ri < ${RAYS}u; ri = ri + 1u ) {
          let rr = ${this.rays}[ local * ${RAYS}u + ri ];
          if ( rr.w < 0.0 ) {
            continue;
          }
          let w = max( 0.0, dot( d, ${fib}( ri ) ) );
          sum += w * rr.xyz;
          wsum += w;
        }
        let est = select( vec3f( 0.0 ), sum / wsum, wsum > 1e-4 );
        let tile = vec2u( probe % ${ATLAS_COLS}u, probe / ${ATLAS_COLS}u );
        let px = tile * ${IRR_T}u + tc;
        let idx = px.y * ${irrW}u + px.x;
        let first = ${this.probeState}[ probe ].y == 0u;
        let h = select( ${this.hysteresisU}, 0.0, first );
        let v = mix( est, ${this.irrState}[ idx ].xyz, h );
        ${this.irrState}[ idx ] = vec4f( v, 1.0 );
        textureStore( ${irrStore}, px, vec4f( v, 1.0 ) );
      }
    `, [64, 1, 1]);

    // 3. Blend distance moments.
    this.kernels.blendDepth = makeKernel(wgslTagFn/* wgsl */ `
      fn ddgiBlendDepth( globalId: vec3u ) -> void {
        let t = globalId.x;
        let local = t / ${DEP_T * DEP_T}u;
        if ( local >= ${countU} ) {
          return;
        }
        let probe = ( ${offsetU} + local ) % ${N}u;
        let k = t % ${DEP_T * DEP_T}u;
        let tc = vec2u( k % ${DEP_T}u, k / ${DEP_T}u );
        let ic = ${border}( tc, ${DEP_N}u );
        let d = ${octDecode}( ( ( vec2f( ic ) - 0.5 ) / ${DEP_N}.0 ) * 2.0 - 1.0 );
        var sum = vec2f( 0.0 );
        var wsum = 0.0;
        for ( var ri = 0u; ri < ${RAYS}u; ri = ri + 1u ) {
          let rr = ${this.rays}[ local * ${RAYS}u + ri ];
          var dist = min( abs( rr.w ), ${this.maxDistU} );
          if ( rr.w < 0.0 ) {
            dist = dist * 0.2;
          }
          let w = pow( max( 0.0, dot( d, ${fib}( ri ) ) ), ${this.sharpnessU} );
          sum += w * vec2f( dist, dist * dist );
          wsum += w;
        }
        let est = select( vec2f( ${this.maxDistU}, ${this.maxDistU} * ${this.maxDistU} ), sum / wsum, wsum > 1e-6 );
        let tile = vec2u( probe % ${ATLAS_COLS}u, probe / ${ATLAS_COLS}u );
        let px = tile * ${DEP_T}u + tc;
        let idx = px.y * ${depW}u + px.x;
        let first = ${this.probeState}[ probe ].y == 0u;
        let h = select( ${this.hysteresisU}, 0.0, first );
        let v = mix( est, ${this.depState}[ idx ], h );
        ${this.depState}[ idx ] = v;
        textureStore( ${depStore}, px, vec4f( v, 0.0, 1.0 ) );
      }
    `, [64, 1, 1]);

    // 4. Probe state: classification by back-face ratio and update counter.
    this.kernels.state = makeKernel(wgslTagFn/* wgsl */ `
      fn ddgiState( globalId: vec3u ) -> void {
        let local = globalId.x;
        if ( local >= ${countU} ) {
          return;
        }
        let probe = ( ${offsetU} + local ) % ${N}u;
        var back = 0u;
        for ( var ri = 0u; ri < ${RAYS}u; ri = ri + 1u ) {
          if ( ${this.rays}[ local * ${RAYS}u + ri ].w < 0.0 ) {
            back = back + 1u;
          }
        }
        let isActive = select( 0u, 1u, f32( back ) < ${this.backfaceU} * ${RAYS}.0 );
        let s = ${this.probeState}[ probe ];
        ${this.probeState}[ probe ] = vec2u( isActive, s.y + 1u );
      }
    `, [64, 1, 1]);

    // Reset: forget all probe history.
    this.kernels.clear = makeKernel(wgslTagFn/* wgsl */ `
      fn ddgiClear( globalId: vec3u ) -> void {
        if ( globalId.x < ${N}u ) {
          ${this.probeState}[ globalId.x ] = vec2u( 0u, 0u );
        }
      }
    `, [64, 1, 1]);

    this.raysPerProbeBuilt = RAYS;
    this.atlas = { irrW, irrH, depW, depH };
    this.needsClear = true;
    this.probeOffset = 0;
  }

  raysPerProbeBuilt = 128;
  atlas = { irrW: 0, irrH: 0, depW: 0, depH: 0 };

  reset(): void {
    this.needsClear = true;
  }

  /** Records trace + blend passes for this frame's subset of probes. */
  update(timer: { begin(n: string): void; end(n: string): void }, prefix = 'ddgi'): void {
    this.ensureBuilt();
    const p = this.params;
    const N = this.probeCount;
    const RAYS = this.raysPerProbeBuilt;
    if (this.needsClear) {
      dispatch1D(this.renderer, this.kernels.clear, N);
      this.needsClear = false;
    }
    const count = Math.min(N, Math.max(1, Math.ceil(N * p.updateFraction)));
    this.probesThisFrame = count;
    this.offsetU.value = this.probeOffset;
    this.countU.value = count;
    this.hysteresisU.value = p.hysteresis;
    this.normalBiasU.value = p.normalBias;
    this.viewBiasU.value = p.viewBias;
    this.sharpnessU.value = p.depthSharpness;
    this.backfaceU.value = p.backfaceThreshold;
    // Uniformly random rotation of the ray set (Shoemake).
    const u1 = Math.random(), u2 = Math.random() * 2 * Math.PI, u3 = Math.random() * 2 * Math.PI;
    const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
    this.q.set(a * Math.sin(u2), a * Math.cos(u2), b * Math.sin(u3), b * Math.cos(u3));
    this.rotationU.value.makeRotationFromQuaternion(this.q);

    timer.begin(`${prefix}.trace`);
    dispatch1D(this.renderer, this.kernels.trace, count * RAYS);
    timer.end(`${prefix}.trace`);
    timer.begin(`${prefix}.blend`);
    dispatch1D(this.renderer, this.kernels.blendIrr, count * (IRR_N + 2) ** 2);
    dispatch1D(this.renderer, this.kernels.blendDepth, count * (DEP_N + 2) ** 2);
    dispatch1D(this.renderer, this.kernels.state, count);
    timer.end(`${prefix}.blend`);
    this.probeOffset = (this.probeOffset + count) % N;
  }

  get memoryBytes(): number {
    return this.ledger.total;
  }

  private disposeResources(): void {
    Object.values(this.kernels).forEach(disposeKernel);
    this.kernels = {};
    this.irradianceTex?.dispose();
    this.depthTex?.dispose();
    for (const n of [this.rays, this.irrState, this.depState, this.probeState]) if (n) disposeStorage(this.renderer, n);
  }

  dispose(): void {
    this.disposeResources();
    this.builtKey = '';
  }
}

/** Method 4: DDGI probes; the gather pass samples the volume at every G-buffer pixel. */
export class DDGIMethod implements GIMethod {
  readonly key: string = 'ddgi';
  readonly label: string = 'DDGI (probes)';
  protected ctx!: GIContext;
  protected rt!: RTScene;
  volume!: DDGIVolume;
  protected output!: StorageTexture;
  private gather: any = null;
  private gatherKey = '';
  private gb!: GBufferAccess;

  async init(ctx: GIContext): Promise<void> {
    this.ctx = ctx;
    this.rt = await ctx.getRTScene();
    const bounds = sceneBounds(this.rt);
    this.volume = new DDGIVolume(ctx.renderer, this.rt, bounds);
    this.output = createOutputTexture(ctx.width, ctx.height);
    this.gb = new GBufferAccess(ctx.gbuffer, ctx.camera);
  }

  private buildGather(): void {
    disposeKernel(this.gather);
    this.gatherKey = this.volume.key;
    const W = this.ctx.width, H = this.ctx.height;
    const out = storeNode(this.output);
    this.gather = makeKernel(wgslTagFn/* wgsl */ `
      fn ddgiGather( globalId: vec3u ) -> void {
        let px = globalId.xy;
        if ( px.x >= ${W}u || px.y >= ${H}u ) {
          return;
        }
        var p: vec3f;
        var n: vec3f;
        if ( ! ${this.gb.surface}( px, vec2u( ${W}u, ${H}u ), &p, &n ) ) {
          textureStore( ${out}, px, vec4f( 0.0, 0.0, 0.0, 1.0 ) );
          return;
        }
        let v = normalize( ${this.gb.camPos} - p );
        textureStore( ${out}, px, vec4f( ${this.volume.sampleFn}( p, n, v ), 1.0 ) );
      }
    `);
  }

  update(_dt: number, _info: FrameInfo): void {}

  reset(): void {
    // DDGI lives in world space: camera jumps do not invalidate it. Explicit resets (key R) do.
    this.volume.reset();
  }

  protected runVolumeAndGather(): Texture {
    const { renderer, timer, width, height } = this.ctx;
    this.rt.writeLights(this.ctx.lights);
    this.volume.update(timer);
    if (this.gatherKey !== this.volume.key) this.buildGather();
    timer.begin('ddgi.gather');
    dispatch2D(renderer, this.gather, width, height);
    timer.end('ddgi.gather');
    this.rt.sampleRayCounter(renderer);
    return this.output;
  }

  run(_gbuffer: GBuffer): Texture {
    return this.runVolumeAndGather();
  }

  stats(): GIStats {
    const ledger = new MemoryLedger();
    for (const [k, v] of Object.entries(this.volume.ledger.breakdown())) ledger.add(k, v);
    ledger.add('salida (rgba16f)', textureBytes(this.ctx.width, this.ctx.height, 8));
    const v = this.volume;
    return {
      raysPerFrame: this.rt.rayCountLast,
      memoryBytes: ledger.total + this.rt.memoryBytes,
      memoryBreakdown: { ...ledger.breakdown(), 'BVH + geometría': this.rt.memoryBytes },
      extra: {
        Probes: `${v.probeCount} (${v.params.countX}×${v.params.countY}×${v.params.countZ}), ${v.probesThisFrame}/frame × ${v.raysPerProbeBuilt} rayos`,
      },
    };
  }

  buildGui(folder: GUI): void {
    const p = this.volume.params;
    const rebuild = () => this.volume.ensureBuilt();
    folder.add(p, 'countX', 2, 32, 1).name('probes X').onFinishChange(rebuild);
    folder.add(p, 'countY', 2, 16, 1).name('probes Y').onFinishChange(rebuild);
    folder.add(p, 'countZ', 2, 32, 1).name('probes Z').onFinishChange(rebuild);
    folder.add(p, 'raysPerProbe', 64, 512, 64).name('rayos por probe').onFinishChange(rebuild);
    folder.add(p, 'updateFraction', { '1': 1, '1/2': 0.5, '1/4': 0.25, '1/8': 0.125, '1/16': 0.0625 }).name('fracción por frame');
    folder.add(p, 'hysteresis', 0, 0.995, 0.001).name('histéresis');
    folder.add(p, 'normalBias', 0, 1, 0.01).name('sesgo normal');
    folder.add(p, 'viewBias', 0, 1, 0.01).name('sesgo vista');
    folder.add(p, 'depthSharpness', 1, 100, 1).name('nitidez distancia');
    folder.add(p, 'backfaceThreshold', 0, 1, 0.01).name('umbral caras traseras');
    folder.add({ reset: () => this.volume.reset() }, 'reset').name('reiniciar probes');
  }

  dispose(): void {
    this.volume.dispose();
    disposeKernel(this.gather);
    this.output.dispose();
  }
}

/** Scene bounds from the BVH geometry, used for the probe grid. */
export function sceneBounds(rt: RTScene): Box3 {
  const g = (rt.bvhData as any).objects[0].geometry;
  g.computeBoundingBox();
  return g.boundingBox.clone();
}
