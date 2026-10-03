import type { Texture } from 'three';
import type { StorageTexture } from 'three/webgpu';
import { instancedArray, uniform } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import type GUI from 'lil-gui';
import type { FrameInfo, GIContext, GIMethod, GIStats } from './types';
import type { RTScene } from '../rt/rtScene';
import type { GBuffer } from '../render/gbuffer';
import { GBufferAccess, createOutputTexture, dispatch1D, dispatch2D, disposeKernel, disposeStorage, makeKernel, storeNode } from '../rt/kernels';
import { AtrousFilter } from '../render/atrous';
import { TemporalFilter } from '../render/temporal';
import { MemoryLedger, textureBytes } from '../metrics/memory';

/* eslint-disable @typescript-eslint/no-explicit-any */

const MAX_BOUNCES = 6; // compile-time bound of the update path length
const PROBE_SLOTS = 8; // linear probing window
const FIXED_SCALE = 4096; // fixed-point scale for atomic radiance accumulation

/**
 * Method 6: spatially hashed radiance cache in the spirit of NVIDIA SHaRC.
 *
 * Cache entry = world-space voxel (size grows with camera distance: about `voxelPixels` projected
 * pixels, snapped to powers of two) x quantized normal (6 directions). It stores the outgoing
 * radiance of the (Lambertian) surfaces in the voxel.
 *
 * Per frame:
 *  1. Update: short paths start at the G-buffer surface of 1 pixel per `updateTile`² tile (random
 *     within the tile). Every hit vertex gets next-event estimation; the path tail is closed with
 *     the cache itself (infinite bounces). Radiance is propagated backwards and accumulated into
 *     the entry of every vertex with fixed-point atomics.
 *  2. Resolve: per entry, this frame's samples are merged into the resolved radiance with a
 *     running mean capped at `maxSamples` (temporal adaptivity); entries not updated for
 *     `staleFrames` frames are evicted.
 *  3. Query: one cosine ray per pixel; the hit's cached radiance is the indirect estimate
 *     (fallback when the entry is missing: direct light only at the hit).
 *  4. Edge-avoiding à-trous spatial denoiser (optional temporal accumulation afterwards).
 */
export class SharcMethod implements GIMethod {
  readonly key = 'sharc';
  readonly label = 'Cache hash (tipo SHaRC)';
  readonly params = {
    log2Capacity: 20,
    updateTile: 4,
    maxBounces: 3,
    voxelPixels: 12,
    maxSamples: 32,
    staleFrames: 64,
    minSamples: 1,
    temporal: false,
  };
  private ctx!: GIContext;
  private rt!: RTScene;
  private gb!: GBufferAccess;
  private output!: StorageTexture;
  private atrous!: AtrousFilter;
  private temporal!: TemporalFilter;
  private keys: any;
  private accum: any;
  private resolved: any;
  private meta: any;
  private kernels: Record<string, any> = {};
  private builtCapacity = 0;
  private needsClear = true;
  private frameCounter = 1;
  private readonly frameU = uniform(1, 'uint');
  private readonly tileU = uniform(4, 'uint');
  private readonly bouncesU = uniform(3, 'uint');
  private readonly voxelScaleU = uniform(0.01);
  private readonly maxSamplesU = uniform(32);
  private readonly staleU = uniform(64, 'uint');
  private readonly minSamplesU = uniform(1);
  private readonly ledger = new MemoryLedger();

  async init(ctx: GIContext): Promise<void> {
    this.ctx = ctx;
    this.rt = await ctx.getRTScene();
    this.gb = new GBufferAccess(ctx.gbuffer, ctx.camera);
    this.output = createOutputTexture(ctx.width, ctx.height);
    this.atrous = new AtrousFilter(ctx.renderer, ctx.gbuffer, ctx.camera);
    this.temporal = new TemporalFilter(ctx.renderer, ctx.gbuffer, ctx.camera, 'sharc');
    this.build();
  }

  private build(): void {
    this.disposeTable();
    const CAP = 1 << this.params.log2Capacity;
    this.builtCapacity = CAP;
    const MASK = CAP - 1;
    this.keys = instancedArray(CAP, 'uint').toAtomic().setName('sharc_keys');
    this.accum = instancedArray(CAP * 4, 'uint').toAtomic().setName('sharc_accum');
    this.resolved = instancedArray(CAP, 'uvec2').setName('sharc_resolved');
    this.meta = instancedArray(CAP, 'uint').setName('sharc_meta');
    this.ledger.clear();
    this.ledger.add('claves (u32)', CAP * 4);
    this.ledger.add('acumuladores (4×u32)', CAP * 16);
    this.ledger.add('radiancia resuelta (4×f16)', CAP * 8);
    this.ledger.add('último frame (u32)', CAP * 4);
    this.ledger.add('salida query (rgba16f)', textureBytes(this.ctx.width, this.ctx.height, 8));
    this.ledger.add('à-trous (2×rgba16f)', this.atrous.memoryBytes);

    const { fns } = this.rt;
    const { keys, accum, resolved, meta } = this;
    const W = this.ctx.width, H = this.ctx.height;
    const gb = this.gb;

    // Voxel key of a surface point: returns (table index, checksum).
    const voxelKey = wgslTagFn/* wgsl */ `
      fn sharc_key( p: vec3f, n: vec3f ) -> vec2u {
        let d = max( distance( p, ${gb.camPos} ), 0.05 );
        let level = clamp( ceil( log2( d * ${this.voxelScaleU} / 0.01 ) ), 0.0, 15.0 );
        let size = 0.01 * exp2( level );
        let c = vec3i( floor( p / size ) );
        let a = abs( n );
        var axis = 0u;
        if ( a.y > a.x && a.y > a.z ) {
          axis = 1u;
        } else if ( a.z > a.x ) {
          axis = 2u;
        }
        let s = select( 0u, 1u, n[ axis ] < 0.0 );
        let w = u32( level ) | ( ( axis * 2u + s ) << 4u );
        let h = ${fns.pcg}( bitcast<u32>( c.x ) ^ ${fns.pcg}( bitcast<u32>( c.y ) ^ ${fns.pcg}( bitcast<u32>( c.z ) ^ ${fns.pcg}( w ) ) ) );
        let check = ${fns.pcg}( h ^ 0x9e3779b9u ) | 1u;
        return vec2u( h & ${MASK}u, check );
      }
    `;

    const findSlot = wgslTagFn/* wgsl */ `
      fn sharc_find( k: vec2u ) -> i32 {
        for ( var i = 0u; i < ${PROBE_SLOTS}u; i = i + 1u ) {
          let slot = ( k.x + i ) & ${MASK}u;
          if ( atomicLoad( &${keys}[ slot ] ) == k.y ) {
            return i32( slot );
          }
        }
        return -1;
      }
    `;

    const insertSlot = wgslTagFn/* wgsl */ `
      fn sharc_insert( k: vec2u ) -> i32 {
        for ( var i = 0u; i < ${PROBE_SLOTS}u; i = i + 1u ) {
          let slot = ( k.x + i ) & ${MASK}u;
          for ( var attempt = 0u; attempt < 2u; attempt = attempt + 1u ) {
            let r = atomicCompareExchangeWeak( &${keys}[ slot ], 0u, k.y );
            if ( r.exchanged || r.old_value == k.y ) {
              return i32( slot );
            }
            if ( r.old_value != 0u ) {
              break;
            }
          }
        }
        return -1;
      }
    `;

    // Resolved radiance of a slot, w = sample count.
    const readSlot = wgslTagFn/* wgsl */ `
      fn sharc_read( slot: i32 ) -> vec4f {
        let r = ${resolved}[ u32( slot ) ];
        return vec4f( unpack2x16float( r.x ), unpack2x16float( r.y ) );
      }
    `;

    // 1. Update paths.
    this.kernels.update = makeKernel(wgslTagFn/* wgsl */ `
      fn sharcUpdate( globalId: vec3u ) -> void {
        let tiles = vec2u( ( ${W}u + ${this.tileU} - 1u ) / ${this.tileU}, ( ${H}u + ${this.tileU} - 1u ) / ${this.tileU} );
        let t = globalId.x;
        if ( t >= tiles.x * tiles.y ) {
          return;
        }
        let tileCoord = vec2u( t % tiles.x, t / tiles.x );
        var state = ${fns.seed}( tileCoord, ${this.frameU}, 0x51a3c0deu );
        let jitter = vec2u( vec2f( ${fns.rand}( &state ), ${fns.rand}( &state ) ) * f32( ${this.tileU} ) );
        let px = min( tileCoord * ${this.tileU} + jitter, vec2u( ${W - 1}u, ${H - 1}u ) );
        var p: vec3f;
        var n: vec3f;
        if ( ! ${gb.surface}( px, vec2u( ${W}u, ${H}u ), &p, &n ) ) {
          return;
        }
        var slots: array<i32, ${MAX_BOUNCES}>;
        var direct: array<vec3f, ${MAX_BOUNCES}>;
        var albedo: array<vec3f, ${MAX_BOUNCES}>;
        var count = 0u;
        var tail = vec3f( 0.0 );
        var rays = 0u;
        var o = p;
        var nn = n;
        var g = n;
        for ( var b = 0u; b < ${this.bouncesU}; b = b + 1u ) {
          var dir = ${fns.cosineDir}( nn, vec2f( ${fns.rand}( &state ), ${fns.rand}( &state ) ) );
          if ( dot( dir, g ) <= 0.0 ) {
            dir = reflect( dir, g );
          }
          rays += 1u;
          let h = ${fns.trace}( o + g * 0.002, dir, 0.0 );
          if ( h.hit == 0u ) {
            tail = ${fns.sky}( dir );
            break;
          }
          let k = ${voxelKey}( h.position, h.normal );
          let isLast = b + 1u >= ${this.bouncesU};
          if ( isLast ) {
            // Close the path with the cache when the entry is already resolved.
            let s = ${findSlot}( k );
            if ( s >= 0 ) {
              let c = ${readSlot}( s );
              if ( c.w >= ${this.minSamplesU} ) {
                tail = c.xyz;
                break;
              }
            }
          }
          slots[ count ] = ${insertSlot}( k );
          direct[ count ] = h.albedo * ${fns.directSampled}( h.position, h.normal, h.geomNormal, &state, &rays ) * 0.3183098862;
          albedo[ count ] = h.albedo;
          count = count + 1u;
          o = h.position;
          nn = h.normal;
          g = h.geomNormal;
        }
        var L = tail;
        for ( var i = i32( count ) - 1; i >= 0; i = i - 1 ) {
          L = direct[ i ] + albedo[ i ] * L;
          let s = slots[ i ];
          if ( s >= 0 ) {
            let q = vec3u( clamp( L, vec3f( 0.0 ), vec3f( 1000.0 ) ) * ${FIXED_SCALE}.0 );
            let base = u32( s ) * 4u;
            atomicAdd( &${accum}[ base ], q.x );
            atomicAdd( &${accum}[ base + 1u ], q.y );
            atomicAdd( &${accum}[ base + 2u ], q.z );
            atomicAdd( &${accum}[ base + 3u ], 1u );
          }
        }
        ${fns.countRays}( px, rays );
      }
    `, [64, 1, 1]);

    // 2. Resolve + eviction.
    this.kernels.resolve = makeKernel(wgslTagFn/* wgsl */ `
      fn sharcResolve( globalId: vec3u ) -> void {
        let i = globalId.x;
        if ( i >= ${CAP}u ) {
          return;
        }
        if ( atomicLoad( &${keys}[ i ] ) == 0u ) {
          return;
        }
        let base = i * 4u;
        let cnt = atomicExchange( &${accum}[ base + 3u ], 0u );
        let r = atomicExchange( &${accum}[ base ], 0u );
        let g = atomicExchange( &${accum}[ base + 1u ], 0u );
        let b = atomicExchange( &${accum}[ base + 2u ], 0u );
        if ( cnt > 0u ) {
          let old = ${readSlot}( i32( i ) );
          let sum = vec3f( f32( r ), f32( g ), f32( b ) ) / ${FIXED_SCALE}.0;
          let total = old.w + f32( cnt );
          let mean = ( old.xyz * old.w + sum ) / total;
          let nNew = min( total, ${this.maxSamplesU} );
          ${resolved}[ i ] = vec2u( pack2x16float( mean.xy ), pack2x16float( vec2f( mean.z, nNew ) ) );
          ${meta}[ i ] = ${this.frameU};
        } else if ( ${this.frameU} - ${meta}[ i ] > ${this.staleU} ) {
          atomicStore( &${keys}[ i ], 0u );
          ${resolved}[ i ] = vec2u( 0u );
        }
      }
    `, [64, 1, 1]);

    // 3. Query: one ray per pixel.
    const out = storeNode(this.output);
    this.kernels.query = makeKernel(wgslTagFn/* wgsl */ `
      fn sharcQuery( globalId: vec3u ) -> void {
        let px = globalId.xy;
        if ( px.x >= ${W}u || px.y >= ${H}u ) {
          return;
        }
        var p: vec3f;
        var n: vec3f;
        if ( ! ${gb.surface}( px, vec2u( ${W}u, ${H}u ), &p, &n ) ) {
          textureStore( ${out}, px, vec4f( 0.0, 0.0, 0.0, 1.0 ) );
          return;
        }
        var state = ${fns.seed}( px, ${this.frameU}, 0x2545f491u );
        var dir = ${fns.cosineDir}( n, vec2f( ${fns.rand}( &state ), ${fns.rand}( &state ) ) );
        var rays = 1u;
        let h = ${fns.trace}( p + n * 0.004, dir, 0.0 );
        var L = vec3f( 0.0 );
        if ( h.hit == 0u ) {
          L = ${fns.sky}( dir );
        } else {
          let s = ${findSlot}( ${voxelKey}( h.position, h.normal ) );
          var found = false;
          if ( s >= 0 ) {
            let c = ${readSlot}( s );
            if ( c.w >= ${this.minSamplesU} ) {
              L = c.xyz;
              found = true;
            }
          }
          if ( ! found ) {
            L = h.albedo * ${fns.directSampled}( h.position, h.normal, h.geomNormal, &state, &rays ) * 0.3183098862;
          }
        }
        textureStore( ${out}, px, vec4f( L, 1.0 ) );
        ${fns.countRays}( px, rays );
      }
    `);

    this.kernels.clear = makeKernel(wgslTagFn/* wgsl */ `
      fn sharcClear( globalId: vec3u ) -> void {
        let i = globalId.x;
        if ( i < ${CAP}u ) {
          atomicStore( &${keys}[ i ], 0u );
          ${resolved}[ i ] = vec2u( 0u );
          ${meta}[ i ] = 0u;
          atomicStore( &${accum}[ i * 4u ], 0u );
          atomicStore( &${accum}[ i * 4u + 1u ], 0u );
          atomicStore( &${accum}[ i * 4u + 2u ], 0u );
          atomicStore( &${accum}[ i * 4u + 3u ], 0u );
        }
      }
    `, [64, 1, 1]);

    // Occupancy statistics (sampled occasionally): number of live keys.
    this.needsClear = true;
  }

  update(_dt: number, _info: FrameInfo): void {}

  reset(): void {
    this.needsClear = true;
    this.temporal.reset();
  }

  // The cache is world-space and survives camera cuts; only the screen-space history is dropped.
  onCameraCut(): void {
    this.temporal.reset();
  }

  run(_gbuffer: GBuffer): Texture {
    if (this.builtCapacity !== 1 << this.params.log2Capacity) this.build();
    const { renderer, timer, width, height, camera } = this.ctx;
    const p = this.params;
    this.rt.writeLights(this.ctx.lights);
    if (this.needsClear) {
      dispatch1D(renderer, this.kernels.clear, this.builtCapacity);
      this.needsClear = false;
    }
    this.frameU.value = this.frameCounter++;
    this.tileU.value = p.updateTile;
    this.bouncesU.value = Math.min(p.maxBounces, MAX_BOUNCES);
    // Voxel size ~ voxelPixels projected pixels at distance d: size = d * 2 tan(fov/2) / H * px.
    this.voxelScaleU.value = (2 * Math.tan((camera.fov * Math.PI) / 360) / height) * p.voxelPixels;
    this.maxSamplesU.value = p.maxSamples;
    this.staleU.value = p.staleFrames;
    this.minSamplesU.value = p.minSamples;

    const tiles = Math.ceil(width / p.updateTile) * Math.ceil(height / p.updateTile);
    timer.begin('sharc.update');
    dispatch1D(renderer, this.kernels.update, tiles);
    timer.end('sharc.update');
    timer.begin('sharc.resolve');
    dispatch1D(renderer, this.kernels.resolve, this.builtCapacity);
    timer.end('sharc.resolve');
    timer.begin('sharc.query');
    dispatch2D(renderer, this.kernels.query, width, height);
    timer.end('sharc.query');
    timer.begin('sharc.atrous');
    let out = this.atrous.run(this.output);
    timer.end('sharc.atrous');
    if (p.temporal) {
      timer.begin('sharc.temporal');
      out = this.temporal.run(out, camera);
      timer.end('sharc.temporal');
    } else {
      this.temporal.reset();
    }
    this.rt.sampleRayCounter(renderer);
    return out;
  }

  stats(): GIStats {
    const extraMem = this.params.temporal ? this.temporal.memoryBytes : 0;
    return {
      raysPerFrame: this.rt.rayCountLast,
      memoryBytes: this.ledger.total + extraMem,
      sharedBytes: this.rt.memoryBytes,
      memoryBreakdown: { ...this.ledger.breakdown(), ...(extraMem ? { 'filtro temporal': extraMem } : {}) },
      extra: {
        'Tabla hash': `2^${this.params.log2Capacity} entradas, update 1/${this.params.updateTile ** 2} px × ${this.params.maxBounces} rebotes`,
      },
    };
  }

  buildGui(folder: GUI): void {
    const p = this.params;
    folder.add(p, 'log2Capacity', 18, 22, 1).name('log2 entradas tabla');
    folder.add(p, 'updateTile', { '1/4 px (2×2)': 2, '1/16 px (4×4)': 4, '1/64 px (8×8)': 8 }).name('píxeles de update');
    folder.add(p, 'maxBounces', 1, MAX_BOUNCES, 1).name('rebotes update');
    folder.add(p, 'voxelPixels', 2, 64, 1).name('voxel (px proyectados)');
    folder.add(p, 'maxSamples', 1, 256, 1).name('muestras máx. (historia)');
    folder.add(p, 'staleFrames', 4, 512, 1).name('frames hasta desalojo');
    folder.add(p, 'minSamples', 0, 16, 1).name('muestras mín. para usar');
    folder.add(this.atrous.params, 'iterations', 0, 5, 1).name('à-trous iteraciones');
    folder.add(this.atrous.params, 'normalPower', 1, 256, 1).name('à-trous potencia normal');
    folder.add(this.atrous.params, 'planeSigma', 0.001, 0.5, 0.001).name('à-trous sigma plano');
    folder.add(p, 'temporal').name('acumulación temporal');
    folder.add({ reset: () => this.reset() }, 'reset').name('vaciar cache');
  }

  private disposeTable(): void {
    Object.values(this.kernels).forEach(disposeKernel);
    this.kernels = {};
    for (const n of [this.keys, this.accum, this.resolved, this.meta]) if (n) disposeStorage(this.ctx.renderer, n);
  }

  dispose(): void {
    this.disposeTable();
    this.output.dispose();
    this.atrous.dispose();
    this.temporal.dispose();
  }
}
