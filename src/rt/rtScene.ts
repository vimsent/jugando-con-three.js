import { BufferAttribute, BufferGeometry, Mesh, Vector3 } from 'three';
import { StorageBufferAttribute, StructTypeNode, type WebGPURenderer } from 'three/webgpu';
import { storage, instancedArray } from 'three/tsl';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import {
  BVHComputeData, wgslTagFn, rayStruct, rayIntersectionResultStruct, bvhNodeBoundsStruct, intersectRayTriangle,
} from 'three-mesh-bvh/webgpu';
import type { MeshLambertNodeMaterial } from 'three/webgpu';
import type { SponzaScene } from '../scene/sponza';
import type { SceneLights } from '../scene/lights';

/* eslint-disable @typescript-eslint/no-explicit-any */
type FnNode = any;

// Number of vec4 slots in the lights buffer (layout documented in `writeLights`).
const LIGHT_SLOTS = 8;
// Atomic slots for ray counting (spreads contention).
const RAY_COUNTER_SLOTS = 256;

/** Result of `gi_trace`: closest hit with interpolated shading normal and per-material albedo. */
export const hitStruct = new StructTypeNode(
  {
    position: 'vec3f',
    hit: 'uint',
    normal: 'vec3f', // shading normal, facing the incoming ray
    t: 'float',
    geomNormal: 'vec3f', // geometric normal, facing the incoming ray
    triangle: 'uint',
    albedo: 'vec3f',
    _pad: 'float',
  },
  'GIHit',
);

/**
 * Static ray tracing scene shared by every ray-traced method: Sponza merged into one world-space
 * mesh, a SAH BVH built with three-mesh-bvh, packed for WGSL with `BVHComputeData`, and a small
 * WGSL library on top of its traversal (`raycastFirstHit`) plus an any-hit variant for shadow rays.
 *
 * Simplifications (documented in the README):
 * - secondary hits use the material's *average* albedo (stored per vertex), not the texture;
 * - alpha-tested geometry (plants) is opaque for rays;
 * - normal maps are ignored at secondary hits (interpolated vertex normals are used).
 */
export class RTScene {
  readonly bvhData: BVHComputeData;
  readonly triangleCount: number;
  readonly vertexCount: number;
  readonly buildMs: number;
  private readonly lightsArray = new Float32Array(LIGHT_SLOTS * 4);
  private readonly lightsAttr = new StorageBufferAttribute(this.lightsArray, 4);
  readonly lightsNode = storage(this.lightsAttr, 'vec4', LIGHT_SLOTS).toReadOnly().setName('gi_lights');
  readonly rayCounter = instancedArray(RAY_COUNTER_SLOTS, 'uint').toAtomic().setName('gi_rayCounter');
  private rayCounterBusy = false;
  rayCountLast = 0; // rays counted during the last sampled frame

  // WGSL library (function nodes usable inside wgslTagFn templates).
  readonly fns: {
    pcg: FnNode;
    rand: FnNode;
    seed: FnNode;
    cosineDir: FnNode;
    sky: FnNode;
    trace: FnNode;
    occluded: FnNode;
    directIrradiance: FnNode;
    directSampled: FnNode;
    pathTrace: FnNode;
    countRays: FnNode;
  };

  constructor(sponza: SponzaScene) {
    const t0 = performance.now();
    const geometry = mergeWorldGeometry(sponza);
    this.triangleCount = geometry.index!.count / 3;
    this.vertexCount = geometry.attributes.position.count;
    geometry.boundsTree = new MeshBVH(geometry, { strategy: SAH, targetLeafSize: 8 } as ConstructorParameters<typeof MeshBVH>[1]);
    const mesh = new Mesh(geometry);
    mesh.updateMatrixWorld(true);
    this.bvhData = new BVHComputeData(mesh, {
      attributes: { position: 'vec4f', normal: 'vec4f', color: 'vec4f' },
      autogenerateBvh: false,
    });
    this.bvhData.update();
    this.buildMs = performance.now() - t0;
    this.fns = this.buildLibrary();
  }

  /** Bytes of BVH nodes + index + interleaved attributes + lights buffer. */
  get memoryBytes(): number {
    const s = this.bvhData.storage as any;
    let total = 0;
    for (const k of ['nodes', 'index', 'attributes', 'transforms']) {
      const arr = s[k]?.proxyNode?.value?.array ?? s[k]?.value?.array;
      if (arr) total += arr.byteLength;
    }
    return total + this.lightsArray.byteLength + RAY_COUNTER_SLOTS * 4;
  }

  /**
   * Lights buffer layout (vec4 slots):
   *   0 sun direction (towards the sun), w = 1 if enabled
   *   1 sun irradiance color (color * intensity)
   *   2 sky zenith radiance (already scaled)
   *   3 sky horizon radiance (already scaled)
   *   4,6 point light position, w = casts shadow
   *   5,7 point light intensity color (candela), w unused
   */
  writeLights(lights: SceneLights): void {
    const a = this.lightsArray;
    const d = lights.sunDirection;
    const sc = lights.sunColor;
    a.set([d.x, d.y, d.z, lights.sun.intensity > 0 ? 1 : 0], 0);
    a.set([sc.r, sc.g, sc.b, 0], 4);
    const s = lights.skyScale;
    a.set([lights.skyZenith.r * s, lights.skyZenith.g * s, lights.skyZenith.b * s, 0], 8);
    a.set([lights.skyHorizon.r * s, lights.skyHorizon.g * s, lights.skyHorizon.b * s, 0], 12);
    lights.points.forEach((p, i) => {
      a.set([p.position.x, p.position.y, p.position.z, p.castShadow ? 1 : 0], 16 + i * 8);
      a.set([p.color.r * p.intensity, p.color.g * p.intensity, p.color.b * p.intensity, 0], 20 + i * 8);
    });
    this.lightsAttr.needsUpdate = true;
  }

  /**
   * Reads back the ray counter (asynchronously) and clears it. Call after the frame's dispatches.
   * Counting uses one atomicAdd per thread into one of 256 slots; its cost is included in the
   * timings of the passes that count.
   */
  sampleRayCounter(renderer: WebGPURenderer): void {
    if (this.rayCounterBusy) return;
    this.rayCounterBusy = true;
    const attr = (this.rayCounter as any).value as StorageBufferAttribute;
    renderer
      .getArrayBufferAsync(attr)
      .then((buf: ArrayBuffer) => {
        const u = new Uint32Array(buf);
        let s = 0;
        for (let i = 0; i < u.length; i++) s += u[i];
        this.rayCountLast = s;
        (attr.array as Uint32Array).fill(0);
        attr.needsUpdate = true;
      })
      .finally(() => (this.rayCounterBusy = false));
  }

  private buildLibrary() {
    const { storage: bvhStorage, fns: bvhFns } = this.bvhData as any;
    const attributes = bvhStorage.attributes;
    const L = this.lightsNode;
    const counter = this.rayCounter;

    const pcg = wgslTagFn/* wgsl */ `
      fn gi_pcg( v: u32 ) -> u32 {
        let state = v * 747796405u + 2891336453u;
        let word = ( ( state >> ( ( state >> 28u ) + 4u ) ) ^ state ) * 277803737u;
        return ( word >> 22u ) ^ word;
      }
    `;

    const seed = wgslTagFn/* wgsl */ `
      fn gi_seed( px: vec2u, frame: u32, salt: u32 ) -> u32 {
        return ${pcg}( px.x + ${pcg}( px.y + ${pcg}( frame + ${pcg}( salt ) ) ) );
      }
    `;

    const rand = wgslTagFn/* wgsl */ `
      fn gi_rand( state: ptr<function, u32> ) -> f32 {
        *state = ${pcg}( *state );
        return f32( *state >> 8u ) * ( 1.0 / 16777216.0 );
      }
    `;

    const cosineDir = wgslTagFn/* wgsl */ `
      fn gi_cosineDir( n: vec3f, u: vec2f ) -> vec3f {
        let s = select( -1.0, 1.0, n.z >= 0.0 );
        let a = -1.0 / ( s + n.z );
        let b = n.x * n.y * a;
        let t1 = vec3f( 1.0 + s * n.x * n.x * a, s * b, -s * n.x );
        let t2 = vec3f( b, s + n.y * n.y * a, -n.y );
        let r = sqrt( u.x );
        let phi = 6.283185307 * u.y;
        return normalize( t1 * ( r * cos( phi ) ) + t2 * ( r * sin( phi ) ) + n * sqrt( max( 0.0, 1.0 - u.x ) ) );
      }
    `;

    const sky = wgslTagFn/* wgsl */ `
      fn gi_sky( dir: vec3f ) -> vec3f {
        let t = sqrt( max( dir.y, 0.0 ) );
        return mix( ${L}[ 3 ].xyz, ${L}[ 2 ].xyz, t );
      }
    `;

    const countRays = wgslTagFn/* wgsl */ `
      fn gi_countRays( px: vec2u, n: u32 ) -> void {
        if ( n > 0u ) {
          atomicAdd( &${counter}[ ( px.x * 7u + px.y * 13u ) & ${RAY_COUNTER_SLOTS - 1}u ], n );
        }
      }
    `;

    const trace = wgslTagFn/* wgsl */ `
      fn gi_trace( origin: vec3f, dir: vec3f, maxDist: f32 ) -> ${hitStruct} {
        var ray: ${rayStruct};
        ray.origin = origin;
        ray.direction = dir;
        ray.maxDist = maxDist;
        var res: ${rayIntersectionResultStruct};
        var h: ${hitStruct};
        h.hit = 0u;
        if ( ! ${bvhFns.raycastFirstHit}( ray, &res ) ) {
          return h;
        }
        let a = ${attributes}[ res.indices.x ];
        let b = ${attributes}[ res.indices.y ];
        let c = ${attributes}[ res.indices.z ];
        let bc = res.barycoord;
        var ng = normalize( res.normal );
        if ( dot( ng, dir ) > 0.0 ) {
          ng = -ng;
        }
        var n = a.normal.xyz * bc.x + b.normal.xyz * bc.y + c.normal.xyz * bc.z;
        n = normalize( n );
        if ( dot( n, ng ) < 0.0 ) {
          n = -n;
        }
        h.hit = 1u;
        h.t = res.dist;
        h.position = origin + dir * res.dist;
        h.normal = n;
        h.geomNormal = ng;
        h.triangle = res.indices.w;
        h.albedo = a.color.xyz;
        return h;
      }
    `;

    // Any-hit traversal for shadow rays: once a hit is found every remaining node is rejected,
    // so the stack drains without testing more triangles.
    const anyHit = this.bvhData.getShapecastFn({
      name: 'gi_anyHitCast',
      shapeStruct: rayStruct,
      resultStruct: rayIntersectionResultStruct,
      intersectsBoundsFn: wgslTagFn/* wgsl */ `
        fn gi_anyHitBounds( ray: ${rayStruct}, bounds: ${bvhNodeBoundsStruct}, result: ptr<function, ${rayIntersectionResultStruct}> ) -> u32 {
          if ( result.didHit ) {
            return 0u;
          }
          let bmin = vec3f( bounds.min[ 0 ], bounds.min[ 1 ], bounds.min[ 2 ] );
          let bmax = vec3f( bounds.max[ 0 ], bounds.max[ 1 ], bounds.max[ 2 ] );
          let inv = 1.0 / ray.direction;
          let t0s = ( bmin - ray.origin ) * inv;
          let t1s = ( bmax - ray.origin ) * inv;
          let tmin = min( t0s, t1s );
          let tmax = max( t0s, t1s );
          let t0 = max( max( tmin.x, tmin.y ), max( tmin.z, 0.0 ) );
          let t1 = min( min( tmax.x, tmax.y ), tmax.z );
          if ( t1 < t0 || ( ray.maxDist > 0.0 && t0 >= ray.maxDist ) ) {
            return 0u;
          }
          return 1u;
        }
      `,
      intersectRangeFn: wgslTagFn/* wgsl */ `
        fn gi_anyHitRange( ray: ${rayStruct}, offset: u32, count: u32, result: ptr<function, ${rayIntersectionResultStruct}> ) -> bool {
          for ( var ti = offset; ti < offset + count; ti = ti + 1u ) {
            let i0 = ${bvhStorage.index}[ ti * 3u ];
            let i1 = ${bvhStorage.index}[ ti * 3u + 1u ];
            let i2 = ${bvhStorage.index}[ ti * 3u + 2u ];
            let tri = ${intersectRayTriangle}( ray, ${attributes}[ i0 ].position.xyz, ${attributes}[ i1 ].position.xyz, ${attributes}[ i2 ].position.xyz, 0.0 );
            if ( tri.didHit && ( ray.maxDist <= 0.0 || tri.dist < ray.maxDist ) ) {
              result.didHit = true;
              result.dist = tri.dist;
              return true;
            }
          }
          return false;
        }
      `,
    });

    const occluded = wgslTagFn/* wgsl */ `
      fn gi_occluded( origin: vec3f, dir: vec3f, maxDist: f32 ) -> bool {
        var ray: ${rayStruct};
        ray.origin = origin;
        ray.direction = dir;
        ray.maxDist = maxDist;
        var res: ${rayIntersectionResultStruct};
        res.didHit = false;
        return ${anyHit}( ray, &res );
      }
    `;

    // Irradiance from the sun and the point lights at p (normal n), with ray-traced visibility.
    // Matches three.js: E_sun = I * max(n.l, 0); E_point = I_cd * max(n.l, 0) / max(d^2, 0.01).
    const directIrradiance = wgslTagFn/* wgsl */ `
      fn gi_directIrradiance( p: vec3f, n: vec3f, ng: vec3f, rays: ptr<function, u32> ) -> vec3f {
        var E = vec3f( 0.0 );
        let o = p + ng * 0.002;
        let sunDir = ${L}[ 0 ].xyz;
        let ndl = dot( n, sunDir );
        if ( ${L}[ 0 ].w > 0.0 && ndl > 0.0 && dot( ng, sunDir ) > -0.05 ) {
          *rays += 1u;
          if ( ! ${occluded}( o, sunDir, 0.0 ) ) {
            E += ${L}[ 1 ].xyz * ndl;
          }
        }
        for ( var i = 0u; i < 2u; i = i + 1u ) {
          let lp = ${L}[ 4u + i * 2u ];
          let d = lp.xyz - p;
          let dist2 = dot( d, d );
          let dist = sqrt( dist2 );
          let l = d / dist;
          let pdl = dot( n, l );
          if ( pdl > 0.0 ) {
            var visible = true;
            if ( lp.w > 0.0 ) {
              *rays += 1u;
              visible = ! ${occluded}( o, l, dist - 0.01 );
            }
            if ( visible ) {
              E += ${L}[ 5u + i * 2u ].xyz * pdl / max( dist2, 0.01 );
            }
          }
        }
        return E;
      }
    `;

    // Same expectation as gi_directIrradiance but with a single shadow ray: one light is chosen
    // with probability proportional to its unshadowed irradiance and its contribution divided by
    // that probability (unbiased; used by the path tracers).
    const directSampled = wgslTagFn/* wgsl */ `
      fn gi_directSampled( p: vec3f, n: vec3f, ng: vec3f, state: ptr<function, u32>, rays: ptr<function, u32> ) -> vec3f {
        var E: array<vec3f, 3>;
        var dirs: array<vec4f, 3>;
        var w: array<f32, 3>;
        let lw = vec3f( 0.2126, 0.7152, 0.0722 );
        let sunDir = ${L}[ 0 ].xyz;
        let ndl = dot( n, sunDir );
        E[ 0 ] = vec3f( 0.0 );
        if ( ${L}[ 0 ].w > 0.0 && ndl > 0.0 && dot( ng, sunDir ) > -0.05 ) {
          E[ 0 ] = ${L}[ 1 ].xyz * ndl;
        }
        dirs[ 0 ] = vec4f( sunDir, 0.0 );
        for ( var i = 0u; i < 2u; i = i + 1u ) {
          let lp = ${L}[ 4u + i * 2u ];
          let d = lp.xyz - p;
          let dist2 = dot( d, d );
          let dist = sqrt( dist2 );
          let l = d / dist;
          let pdl = max( dot( n, l ), 0.0 );
          E[ i + 1u ] = ${L}[ 5u + i * 2u ].xyz * pdl / max( dist2, 0.01 );
          dirs[ i + 1u ] = vec4f( l, select( -1.0, dist - 0.01, lp.w > 0.0 ) );
        }
        var total = 0.0;
        for ( var i = 0u; i < 3u; i = i + 1u ) {
          w[ i ] = dot( E[ i ], lw );
          total += w[ i ];
        }
        if ( total <= 0.0 ) {
          return vec3f( 0.0 );
        }
        let u = ${rand}( state ) * total;
        var k = 2u;
        if ( u < w[ 0 ] ) {
          k = 0u;
        } else if ( u < w[ 0 ] + w[ 1 ] ) {
          k = 1u;
        }
        let pk = w[ k ] / total;
        let dk = dirs[ k ];
        if ( dk.w >= 0.0 ) {
          *rays += 1u;
          if ( ${occluded}( p + ng * 0.002, dk.xyz, dk.w ) ) {
            return vec3f( 0.0 );
          }
        }
        return E[ k ] / pk;
      }
    `;

    // Unidirectional path tracer for indirect diffuse light leaving from (p, n). Returns the
    // cosine-weighted mean incoming indirect radiance (= E_ind / pi): cosine sampling, next event
    // estimation of the sun / point lights at every vertex, sky on escape, Russian roulette.
    const pathTrace = wgslTagFn/* wgsl */ `
      fn gi_pathTrace( p: vec3f, n: vec3f, ng: vec3f, state: ptr<function, u32>, maxBounces: u32, rays: ptr<function, u32> ) -> vec3f {
        var L = vec3f( 0.0 );
        var T = vec3f( 1.0 );
        var o = p;
        var nn = n;
        var g = ng;
        for ( var b = 0u; b < maxBounces; b = b + 1u ) {
          var dir = ${cosineDir}( nn, vec2f( ${rand}( state ), ${rand}( state ) ) );
          if ( dot( dir, g ) <= 0.0 ) {
            dir = reflect( dir, g );
          }
          *rays += 1u;
          let h = ${trace}( o + g * 0.002, dir, 0.0 );
          if ( h.hit == 0u ) {
            L += T * ${sky}( dir );
            break;
          }
          T *= h.albedo;
          L += T * ${directSampled}( h.position, h.normal, h.geomNormal, state, rays ) * 0.3183098862;
          if ( b >= 2u ) {
            let q = clamp( max( T.x, max( T.y, T.z ) ), 0.05, 0.95 );
            if ( ${rand}( state ) > q ) {
              break;
            }
            T /= q;
          }
          o = h.position;
          nn = h.normal;
          g = h.geomNormal;
        }
        return L;
      }
    `;

    return { pcg, rand, seed, cosineDir, sky, trace, occluded, directIrradiance, directSampled, pathTrace, countRays };
  }
}

/** Bakes every Sponza mesh into a single world-space, indexed geometry with a per-vertex albedo. */
function mergeWorldGeometry(sponza: SponzaScene): BufferGeometry {
  let vtx = 0;
  let idx = 0;
  for (const m of sponza.meshes) {
    vtx += m.geometry.attributes.position.count;
    idx += m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position.count;
  }
  const pos = new Float32Array(vtx * 3);
  const nrm = new Float32Array(vtx * 3);
  const col = new Float32Array(vtx * 3);
  const index = new Uint32Array(idx);
  const v = new Vector3();
  let vo = 0;
  let io = 0;
  for (const m of sponza.meshes) {
    const g = m.geometry;
    const P = g.attributes.position;
    const N = g.attributes.normal;
    const albedo = sponza.averageAlbedo.get(m.material as MeshLambertNodeMaterial)!;
    for (let i = 0; i < P.count; i++) {
      v.fromBufferAttribute(P, i).applyMatrix4(m.matrixWorld);
      pos.set([v.x, v.y, v.z], (vo + i) * 3);
      v.fromBufferAttribute(N, i).transformDirection(m.matrixWorld);
      nrm.set([v.x, v.y, v.z], (vo + i) * 3);
      col.set([albedo.r, albedo.g, albedo.b], (vo + i) * 3);
    }
    if (g.index) {
      for (let i = 0; i < g.index.count; i++) index[io + i] = g.index.getX(i) + vo;
      io += g.index.count;
    } else {
      for (let i = 0; i < P.count; i++) index[io + i] = vo + i;
      io += P.count;
    }
    vo += P.count;
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(pos, 3));
  geometry.setAttribute('normal', new BufferAttribute(nrm, 3));
  geometry.setAttribute('color', new BufferAttribute(col, 3));
  geometry.setIndex(new BufferAttribute(index, 1));
  return geometry;
}
