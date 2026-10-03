import { HalfFloatType, LinearFilter, Matrix4, RGBAFormat, type PerspectiveCamera } from 'three';
import { StorageTexture, type WebGPURenderer } from 'three/webgpu';
import { globalId, texture, textureStore, uniform } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import type { GBuffer } from '../render/gbuffer';

/* eslint-disable @typescript-eslint/no-explicit-any */
type FnNode = any;

/** rgba16float storage texture usable as a GI output and sampled by the composite pass. */
export function createOutputTexture(width: number, height: number): StorageTexture {
  const t = new StorageTexture(width, height);
  t.type = HalfFloatType;
  t.format = RGBAFormat;
  t.minFilter = LinearFilter;
  t.magFilter = LinearFilter;
  t.generateMipmaps = false;
  return t;
}

/**
 * Shared G-buffer access for compute kernels. `surface` reconstructs the world-space position from
 * hardware depth (WebGPU NDC z in [0, 1]) and the world-space normal from the view-space normal.
 */
export class GBufferAccess {
  readonly depth: FnNode;
  readonly normal: FnNode;
  readonly albedo: FnNode;
  readonly direct: FnNode;
  readonly projInv: FnNode;
  readonly camWorld: FnNode;
  readonly viewProj: FnNode; // projection * view, for reprojection into the current frame
  readonly camPos: FnNode;
  readonly surface: FnNode;
  private readonly viewProjMatrix = new Matrix4();

  constructor(gbuffer: GBuffer, camera: PerspectiveCamera) {
    this.depth = texture(gbuffer.depth);
    this.normal = texture(gbuffer.normal);
    this.albedo = texture(gbuffer.albedo);
    this.direct = texture(gbuffer.direct);
    this.projInv = uniform(camera.projectionMatrixInverse);
    this.camWorld = uniform(camera.matrixWorld);
    this.camPos = uniform(camera.position);
    this.viewProj = uniform(this.viewProjMatrix);
    this.surface = wgslTagFn/* wgsl */ `
      fn gb_surface( px: vec2u, dims: vec2u, outPos: ptr<function, vec3f>, outNormal: ptr<function, vec3f> ) -> bool {
        let depth = textureLoad( ${this.depth}, px, 0 );
        if ( depth >= 1.0 ) {
          return false;
        }
        let uv = ( vec2f( px ) + 0.5 ) / vec2f( dims );
        var vp = ${this.projInv} * vec4f( uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0 );
        vp = vp / vp.w;
        *outPos = ( ${this.camWorld} * vec4f( vp.xyz, 1.0 ) ).xyz;
        let nv = textureLoad( ${this.normal}, px, 0 ).xyz;
        *outNormal = normalize( ( ${this.camWorld} * vec4f( nv, 0.0 ) ).xyz );
        return true;
      }
    `;
  }

  /** Keeps the cached view-projection matrix in sync (call once per frame). */
  update(camera: PerspectiveCamera): void {
    this.viewProjMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  }
}

/** Builds a compute node from a `fn name( globalId: vec3u ) -> void` WGSL tag function. */
export function makeKernel(fn: FnNode, workgroupSize: [number, number, number] = [8, 8, 1]): FnNode {
  return fn({ globalId }).computeKernel(workgroupSize);
}

export function dispatch2D(renderer: WebGPURenderer, kernel: FnNode, w: number, h: number, wg = [8, 8]): void {
  renderer.compute(kernel, [Math.ceil(w / wg[0]), Math.ceil(h / wg[1]), 1]);
}

export function dispatch1D(renderer: WebGPURenderer, kernel: FnNode, n: number, wg = 64): void {
  renderer.compute(kernel, [Math.ceil(n / wg), 1, 1]);
}

/** Storage-texture write node for a texture created with `createOutputTexture`. */
export function storeNode(tex: StorageTexture): FnNode {
  return textureStore(tex as any).toWriteOnly();
}

/** Frees the GPU buffer behind a storage node (three.js r186 has no public API for this). */
export function disposeStorage(renderer: WebGPURenderer, node: FnNode): void {
  const attr = node?.value;
  if (!attr) return;
  (renderer as any)._attributes?.delete(attr);
}

/** Frees a compute pipeline and its bindings. */
export function disposeKernel(kernel: FnNode): void {
  kernel?.dispose?.();
}
