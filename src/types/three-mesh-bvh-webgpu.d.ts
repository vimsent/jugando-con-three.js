// Declarations for the parts of `three-mesh-bvh/webgpu` that ship without types (the module marks
// them as "temporary exports"). Function nodes are callable TSL nodes; they are typed loosely.
import 'three-mesh-bvh/webgpu';

declare module 'three-mesh-bvh/webgpu' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  type WgslNode = any;
  export function wgslTagFn(strings: TemplateStringsArray, ...args: unknown[]): WgslNode;
  export function wgslTagCode(strings: TemplateStringsArray, ...args: unknown[]): WgslNode;
  export const rayStruct: WgslNode;
  export const rayIntersectionResultStruct: WgslNode;
  export const bvhNodeBoundsStruct: WgslNode;
  export const intersectRayTriangle: WgslNode;
}
