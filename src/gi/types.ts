import type { PerspectiveCamera, Scene, Texture } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import type GUI from 'lil-gui';
import type { GBuffer } from '../render/gbuffer';
import type { SceneLights } from '../scene/lights';
import type { GpuTimer } from '../metrics/gpuTimer';
import type { RTScene } from '../rt/rtScene';

/** Everything a GI method may need; shared, owned by the app. */
export interface GIContext {
  renderer: WebGPURenderer;
  scene: Scene;
  camera: PerspectiveCamera;
  lights: SceneLights;
  width: number;
  height: number;
  timer: GpuTimer;
  /** Lazily built BVH + shared WGSL tracing library (only ray-traced methods ask for it). */
  getRTScene(): Promise<RTScene>;
}

export interface GIStats {
  raysPerFrame: number;
  memoryBytes: number;
  memoryBreakdown?: Record<string, number>;
  extra?: Record<string, string | number>;
}

/** Per-frame information passed to `update`. */
export interface FrameInfo {
  frame: number;
  time: number;
  cameraMoved: boolean;
  lightsChanged: boolean;
}

/**
 * A GI method turns the G-buffer into a buffer of indirect diffuse lighting.
 *
 * Convention: the returned texture holds E_ind / π (linear, pre-tonemap), i.e. the cosine-weighted
 * mean incoming indirect radiance. The composite pass computes
 *     final = direct + albedo * indirect
 * which is the outgoing radiance of a Lambertian surface (albedo/π · E_ind).
 */
export interface GIMethod {
  readonly key: string;
  readonly label: string; // Spanish, shown in the HUD
  init(ctx: GIContext): Promise<void>;
  update(dt: number, info: FrameInfo): void;
  /** Records the method's passes (wrapped in ctx.timer.begin/end) and returns the indirect texture. */
  run(gbuffer: GBuffer): Texture;
  stats(): GIStats;
  /** Adds the method's parameters to a lil-gui folder. */
  buildGui?(folder: GUI): void;
  /** Drops temporal history (bookmark jump, method switch, explicit reset). */
  reset?(): void;
  dispose(): void;
}

export type GIMethodFactory = () => GIMethod;
