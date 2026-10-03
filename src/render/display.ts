import { DataTexture, HalfFloatType, RGBAFormat, Texture } from 'three';

function blank(): DataTexture {
  const t = new DataTexture(new Uint16Array(4), 1, 1, RGBAFormat, HalfFloatType);
  t.needsUpdate = true;
  return t;
}
import { NodeMaterial, QuadMesh, type Node, type WebGPURenderer } from 'three/webgpu';
import {
  Fn, abs, acesFilmicToneMapping, clamp, dot, float, mix, select, smoothstep, texture, uniform, uv, vec3, vec4,
} from 'three/tsl';
import type { GBuffer } from './gbuffer';

export const VIEW_MODES = ['final', 'indirecta', 'directa', 'albedo', 'normal', 'diferencia'] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

export interface DisplayInputs {
  finalA: Texture;
  indirectA: Texture;
  finalB: Texture | null; // right side of the split view (method B or the reference image)
  indirectB: Texture | null;
  reference: Texture | null; // reference final image for the difference view
}

/**
 * Final on-screen pass: view-mode selection, side-by-side split (A | B), difference heat map
 * against the reference, exposure and ACES tonemapping. Everything upstream is linear HDR.
 */
export class Display {
  readonly mode = uniform(0, 'int');
  readonly split = uniform(1.0); // x position of the split, 1 = A only
  readonly exposure = uniform(1.0);
  readonly diffGain = uniform(4.0);
  private readonly quad: QuadMesh;
  private readonly material: NodeMaterial;
  private readonly nodes = {
    finalA: texture(new Texture()),
    indirectA: texture(new Texture()),
    finalB: texture(new Texture()),
    indirectB: texture(new Texture()),
    reference: texture(new Texture()),
  };
  private readonly bound = new Map<string, Texture>();
  // One distinct placeholder per optional input: three.js shares a binding between texture nodes
  // that point at the same texture, so two inputs must never alias the same texture.
  private readonly placeholders = {
    finalB: blank(),
    indirectB: blank(),
    reference: blank(),
  };
  private readonly hasRef = uniform(0);

  constructor(gbuffer: GBuffer) {
    const n = this.nodes;
    const material = (this.material = new NodeMaterial());
    material.fragmentNode = Fn(() => {
      const st = uv();
      const right = st.x.greaterThan(this.split);
      const finalA = n.finalA.sample(st).rgb;
      const finalB = n.finalB.sample(st).rgb;
      const indA = n.indirectA.sample(st).rgb;
      const indB = n.indirectB.sample(st).rgb;
      const ref = n.reference.sample(st).rgb;

      const final = select(right, finalB, finalA);
      const indirect = select(right, indB, indA);
      const direct = texture(gbuffer.direct, st).rgb;
      const albedo = texture(gbuffer.albedo, st).rgb;
      const normal = texture(gbuffer.normal, st).rgb.mul(0.5).add(0.5);

      // Absolute linear difference against the reference (luminance), mapped to a heat ramp.
      const lum = vec3(0.2126, 0.7152, 0.0722);
      const err = clamp(dot(abs(final.sub(ref)), lum).mul(this.diffGain), 0, 1);
      const heat = mix(
        mix(vec3(0, 0, 0), vec3(0.55, 0.05, 0.6), smoothstep(0, 0.33, err)),
        mix(vec3(0.95, 0.35, 0.05), vec3(1, 1, 0.6), smoothstep(0.66, 1, err)),
        smoothstep(0.25, 0.7, err),
      );

      const m = this.mode;
      const hdr = select(m.equal(1), indirect, select(m.equal(2), direct, final));
      const toned = acesFilmicToneMapping(hdr, this.exposure);
      const ldr = select(m.equal(3), albedo, select(m.equal(4), normal, select(m.equal(5), select(this.hasRef.greaterThan(0), heat, vec3(0.2, 0, 0.2)), toned)));
      // Thin split line.
      const line = abs(st.x.sub(this.split)).lessThan(float(0.0009)).and(this.split.lessThan(1).and(this.split.greaterThan(0)) as never) as unknown as Node<'bool'>;
      return vec4(select(line, vec3(1), ldr) as Node<'vec3'>, 1);
    })();
    this.quad = new QuadMesh(material);
  }

  setModeByName(name: ViewMode): void {
    this.mode.value = VIEW_MODES.indexOf(name);
  }

  render(renderer: WebGPURenderer, inputs: DisplayInputs): void {
    const dirty = [
      this.bind('finalA', inputs.finalA),
      this.bind('indirectA', inputs.indirectA),
      this.bind('finalB', inputs.finalB ?? this.placeholders.finalB),
      this.bind('indirectB', inputs.indirectB ?? this.placeholders.indirectB),
      this.bind('reference', inputs.reference ?? this.placeholders.reference),
    ].some(Boolean);
    this.hasRef.value = inputs.reference ? 1 : 0;
    if (dirty) this.material.needsUpdate = true;
    renderer.setRenderTarget(null);
    this.quad.render(renderer);
  }

  private bind(key: keyof Display['nodes'], tex: Texture): boolean {
    if (this.bound.get(key) === tex) return false;
    this.bound.set(key, tex);
    this.nodes[key].value = tex;
    return true;
  }
}
