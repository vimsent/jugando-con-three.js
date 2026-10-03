import { Color, DirectionalLight, MathUtils, Object3D, PointLight, Scene, Vector3 } from 'three';

export interface LightParams {
  hour: number; // 6..18
  animateSun: boolean;
  sunSpeed: number; // hours per second
  sunIntensity: number; // illuminance-like units (three.js directional light intensity)
  skyIntensity: number; // radiance multiplier of the analytic sky
  pointIntensity: number; // candela
  animatePoint: boolean;
  pointSpeed: number;
  pointShadows: boolean;
}

// Point light colors; light 0 is static (torch in the gallery), light 1 moves along the atrium.
const POINT_COLORS = [new Color(1.0, 0.62, 0.32), new Color(0.45, 0.7, 1.0)];
const POINT_STATIC_POS = new Vector3(-6, 2.0, 5.4);

/**
 * Direct lighting shared by every GI method: one shadowed sun and two shadowed point lights, plus
 * an analytic sky that is only seen by indirect rays (it is part of the GI signal, not of the
 * direct term). `version` increments whenever anything that changes the lighting moves, which lets
 * methods with temporal caches and the convergence tracker react.
 */
export class SceneLights {
  readonly sun: DirectionalLight;
  readonly points: PointLight[];
  readonly params: LightParams = {
    hour: 13.5,
    animateSun: false,
    sunSpeed: 0.5,
    sunIntensity: 7,
    skyIntensity: 1.0,
    pointIntensity: 12,
    animatePoint: false,
    pointSpeed: 0.35,
    pointShadows: true,
  };
  readonly skyZenith = new Color(0.32, 0.5, 0.95);
  readonly skyHorizon = new Color(0.75, 0.82, 0.95);
  readonly sunDirection = new Vector3(); // unit vector pointing towards the sun
  readonly sunColor = new Color(); // color * intensity (linear)
  version = 0;
  private pointPhase = 0;

  constructor(scene: Scene) {
    this.sun = new DirectionalLight(0xffffff, 1);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(4096, 4096);
    const cam = this.sun.shadow.camera;
    cam.left = -17;
    cam.right = 17;
    cam.top = 17;
    cam.bottom = -17;
    cam.near = 1;
    cam.far = 80;
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    const target = new Object3D();
    target.position.set(0, 4, 0);
    scene.add(target);
    this.sun.target = target;
    scene.add(this.sun);

    this.points = POINT_COLORS.map((color) => {
      const l = new PointLight(color, 1, 0, 2);
      l.castShadow = true;
      l.shadow.mapSize.set(512, 512);
      l.shadow.bias = -0.002;
      l.shadow.camera.near = 0.05;
      l.shadow.camera.far = 40;
      scene.add(l);
      return l;
    });
    this.points[0].position.copy(POINT_STATIC_POS);
    this.apply();
  }

  /** Advances animations; returns true when the lighting changed this frame. */
  update(dt: number): boolean {
    const p = this.params;
    let changed = false;
    if (p.animateSun) {
      p.hour += dt * p.sunSpeed;
      if (p.hour > 18) p.hour = 6 + (p.hour - 18);
      changed = true;
    }
    if (p.animatePoint) {
      this.pointPhase += dt * p.pointSpeed;
      changed = true;
    }
    if (changed) this.apply();
    return changed;
  }

  /** Recomputes light objects from the parameters and bumps the version. */
  apply(): void {
    const p = this.params;
    const t = MathUtils.clamp((p.hour - 6) / 12, 0, 1);
    const elevation = Math.sin(Math.PI * t) * MathUtils.degToRad(76) + MathUtils.degToRad(4);
    const azimuth = MathUtils.degToRad(MathUtils.lerp(165, 15, t));
    this.sunDirection
      .set(Math.cos(azimuth) * Math.cos(elevation), Math.sin(elevation), Math.sin(azimuth) * Math.cos(elevation))
      .normalize();
    const day = MathUtils.smoothstep(Math.sin(elevation), 0.02, 0.2);
    const warm = MathUtils.smoothstep(Math.sin(elevation), 0.05, 0.6);
    this.sun.color.setRGB(1.0, 0.5, 0.25).lerp(new Color(1.0, 0.96, 0.9), warm);
    this.sun.intensity = p.sunIntensity * day;
    this.sunColor.copy(this.sun.color).multiplyScalar(this.sun.intensity);
    this.sun.position.copy(this.sun.target.position).addScaledVector(this.sunDirection, 40);
    this.sun.updateMatrixWorld();
    this.sun.target.updateMatrixWorld();

    // The moving point light sweeps the long axis of the atrium at head height.
    const ph = this.pointPhase;
    this.points[1].position.set(Math.sin(ph) * 11, 1.8 + 0.6 * Math.sin(ph * 2.3), 0.6 * Math.cos(ph * 1.7));
    for (const l of this.points) {
      l.intensity = p.pointIntensity;
      l.castShadow = p.pointShadows;
      l.updateMatrixWorld();
    }
    this.version++;
  }

  /** Sky dimming follows the sun so night-ish hours are not lit by a noon sky. */
  get skyScale(): number {
    return this.params.skyIntensity * (0.15 + 0.85 * MathUtils.smoothstep(this.sunDirection.y, -0.05, 0.4));
  }

  getState(): Record<string, unknown> {
    return { ...this.params, pointPhase: this.pointPhase };
  }

  setState(s: Partial<LightParams> & { pointPhase?: number }): void {
    const { pointPhase, ...rest } = s;
    Object.assign(this.params, rest);
    if (pointPhase !== undefined) this.pointPhase = pointPhase;
    this.apply();
  }
}
