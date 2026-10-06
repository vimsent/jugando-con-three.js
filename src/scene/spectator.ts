import { Euler, PerspectiveCamera, Vector3 } from 'three';

/**
 * Minecraft-style spectator camera: click the canvas to capture the mouse (pointer lock), mouse to
 * look, WASD to move horizontally, Space / Shift to go up / down, Ctrl or double-tap W to sprint,
 * mouse wheel to change the fly speed. Esc releases the mouse. Movement ignores pitch, so W never
 * makes you sink or climb, and there is no collision (fly through walls).
 */
export class SpectatorControls {
  speed = 4; // m/s
  sensitivity = 0.0022; // rad per pixel
  private readonly keys = new Set<string>();
  private readonly euler = new Euler(0, 0, 0, 'YXZ');
  private readonly move = new Vector3();
  private sprint = false;
  private lastWTap = 0;

  constructor(private readonly camera: PerspectiveCamera, private readonly dom: HTMLElement) {
    dom.addEventListener('click', () => {
      if (!this.locked) void dom.requestPointerLock();
    });
    document.addEventListener('pointerlockchange', () => {
      if (!this.locked) this.releaseKeys();
    });
    document.addEventListener('mousemove', (e) => {
      if (!this.locked) return;
      this.euler.setFromQuaternion(camera.quaternion);
      this.euler.y -= e.movementX * this.sensitivity;
      this.euler.x -= e.movementY * this.sensitivity;
      this.euler.x = Math.max(-Math.PI / 2 + 1e-3, Math.min(Math.PI / 2 - 1e-3, this.euler.x));
      camera.quaternion.setFromEuler(this.euler);
    });
    dom.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.speed = Math.max(0.25, Math.min(40, this.speed * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    }, { passive: false });
    window.addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      if (e.code === 'KeyW' && !e.repeat && !this.keys.has('KeyW')) {
        const t = performance.now();
        if (t - this.lastWTap < 300) this.sprint = true;
        this.lastWTap = t;
      }
      this.keys.add(e.code);
      if (this.locked && (e.code === 'Space' || e.code.startsWith('Control'))) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => {
      this.keys.delete(e.code);
      if (e.code === 'KeyW') this.sprint = false;
    });
    window.addEventListener('blur', () => this.releaseKeys());
  }

  get locked(): boolean {
    return document.pointerLockElement === this.dom;
  }

  private releaseKeys(): void {
    this.keys.clear();
    this.sprint = false;
  }

  private down(...codes: string[]): boolean {
    return codes.some((c) => this.keys.has(c));
  }

  /** Moves the camera; dt in seconds (wall-clock, so flying still works while the sim is paused). */
  update(dt: number): void {
    if (!this.locked || dt <= 0) return;
    const fwd = +this.down('KeyW') - +this.down('KeyS');
    const right = +this.down('KeyD') - +this.down('KeyA');
    const up = +this.down('Space') - +this.down('ShiftLeft', 'ShiftRight');
    if (!fwd && !right && !up) return;
    const yaw = this.euler.setFromQuaternion(this.camera.quaternion).y;
    // Horizontal basis from yaw only (forward is -Z in camera space).
    this.move.set(-Math.sin(yaw) * fwd + Math.cos(yaw) * right, 0, -Math.cos(yaw) * fwd - Math.sin(yaw) * right);
    if (this.move.lengthSq() > 0) this.move.normalize();
    this.move.y = up;
    const sprinting = this.sprint || this.down('ControlLeft', 'ControlRight');
    this.camera.position.addScaledVector(this.move, this.speed * (sprinting ? 2.5 : 1) * dt);
  }
}
