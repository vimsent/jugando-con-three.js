import { formatBytes } from '../metrics/memory';

export interface HudData {
  method: string;
  compare: string | null;
  bookmark: string;
  view: string;
  fps: number;
  passes: [string, number][];
  gpuTotal: number;
  timerSupported: boolean;
  rays: number;
  memory: number;
  error: { rmse: number; rel: number; refSamples: number } | null;
  converge: string | null;
  extra: Record<string, string | number>;
  status: string;
}

const fmtRays = (r: number) => (r >= 1e6 ? (r / 1e6).toFixed(2) + ' M' : r >= 1e3 ? (r / 1e3).toFixed(1) + ' k' : String(r));

/** Text HUD in Spanish (top-left). */
export class Hud {
  private visible = true;
  constructor(private readonly el: HTMLElement) {}

  toggle(): void {
    this.visible = !this.visible;
    this.el.style.display = this.visible ? '' : 'none';
  }

  update(d: HudData): void {
    if (!this.visible) return;
    const lines: string[] = [];
    lines.push(`Método: ${d.method}${d.compare ? `  |  B: ${d.compare}` : ''}`);
    lines.push(`Vista: ${d.view}   Cámara: ${d.bookmark}   ${d.fps.toFixed(0)} fps`);
    lines.push('');
    lines.push(d.timerSupported ? 'GPU por pase (ms, promedio):' : 'Tiempo por pase (ms, CPU: timestamp-query no disponible):');
    for (const [name, ms] of d.passes) lines.push(`  ${name.padEnd(24)} ${ms.toFixed(3).padStart(8)}`);
    lines.push(`  ${'TOTAL'.padEnd(24)} ${d.gpuTotal.toFixed(3).padStart(8)}`);
    lines.push('');
    lines.push(`Rayos/frame: ${fmtRays(d.rays)}   Memoria GI: ${formatBytes(d.memory)}`);
    if (d.error) {
      lines.push(`Error vs referencia (${d.error.refSamples} spp): RMSE ${d.error.rmse.toExponential(3)}  rel ${(d.error.rel * 100).toFixed(2)} %`);
    } else {
      lines.push('Error vs referencia: (sin imagen de referencia para esta cámara/luz)');
    }
    if (d.converge) lines.push(`Convergencia: ${d.converge}`);
    for (const [k, v] of Object.entries(d.extra)) lines.push(`${k}: ${v}`);
    if (d.status) lines.push('', d.status);
    lines.push('', '[0-7] método  [Shift+1-5] cámara  [V] vista  [S] split  [R] reset  [H] HUD');
    this.el.textContent = lines.join('\n');
  }
}
