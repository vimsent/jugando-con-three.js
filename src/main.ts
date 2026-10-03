import { App } from './app';

const canvas = document.getElementById('view') as HTMLCanvasElement;
const hud = document.getElementById('hud') as HTMLElement;
const msg = document.getElementById('msg') as HTMLElement;

const app = new App(canvas, hud);
// Automation hook used by scripts/capture.mjs (Playwright).
(window as unknown as { __lab: unknown }).__lab = { app, ready: false, error: null as string | null };

app
  .init((s) => (msg.textContent = s))
  .then(() => {
    (window as unknown as { __lab: { ready: boolean } }).__lab.ready = true;
  })
  .catch((err: unknown) => {
    console.error(err);
    const text = err instanceof Error ? err.message : String(err);
    msg.textContent = `Error: ${text}`;
    (window as unknown as { __lab: { error: string } }).__lab.error = text;
  });
