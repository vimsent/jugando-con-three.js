// Explicit GPU memory bookkeeping. WebGPU does not report allocation sizes, so each method declares
// what it allocates through a MemoryLedger; sizes are computed from dimensions and formats.

export function textureBytes(width: number, height: number, bytesPerTexel: number, depth = 1): number {
  return width * height * depth * bytesPerTexel;
}

export class MemoryLedger {
  private readonly entries = new Map<string, number>();

  add(name: string, bytes: number): void {
    this.entries.set(name, (this.entries.get(name) ?? 0) + bytes);
  }

  clear(): void {
    this.entries.clear();
  }

  get total(): number {
    let s = 0;
    for (const v of this.entries.values()) s += v;
    return s;
  }

  breakdown(): Record<string, number> {
    return Object.fromEntries(this.entries);
  }
}

export function formatBytes(b: number): string {
  if (b >= 1 << 30) return (b / (1 << 30)).toFixed(2) + ' GiB';
  if (b >= 1 << 20) return (b / (1 << 20)).toFixed(1) + ' MiB';
  if (b >= 1 << 10) return (b / (1 << 10)).toFixed(1) + ' KiB';
  return b + ' B';
}
