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

/** Bytes of a render target's color attachments (+ depth), from format/type. */
export function renderTargetBytes(rt: { width: number; height: number; textures: { type: number; format: number }[]; depthBuffer?: boolean; depthTexture?: unknown }): number {
  let total = 0;
  for (const t of rt.textures) total += rt.width * rt.height * texelBytes(t.format, t.type);
  if (rt.depthBuffer || rt.depthTexture) total += rt.width * rt.height * 4;
  return total;
}

// three.js constants (kept numeric to avoid importing three here)
const FLOAT = 1015, HALF = 1016, RED = 1028, RG = 1030, RGB = 1022, RGBA = 1023, UNSIGNED_INT_10F_11F_11F_REV = 35899;
export function texelBytes(format: number, type: number): number {
  const channels = format === RED ? 1 : format === RG ? 2 : format === RGB ? 3 : format === RGBA ? 4 : 4;
  if (type === UNSIGNED_INT_10F_11F_11F_REV) return 4;
  const bpc = type === FLOAT ? 4 : type === HALF ? 2 : 1;
  return channels * bpc;
}
