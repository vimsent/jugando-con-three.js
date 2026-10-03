import { DataTexture, HalfFloatType, LinearFilter, RGBAFormat, type Texture } from 'three';
import { RenderTarget, type WebGPURenderer } from 'three/webgpu';
import { textureBytes } from './memory';

export interface ReferenceEntry {
  key: string;
  bookmark: number;
  lightKey: string;
  spp: number;
  final: Texture; // composite with the reference indirect (linear HDR, rgba16f)
  indirect: Texture; // reference E_ind / π (rgba16f)
}

interface StoredEntry {
  key: string;
  bookmark: number;
  lightKey: string;
  spp: number;
  width: number;
  height: number;
  final: Uint16Array;
  indirect: Uint16Array;
}

// Bump when anything that changes the reference image changes (scene, lights, estimator).
const DB_NAME = 'sponza-gi-lab';
const DB_VERSION = 1;
const STORE = 'references-v1';

/**
 * Ground-truth images per (camera bookmark, light state). Entries live on the GPU and are also
 * persisted in IndexedDB (raw half floats), so references survive page reloads.
 */
export class ReferenceCache {
  private readonly entries = new Map<string, ReferenceEntry>();
  private readonly loading = new Set<string>();
  private readonly missing = new Set<string>();
  private db: Promise<IDBDatabase | null>;

  constructor(private readonly width: number, private readonly height: number) {
    this.db = openDb();
  }

  static key(bookmark: number, lightKey: string): string {
    return `${bookmark}|${lightKey}`;
  }

  /** Returns the entry if it is on the GPU; otherwise starts loading it from IndexedDB. */
  get(key: string): ReferenceEntry | undefined {
    const e = this.entries.get(key);
    if (!e && !this.loading.has(key) && !this.missing.has(key)) void this.load(key);
    return e;
  }

  /** Copies the given textures (rgba16f, full resolution) into a cache entry and persists it. */
  async store(renderer: WebGPURenderer, e: Omit<ReferenceEntry, 'final' | 'indirect'>, final: Texture, indirect: Texture): Promise<ReferenceEntry> {
    const finalRT = this.makeTarget(renderer);
    const indirectRT = this.makeTarget(renderer);
    renderer.copyTextureToTexture(final, finalRT.texture);
    renderer.copyTextureToTexture(indirect, indirectRT.texture);
    const entry: ReferenceEntry = { ...e, final: finalRT.texture, indirect: indirectRT.texture };
    this.entries.get(e.key)?.final.dispose();
    this.entries.set(e.key, entry);
    this.missing.delete(e.key);
    const [f, i] = await Promise.all([
      renderer.readRenderTargetPixelsAsync(finalRT, 0, 0, this.width, this.height),
      renderer.readRenderTargetPixelsAsync(indirectRT, 0, 0, this.width, this.height),
    ]);
    const stored: StoredEntry = {
      ...e,
      width: this.width,
      height: this.height,
      final: new Uint16Array(f as Uint16Array),
      indirect: new Uint16Array(i as Uint16Array),
    };
    await this.put(stored);
    return entry;
  }

  /** Raw half-float pixels of an entry (for exporting images from the benchmark). */
  async readPixels(renderer: WebGPURenderer, tex: Texture): Promise<Uint16Array> {
    const rt = this.makeTarget(renderer);
    renderer.copyTextureToTexture(tex, rt.texture);
    const px = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, this.width, this.height)) as Uint16Array;
    rt.dispose();
    return new Uint16Array(px);
  }

  list(): ReferenceEntry[] {
    return [...this.entries.values()];
  }

  async clear(): Promise<void> {
    for (const e of this.entries.values()) {
      e.final.dispose();
      e.indirect.dispose();
    }
    this.entries.clear();
    this.missing.clear();
    const db = await this.db;
    if (!db) return;
    await txDone(db.transaction(STORE, 'readwrite').objectStore(STORE).clear());
  }

  get memoryBytes(): number {
    return this.entries.size * 2 * textureBytes(this.width, this.height, 8);
  }

  private makeTarget(renderer: WebGPURenderer): RenderTarget {
    const rt = new RenderTarget(this.width, this.height, { type: HalfFloatType, depthBuffer: false });
    rt.texture.minFilter = LinearFilter;
    rt.texture.generateMipmaps = false;
    renderer.initRenderTarget(rt);
    return rt;
  }

  private async load(key: string): Promise<void> {
    this.loading.add(key);
    try {
      const db = await this.db;
      if (!db) return void this.missing.add(key);
      const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
      const s = (await txDone(req)) as StoredEntry | undefined;
      if (!s || s.width !== this.width || s.height !== this.height) return void this.missing.add(key);
      const mk = (data: Uint16Array) => {
        const t = new DataTexture(data, s.width, s.height, RGBAFormat, HalfFloatType);
        t.minFilter = LinearFilter;
        t.magFilter = LinearFilter;
        t.needsUpdate = true;
        return t;
      };
      this.entries.set(key, { key, bookmark: s.bookmark, lightKey: s.lightKey, spp: s.spp, final: mk(s.final), indirect: mk(s.indirect) });
    } finally {
      this.loading.delete(key);
    }
  }

  private async put(s: StoredEntry): Promise<void> {
    const db = await this.db;
    if (!db) return;
    await txDone(db.transaction(STORE, 'readwrite').objectStore(STORE).put(s, s.key));
  }
}

function openDb(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

function txDone<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
