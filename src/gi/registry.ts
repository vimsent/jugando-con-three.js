import type { GIMethod } from './types';
import { FlatAmbient } from './flat';
import { ReferencePathTracer } from './reference';
import { GTAOMethod } from './gtao';
import { SSGIMethod } from './ssgi';
import { DDGIMethod } from './ddgi';
import { SharcMethod } from './sharc';
import { DDGISSDOMethod } from './hybrid';

export interface MethodEntry {
  /** Hotkey digit (0 = reference). */
  hotkey: number;
  key: string;
  label: string;
  create: () => Promise<GIMethod>;
}

// Methods are created lazily and disposed when switched away from, so that the
// memory reported for a method only covers that method.
export const METHODS: MethodEntry[] = [
  { hotkey: 0, key: 'reference', label: 'Referencia (path tracing)', create: async () => new ReferencePathTracer() },
  { hotkey: 1, key: 'flat', label: 'Sin GI (ambiente plano)', create: async () => new FlatAmbient() },
  { hotkey: 2, key: 'gtao', label: 'GTAO (AO × ambiente)', create: async () => new GTAOMethod() },
  { hotkey: 3, key: 'ssgi', label: 'SSGI (espacio de pantalla)', create: async () => new SSGIMethod() },
  { hotkey: 4, key: 'ddgi', label: 'DDGI (probes)', create: async () => new DDGIMethod() },
  { hotkey: 5, key: 'ddgi-ssdo', label: 'DDGI + SSDO', create: async () => new DDGISSDOMethod() },
  { hotkey: 6, key: 'sharc', label: 'Cache hash (tipo SHaRC)', create: async () => new SharcMethod() },
];

export function findMethod(key: string): MethodEntry | undefined {
  return METHODS.find((m) => m.key === key);
}
