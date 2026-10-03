import type { GIMethod } from './types';
import { FlatAmbient } from './flat';

export interface MethodEntry {
  /** Hotkey digit (0 = reference). */
  hotkey: number;
  key: string;
  label: string;
  create: () => Promise<GIMethod>;
}

// Methods are created lazily (dynamic imports) and disposed when switched away from, so that the
// memory reported for a method only covers that method.
export const METHODS: MethodEntry[] = [
  { hotkey: 1, key: 'flat', label: 'Sin GI (ambiente plano)', create: async () => new FlatAmbient() },
];

export function findMethod(key: string): MethodEntry | undefined {
  return METHODS.find((m) => m.key === key);
}
