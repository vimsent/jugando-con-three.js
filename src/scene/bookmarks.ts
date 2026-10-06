import { PerspectiveCamera, Vector3 } from 'three';

export interface CameraBookmark {
  name: string; // shown in the HUD (Spanish)
  position: [number, number, number];
  target: [number, number, number];
}

// Fixed viewpoints used for the reference images and the benchmark. Sponza is ~30 m long along X.
export const BOOKMARKS: CameraBookmark[] = [
  { name: 'Atrio (eje largo)', position: [-10.5, 1.7, -0.3], target: [10, 3.2, 0.3] },
  { name: 'Galería inferior', position: [-9.5, 1.6, -5.6], target: [8, 2.0, -5.2] },
  { name: 'Galería superior', position: [9.5, 6.6, 5.4], target: [-8, 5.2, 4.4] },
  { name: 'Cortinas', position: [1.5, 1.5, -5.2], target: [-2.5, 4.0, 3.8] },
  { name: 'Vista elevada', position: [9.5, 6.6, 0], target: [-8, 1.2, 0] },
];

export function applyBookmark(camera: PerspectiveCamera, b: CameraBookmark): void {
  camera.position.set(...b.position);
  camera.lookAt(new Vector3(...b.target));
  camera.updateMatrixWorld();
}
