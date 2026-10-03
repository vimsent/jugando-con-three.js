import { Color, DoubleSide, Group, Mesh, MeshStandardMaterial, SRGBColorSpace, Texture } from 'three';
import { MeshLambertNodeMaterial } from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export interface SponzaScene {
  root: Group;
  meshes: Mesh[];
  // Average linear albedo per G-buffer material, used as the albedo of secondary ray hits.
  averageAlbedo: Map<MeshLambertNodeMaterial, Color>;
}

/**
 * Loads Sponza and replaces its PBR materials by Lambert materials: the lab only studies diffuse
 * transport, so specular is dropped on purpose. Base color, normal map and alpha test are kept.
 */
export async function loadSponza(url: string, onProgress?: (f: number) => void): Promise<SponzaScene> {
  const gltf = await new GLTFLoader().loadAsync(url, (e) => {
    if (e.total) onProgress?.(e.loaded / e.total);
  });
  const root = gltf.scene as Group;
  root.updateMatrixWorld(true);

  const meshes: Mesh[] = [];
  const converted = new Map<MeshStandardMaterial, MeshLambertNodeMaterial>();
  const averageAlbedo = new Map<MeshLambertNodeMaterial, Color>();

  root.traverse((o) => {
    if (!(o as Mesh).isMesh) return;
    const mesh = o as Mesh;
    const src = mesh.material as MeshStandardMaterial;
    let mat = converted.get(src);
    if (!mat) {
      mat = new MeshLambertNodeMaterial();
      mat.name = src.name;
      mat.color.copy(src.color);
      mat.map = src.map;
      mat.normalMap = src.normalMap;
      if (src.normalMap) mat.normalScale.copy(src.normalScale);
      mat.alphaTest = src.alphaTest;
      mat.transparent = false;
      mat.side = src.side === DoubleSide ? DoubleSide : src.side;
      converted.set(src, mat);
      averageAlbedo.set(mat, computeAverageAlbedo(src.color, src.map, src.alphaTest > 0));
    }
    mesh.material = mat;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    meshes.push(mesh);
  });

  return { root, meshes, averageAlbedo };
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/**
 * Averages a base color texture in linear space on a 64x64 downsample. With `alphaWeighted`, texels
 * are weighted by alpha so cut-out regions of masked materials (plants) do not darken the result.
 */
function computeAverageAlbedo(factor: Color, map: Texture | null, alphaWeighted: boolean): Color {
  const out = factor.clone();
  const image = map?.image as CanvasImageSource | undefined;
  if (!map || !image) return out;
  const size = 64;
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return out;
  ctx.drawImage(image, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  const isSRGB = map.colorSpace === SRGBColorSpace;
  let r = 0, g = 0, b = 0, w = 0;
  for (let i = 0; i < data.length; i += 4) {
    const a = alphaWeighted ? data[i + 3] / 255 : 1;
    const dec = (v: number) => (isSRGB ? srgbToLinear(v / 255) : v / 255);
    r += dec(data[i]) * a;
    g += dec(data[i + 1]) * a;
    b += dec(data[i + 2]) * a;
    w += a;
  }
  if (w > 0) out.multiply(new Color(r / w, g / w, b / w));
  return out;
}
