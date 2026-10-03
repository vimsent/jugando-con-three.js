# Sponza GI Lab

Laboratorio de juguete en three.js (WebGPU) para comparar métodos de **iluminación global difusa**
sobre Sponza, en costo (ms de GPU por pase, rayos, memoria) y en calidad (error contra una
referencia path-traced).

> **Importante — trazado de rayos por software.** WebGPU no expone los RT cores de la GPU. Todo el
> trazado de rayos de este proyecto (referencia, DDGI, cache hash, ReSTIR) recorre un BVH en
> *compute shaders*. Los tiempos sirven para **comparar los métodos entre sí** en este entorno;
> **no** predicen el rendimiento de esos métodos con trazado por hardware (DXR/Vulkan RT), donde el
> costo relativo de un rayo frente al de un pase de pantalla cambia mucho.

## Requisitos

- Node ≥ 20, npm.
- Un navegador con WebGPU. Probado con Chrome/Chromium en Fedora (Wayland) con una RTX 4060 Ti
  (driver NVIDIA 580, Vulkan).

### WebGPU en Chrome sobre Linux

En Linux, WebGPU sigue detrás de flags en Chrome estable. Abre `chrome://gpu` y busca
**WebGPU: Hardware accelerated**. Si dice *Disabled* o *Software only*:

1. `chrome://flags/#enable-unsafe-webgpu` → **Enabled**
2. `chrome://flags/#enable-vulkan` → **Enabled**
3. Reinicia Chrome y vuelve a mirar `chrome://gpu`.

Equivalente por línea de comandos:

```sh
google-chrome --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan
# Chrome Dev en Flatpak:
flatpak run com.google.ChromeDev --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan
```

En la consola de DevTools, `(await navigator.gpu.requestAdapter()).info` debería mostrar
`vendor: "nvidia"`, `architecture: "lovelace"`. Para medir tiempos de GPU hace falta la feature
`timestamp-query` (disponible con esos flags). Sin ella, el HUD cae a tiempos de CPU y lo indica.

## Uso

```sh
npm install
npm run fetch-assets   # descarga Sponza (~50 MB) a public/assets/sponza
npm run dev            # http://localhost:5173
```

Controles: `1`–`7` método, `0` referencia, `Shift+1`–`5` bookmarks de cámara, `V` modo de vista,
`S` split A|B, `R` reinicia el historial temporal, `H` oculta el HUD. El resto de los parámetros
está en el panel lil-gui.

`npm run capture` abre la app en Chromium headless (Playwright) sobre la GPU real y guarda
capturas; ver `scripts/capture.mjs`.

## Escena y licencia

Sponza viene de [KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza).
© 2016 Crytek, bajo la *Cryengine Limited License Agreement*; modelo original de Marko Dabrovic
(2002), versión mejorada de Frank Meinl (Crytek, 2010), texturas PBR de Alexandre Pestana, con
correcciones de Morgan McGuire. Los metadatos son CC-BY 4.0. Por esa licencia el modelo **no se
versiona en este repositorio**: `npm run fetch-assets` lo descarga y copia junto al modelo su
`README.md`, `LICENSE.md` y el texto de la licencia (`LicenseRef-CRYENGINE-Agreement.txt`).

## Arquitectura

- **G-buffer** (MRT): radiancia directa, albedo lineal, normal en espacio de vista y profundidad
  (la posición se reconstruye a partir de la profundidad).
- **Luz directa común**: sol direccional con shadow map 4096² y dos luces puntuales con sombras
  cúbicas (una fija en la galería y otra que se mueve por el atrio). Materiales Lambert: se
  descarta el especular a propósito.
- **Interfaz `GIMethod`** (`src/gi/types.ts`): cada método recibe el G-buffer y devuelve una
  textura de **iluminación indirecta difusa** en la convención `E_ind / π`. El pase común compone
  `final = directa + albedo · indirecta` en HDR lineal.
- **Medición de GPU por pase**: three.js r186 escribe un par de timestamps por *render pass* y por
  despacho de compute, con un UID `r|c:<llamada>:<id>:f<frame>`. `GpuTimer`
  (`src/metrics/gpuTimer.ts`) registra qué llamadas pertenecen a cada pase con nombre, resuelve
  con `renderer.resolveTimestampsAsync()` y suma las duraciones de cada pase.

## Supuestos y desvíos documentados

- La luz directa se evalúa al llenar el G-buffer y no en un pase de iluminación diferido aparte:
  three.js liga las sombras y las luces a los materiales. Es idéntica para todos los métodos.
- Resolución interna fija de 1280×720 (el canvas se escala con CSS), para que los benchmarks no
  dependan del tamaño de la ventana.

_(Esta sección y la tabla de resultados se completan en los hitos siguientes.)_
