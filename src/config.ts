// Global, benchmark-relevant constants. The internal resolution is fixed so that every method is
// measured at the same pixel count regardless of the browser window size.
export const RENDER_WIDTH = 1280;
export const RENDER_HEIGHT = 720;

export const SPONZA_URL = '/assets/sponza/glTF/Sponza.gltf';

// Benchmark defaults (overridable from the URL / automation API).
export const BENCH_WARMUP_FRAMES = 300;
export const BENCH_MEASURE_FRAMES = 600;

// Relative-error threshold used to decide that a method has "converged" after a lighting change.
export const CONVERGENCE_REL_ERROR = 0.05;
