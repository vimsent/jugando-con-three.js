// Downloads Sponza (glTF) from KhronosGroup/glTF-Sample-Assets into public/assets/sponza,
// keeping the model's README/LICENSE files and the referenced Cryengine license text.
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const dest = join(process.cwd(), 'public/assets/sponza');
if (existsSync(join(dest, 'glTF/Sponza.gltf')) && !process.argv.includes('--force')) {
  console.log('Sponza already present at', dest);
  process.exit(0);
}

const tmp = mkdtempSync(join(tmpdir(), 'sponza-'));
const git = (...args) => execFileSync('git', args, { cwd: tmp, stdio: 'inherit' });
try {
  git('clone', '--depth', '1', '--filter=blob:none', '--sparse',
    'https://github.com/KhronosGroup/glTF-Sample-Assets.git', 'repo');
  execFileSync('git', ['sparse-checkout', 'set', '--no-cone', '/Models/Sponza/', '/LICENSES/LicenseRef-CRYENGINE-Agreement.txt'],
    { cwd: join(tmp, 'repo'), stdio: 'inherit' });
  mkdirSync(dest, { recursive: true });
  const src = join(tmp, 'repo/Models/Sponza');
  for (const f of ['glTF', 'README.md', 'LICENSE.md', 'metadata.json']) cpSync(join(src, f), join(dest, f), { recursive: true });
  cpSync(join(tmp, 'repo/LICENSES/LicenseRef-CRYENGINE-Agreement.txt'), join(dest, 'LicenseRef-CRYENGINE-Agreement.txt'));
  console.log('Sponza copied to', dest);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
