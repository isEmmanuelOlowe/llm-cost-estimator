import fs from 'node:fs/promises';
import path from 'node:path';

const repoRoot = process.cwd();
const sourcePath = path.join(repoRoot, 'src/data/gpu-catalog.overrides.json');
const outputPath = path.join(repoRoot, 'src/estimator/gpus.json');

const gpus = JSON.parse(await fs.readFile(sourcePath, 'utf8'));

for (const gpu of gpus) {
  for (const key of [
    'name',
    'memory_gb',
    'fp32_tflops',
    'memory_bandwidth_gb_s',
  ]) {
    if (!(key in gpu)) {
      throw new Error(`GPU entry ${gpu.name ?? '(unknown)'} is missing ${key}`);
    }
  }
}

await fs.writeFile(outputPath, `${JSON.stringify(gpus, null, 2)}\n`);
console.log(
  `Wrote ${gpus.length} GPU entries to ${path.relative(repoRoot, outputPath)}`,
);
