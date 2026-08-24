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
    'vendor',
    'architecture',
    'memory_type',
    'memory_model',
    'device_count',
    'per_device_memory_gb',
    'source_url',
    'source_checked_at',
  ]) {
    if (!(key in gpu)) {
      throw new Error(`GPU entry ${gpu.name ?? '(unknown)'} is missing ${key}`);
    }
  }
  if (!Number.isFinite(gpu.memory_gb) || gpu.memory_gb <= 0) {
    throw new Error(`GPU entry ${gpu.name} has invalid aggregate memory`);
  }
  if (!Number.isFinite(gpu.device_count) || gpu.device_count < 1) {
    throw new Error(`GPU entry ${gpu.name} has invalid device count`);
  }
  if (
    !Number.isFinite(gpu.per_device_memory_gb) ||
    gpu.per_device_memory_gb <= 0 ||
    gpu.per_device_memory_gb * gpu.device_count > gpu.memory_gb * 1.01
  ) {
    throw new Error(`GPU entry ${gpu.name} has inconsistent per-device memory`);
  }
}

await fs.writeFile(outputPath, `${JSON.stringify(gpus, null, 2)}\n`);
console.log(
  `Wrote ${gpus.length} GPU entries to ${path.relative(repoRoot, outputPath)}`,
);
