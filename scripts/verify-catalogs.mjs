import fs from 'node:fs/promises';
import path from 'node:path';

const repoRoot = process.cwd();

async function readJson(relativePath) {
  return JSON.parse(
    await fs.readFile(path.join(repoRoot, relativePath), 'utf8'),
  );
}

const models = await readJson('src/data/model-presets.generated.json');
const gpus = await readJson('src/estimator/gpus.json');

for (const id of [
  'Qwen/Qwen3.5-27B',
  'Qwen/Qwen3.5-35B-A3B',
  'Qwen/Qwen3-Coder-Next',
  'openai/gpt-oss-20b',
  'zai-org/GLM-4.7-Flash',
  'moonshotai/Kimi-K2.5',
]) {
  const entry = models.find((model) => model.id === id);
  if (!entry) throw new Error(`Missing required model preset: ${id}`);
  for (const key of [
    'label',
    'family',
    'modelType',
    'modality',
    'parameterCount',
    'contextLength',
    'summary',
  ]) {
    if (!entry[key]) throw new Error(`Model preset ${id} is missing ${key}`);
  }
}

for (const name of [
  'NVIDIA GeForce RTX 5090',
  'NVIDIA RTX PRO 6000 Blackwell',
  'NVIDIA H200 SXM 141GB',
  'NVIDIA DGX B200 (8x B200)',
]) {
  if (!gpus.find((gpu) => gpu.name === name)) {
    throw new Error(`Missing required GPU entry: ${name}`);
  }
}

console.log('Catalog verification passed.');
