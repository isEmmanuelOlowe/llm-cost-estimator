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
  'moonshotai/Kimi-K3',
  'moonshotai/Kimi-K2.6',
  'moonshotai/Kimi-K2.7-Code',
  'google/gemma-4-12B',
  'google/gemma-4-26B-A4B',
  'google/gemma-3-4b-it',
  'Qwen/Qwen3.8-27B',
  'Qwen/Qwen3.8-2.4T-A95B',
  'Qwen/Qwen3-32B',
  'zai-org/GLM-5',
  'deepseek-ai/DeepSeek-V3.2',
  'openai/gpt-oss-120b',
  'meta-models/Muse-Glimmer-30B',
  'thinkingmachines/Inkling',
  'thinkingmachines/Inkling-Small',
  'deepseek-ai/DeepSeek-V4-Pro',
  'deepseek-ai/DeepSeek-V4-Flash',
  'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16',
  'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
  'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16',
  'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
  'MiniMaxAI/MiniMax-M3',
  'zai-org/GLM-5.2',
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
  if (
    [
      'thinkingmachines/Inkling-Small',
      'deepseek-ai/DeepSeek-V4-Pro',
      'deepseek-ai/DeepSeek-V4-Flash',
      'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
      'moonshotai/Kimi-K3',
      'moonshotai/Kimi-K2.6',
      'moonshotai/Kimi-K2.7-Code',
      'zai-org/GLM-5',
      'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
    ].includes(id) &&
    !entry.kvCacheArchitecture
  ) {
    throw new Error(
      `Model preset ${id} is missing cache architecture metadata`,
    );
  }
}

for (const name of [
  'NVIDIA GeForce RTX 5090',
  'NVIDIA RTX PRO 6000 Blackwell',
  'NVIDIA H200 SXM 141GB',
  'NVIDIA DGX B200 (8x B200)',
  'NVIDIA B200 SXM 180GB',
  'NVIDIA B100 SXM 192GB',
  'NVIDIA B300 SXM 288GB',
  'NVIDIA HGX B300 (8x B300)',
  'NVIDIA DGX B300 (8x B300)',
  'NVIDIA GB200 NVL72 (72x B200)',
  'NVIDIA GB300 NVL72 (72x B300)',
  'NVIDIA DGX Station GB300',
  'NVIDIA GeForce RTX 5080',
  'NVIDIA GeForce RTX 5070 Ti',
  'NVIDIA DGX Spark',
  'AMD Ryzen AI Max+ PRO 395 (Strix Halo)',
  'AMD Instinct MI300X',
  'Intel Arc Pro B60',
  'Intel Arc Pro B70',
  'Apple Mac mini M4 (32GB)',
  'Apple Mac mini M4 Pro (64GB)',
  'Apple Mac Studio M4 Max (128GB)',
  'Apple Mac Studio M3 Ultra (256GB)',
  'Apple MacBook Air M5 (32GB)',
  'Apple MacBook Pro M5 Pro (64GB)',
  'Apple MacBook Pro M5 Max (128GB)',
  'Apple MacBook Air M4 (32GB)',
  'Apple MacBook Pro M4 Pro (64GB)',
  'Apple MacBook Pro M4 Max (128GB)',
]) {
  const gpu = gpus.find((entry) => entry.name === name);
  if (!gpu) {
    throw new Error(`Missing required GPU entry: ${name}`);
  }
  if (!gpu.source_url || !gpu.source_checked_at || !gpu.vendor) {
    throw new Error(`GPU ${name} is missing provenance metadata`);
  }
}

console.log('Catalog verification passed.');
