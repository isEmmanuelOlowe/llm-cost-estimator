import fs from 'node:fs/promises';
import path from 'node:path';

const repoRoot = process.cwd();
const overridesPath = path.join(
  repoRoot,
  'src/data/model-presets.overrides.json',
);
const outputPath = path.join(repoRoot, 'src/data/model-presets.generated.json');

const overrides = JSON.parse(await fs.readFile(overridesPath, 'utf8'));

const dtypeToBits = {
  float32: 32,
  'torch.float32': 32,
  float16: 16,
  'torch.float16': 16,
  bfloat16: 16,
  'torch.bfloat16': 16,
  int8: 8,
  int4: 4,
};

function parseLicense(tags = []) {
  return tags.find((tag) => tag.startsWith('license:'))?.split(':')[1] ?? null;
}

async function fetchOptionalJson(url) {
  const response = await fetch(url);
  if (!response.ok) return null;
  return response.json();
}

const generated = [];

for (const override of overrides) {
  const apiResponse = await fetch(
    `https://huggingface.co/api/models/${override.id}`,
  );
  if (!apiResponse.ok) {
    throw new Error(
      `Failed to fetch model info for ${override.id}: ${apiResponse.status}`,
    );
  }

  const apiData = await apiResponse.json();
  const configData =
    (await fetchOptionalJson(
      `https://huggingface.co/${override.id}/raw/main/config.json`,
    )) ?? {};
  const resolvedConfig = apiData.config ?? {};

  generated.push({
    id: override.id,
    label: override.label,
    family: override.family,
    modelType: override.modelType,
    modality: override.modality,
    parameterCount: override.parameterCount,
    activeParameterCount: override.activeParameterCount ?? null,
    contextLength:
      configData.max_position_embeddings ??
      configData.seq_length ??
      configData.n_positions ??
      override.contextLength,
    hiddenSize: configData.hidden_size ?? override.hiddenSize ?? null,
    numLayers: configData.num_hidden_layers ?? override.numLayers ?? null,
    numHeads: configData.num_attention_heads ?? override.numHeads ?? null,
    intermediateSize:
      configData.intermediate_size ?? override.intermediateSize ?? null,
    vocabSize: configData.vocab_size ?? override.vocabSize ?? null,
    dtypeBits:
      dtypeToBits[configData.torch_dtype] ?? override.dtypeBits ?? null,
    pipelineTag: apiData.pipeline_tag ?? null,
    modelTypeTag: resolvedConfig.model_type ?? null,
    architectures: resolvedConfig.architectures ?? [],
    license: parseLicense(apiData.tags),
    tags: apiData.tags ?? [],
    engineSupport: override.engineSupport,
    summary: override.summary,
  });
}

await fs.writeFile(outputPath, `${JSON.stringify(generated, null, 2)}\n`);
console.log(
  `Wrote ${generated.length} model presets to ${path.relative(repoRoot, outputPath)}`,
);
