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

const checkedAt = new Date().toISOString().slice(0, 10);

const featuredModelIds = [
  'google/gemma-4-12B',
  'Qwen/Qwen3.8-27B',
  'zai-org/GLM-5',
  'moonshotai/Kimi-K3',
  'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
];

const familyOrder = [
  'Gemma 4',
  'Qwen3.8',
  'Qwen3',
  'Qwen3.5',
  'Qwen3-Coder',
  'DeepSeek V4',
  'DeepSeek',
  'Inkling',
  'Muse Glimmer',
  'MiniMax M3',
  'Nemotron 3',
  'GLM',
  'GLM-4.7',
  'gpt-oss',
  'Kimi',
  'Gemma 3',
];

function orderedIndex(value, values) {
  const index = values.indexOf(value);
  return index === -1 ? values.length : index;
}

function comparePresets(left, right) {
  const leftFeatured = featuredModelIds.indexOf(left.id);
  const rightFeatured = featuredModelIds.indexOf(right.id);
  if (leftFeatured !== -1 || rightFeatured !== -1) {
    return (
      (leftFeatured === -1 ? featuredModelIds.length : leftFeatured) -
      (rightFeatured === -1 ? featuredModelIds.length : rightFeatured)
    );
  }

  const familyDifference =
    orderedIndex(left.family, familyOrder) -
    orderedIndex(right.family, familyOrder);
  return (
    familyDifference ||
    (left.parameterCount ?? Number.POSITIVE_INFINITY) -
      (right.parameterCount ?? Number.POSITIVE_INFINITY) ||
    left.label.localeCompare(right.label)
  );
}

function parseLicense(tags = []) {
  return tags.find((tag) => tag.startsWith('license:'))?.split(':')[1] ?? null;
}

function encodeRepoPath(id) {
  return id.split('/').map(encodeURIComponent).join('/');
}

async function fetchJson(url, { optional = false, attempts = 3 } = {}) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: { accept: 'application/json' },
      });
      if (!response.ok) {
        if (optional && response.status === 404) return null;
        throw new Error(`${response.status} ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      lastError = error;
      if (attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, 250 * 2 ** attempt));
      }
    } finally {
      clearTimeout(timeoutId);
    }
  }
  if (optional && lastError?.message?.startsWith('404 ')) return null;
  throw new Error(
    `Failed to fetch ${url}: ${lastError?.message ?? 'unknown error'}`,
  );
}

function positiveNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
    ? value
    : null;
}

function configRecords(config) {
  return [
    config,
    ...['text_config', 'language_config', 'llm_config', 'model_config', 'model']
      .map((key) => config[key])
      .filter(
        (value) => value && typeof value === 'object' && !Array.isArray(value),
      ),
  ];
}

function getConfigValue(config, keys) {
  for (const record of configRecords(config)) {
    for (const key of keys) {
      if (record[key] !== undefined && record[key] !== null) {
        return record[key];
      }
    }
  }
  return null;
}

function getConfigNumber(config, keys) {
  for (const key of keys) {
    const value = positiveNumber(getConfigValue(config, [key]));
    if (value) return value;
  }
  return null;
}

function getConfigString(config, keys) {
  const value = getConfigValue(config, keys);
  return typeof value === 'string' && value ? value : null;
}

function getConfigBoolean(config, keys) {
  const value = getConfigValue(config, keys);
  return typeof value === 'boolean' ? value : null;
}

function getConfigArray(config, keys) {
  const value = getConfigValue(config, keys);
  return Array.isArray(value) ? value : [];
}

function deriveModalityArchitecture(config, modelType) {
  const vision =
    config.vision_config && typeof config.vision_config === 'object'
      ? config.vision_config
      : null;
  const audio =
    config.audio_config && typeof config.audio_config === 'object'
      ? config.audio_config
      : null;
  const normalizedModelType = modelType?.toLowerCase() ?? '';
  const isUnified = normalizedModelType.includes('unified');
  if (!vision && !audio && !isUnified) return null;

  const text =
    config.text_config && typeof config.text_config === 'object'
      ? config.text_config
      : null;
  const textHiddenSize = positiveNumber(text?.hidden_size);
  const patchSize = positiveNumber(vision?.patch_size);
  const poolingKernelSize = positiveNumber(vision?.pooling_kernel_size);
  const pooledPatchSize =
    positiveNumber(vision?.model_patch_size) ??
    (patchSize && poolingKernelSize ? patchSize * poolingKernelSize : null);

  return {
    family: modelType ?? 'multimodal',
    evidence: 'config',
    vision: vision
      ? {
          encoderFree: isUnified,
          patchSize,
          pooledPatchSize,
          rawChannels:
            positiveNumber(vision.num_channels) ??
            positiveNumber(vision.in_channels) ??
            3,
          embedDim:
            positiveNumber(vision.mm_embed_dim) ??
            positiveNumber(vision.hidden_size),
          outputDim:
            positiveNumber(vision.output_proj_dims) ??
            positiveNumber(vision.projection_dim),
          softTokens:
            positiveNumber(vision.num_soft_tokens) ??
            positiveNumber(vision.num_image_tokens),
        }
      : null,
    video: Boolean(config.video_token_id) || isUnified,
    audio: audio
      ? {
          encoderFree: isUnified,
          featureDim:
            positiveNumber(audio.audio_embed_dim) ??
            positiveNumber(audio.hidden_size) ??
            positiveNumber(audio.input_dim),
          samplesPerToken:
            positiveNumber(audio.audio_samples_per_token) ??
            positiveNumber(audio.samples_per_token),
          outputDim:
            textHiddenSize ??
            positiveNumber(audio.output_proj_dims) ??
            positiveNumber(audio.hidden_size),
        }
      : null,
  };
}

function deriveKvCacheArchitecture(config, modelType) {
  const normalizedModelType = modelType?.toLowerCase() ?? '';
  const numLayers = getConfigNumber(config, [
    'num_hidden_layers',
    'num_layers',
    'n_layer',
  ]);
  const countPattern = (pattern, character) =>
    typeof pattern === 'string'
      ? [...pattern].filter((entry) => entry === character).length
      : 0;

  if (normalizedModelType === 'deepseek_v4') {
    const ratios = Array.isArray(config.compress_ratios)
      ? config.compress_ratios.filter((value) => typeof value === 'number')
      : [];
    const layerTypes = Array.isArray(config.layer_types)
      ? config.layer_types.filter((value) => typeof value === 'string')
      : [];
    const layerCount = Math.floor(
      numLayers ?? (layerTypes.length || ratios.length),
    );
    const compressedSparseLayers = layerTypes.length
      ? layerTypes.filter((type) => type === 'compressed_sparse_attention')
          .length
      : ratios.length
        ? ratios.filter((ratio) => ratio === 4).length
        : Math.floor(Math.max(0, layerCount - 2) / 2);
    const heavilyCompressedLayers = layerTypes.length
      ? layerTypes.filter((type) => type === 'heavily_compressed_attention')
          .length
      : ratios.length
        ? ratios.filter((ratio) => ratio === 128).length
        : layerCount - compressedSparseLayers;
    const compressRates =
      config.compress_rates && typeof config.compress_rates === 'object'
        ? config.compress_rates
        : null;
    return {
      mode: 'deepseek-v4',
      keyValueShared: true,
      localAttentionLayers: layerCount,
      fullAttentionLayers: 0,
      slidingWindow: getConfigNumber(config, ['sliding_window']) ?? 128,
      compressedSparseLayers,
      heavilyCompressedLayers,
      compressedSparseRate:
        positiveNumber(compressRates?.compressed_sparse_attention) ??
        getConfigNumber(config, ['compress_rate_csa']) ??
        4,
      heavilyCompressedRate:
        positiveNumber(compressRates?.heavily_compressed_attention) ??
        getConfigNumber(config, ['compress_rate_hca']) ??
        128,
      indexHeadDim: getConfigNumber(config, ['index_head_dim']),
      indexerLayers: compressedSparseLayers,
      label:
        'DeepSeek V4 sliding + compressed sparse + heavily compressed cache',
    };
  }

  const text =
    config.text_config && typeof config.text_config === 'object'
      ? config.text_config
      : null;
  const latentKvRank = getConfigNumber(config, ['kv_lora_rank']);
  const latentRopeDim = getConfigNumber(config, [
    'qk_rope_head_dim',
    'rope_head_dim',
  ]);
  const linearAttentionConfig =
    text?.linear_attn_config && typeof text.linear_attn_config === 'object'
      ? text.linear_attn_config
      : null;
  const fullAttentionLayerIds = Array.isArray(
    linearAttentionConfig?.full_attn_layers,
  )
    ? linearAttentionConfig.full_attn_layers.filter(
        (value) => typeof value === 'number',
      )
    : [];
  if (latentKvRank && latentRopeDim && numLayers) {
    const fullAttentionLayers = fullAttentionLayerIds.length || numLayers;
    const recurrentStateLayers = Math.max(0, numLayers - fullAttentionLayers);
    return {
      mode:
        recurrentStateLayers > 0 ? 'hybrid-latent-state' : 'latent-attention',
      keyValueShared: true,
      fullAttentionLayers,
      noAttentionLayers: recurrentStateLayers,
      recurrentStateLayers: recurrentStateLayers || null,
      latentKvRank,
      latentRopeDim,
      label:
        recurrentStateLayers > 0
          ? 'Hybrid latent attention + linear/state cache'
          : 'Multi-head latent attention compressed KV cache',
    };
  }
  const localLayerIds = Array.isArray(text?.local_layer_ids)
    ? text.local_layer_ids.filter((value) => typeof value === 'number')
    : [];
  const textLayerCount = positiveNumber(text?.num_hidden_layers);
  if (localLayerIds.length > 0 && textLayerCount) {
    return {
      mode: 'hybrid-sliding-window',
      keyValueShared: false,
      slidingWindow:
        positiveNumber(text.sliding_window_size) ??
        positiveNumber(text.sliding_window),
      localAttentionLayers: localLayerIds.length,
      fullAttentionLayers: Math.max(0, textLayerCount - localLayerIds.length),
      label: 'Hybrid local/global attention cache',
    };
  }

  const genericLayerTypes = Array.isArray(config.layers_block_type)
    ? config.layers_block_type.filter((value) => typeof value === 'string')
    : [];
  if (genericLayerTypes.length > 0 && numLayers) {
    const localAttentionLayers = genericLayerTypes.filter((type) =>
      /sliding|local|window/i.test(type),
    ).length;
    const fullAttentionLayers = genericLayerTypes.filter((type) =>
      /full_attention|global_attention|^attention$/i.test(type),
    ).length;
    const recurrentStateLayers = genericLayerTypes.filter((type) =>
      /mamba|state|linear_attention|recurrent/i.test(type),
    ).length;
    const noAttentionLayers = Math.max(
      0,
      genericLayerTypes.length - localAttentionLayers - fullAttentionLayers,
    );
    if (localAttentionLayers > 0 || fullAttentionLayers > 0) {
      return {
        mode:
          recurrentStateLayers > 0
            ? 'hybrid-state-space'
            : 'hybrid-sliding-window',
        keyValueShared: false,
        slidingWindow: getConfigNumber(config, [
          'sliding_window',
          'sliding_window_size',
        ]),
        localAttentionLayers,
        fullAttentionLayers,
        noAttentionLayers: recurrentStateLayers > 0 ? noAttentionLayers : null,
        recurrentStateLayers:
          recurrentStateLayers > 0 ? recurrentStateLayers : null,
        label: 'Config-declared hybrid attention schedule',
      };
    }
  }

  const hybridPattern = getConfigString(config, ['hybrid_override_pattern']);
  if (hybridPattern && numLayers) {
    const fullAttentionLayers = countPattern(hybridPattern, '*');
    const recurrentStateLayers = countPattern(hybridPattern, 'M');
    const noAttentionLayers = Math.max(
      0,
      hybridPattern.length - fullAttentionLayers,
    );
    const mambaHeads = getConfigNumber(config, ['mamba_num_heads']);
    const mambaHeadDim = getConfigNumber(config, ['mamba_head_dim']);
    const stateSize = getConfigNumber(config, ['ssm_state_size']);
    const groups = getConfigNumber(config, ['n_groups', 'mamba_n_groups']);
    const convKernel = getConfigNumber(config, ['conv_kernel', 'mamba_d_conv']);
    const stateBits =
      dtypeToBits[getConfigString(config, ['mamba_ssm_cache_dtype'])] ?? 32;
    const mambaIntermediate =
      mambaHeads && mambaHeadDim ? mambaHeads * mambaHeadDim : 0;
    const convChannels =
      mambaIntermediate && groups && stateSize
        ? mambaIntermediate + 2 * groups * stateSize
        : 0;
    const recurrentStateBytesPerLayer =
      mambaIntermediate && stateSize && convChannels && convKernel
        ? (mambaIntermediate * stateSize + convChannels * convKernel) *
          (stateBits / 8)
        : null;
    return {
      mode: 'hybrid-state-space',
      keyValueShared: false,
      fullAttentionLayers,
      noAttentionLayers,
      recurrentStateLayers,
      recurrentStateBytesPerLayer,
      label: 'Hybrid attention + state-space cache',
    };
  }

  const slidingWindow = getConfigNumber(config, [
    'sliding_window',
    'sliding_window_size',
  ]);
  if (slidingWindow && numLayers) {
    return {
      mode: 'sliding-window',
      keyValueShared: false,
      slidingWindow,
      localAttentionLayers: numLayers,
      fullAttentionLayers: 0,
      label: `Sliding-window cache capped at ${slidingWindow.toLocaleString()} tokens`,
    };
  }

  return null;
}

function getSafetensorsParameters(apiData) {
  const parameters = apiData.safetensors?.parameters ?? {};
  const byDtype = Object.fromEntries(
    Object.entries(parameters).filter(([, value]) => positiveNumber(value)),
  );
  const total =
    positiveNumber(apiData.safetensors?.total) ??
    Object.values(byDtype).reduce((sum, value) => sum + value, 0);
  return { total, byDtype };
}

function transformersModelDirectory(modelType) {
  const aliases = {
    gemma3_text: 'gemma3',
    gemma4_unified: 'gemma4_unified',
    qwen3_5_moe_text: 'qwen3_5',
    minimax_m3_vl: 'minimax_m3_vl',
  };
  return aliases[modelType?.toLowerCase()] ?? modelType?.toLowerCase();
}

const generated = [];

for (const override of overrides) {
  const apiData = await fetchJson(
    `https://huggingface.co/api/models/${encodeRepoPath(override.id)}`,
  );
  const revision = apiData.sha;
  if (!/^[0-9a-f]{40}$/i.test(revision ?? '')) {
    throw new Error(
      `Hugging Face API did not return an immutable commit SHA for ${override.id}`,
    );
  }
  let configData = {};
  let configurationSource = 'huggingface-config';
  let configurationWarning = null;
  try {
    configData =
      (await fetchJson(
        `https://huggingface.co/${encodeRepoPath(override.id)}/resolve/${revision}/config.json`,
        { optional: false },
      )) ?? {};
  } catch (error) {
    if (!apiData.gated) throw error;
    configurationSource = 'override-gated-config';
    configurationWarning =
      'Pinned config.json requires the model license; architecture fields use explicit override values until an authenticated refresh is available.';
  }
  const resolvedConfig = apiData.config ?? {};
  const safetensors = getSafetensorsParameters(apiData);
  const parameterCount = safetensors.total ?? override.parameterCount;
  const siblings = Array.isArray(apiData.siblings)
    ? apiData.siblings.map((sibling) => sibling.rfilename).filter(Boolean)
    : [];
  let indexData = null;
  if (siblings.includes('model.safetensors.index.json')) {
    try {
      indexData = await fetchJson(
        `https://huggingface.co/${encodeRepoPath(override.id)}/resolve/${revision}/model.safetensors.index.json`,
        { optional: true },
      );
    } catch (error) {
      if (!apiData.gated) throw error;
    }
  }
  const modelType =
    resolvedConfig.model_type ?? getConfigString(configData, ['model_type']);
  const modalityArchitecture = deriveModalityArchitecture(
    configData,
    modelType,
  );
  const kvCacheArchitecture = deriveKvCacheArchitecture(configData, modelType);
  const architectures =
    resolvedConfig.architectures?.length > 0
      ? resolvedConfig.architectures
      : getConfigArray(configData, ['architectures']);
  const numHeads =
    getConfigNumber(configData, [
      'num_attention_heads',
      'num_heads',
      'n_head',
    ]) ??
    override.numHeads ??
    null;
  const numKeyValueGroups = getConfigNumber(configData, [
    'num_key_value_groups',
  ]);
  const numKeyValueHeads =
    getConfigNumber(configData, [
      'num_key_value_heads',
      'num_kv_heads',
      'n_head_kv',
    ]) ??
    (numKeyValueGroups && numHeads
      ? Math.ceil(numHeads / numKeyValueGroups)
      : (override.numKeyValueHeads ?? null));
  const hiddenAct = getConfigString(configData, [
    'hidden_act',
    'hidden_activation',
    'activation_function',
  ]);

  generated.push({
    id: override.id,
    label: override.label,
    family: override.family,
    modelType: override.modelType,
    modality: override.modality,
    parameterCount,
    activeParameterCount: override.activeParameterCount ?? null,
    parameterSource: safetensors.total ? 'huggingface-safetensors' : 'override',
    reportedParameterCount: safetensors.total,
    configurationSource,
    configurationWarning,
    accessStatus: apiData.gated ? 'gated' : 'public',
    remoteCodeRequired: override.remoteCodeRequired ?? false,
    contextLength:
      getConfigNumber(configData, [
        'max_position_embeddings',
        'max_seq_len',
        'model_max_length',
        'max_length',
        'seq_length',
        'n_positions',
      ]) ?? override.contextLength,
    hiddenSize:
      getConfigNumber(configData, [
        'hidden_size',
        'd_model',
        'n_embd',
        'hidden_dim',
      ]) ??
      override.hiddenSize ??
      null,
    numLayers:
      getConfigNumber(configData, [
        'num_hidden_layers',
        'num_layers',
        'n_layer',
        'num_decoder_layers',
      ]) ??
      override.numLayers ??
      null,
    numHeads,
    numKeyValueHeads,
    headDim:
      getConfigNumber(configData, ['head_dim', 'attention_head_size']) ??
      override.headDim ??
      null,
    intermediateSize:
      getConfigNumber(configData, [
        'intermediate_size',
        'ffn_dim',
        'd_ff',
        'ffn_hidden_size',
      ]) ??
      override.intermediateSize ??
      null,
    expertIntermediateSize:
      getConfigNumber(configData, [
        'moe_intermediate_size',
        'expert_intermediate_size',
      ]) ??
      override.expertIntermediateSize ??
      null,
    sharedExpertIntermediateSize:
      getConfigNumber(configData, [
        'shared_expert_intermediate_size',
        'shared_mlp_intermediate_size',
      ]) ??
      override.sharedExpertIntermediateSize ??
      null,
    numSharedExperts:
      getConfigNumber(configData, ['n_shared_experts', 'num_shared_experts']) ??
      override.numSharedExperts ??
      null,
    isEncoderDecoder:
      getConfigBoolean(configData, ['is_encoder_decoder']) ?? null,
    vocabSize:
      getConfigNumber(configData, ['vocab_size', 'n_vocab']) ??
      override.vocabSize ??
      null,
    dtypeBits:
      dtypeToBits[getConfigString(configData, ['torch_dtype', 'dtype'])] ??
      override.dtypeBits ??
      null,
    numExperts:
      getConfigNumber(configData, [
        'num_local_experts',
        'num_experts',
        'n_routed_experts',
      ]) ??
      override.numExperts ??
      null,
    numExpertsPerToken:
      getConfigNumber(configData, [
        'num_experts_per_tok',
        'num_selected_experts',
        'top_k',
      ]) ??
      override.numExpertsPerToken ??
      null,
    gatedMlp:
      getConfigBoolean(configData, ['gated_mlp', 'gated_mlp_projections']) ??
      (hiddenAct ? /silu|swiglu|geglu|glu/i.test(hiddenAct) : null),
    tieWordEmbeddings:
      getConfigBoolean(configData, ['tie_word_embeddings', 'tie_embeddings']) ??
      null,
    pipelineTag: apiData.pipeline_tag ?? null,
    modelTypeTag: modelType,
    architectures,
    modalityArchitecture,
    kvCacheArchitecture,
    license: parseLicense(apiData.tags),
    tags: apiData.tags ?? [],
    revision,
    lastModified: apiData.lastModified ?? null,
    sourceCheckedAt: checkedAt,
    sourceUrls: {
      api: `https://huggingface.co/${override.id}`,
      config: `https://huggingface.co/${override.id}/blob/${revision}/config.json`,
      transformers: modelType
        ? `https://github.com/huggingface/transformers/tree/main/src/transformers/models/${transformersModelDirectory(modelType)}`
        : null,
    },
    safetensors: safetensors.total
      ? {
          total: safetensors.total,
          parameters: safetensors.byDtype,
          totalSizeBytes: indexData?.metadata?.total_size ?? null,
          files: siblings.filter((file) => file.endsWith('.safetensors')),
        }
      : null,
    engineSupport: override.engineSupport,
    summary: override.summary,
  });
}

generated.sort(comparePresets);
await fs.writeFile(outputPath, `${JSON.stringify(generated, null, 2)}\n`);
console.log(
  `Wrote ${generated.length} model presets to ${path.relative(repoRoot, outputPath)}`,
);
