import { fetchJson, HttpError } from './http';
import type { ModalityArchitecture } from './model-architecture';
import {
  estimateTransformerParameterBreakdown,
  type KvCacheArchitecture,
  type PrecisionBits,
  type TransformerParameterBreakdown,
} from '../estimator/estimator';

export type JsonRecord = Record<string, unknown>;

export type ParameterSource =
  | 'huggingface-safetensors'
  | 'huggingface-config'
  | 'estimated-from-architecture'
  | 'unknown';

export type EvidenceConfidence =
  | 'authoritative'
  | 'reported'
  | 'derived'
  | 'unavailable';

export interface ModelEvidence {
  label: string;
  kind: 'hub-api' | 'config' | 'weights' | 'transformers' | 'model-card';
  confidence: EvidenceConfidence;
  url: string;
  detail: string;
}

export interface QuantizationMetadata {
  method?: string;
  bits?: number;
  groupSize?: number;
  raw: JsonRecord;
}

export interface ModelArchitecture {
  modelType?: string;
  architectures: string[];
  hiddenSize?: number;
  numLayers?: number;
  numAttentionHeads?: number;
  numKeyValueHeads?: number;
  headDim?: number;
  intermediateSize?: number;
  expertIntermediateSize?: number;
  sharedExpertIntermediateSize?: number;
  numSharedExperts?: number;
  moeLayerFrequency?: number;
  numDenseLayers?: number;
  vocabSize?: number;
  contextLength?: number;
  numExperts?: number;
  numExpertsPerToken?: number;
  gatedMlp?: boolean;
  tieWordEmbeddings?: boolean;
  isEncoderDecoder?: boolean;
  attentionBias?: boolean;
  mlpBias?: boolean;
  hiddenActivation?: string;
  ropeScaling?: JsonRecord;
  dtype?: string;
  dtypeBits?: PrecisionBits;
  modality?: string;
  modalityArchitecture?: ModalityArchitecture;
  kvCacheArchitecture?: KvCacheArchitecture;
}

export interface ParsedModelConfig {
  architecture: ModelArchitecture;
  parameterCount?: number;
  activeParameterCount?: number;
  parameterSource: ParameterSource;
  quantization?: QuantizationMetadata;
  parameterBreakdown?: TransformerParameterBreakdown;
  warnings: string[];
  raw: JsonRecord;
}

export interface SafetensorsSummary {
  parameterCount?: number;
  parameterCountByDtype: Record<string, number>;
  totalSizeBytes?: number;
  files: string[];
  inspectedFiles: number;
  source: 'hub-api' | 'range-header' | 'unavailable';
  sharedTensorWarning: boolean;
}

export interface TransformersSourceFile {
  name: string;
  url: string;
  rawUrl?: string;
}

export interface TransformersCacheAnalysis {
  file: string;
  signals: string[];
  detail: string;
  confidence: EvidenceConfidence;
}

export interface TransformersSource {
  modelType: string;
  directoryUrl: string;
  files: TransformersSourceFile[];
  preview?: {
    name: string;
    url: string;
    content: string;
  };
  remoteCodeFiles: string[];
  transformersVersion?: string;
  cacheAnalysis?: TransformersCacheAnalysis;
}

export interface ModelInspection {
  id: string;
  revision: string;
  sha?: string;
  lastModified?: string;
  author?: string;
  pipelineTag?: string;
  libraryName?: string;
  modelType?: string;
  architectures: string[];
  tags: string[];
  license?: string;
  gated?: boolean | string;
  private?: boolean;
  parameterCount: number;
  activeParameterCount?: number;
  parameterSource: ParameterSource;
  parameterCountByDtype: Record<string, number>;
  parameterBreakdown?: TransformerParameterBreakdown;
  architecture: ModelArchitecture;
  quantization?: QuantizationMetadata;
  safetensors?: SafetensorsSummary;
  transformers?: TransformersSource;
  remoteCodeFiles: string[];
  files: string[];
  generationConfig?: JsonRecord;
  tokenizerConfig?: JsonRecord;
  cardExcerpt?: string;
  evidence: ModelEvidence[];
  warnings: string[];
  fetchedAt: string;
}

interface HubSibling {
  rfilename?: unknown;
  path?: unknown;
  size?: unknown;
}

interface GitHubSourceEntry {
  name?: unknown;
  html_url?: unknown;
  download_url?: unknown;
  type?: unknown;
}

const DIRECT_PARAMETER_KEYS = [
  'num_parameters',
  'number_of_parameters',
  'n_parameters',
  'n_params',
  'num_params',
  'total_params',
  'parameter_count',
];

const CONFIG_SECTIONS = [
  'text_config',
  'language_config',
  'llm_config',
  'model_config',
  'model',
] as const;

const MAX_HEADER_BYTES = 2 * 1024 * 1024;
const MAX_CARD_BYTES = 32 * 1024;
const MAX_SOURCE_PREVIEW_CHARS = 6000;
const MAX_SOURCE_ANALYSIS_CHARS = 160 * 1024;
const MAX_SAFE_TENSOR_FILES = 128;
const SAFE_TENSOR_CONCURRENCY = 8;
const MAX_RANGE_INSPECTION_BYTES = 16 * 1024 * 1024;
const INSPECTION_CACHE_TTL_MS = 10 * 60 * 1000;
const inspectionCache = new Map<
  string,
  { expiresAt: number; inspection: ModelInspection }
>();

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finitePositive(value: number | undefined): number | undefined {
  return value !== undefined && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

export function parseNumericValue(
  value: unknown,
  { allowUnits = true }: { allowUnits?: boolean } = {},
): number | undefined {
  if (typeof value === 'number') return finitePositive(value);
  if (typeof value !== 'string') return undefined;

  const trimmed = value.trim().replace(/,/g, '');
  if (!trimmed) return undefined;

  const match = trimmed.match(
    /^([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(k|m|b|t|thousand|million|billion|trillion)?(?:\s*parameters?)?$/i,
  );
  if (!match) return undefined;

  const number = Number(match[1]);
  if (!Number.isFinite(number) || number <= 0) return undefined;
  if (!allowUnits || !match[2]) return number;

  const multiplier: Record<string, number> = {
    k: 1e3,
    thousand: 1e3,
    m: 1e6,
    million: 1e6,
    b: 1e9,
    billion: 1e9,
    t: 1e12,
    trillion: 1e12,
  };
  return number * multiplier[match[2].toLowerCase()];
}

function recordsFromConfig(config: JsonRecord): JsonRecord[] {
  return [
    config,
    ...CONFIG_SECTIONS.flatMap((section) =>
      isRecord(config[section]) ? [config[section]] : [],
    ),
  ];
}

function firstValue(records: JsonRecord[], keys: readonly string[]): unknown {
  for (const record of records) {
    for (const key of keys) {
      if (record[key] !== undefined && record[key] !== null) {
        return record[key];
      }
    }
  }
  return undefined;
}

function firstNumber(
  records: JsonRecord[],
  keys: readonly string[],
): number | undefined {
  return parseNumericValue(firstValue(records, keys));
}

function firstString(
  records: JsonRecord[],
  keys: readonly string[],
): string | undefined {
  const value = firstValue(records, keys);
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function firstBoolean(
  records: JsonRecord[],
  keys: readonly string[],
): boolean | undefined {
  const value = firstValue(records, keys);
  return typeof value === 'boolean' ? value : undefined;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((entry): entry is string => typeof entry === 'string');
}

function findRemoteCodeIndicators(
  configs: JsonRecord[],
  files: string[],
): string[] {
  const indicators = new Set<string>();
  for (const config of configs) {
    for (const key of ['auto_map', 'custom_code', 'code_revision']) {
      if (config[key] !== undefined) indicators.add(key);
    }
  }
  for (const file of files) {
    if (file.endsWith('.py') || file.includes('custom_generate')) {
      indicators.add(file);
    }
  }
  return [...indicators];
}

function firstStringArray(
  records: JsonRecord[],
  keys: readonly string[],
): string[] {
  for (const record of records) {
    const values = stringArray(firstValue([record], keys));
    if (values.length) return values;
  }
  return [];
}

function parseModalityArchitecture(
  config: JsonRecord,
  modelType: string | undefined,
): ModalityArchitecture | undefined {
  const visionConfig = isRecord(config.vision_config)
    ? config.vision_config
    : undefined;
  const audioConfig = isRecord(config.audio_config)
    ? config.audio_config
    : undefined;
  const normalizedModelType = modelType?.toLowerCase() ?? '';
  const isUnified = normalizedModelType.includes('unified');
  const hasMediaConfig = Boolean(visionConfig || audioConfig);
  if (!hasMediaConfig && !isUnified) return undefined;

  const textConfig = isRecord(config.text_config)
    ? config.text_config
    : undefined;
  const textHiddenSize = textConfig
    ? firstNumber([textConfig], ['hidden_size', 'd_model'])
    : undefined;
  const patchSize = visionConfig
    ? firstNumber([visionConfig], ['patch_size'])
    : undefined;
  const poolingKernelSize = visionConfig
    ? firstNumber([visionConfig], ['pooling_kernel_size'])
    : undefined;
  const pooledPatchSize =
    (visionConfig &&
      (firstNumber([visionConfig], ['model_patch_size']) ??
        (patchSize && poolingKernelSize
          ? patchSize * poolingKernelSize
          : undefined))) ??
    undefined;

  return {
    family: modelType ?? 'multimodal',
    evidence: 'config',
    vision: visionConfig
      ? {
          encoderFree: isUnified,
          patchSize,
          pooledPatchSize,
          rawChannels:
            firstNumber([visionConfig], ['num_channels', 'in_channels']) ?? 3,
          embedDim: firstNumber(
            [visionConfig],
            ['mm_embed_dim', 'hidden_size'],
          ),
          outputDim: firstNumber(
            [visionConfig],
            ['output_proj_dims', 'projection_dim'],
          ),
          softTokens: firstNumber(
            [visionConfig],
            ['num_soft_tokens', 'num_image_tokens'],
          ),
        }
      : undefined,
    video: Boolean(config.video_token_id) || isUnified,
    audio: audioConfig
      ? {
          encoderFree: isUnified,
          featureDim: firstNumber(
            [audioConfig],
            ['audio_embed_dim', 'hidden_size', 'input_dim'],
          ),
          samplesPerToken: firstNumber(
            [audioConfig],
            ['audio_samples_per_token', 'samples_per_token'],
          ),
          outputDim:
            textHiddenSize ??
            firstNumber([audioConfig], ['output_proj_dims', 'hidden_size']),
        }
      : undefined,
  };
}

function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (entry): entry is number =>
      typeof entry === 'number' && Number.isFinite(entry),
  );
}

function countPattern(pattern: string, character: string): number {
  return [...pattern].filter((entry) => entry === character).length;
}

function parseKvCacheArchitecture(
  config: JsonRecord,
  modelType: string | undefined,
): KvCacheArchitecture | undefined {
  const records = recordsFromConfig(config);
  const normalizedModelType = modelType?.toLowerCase() ?? '';
  const numLayers = firstNumber(records, [
    'num_hidden_layers',
    'num_layers',
    'n_layer',
  ]);

  if (normalizedModelType === 'deepseek_v4') {
    const ratios = numberArray(config.compress_ratios);
    const layerTypes = stringArray(config.layer_types);
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
    const compressRates = isRecord(config.compress_rates)
      ? config.compress_rates
      : undefined;
    return {
      mode: 'deepseek-v4',
      keyValueShared: true,
      localAttentionLayers: layerCount,
      fullAttentionLayers: 0,
      slidingWindow: firstNumber(records, ['sliding_window']) ?? 128,
      compressedSparseLayers,
      heavilyCompressedLayers,
      compressedSparseRate:
        (compressRates
          ? firstNumber([compressRates], ['compressed_sparse_attention'])
          : undefined) ??
        firstNumber(records, ['compress_rate_csa']) ??
        4,
      heavilyCompressedRate:
        (compressRates
          ? firstNumber([compressRates], ['heavily_compressed_attention'])
          : undefined) ??
        firstNumber(records, ['compress_rate_hca']) ??
        128,
      indexHeadDim: firstNumber(records, ['index_head_dim']),
      indexerLayers: compressedSparseLayers,
      label:
        'DeepSeek V4 sliding + compressed sparse + heavily compressed cache',
    };
  }

  const textConfig = isRecord(config.text_config)
    ? config.text_config
    : undefined;
  const latentKvRank = firstNumber(records, ['kv_lora_rank']);
  const latentRopeDim = firstNumber(records, [
    'qk_rope_head_dim',
    'rope_head_dim',
  ]);
  const linearAttentionConfig =
    textConfig && isRecord(textConfig.linear_attn_config)
      ? textConfig.linear_attn_config
      : undefined;
  const fullAttentionLayerIds = linearAttentionConfig
    ? numberArray(linearAttentionConfig.full_attn_layers)
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
      recurrentStateLayers: recurrentStateLayers || undefined,
      latentKvRank,
      latentRopeDim,
      label:
        recurrentStateLayers > 0
          ? 'Hybrid latent attention + linear/state cache'
          : 'Multi-head latent attention compressed KV cache',
    };
  }
  const localLayerIds = textConfig
    ? numberArray(textConfig.local_layer_ids)
    : [];
  const textLayerCount = textConfig
    ? firstNumber([textConfig], ['num_hidden_layers', 'num_layers'])
    : undefined;
  if (textConfig && localLayerIds.length > 0 && textLayerCount) {
    return {
      mode: 'hybrid-sliding-window',
      keyValueShared: false,
      slidingWindow: firstNumber(
        [textConfig],
        ['sliding_window_size', 'sliding_window'],
      ),
      localAttentionLayers: localLayerIds.length,
      fullAttentionLayers: Math.max(0, textLayerCount - localLayerIds.length),
      label: 'Hybrid local/global attention cache',
    };
  }

  const genericLayerTypes = firstStringArray(records, [
    'layer_types',
    'layers_block_type',
  ]);
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
        slidingWindow: firstNumber(records, [
          'sliding_window',
          'sliding_window_size',
        ]),
        localAttentionLayers,
        fullAttentionLayers,
        noAttentionLayers:
          recurrentStateLayers > 0 ? noAttentionLayers : undefined,
        recurrentStateLayers:
          recurrentStateLayers > 0 ? recurrentStateLayers : undefined,
        label: 'Config-declared hybrid attention schedule',
      };
    }
  }

  const hybridPattern = firstString(records, ['hybrid_override_pattern']);
  if (hybridPattern && numLayers) {
    const fullAttentionLayers = countPattern(hybridPattern, '*');
    const recurrentStateLayers = countPattern(hybridPattern, 'M');
    const noAttentionLayers = Math.max(
      0,
      hybridPattern.length - fullAttentionLayers,
    );
    const mambaHeads = firstNumber(records, ['mamba_num_heads']);
    const mambaHeadDim = firstNumber(records, ['mamba_head_dim']);
    const stateSize = firstNumber(records, ['ssm_state_size']);
    const groups = firstNumber(records, ['n_groups', 'mamba_n_groups']);
    const convKernel = firstNumber(records, ['conv_kernel', 'mamba_d_conv']);
    const stateBits = parsePrecisionBits(
      firstString(records, ['mamba_ssm_cache_dtype']),
    );
    const mambaIntermediate =
      mambaHeads && mambaHeadDim ? mambaHeads * mambaHeadDim : 0;
    const convChannels =
      mambaIntermediate && groups && stateSize
        ? mambaIntermediate + 2 * groups * stateSize
        : 0;
    const recurrentStateBytesPerLayer =
      mambaIntermediate && stateSize && convChannels && convKernel
        ? (mambaIntermediate * stateSize + convChannels * convKernel) *
          ((stateBits ?? 32) / 8)
        : undefined;
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

  if (
    numLayers &&
    /mamba|rwkv|state[-_ ]space|recurrent/i.test(normalizedModelType)
  ) {
    return {
      mode: 'hybrid-state-space',
      keyValueShared: false,
      fullAttentionLayers: 0,
      noAttentionLayers: numLayers,
      recurrentStateLayers: numLayers,
      label: 'State-space model with no conventional KV attention cache',
    };
  }

  const slidingWindow = firstNumber(records, [
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
      label: `Sliding-window cache capped at ${slidingWindow.toLocaleString('en-US')} tokens`,
    };
  }

  return undefined;
}

function parsePrecisionBits(value: unknown): PrecisionBits | undefined {
  if (typeof value === 'number') {
    if (value === 4 || value === 8 || value === 16 || value === 32) {
      return value;
    }
    return undefined;
  }

  if (typeof value !== 'string') return undefined;
  const normalized = value.toLowerCase();
  if (normalized.includes('float32') || normalized === 'fp32') return 32;
  if (
    normalized.includes('float16') ||
    normalized.includes('bfloat16') ||
    normalized === 'fp16' ||
    normalized === 'bf16'
  ) {
    return 16;
  }
  if (
    normalized.includes('float8') ||
    normalized.includes('int8') ||
    normalized === 'fp8' ||
    normalized === 'int8'
  ) {
    return 8;
  }
  if (
    normalized.includes('int4') ||
    normalized.includes('nf4') ||
    normalized.includes('fp4')
  ) {
    return 4;
  }
  return undefined;
}

function parseQuantization(
  records: JsonRecord[],
): QuantizationMetadata | undefined {
  const raw = firstValue(records, ['quantization_config']);
  if (!isRecord(raw)) return undefined;

  return {
    method: firstString([raw], ['quant_method', 'method', 'quant_type']),
    bits: firstNumber([raw], ['bits', 'bit_width', 'wbits']),
    groupSize: firstNumber([raw], ['group_size', 'groupsize']),
    raw,
  };
}

function parseDirectParameterCount(records: JsonRecord[]): number | undefined {
  return firstNumber(records, DIRECT_PARAMETER_KEYS);
}

function parseActiveParameterCount(records: JsonRecord[]): number | undefined {
  return firstNumber(records, [
    'active_parameters',
    'active_parameter_count',
    'num_active_parameters',
    'active_params',
  ]);
}

function parseArchitecture(config: JsonRecord): ModelArchitecture {
  const records = recordsFromConfig(config);
  const modelType = firstString(records, ['model_type', 'modelType']);
  const modalityArchitecture = parseModalityArchitecture(config, modelType);
  const kvCacheArchitecture = parseKvCacheArchitecture(config, modelType);
  const architectures = firstStringArray(records, [
    'architectures',
    'architecture',
  ]);
  const hiddenSize = firstNumber(records, [
    'hidden_size',
    'd_model',
    'n_embd',
    'hidden_dim',
    'model_dim',
  ]);
  const numLayers = firstNumber(records, [
    'num_hidden_layers',
    'num_layers',
    'n_layer',
    'num_decoder_layers',
    'decoder_layers',
  ]);
  const numAttentionHeads = firstNumber(records, [
    'num_attention_heads',
    'num_heads',
    'n_head',
    'n_head_q',
    'num_q_heads',
  ]);
  const numKeyValueHeads = firstNumber(records, [
    'num_key_value_heads',
    'num_kv_heads',
    'num_kv_head',
    'n_head_kv',
  ]);
  const resolvedNumKeyValueHeads =
    numKeyValueHeads ??
    (firstNumber(records, ['num_key_value_groups']) && numAttentionHeads
      ? Math.ceil(
          numAttentionHeads / firstNumber(records, ['num_key_value_groups'])!,
        )
      : undefined);
  const intermediateSize = firstNumber(records, [
    'intermediate_size',
    'ffn_dim',
    'd_ff',
    'ffn_hidden_size',
  ]);
  const expertIntermediateSize = firstNumber(records, [
    'moe_intermediate_size',
    'expert_intermediate_size',
  ]);
  const sharedExpertIntermediateSize = firstNumber(records, [
    'shared_expert_intermediate_size',
    'shared_mlp_intermediate_size',
  ]);
  const numSharedExperts = firstNumber(records, [
    'n_shared_experts',
    'num_shared_experts',
  ]);
  const moeLayerFrequency = firstNumber(records, [
    'moe_layer_freq',
    'decoder_sparse_step',
  ]);
  const numDenseLayers = firstNumber(records, [
    'num_dense_layers',
    'num_dense_mlp_layers',
  ]);
  const vocabSize = firstNumber(records, ['vocab_size', 'n_vocab']);
  const contextLength = firstNumber(records, [
    'max_position_embeddings',
    'max_sequence_length',
    'max_seq_len',
    'model_max_length',
    'max_length',
    'seq_length',
    'n_positions',
  ]);
  const numExperts = firstNumber(records, [
    'num_local_experts',
    'num_experts',
    'n_routed_experts',
    'n_experts',
  ]);
  const numExpertsPerToken = firstNumber(records, [
    'num_experts_per_tok',
    'num_selected_experts',
    'num_experts_per_token',
    'top_k',
    'num_experts_per_token',
  ]);
  const hiddenActivation = firstString(records, [
    'hidden_act',
    'hidden_activation',
    'activation_function',
  ]);
  const dtype = firstString(records, ['torch_dtype', 'dtype', 'weight_dtype']);
  const quantization = parseQuantization(records);
  const quantizationBits = parsePrecisionBits(quantization?.bits);
  const dtypeBits = quantizationBits ?? parsePrecisionBits(dtype);
  const gatedMlp =
    firstBoolean(records, ['gated_mlp', 'gated_mlp_projections']) ??
    Boolean(
      hiddenActivation && /silu|swiglu|geglu|glu/i.test(hiddenActivation),
    );
  const tieWordEmbeddings = firstBoolean(records, [
    'tie_word_embeddings',
    'tie_embeddings',
  ]);
  const attentionBias = firstBoolean(records, [
    'attention_bias',
    'qkv_bias',
    'use_bias',
  ]);
  const mlpBias = firstBoolean(records, ['mlp_bias', 'ffn_bias']);
  const isEncoderDecoder = firstBoolean(records, [
    'is_encoder_decoder',
    'encoder_decoder',
  ]);
  const ropeScaling = firstValue(records, ['rope_scaling']);

  return {
    modelType,
    architectures,
    hiddenSize,
    numLayers,
    numAttentionHeads,
    numKeyValueHeads: resolvedNumKeyValueHeads,
    headDim: firstNumber(records, ['head_dim', 'attention_head_size']),
    intermediateSize,
    expertIntermediateSize,
    sharedExpertIntermediateSize,
    numSharedExperts,
    moeLayerFrequency,
    numDenseLayers,
    vocabSize,
    contextLength,
    numExperts,
    numExpertsPerToken,
    gatedMlp,
    tieWordEmbeddings,
    isEncoderDecoder,
    attentionBias,
    mlpBias,
    hiddenActivation,
    ropeScaling: isRecord(ropeScaling) ? ropeScaling : undefined,
    dtype,
    dtypeBits,
    modality: modalityArchitecture ? 'multimodal' : undefined,
    modalityArchitecture,
    kvCacheArchitecture,
  };
}

export function parseModelConfig(
  config: JsonRecord,
  additionalSources: JsonRecord[] = [],
): ParsedModelConfig {
  const records = [
    ...additionalSources.flatMap((source) =>
      isRecord(source) ? recordsFromConfig(source) : [],
    ),
    ...recordsFromConfig(config),
  ];
  const architecture = parseArchitecture(config);
  const explicitParameterCount = parseDirectParameterCount(records);
  const activeParameterCount = parseActiveParameterCount(records);
  const canEstimate = Boolean(
    architecture.vocabSize &&
    architecture.hiddenSize &&
    architecture.numLayers &&
    architecture.numAttentionHeads,
  );
  const isMoE = Boolean(architecture.numExperts && architecture.numExperts > 1);
  const parameterBreakdown =
    canEstimate && !isMoE
      ? estimateTransformerParameterBreakdown({
          vocabSize: architecture.vocabSize!,
          hiddenSize: architecture.hiddenSize!,
          numLayers: architecture.numLayers!,
          numAttentionHeads: architecture.numAttentionHeads!,
          numKeyValueHeads: architecture.numKeyValueHeads,
          headDim: architecture.headDim,
          intermediateSize: architecture.intermediateSize,
          numExperts: architecture.numExperts,
          numExpertsPerToken: architecture.numExpertsPerToken,
          gatedMlp: architecture.gatedMlp,
          tieWordEmbeddings: architecture.tieWordEmbeddings,
          attentionBias: architecture.attentionBias,
          mlpBias: architecture.mlpBias,
        })
      : undefined;
  const estimatedParameterCount = parameterBreakdown?.totalParameters;
  const parameterCount = explicitParameterCount ?? estimatedParameterCount;
  const warnings: string[] = [];

  if (!explicitParameterCount && estimatedParameterCount) {
    warnings.push(
      'Parameter count is derived from the normalized architecture; verify it against the Hub safetensors metadata when available.',
    );
  }
  if (isMoE) {
    warnings.push(
      'MoE parameter composition is withheld until expert width, shared experts, and the sparse-layer schedule are all modeled; use the Hub safetensors total instead.',
    );
  }
  if (
    architecture.numKeyValueHeads &&
    architecture.numAttentionHeads &&
    architecture.numKeyValueHeads < architecture.numAttentionHeads
  ) {
    warnings.push(
      'Grouped-query or multi-query attention was detected; KV cache uses the smaller key/value head count.',
    );
  }
  if (architecture.numExperts && architecture.numExperts > 1) {
    warnings.push(
      `Mixture-of-experts architecture detected (${architecture.numExperts} experts); total and active parameters are different quantities.`,
    );
  }

  return {
    architecture,
    parameterCount,
    activeParameterCount,
    parameterSource: explicitParameterCount
      ? 'huggingface-config'
      : estimatedParameterCount
        ? 'estimated-from-architecture'
        : 'unknown',
    quantization: parseQuantization(records),
    parameterBreakdown,
    warnings,
    raw: config,
  };
}

export function normalizeModelId(input: string): string {
  const trimmed = input.trim();
  if (!trimmed) throw new Error('A Hugging Face model ID is required.');

  let path = trimmed;
  try {
    if (/^https?:\/\//i.test(trimmed)) {
      const url = new URL(trimmed);
      if (!/huggingface\.co$/i.test(url.hostname)) {
        throw new Error('Only huggingface.co model URLs are supported.');
      }
      path = url.pathname;
    }
  } catch (error) {
    if (error instanceof Error && error.message.includes('Only')) throw error;
  }

  const parts = path
    .replace(/^\/+|\/+$/g, '')
    .split('/')
    .filter(Boolean);
  if (parts[0] === 'models') parts.shift();
  const modelParts = parts.slice(0, 2);
  if (modelParts.length !== 2) {
    throw new Error('Use a Hugging Face model ID such as org/model.');
  }
  return modelParts.map((part) => decodeURIComponent(part)).join('/');
}

function encodeRepoPath(value: string): string {
  return value.split('/').map(encodeURIComponent).join('/');
}

function encodeFilePath(value: string): string {
  return value.split('/').map(encodeURIComponent).join('/');
}

function hubApiUrl(modelId: string): string {
  return `https://huggingface.co/api/models/${encodeRepoPath(modelId)}`;
}

function hubFileUrl(
  modelId: string,
  revision: string,
  filePath: string,
): string {
  return `https://huggingface.co/${encodeRepoPath(modelId)}/resolve/${encodeURIComponent(revision)}/${encodeFilePath(filePath)}`;
}

function siblingPath(sibling: HubSibling): string | undefined {
  const value = sibling.rfilename ?? sibling.path;
  return typeof value === 'string' ? value : undefined;
}

function stringField(record: JsonRecord, key: string): string | undefined {
  return typeof record[key] === 'string' ? (record[key] as string) : undefined;
}

function apiTags(api: JsonRecord): string[] {
  return stringArray(api.tags);
}

function parseLicense(api: JsonRecord): string | undefined {
  const cardData = isRecord(api.cardData) ? api.cardData : undefined;
  const license =
    stringField(cardData ?? {}, 'license') ??
    apiTags(api)
      .find((tag) => tag.startsWith('license:'))
      ?.slice(8);
  return license || undefined;
}

function apiParameterCounts(api: JsonRecord): {
  total?: number;
  byDtype: Record<string, number>;
} {
  const safetensors = isRecord(api.safetensors) ? api.safetensors : {};
  const rawParameters = isRecord(safetensors.parameters)
    ? safetensors.parameters
    : {};
  const byDtype: Record<string, number> = {};
  for (const [dtype, value] of Object.entries(rawParameters)) {
    const count = parseNumericValue(value);
    if (count) byDtype[dtype] = count;
  }
  const explicitTotal = parseNumericValue(safetensors.total);
  const total =
    explicitTotal ?? Object.values(byDtype).reduce((sum, n) => sum + n, 0);
  return { total: total > 0 ? total : undefined, byDtype };
}

async function fetchText(
  url: string,
  {
    timeoutMs = 10000,
    maxBytes,
    range,
  }: { timeoutMs?: number; maxBytes?: number; range?: string } = {},
): Promise<string> {
  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: range ? { Range: range } : undefined,
    });
    if (!response.ok) {
      throw new HttpError(
        `Request failed with status ${response.status}`,
        response.status,
        url,
      );
    }
    const contentLength = Number(response.headers.get('content-length'));
    if (maxBytes && contentLength > maxBytes) {
      throw new Error(`Response exceeds the ${maxBytes}-byte safety limit.`);
    }
    const text = await response.text();
    if (maxBytes && text.length > maxBytes) return text.slice(0, maxBytes);
    return text;
  } finally {
    globalThis.clearTimeout(timeoutId);
  }
}

async function fetchOptionalJson(
  url: string,
  timeoutMs = 10000,
): Promise<JsonRecord | null> {
  try {
    return await fetchJson<JsonRecord>(url, { timeoutMs });
  } catch (error) {
    if (error instanceof HttpError && error.status === 404) return null;
    throw error;
  }
}

interface RangeInspectionBudget {
  consumedBytes: number;
  maxBytes: number;
}

async function fetchRange(
  url: string,
  start: number,
  end: number,
  budget?: RangeInspectionBudget,
): Promise<Uint8Array> {
  const expectedLength = end - start + 1;
  if (expectedLength <= 0) throw new Error('Invalid byte range.');
  if (budget) {
    if (budget.consumedBytes + expectedLength > budget.maxBytes) {
      throw new Error('Aggregate safetensors inspection budget exceeded.');
    }
    budget.consumedBytes += expectedLength;
  }
  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { Range: `bytes=${start}-${end}` },
    });
    if (!response.ok) {
      throw new HttpError(
        `Range request failed with status ${response.status}`,
        response.status,
        url,
      );
    }
    if (response.status !== 206) {
      throw new Error('The host did not honor the safe range request.');
    }
    const contentLength = Number(response.headers.get('content-length'));
    const contentRange = response.headers.get('content-range') ?? '';
    const rangeMatch = contentRange.match(/^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i);
    if (
      contentLength !== expectedLength ||
      !rangeMatch ||
      Number(rangeMatch[1]) !== start ||
      Number(rangeMatch[2]) !== end
    ) {
      throw new Error('The host returned an unverifiable byte range.');
    }
    const body = new Uint8Array(await response.arrayBuffer());
    if (body.byteLength !== expectedLength) {
      throw new Error('The host returned an incomplete byte range.');
    }
    return body;
  } finally {
    globalThis.clearTimeout(timeoutId);
  }
}

function product(values: unknown): number {
  if (!Array.isArray(values) || values.length === 0) return 0;
  return values.reduce<number>((total, value) => {
    const dimension = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(dimension) && dimension > 0 ? total * dimension : 0;
  }, 1);
}

async function mapWithConcurrency<T, R>(
  values: T[],
  concurrency: number,
  mapper: (value: T) => Promise<R>,
): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(values.length);
  let nextIndex = 0;
  const worker = async () => {
    while (nextIndex < values.length) {
      const index = nextIndex;
      nextIndex += 1;
      try {
        results[index] = {
          status: 'fulfilled',
          value: await mapper(values[index]),
        };
      } catch (reason) {
        results[index] = { status: 'rejected', reason };
      }
    }
  };
  await Promise.all(
    Array.from(
      { length: Math.min(Math.max(1, concurrency), values.length) },
      () => worker(),
    ),
  );
  return results;
}

async function parseSafetensorsFileHeaderWithBudget(
  url: string,
  budget: RangeInspectionBudget,
): Promise<{
  parameterCount: number;
  parameterCountByDtype: Record<string, number>;
}> {
  const prefix = await fetchRange(url, 0, 7, budget);
  if (prefix.byteLength < 8)
    throw new Error('Invalid safetensors header prefix.');
  const headerLength = Number(
    new DataView(prefix.buffer, prefix.byteOffset, 8).getBigUint64(0, true),
  );
  if (
    !Number.isSafeInteger(headerLength) ||
    headerLength <= 0 ||
    headerLength > MAX_HEADER_BYTES
  ) {
    throw new Error('Safetensors header is outside the safe inspection limit.');
  }
  const headerBytes = await fetchRange(url, 8, 7 + headerLength, budget);
  const header = JSON.parse(
    new TextDecoder().decode(headerBytes),
  ) as JsonRecord;
  const parameterCountByDtype: Record<string, number> = {};
  for (const [name, tensor] of Object.entries(header)) {
    if (name === '__metadata__' || !isRecord(tensor)) continue;
    const count = product(tensor.shape);
    const dtype = typeof tensor.dtype === 'string' ? tensor.dtype : 'unknown';
    if (count > 0)
      parameterCountByDtype[dtype] =
        (parameterCountByDtype[dtype] ?? 0) + count;
  }
  return {
    parameterCount: Object.values(parameterCountByDtype).reduce(
      (sum, value) => sum + value,
      0,
    ),
    parameterCountByDtype,
  };
}

export async function parseSafetensorsFileHeader(url: string): Promise<{
  parameterCount: number;
  parameterCountByDtype: Record<string, number>;
}> {
  return parseSafetensorsFileHeaderWithBudget(url, {
    consumedBytes: 0,
    maxBytes: MAX_HEADER_BYTES * 2,
  });
}

async function inspectSafetensors(
  modelId: string,
  revision: string,
  api: JsonRecord,
  files: string[],
  index: JsonRecord | null,
  warnings: string[],
): Promise<SafetensorsSummary | undefined> {
  const apiCounts = apiParameterCounts(api);
  const repositorySafetensorsFiles = files.filter((file) =>
    file.endsWith('.safetensors'),
  );
  const weightMap = isRecord(index?.weight_map) ? index.weight_map : undefined;
  const indexedSafetensorsFiles = weightMap
    ? [
        ...new Set(
          Object.values(weightMap).filter(
            (file): file is string => typeof file === 'string',
          ),
        ),
      ]
    : [];
  const safetensorsFiles = indexedSafetensorsFiles.length
    ? indexedSafetensorsFiles
    : repositorySafetensorsFiles;
  if (!apiCounts.total && !safetensorsFiles.length) return undefined;

  if (
    !apiCounts.total &&
    !indexedSafetensorsFiles.length &&
    safetensorsFiles.length > 1
  ) {
    warnings.push(
      'Multiple safetensors layouts were found without a weight index; parameter totals were withheld to avoid summing duplicate or alternate checkpoints.',
    );
    return {
      parameterCountByDtype: {},
      files: safetensorsFiles,
      inspectedFiles: 0,
      source: 'unavailable',
      sharedTensorWarning: true,
    };
  }

  const indexMetadata = isRecord(index?.metadata) ? index.metadata : undefined;
  if (apiCounts.total) {
    return {
      parameterCount: apiCounts.total,
      parameterCountByDtype: apiCounts.byDtype,
      totalSizeBytes: parseNumericValue(indexMetadata?.total_size),
      files: safetensorsFiles,
      inspectedFiles: 0,
      source: 'hub-api',
      sharedTensorWarning: false,
    };
  }

  if (safetensorsFiles.length > MAX_SAFE_TENSOR_FILES) {
    warnings.push(
      `Found ${safetensorsFiles.length} indexed safetensors shards; range inspection was capped at ${MAX_SAFE_TENSOR_FILES}.`,
    );
    return {
      parameterCountByDtype: {},
      files: safetensorsFiles,
      inspectedFiles: 0,
      source: 'unavailable',
      sharedTensorWarning: true,
    };
  }
  const filesToInspect = safetensorsFiles;
  const budget: RangeInspectionBudget = {
    consumedBytes: 0,
    maxBytes: MAX_RANGE_INSPECTION_BYTES,
  };
  const results = await mapWithConcurrency(
    filesToInspect,
    SAFE_TENSOR_CONCURRENCY,
    async (file) =>
      parseSafetensorsFileHeaderWithBudget(
        hubFileUrl(modelId, revision, file),
        budget,
      ),
  );
  const byDtype: Record<string, number> = {};
  let total = 0;
  let inspectedFiles = 0;
  for (const result of results) {
    if (result.status === 'fulfilled') {
      inspectedFiles += 1;
      total += result.value.parameterCount;
      for (const [dtype, count] of Object.entries(
        result.value.parameterCountByDtype,
      )) {
        byDtype[dtype] = (byDtype[dtype] ?? 0) + count;
      }
    }
  }
  if (inspectedFiles !== filesToInspect.length) {
    warnings.push(
      'Some safetensors headers could not be inspected from the browser.',
    );
    return {
      parameterCountByDtype: byDtype,
      files: safetensorsFiles,
      inspectedFiles,
      source: 'unavailable',
      sharedTensorWarning: true,
    };
  }
  if (!total) return undefined;
  warnings.push(
    'Safetensors header inspection counts serialized tensor elements; tied/shared tensors may make this differ from unique trainable parameters.',
  );
  return {
    parameterCount: total,
    parameterCountByDtype: byDtype,
    totalSizeBytes: parseNumericValue(indexMetadata?.total_size),
    files: safetensorsFiles,
    inspectedFiles,
    source: 'range-header',
    sharedTensorWarning: true,
  };
}

function modelCardExcerpt(text: string): string {
  return text.slice(0, MAX_CARD_BYTES).trim();
}

function sourceDirectoryUrl(modelType: string): string {
  const directory = transformersModelDirectory(modelType);
  return `https://github.com/huggingface/transformers/tree/main/src/transformers/models/${encodeURIComponent(directory)}`;
}

function transformersModelDirectory(modelType: string): string {
  const aliases: Record<string, string> = {
    gemma3_text: 'gemma3',
    gemma4_unified: 'gemma4_unified',
    qwen3_5_moe_text: 'qwen3_5',
    minimax_m3_vl: 'minimax_m3_vl',
  };
  return aliases[modelType.toLowerCase()] ?? modelType.toLowerCase();
}

function analyzeTransformersCacheSource(
  file: string,
  source: string,
): TransformersCacheAnalysis | undefined {
  const signals: string[] = [];
  if (/key_states\s*=\s*value_states|values\s*=\s*keys/i.test(source)) {
    signals.push('source shares K and V storage');
  }
  if (/SlidingWindow|sliding_window|DynamicSlidingWindowLayer/i.test(source)) {
    signals.push('source contains sliding-window cache support');
  }
  if (/compress(?:or|ed)|latent.?cache|compressed.?attention/i.test(source)) {
    signals.push('source contains compressed or latent cache paths');
  }
  if (/Mamba|SSM|state.?space|recurrent_state|conv_states/i.test(source)) {
    signals.push('source contains recurrent/state-space cache paths');
  }
  if (/past_key_values\.update|DynamicCache|Cache\(/i.test(source)) {
    signals.push('source updates a Transformers Cache implementation');
  }
  if (!signals.length) return undefined;
  return {
    file,
    signals,
    detail:
      'Static source inspection found cache-related implementation signals. Numeric layer schedules still come from the pinned config when available.',
    confidence: 'derived',
  };
}

async function fetchTransformersSource(
  modelType: string | undefined,
  config: JsonRecord,
  warnings: string[],
  remoteCodeFiles: string[],
): Promise<TransformersSource | undefined> {
  if (!modelType || !/^[a-z0-9_-]+$/i.test(modelType)) return undefined;
  const normalizedModelType = transformersModelDirectory(modelType);
  const directoryUrl = sourceDirectoryUrl(normalizedModelType);
  const apiUrl = `https://api.github.com/repos/huggingface/transformers/contents/src/transformers/models/${encodeURIComponent(normalizedModelType)}?ref=main`;
  let entries: GitHubSourceEntry[] = [];
  try {
    const response = await fetchJson<GitHubSourceEntry[]>(apiUrl, {
      timeoutMs: 10000,
      init: { headers: { Accept: 'application/vnd.github+json' } },
    });
    entries = Array.isArray(response) ? response : [];
  } catch {
    warnings.push(
      `Transformers source directory lookup was unavailable; the upstream source link remains available for ${modelType}.`,
    );
  }

  const files = entries
    .filter(
      (entry) =>
        entry.type === 'file' &&
        typeof entry.name === 'string' &&
        typeof entry.html_url === 'string',
    )
    .map((entry) => ({
      name: entry.name as string,
      url: entry.html_url as string,
      rawUrl:
        typeof entry.download_url === 'string' ? entry.download_url : undefined,
    }));
  const candidate =
    files.find((file) =>
      file.name.startsWith(`modular_${normalizedModelType}`),
    ) ??
    files.find((file) =>
      file.name.startsWith(`modeling_${normalizedModelType}`),
    ) ??
    files.find((file) => file.name.startsWith('configuration_'));
  let preview: TransformersSource['preview'];
  let cacheAnalysis: TransformersSource['cacheAnalysis'];
  if (candidate?.rawUrl) {
    try {
      const source = await fetchText(candidate.rawUrl, {
        maxBytes: MAX_SOURCE_ANALYSIS_CHARS,
      });
      preview = {
        name: candidate.name,
        url: candidate.url,
        content: source.slice(0, MAX_SOURCE_PREVIEW_CHARS),
      };
      cacheAnalysis = analyzeTransformersCacheSource(candidate.name, source);
    } catch {
      warnings.push(
        `Could not load a preview of ${candidate.name} from Transformers.`,
      );
    }
  }

  const transformersVersion = firstString([config], ['transformers_version']);
  if (transformersVersion) {
    warnings.push(
      `Checkpoint metadata reports Transformers ${transformersVersion}; the preview follows the current upstream main branch and should be version-checked before execution.`,
    );
  }
  return {
    modelType: normalizedModelType,
    directoryUrl,
    files,
    preview,
    remoteCodeFiles,
    transformersVersion,
    cacheAnalysis,
  };
}

function mergeSourceCacheAnalysis(
  architecture: KvCacheArchitecture | undefined,
  analysis: TransformersCacheAnalysis | undefined,
): KvCacheArchitecture | undefined {
  if (!analysis) return architecture;
  const sharedStorage = analysis.signals.some((signal) =>
    signal.includes('shares K and V'),
  );
  return {
    ...(architecture ?? {}),
    mode: architecture?.mode ?? 'source-derived',
    keyValueShared:
      architecture?.keyValueShared ?? (sharedStorage ? true : undefined),
    label:
      architecture?.label ??
      'Source-derived cache signals; numeric layer schedule was not exposed by config',
    confidence: architecture?.mode ? 'config+source' : 'source-derived',
    sourceSignals: analysis.signals,
  };
}

function inspectionError(error: unknown, modelId: string): Error {
  if (error instanceof HttpError) {
    if (error.status === 401 || error.status === 403) {
      return new Error(
        `${modelId} is gated or private. Public browser inspection cannot authenticate without exposing a user token.`,
      );
    }
    if (error.status === 404) {
      return new Error(`${modelId} was not found on the Hugging Face Hub.`);
    }
  }
  return error instanceof Error
    ? error
    : new Error('Unable to retrieve Hugging Face model metadata.');
}

export async function inspectHuggingFaceModel(
  input: string,
): Promise<ModelInspection> {
  const id = normalizeModelId(input);
  const cached = inspectionCache.get(id);
  if (cached && cached.expiresAt > Date.now()) return cached.inspection;
  inspectionCache.delete(id);
  const apiUrl = hubApiUrl(id);
  let api: JsonRecord;
  try {
    api = await fetchJson<JsonRecord>(apiUrl, { timeoutMs: 12000 });
  } catch (error) {
    throw inspectionError(error, id);
  }

  const sha = stringField(api, 'sha');
  if (!sha || !/^[0-9a-f]{40}$/i.test(sha)) {
    throw new Error(
      'Hugging Face did not return an immutable commit SHA; inspection was stopped instead of reading mutable main.',
    );
  }
  const revision = sha;
  const siblings = Array.isArray(api.siblings)
    ? (api.siblings as unknown[]).filter(isRecord)
    : [];
  const files = siblings
    .map((sibling) => siblingPath(sibling))
    .filter((path): path is string => Boolean(path));
  const fileSet = new Set(files);
  const warnings: string[] = [];
  const configUrl = hubFileUrl(id, revision, 'config.json');
  let config: JsonRecord = {};
  try {
    config =
      (await fetchJson<JsonRecord>(configUrl, { timeoutMs: 12000 })) ?? {};
  } catch (error) {
    throw new Error(
      `Unable to read pinned config.json for ${id}: ${inspectionError(error, id).message}`,
    );
  }

  const optionalArtifacts = await Promise.all([
    fileSet.has('generation_config.json')
      ? fetchOptionalJson(hubFileUrl(id, revision, 'generation_config.json'))
      : Promise.resolve(null),
    fileSet.has('tokenizer_config.json')
      ? fetchOptionalJson(hubFileUrl(id, revision, 'tokenizer_config.json'))
      : Promise.resolve(null),
    fileSet.has('model.safetensors.index.json')
      ? fetchOptionalJson(
          hubFileUrl(id, revision, 'model.safetensors.index.json'),
        )
      : Promise.resolve(null),
  ]).catch((error) => {
    warnings.push(
      `Optional Hub metadata fetch failed: ${inspectionError(error, id).message}`,
    );
    return [null, null, null] as const;
  });
  const [generationConfig, tokenizerConfig, safetensorsIndex] =
    optionalArtifacts;

  let cardExcerpt: string | undefined;
  if (fileSet.has('README.md')) {
    try {
      cardExcerpt = modelCardExcerpt(
        await fetchText(hubFileUrl(id, revision, 'README.md'), {
          maxBytes: MAX_CARD_BYTES,
          range: `bytes=0-${MAX_CARD_BYTES - 1}`,
        }),
      );
    } catch {
      warnings.push('Model card could not be loaded from the pinned revision.');
    }
  }

  const apiConfig = isRecord(api.config) ? api.config : {};
  const parsed = parseModelConfig(config, [apiConfig, api]);
  const remoteCodeFiles = findRemoteCodeIndicators(
    [config, apiConfig, api],
    files,
  );
  if (remoteCodeFiles.length > 0) {
    warnings.push(
      'This repository advertises custom or executable model files; they are linked for review but never executed by the app.',
    );
  }
  const safetensors = await inspectSafetensors(
    id,
    revision,
    api,
    files,
    safetensorsIndex,
    warnings,
  );
  const parameterCount =
    safetensors?.parameterCount ?? parsed.parameterCount ?? 0;
  const parameterSource: ParameterSource = safetensors?.parameterCount
    ? 'huggingface-safetensors'
    : parsed.parameterSource;
  const parameterBreakdown = parsed.parameterBreakdown;
  const modelType =
    parsed.architecture.modelType ??
    stringField(apiConfig, 'model_type') ??
    stringField(api, 'model_type');
  const transformers = await fetchTransformersSource(
    modelType,
    config,
    warnings,
    remoteCodeFiles,
  );
  const architecture: ModelArchitecture = {
    ...parsed.architecture,
    kvCacheArchitecture: mergeSourceCacheAnalysis(
      parsed.architecture.kvCacheArchitecture,
      transformers?.cacheAnalysis,
    ),
  };
  const evidence: ModelEvidence[] = [
    {
      label: 'Hugging Face model API',
      kind: 'hub-api',
      confidence: 'authoritative',
      url: `${apiUrl}?revision=${encodeURIComponent(revision)}`,
      detail: `Resolved revision ${revision}; model metadata, file inventory, tags, and Hub-reported weights.`,
    },
    {
      label: 'Pinned config.json',
      kind: 'config',
      confidence: Object.keys(config).length ? 'reported' : 'unavailable',
      url: configUrl,
      detail:
        'Architecture fields are model-defined JSON and are normalized without executing model code.',
    },
  ];
  if (safetensors) {
    evidence.push({
      label: 'Safetensors metadata',
      kind: 'weights',
      confidence:
        safetensors.source === 'hub-api' ? 'authoritative' : 'derived',
      url: safetensors.files[0]
        ? hubFileUrl(id, revision, safetensors.files[0])
        : apiUrl,
      detail:
        safetensors.source === 'hub-api'
          ? 'Parameter totals reported by the Hub from safetensors metadata.'
          : 'Parameter totals derived from safetensors headers using bounded HTTP range requests.',
    });
  }
  if (transformers) {
    evidence.push({
      label: 'Hugging Face Transformers implementation',
      kind: 'transformers',
      confidence: 'reported',
      url: transformers.directoryUrl,
      detail:
        'Read-only upstream source links/preview; no remote Python is imported or executed.',
    });
  }
  if (cardExcerpt) {
    evidence.push({
      label: 'Pinned model card',
      kind: 'model-card',
      confidence: 'reported',
      url: hubFileUrl(id, revision, 'README.md'),
      detail: 'Untrusted Markdown excerpt shown as text only.',
    });
  }
  if (!parameterCount) {
    warnings.push(
      'No authoritative parameter total was available for this repository.',
    );
  }

  const apiCounts = apiParameterCounts(api);
  const inspection: ModelInspection = {
    id,
    revision,
    sha,
    lastModified: stringField(api, 'lastModified'),
    author: stringField(api, 'author'),
    pipelineTag: stringField(api, 'pipeline_tag'),
    libraryName: stringField(api, 'library_name'),
    modelType,
    architectures: parsed.architecture.architectures,
    tags: apiTags(api),
    license: parseLicense(api),
    gated:
      typeof api.gated === 'boolean' || typeof api.gated === 'string'
        ? (api.gated as boolean | string)
        : undefined,
    private: typeof api.private === 'boolean' ? api.private : undefined,
    parameterCount,
    activeParameterCount: parsed.activeParameterCount,
    parameterSource,
    parameterCountByDtype:
      safetensors?.parameterCountByDtype ?? apiCounts.byDtype,
    parameterBreakdown,
    architecture,
    quantization: parsed.quantization,
    safetensors,
    transformers,
    remoteCodeFiles,
    files,
    generationConfig: generationConfig ?? undefined,
    tokenizerConfig: tokenizerConfig ?? undefined,
    cardExcerpt,
    evidence,
    warnings: [...new Set([...parsed.warnings, ...warnings])],
    fetchedAt: new Date().toISOString(),
  };
  inspectionCache.set(id, {
    expiresAt: Date.now() + INSPECTION_CACHE_TTL_MS,
    inspection,
  });
  return inspection;
}
