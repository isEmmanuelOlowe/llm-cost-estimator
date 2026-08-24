import gpus from './gpus.json';

export type PrecisionBits = 4 | 8 | 16 | 32;

export type WeightFormat =
  | 'bf16'
  | 'fp16'
  | 'fp8'
  | 'int8'
  | 'int4'
  | 'nvfp4'
  | 'mxfp4';

export type ExecutionMode = 'inference' | 'training';

export type OptimizerType = 'none' | 'adam' | 'adamw' | 'adafactor' | 'lamb';

export type EstimateClassification = 'exact' | 'heuristic';

/**
 * Architecture facts that change persistent inference-cache storage.
 *
 * This is deliberately separate from attention head counts: MQA/GQA changes
 * the width of a KV tensor, while sliding, compressed, or state-space layers
 * change how many tokens (or whether any KV tensor) is retained.
 */
export interface KvCacheArchitecture {
  mode?: string;
  keyValueShared?: boolean | null;
  slidingWindow?: number | null;
  localAttentionLayers?: number | null;
  fullAttentionLayers?: number | null;
  noAttentionLayers?: number | null;
  compressedSparseLayers?: number | null;
  heavilyCompressedLayers?: number | null;
  compressedSparseRate?: number | null;
  heavilyCompressedRate?: number | null;
  indexHeadDim?: number | null;
  indexerLayers?: number | null;
  latentKvRank?: number | null;
  latentRopeDim?: number | null;
  recurrentStateLayers?: number | null;
  recurrentStateBytesPerLayer?: number | null;
  label?: string | null;
  confidence?: string | null;
  sourceSignals?: string[] | null;
}

export interface MemoryEstimationInput {
  parameterCount: number; // Raw parameter count
  weightPrecisionBits: PrecisionBits;
  mode: ExecutionMode;
  hiddenSize: number;
  numLayers: number;
  numAttentionHeads?: number;
  numKeyValueHeads?: number;
  headDim?: number;
  sequenceLength: number;
  batchSize: number;
  kvCachePrecisionBits?: PrecisionBits;
  activationMultiplierOverride?: number;
  optimizer?: OptimizerType;
  overheadFactor?: number;
  kvCacheArchitecture?: KvCacheArchitecture;
}

export interface MemoryBreakdown {
  weightsGB: number;
  activationsGB: number;
  kvCacheGB: number;
  stateCacheGB: number;
  kvCacheBytesPerToken: number;
  kvCacheTokens: number;
  kvCacheMode: string;
  kvCacheDescription: string;
  kvAttentionLayers: number;
  optimizerGB: number;
  baseTotalGB: number;
  overheadGB: number;
  totalGB: number;
}

export interface ThroughputInput {
  parameterCount: number;
  gpuTFlops: number;
  precisionTFlops?: number;
  efficiency?: number; // 0-1 multiplier capturing kernels/framework efficiency
  memoryBandwidthGBs?: number;
  weightPrecisionBits?: PrecisionBits;
  batchSize?: number;
  memoryEfficiency?: number;
  memoryTrafficMultiplier?: number;
  computeFormat?: WeightFormat;
}

export interface ThroughputEstimate {
  tokensPerSecond: number;
  millisecondsPerToken: number;
  computeBoundTokensPerSecond: number;
  memoryBoundTokensPerSecond: number;
  bottleneck: 'compute' | 'memory' | 'unavailable';
}

export interface CloudInstanceCostInput {
  hourlyRate: number;
  durationHours: number;
}

export interface CloudCostEstimate {
  totalCost: number;
  hourlyRate: number;
  durationHours: number;
}

export interface RecommendedGpu {
  name: string;
  memoryGB: number;
  fp32TFlops: number;
  memoryHeadroomGB: number;
  vendor?: string;
  architecture?: string;
  memoryType?: string;
  memoryModel?: string;
  deviceCount: number;
  perDeviceMemoryGB: number;
  requiredDevices: number;
  aggregateFit: boolean;
}

export interface HardwareLike {
  name: string;
  memory_gb: number;
  fp32_tflops: number;
  memory_bandwidth_gb_s?: number;
  vendor?: string;
  architecture?: string;
  memory_type?: string;
  memory_model?: string;
  device_count?: number;
  per_device_memory_gb?: number;
  fp16_tflops?: number;
  bf16_tflops?: number;
  fp8_tflops?: number;
  fp4_tflops?: number;
  int8_tops?: number;
  status?: string;
  category?: string;
  ai_tops?: number;
  ai_precision?: string;
  power_w?: number;
  interconnect?: string;
  form_factor?: string;
  source_url?: string;
  source_checked_at?: string;
  notes?: string;
}

export interface HardwareFitEstimate {
  requiredMemoryGB: number;
  aggregateMemoryGB: number;
  perDeviceMemoryGB: number;
  deviceCount: number;
  requiredDevices: number;
  aggregateHeadroomGB: number;
  perDeviceHeadroomGB: number;
  fits: boolean;
}

export interface ArchitectureEstimate {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  intermediateSize: number;
}

export interface KvCacheEstimate {
  bytesPerToken: number;
  totalTokens: number;
  totalBytes: number;
  totalGB: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  stateCacheGB: number;
  cacheMode: string;
  cacheDescription: string;
  attentionLayers: number;
}

export function resolveEffectiveParameterCount(
  parameterCount: number,
  activeParameterCount?: number | null,
): number {
  if (activeParameterCount && activeParameterCount > 0) {
    return activeParameterCount;
  }

  return parameterCount;
}

const BYTES_PER_GB = 1024 ** 3;

const DEFAULT_ACTIVATION_MULTIPLIER: Record<ExecutionMode, number> = {
  inference: 0.2,
  training: 2,
};

const OPTIMIZER_MULTIPLIER: Record<OptimizerType, number> = {
  none: 0,
  adam: 4,
  adamw: 4,
  lamb: 4,
  adafactor: 1.5,
};

const DEFAULT_OVERHEAD = 1.15;

const LLAMA_STYLE_ARCHETYPES: Array<{
  maxBillions: number;
  hiddenSize: number;
  numLayers: number;
}> = [
  { maxBillions: 1.5, hiddenSize: 2048, numLayers: 24 },
  { maxBillions: 3.5, hiddenSize: 2560, numLayers: 28 },
  { maxBillions: 8, hiddenSize: 4096, numLayers: 32 },
  { maxBillions: 16, hiddenSize: 5120, numLayers: 40 },
  { maxBillions: 40, hiddenSize: 6656, numLayers: 60 },
  { maxBillions: 80, hiddenSize: 8192, numLayers: 80 },
  { maxBillions: Number.POSITIVE_INFINITY, hiddenSize: 10240, numLayers: 96 },
];

export function bitsToBytes(bits: PrecisionBits): number {
  return bits / 8;
}

export function estimateLlamaStyleArchitecture(
  parameterCount: number,
): ArchitectureEstimate {
  if (!Number.isFinite(parameterCount) || parameterCount <= 0) {
    return { hiddenSize: 0, numLayers: 0, numHeads: 0, intermediateSize: 0 };
  }

  const paramsInBillions = parameterCount / 10 ** 9;
  const archetype = LLAMA_STYLE_ARCHETYPES.find(
    (entry) => paramsInBillions <= entry.maxBillions,
  );

  if (!archetype) {
    return { hiddenSize: 0, numLayers: 0, numHeads: 0, intermediateSize: 0 };
  }

  const { hiddenSize, numLayers } = archetype;
  const numHeads = Math.max(1, Math.round(hiddenSize / 128));
  const intermediateSize = hiddenSize * 4;

  return { hiddenSize, numLayers, numHeads, intermediateSize };
}

export function calculateWeightMemoryGB(
  parameterCount: number,
  weightPrecisionBits: PrecisionBits,
): number {
  if (parameterCount <= 0) return 0;
  const bytes = parameterCount * bitsToBytes(weightPrecisionBits);
  return bytes / BYTES_PER_GB;
}

export function calculateActivationMemoryGB(
  parameterCount: number,
  weightPrecisionBits: PrecisionBits,
  mode: ExecutionMode,
  activationMultiplierOverride?: number,
): number {
  if (parameterCount <= 0) return 0;
  const multiplier =
    activationMultiplierOverride ?? DEFAULT_ACTIVATION_MULTIPLIER[mode];
  const weightMemoryGB = calculateWeightMemoryGB(
    parameterCount,
    weightPrecisionBits,
  );
  return weightMemoryGB * multiplier;
}

export function calculateKvCacheMemoryGB({
  sequenceLength,
  batchSize,
  numLayers,
  hiddenSize,
  precisionBits,
  numAttentionHeads,
  numKeyValueHeads,
  headDim,
  kvCacheArchitecture,
}: {
  sequenceLength: number;
  batchSize: number;
  numLayers: number;
  hiddenSize: number;
  precisionBits: PrecisionBits;
  numAttentionHeads?: number;
  numKeyValueHeads?: number;
  headDim?: number;
  kvCacheArchitecture?: KvCacheArchitecture;
}): number {
  return estimateKvCache({
    sequenceLength,
    batchSize,
    numLayers,
    hiddenSize,
    precisionBits,
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    kvCacheArchitecture,
  }).totalGB;
}

interface ArchitecturalCacheBytes {
  kvBytes: number;
  stateBytes: number;
  cacheMode: string;
  cacheDescription: string;
  attentionLayers: number;
}

function estimateArchitecturalCacheBytes({
  architecture,
  sequenceLength,
  batchSize,
  numLayers,
  numKeyValueHeads,
  headDim,
  bytesPerElement,
}: {
  architecture?: KvCacheArchitecture;
  sequenceLength: number;
  batchSize: number;
  numLayers: number;
  numKeyValueHeads: number;
  headDim: number;
  bytesPerElement: number;
}): ArchitecturalCacheBytes {
  const keyValueCopies = architecture?.keyValueShared ? 1 : 2;
  const fullAttentionLayers = Math.max(
    0,
    Math.floor(
      architecture?.fullAttentionLayers ??
        (architecture?.localAttentionLayers != null
          ? 0
          : numLayers - (architecture?.noAttentionLayers ?? 0)),
    ),
  );
  const localAttentionLayers = Math.max(
    0,
    Math.floor(architecture?.localAttentionLayers ?? 0),
  );
  const noAttentionLayers = Math.max(
    0,
    Math.floor(
      architecture?.noAttentionLayers ??
        Math.max(0, numLayers - fullAttentionLayers - localAttentionLayers),
    ),
  );
  const slidingWindow = Math.max(
    1,
    Math.floor(architecture?.slidingWindow ?? sequenceLength),
  );
  const localTokens = Math.min(sequenceLength, slidingWindow);
  const standardKvBytes =
    (fullAttentionLayers * sequenceLength +
      localAttentionLayers * localTokens) *
    numKeyValueHeads *
    headDim *
    bytesPerElement *
    keyValueCopies *
    batchSize;

  if (
    architecture?.latentKvRank &&
    architecture.latentKvRank > 0 &&
    architecture.latentRopeDim &&
    architecture.latentRopeDim > 0
  ) {
    const latentAttentionLayers =
      fullAttentionLayers || Math.max(0, numLayers - noAttentionLayers);
    const latentBytes =
      latentAttentionLayers *
      sequenceLength *
      (architecture.latentKvRank + architecture.latentRopeDim) *
      bytesPerElement *
      batchSize;
    const stateBytesPerLayer = Math.max(
      0,
      architecture.recurrentStateBytesPerLayer ?? 0,
    );
    const recurrentStateLayers = Math.max(
      0,
      Math.floor(architecture.recurrentStateLayers ?? 0),
    );
    return {
      kvBytes: latentBytes,
      stateBytes: stateBytesPerLayer * recurrentStateLayers * batchSize,
      cacheMode: architecture.mode ?? 'latent-attention',
      cacheDescription:
        architecture.label ??
        `${latentAttentionLayers} attention layers cache a compressed latent rank plus rotary position state; other layers use a separate recurrent/linear state.`,
      attentionLayers: latentAttentionLayers,
    };
  }

  if (architecture?.mode === 'deepseek-v4') {
    const compressedSparseLayers = Math.max(
      0,
      Math.floor(architecture.compressedSparseLayers ?? 0),
    );
    const heavilyCompressedLayers = Math.max(
      0,
      Math.floor(architecture.heavilyCompressedLayers ?? 0),
    );
    const compressedSparseRate = Math.max(
      1,
      Math.floor(architecture.compressedSparseRate ?? 4),
    );
    const heavilyCompressedRate = Math.max(
      1,
      Math.floor(architecture.heavilyCompressedRate ?? 128),
    );
    const compressedSparseEntries = Math.floor(
      sequenceLength / compressedSparseRate,
    );
    const heavilyCompressedEntries = Math.floor(
      sequenceLength / heavilyCompressedRate,
    );
    const indexHeadDim = Math.max(
      1,
      Math.floor(architecture.indexHeadDim ?? headDim),
    );
    const compressedBytes =
      (compressedSparseLayers *
        compressedSparseEntries *
        (headDim + indexHeadDim) +
        heavilyCompressedLayers * heavilyCompressedEntries * headDim) *
      bytesPerElement *
      batchSize;
    const attentionLayers =
      fullAttentionLayers + localAttentionLayers ||
      numLayers - noAttentionLayers;
    return {
      kvBytes: standardKvBytes + compressedBytes,
      stateBytes: 0,
      cacheMode: 'deepseek-v4',
      cacheDescription: `${localAttentionLayers} sliding layers + ${compressedSparseLayers} compressed-sparse layers + ${heavilyCompressedLayers} heavily-compressed layers; shared K=V storage and compressed long-range entries are included. Transient compressor workspaces are omitted.`,
      attentionLayers,
    };
  }

  const stateBytesPerLayer = Math.max(
    0,
    architecture?.recurrentStateBytesPerLayer ?? 0,
  );
  const recurrentStateLayers = Math.max(
    0,
    Math.floor(architecture?.recurrentStateLayers ?? 0),
  );
  const stateBytes = stateBytesPerLayer * recurrentStateLayers * batchSize;
  const attentionLayers =
    fullAttentionLayers + localAttentionLayers || numLayers - noAttentionLayers;

  if (architecture?.mode === 'hybrid-state-space') {
    return {
      kvBytes: standardKvBytes,
      stateBytes,
      cacheMode: 'hybrid-state-space',
      cacheDescription: `${attentionLayers} attention layers retain KV state; ${recurrentStateLayers} recurrent layers use separate state/conv caches instead of KV tensors. Recurrent state is derived from the configured state dimensions.`,
      attentionLayers,
    };
  }
  if (architecture?.mode === 'hybrid-sliding-window') {
    return {
      kvBytes: standardKvBytes,
      stateBytes,
      cacheMode: 'hybrid-sliding-window',
      cacheDescription: `${localAttentionLayers} layers are capped at a ${slidingWindow}-token window; ${fullAttentionLayers} layers retain full-context KV state.`,
      attentionLayers,
    };
  }
  if (architecture?.mode === 'sliding-window') {
    return {
      kvBytes: standardKvBytes,
      stateBytes,
      cacheMode: 'sliding-window',
      cacheDescription: `KV state is capped at a ${slidingWindow}-token window for the configured sliding layers.`,
      attentionLayers,
    };
  }
  return {
    kvBytes: standardKvBytes,
    stateBytes,
    cacheMode: architecture?.keyValueShared ? 'shared-kv' : 'standard',
    cacheDescription: architecture?.keyValueShared
      ? 'The implementation shares K and V storage for each cached entry.'
      : 'Standard K/V storage across the configured attention layers.',
    attentionLayers,
  };
}

export function estimateKvCache({
  sequenceLength,
  batchSize,
  numLayers,
  hiddenSize,
  precisionBits,
  numAttentionHeads,
  numKeyValueHeads,
  headDim,
  kvCacheArchitecture,
}: {
  sequenceLength: number;
  batchSize: number;
  numLayers: number;
  hiddenSize: number;
  precisionBits: PrecisionBits;
  numAttentionHeads?: number;
  numKeyValueHeads?: number;
  headDim?: number;
  kvCacheArchitecture?: KvCacheArchitecture;
}): KvCacheEstimate {
  const inferredAttentionHeads = Math.max(1, Math.round(hiddenSize / 128));
  const resolvedAttentionHeads = Math.max(
    1,
    Math.floor(numAttentionHeads ?? inferredAttentionHeads),
  );
  const resolvedHeadDim =
    headDim && headDim > 0 ? headDim : hiddenSize / resolvedAttentionHeads;
  const resolvedKvHeads = Math.max(
    1,
    Math.min(
      resolvedAttentionHeads,
      Math.floor(numKeyValueHeads ?? resolvedAttentionHeads),
    ),
  );

  if (
    sequenceLength <= 0 ||
    batchSize <= 0 ||
    numLayers <= 0 ||
    hiddenSize <= 0 ||
    resolvedHeadDim <= 0
  ) {
    return {
      bytesPerToken: 0,
      totalTokens: 0,
      totalBytes: 0,
      totalGB: 0,
      numAttentionHeads: resolvedAttentionHeads,
      numKeyValueHeads: resolvedKvHeads,
      headDim: resolvedHeadDim,
      stateCacheGB: 0,
      cacheMode: kvCacheArchitecture?.mode ?? 'standard',
      cacheDescription:
        'No cache estimate is available for the supplied dimensions.',
      attentionLayers: 0,
    };
  }

  const bytesPerElement = bitsToBytes(precisionBits);
  const totalTokens = sequenceLength * batchSize;
  const architecturalBytes = estimateArchitecturalCacheBytes({
    architecture: kvCacheArchitecture,
    sequenceLength,
    batchSize,
    numLayers,
    numKeyValueHeads: resolvedKvHeads,
    headDim: resolvedHeadDim,
    bytesPerElement,
  });
  const totalBytes = architecturalBytes.kvBytes;
  const bytesPerToken = totalBytes / totalTokens;

  return {
    bytesPerToken,
    totalTokens,
    totalBytes,
    totalGB: totalBytes / BYTES_PER_GB,
    numAttentionHeads: resolvedAttentionHeads,
    numKeyValueHeads: resolvedKvHeads,
    headDim: resolvedHeadDim,
    stateCacheGB: architecturalBytes.stateBytes / BYTES_PER_GB,
    cacheMode: architecturalBytes.cacheMode,
    cacheDescription: architecturalBytes.cacheDescription,
    attentionLayers: architecturalBytes.attentionLayers,
  };
}

export function calculateOptimizerMemoryGB(
  parameterCount: number,
  weightPrecisionBits: PrecisionBits,
  optimizer: OptimizerType,
): number {
  if (parameterCount <= 0) return 0;
  const multiplier = OPTIMIZER_MULTIPLIER[optimizer] ?? 0;
  const weightBytes = parameterCount * bitsToBytes(weightPrecisionBits);
  const optimizerBytes = weightBytes * multiplier;
  return optimizerBytes / BYTES_PER_GB;
}

export function estimateMemory({
  parameterCount,
  weightPrecisionBits,
  mode,
  hiddenSize,
  numLayers,
  numAttentionHeads,
  numKeyValueHeads,
  headDim,
  sequenceLength,
  batchSize,
  kvCachePrecisionBits,
  activationMultiplierOverride,
  optimizer = mode === 'training' ? 'adamw' : 'none',
  overheadFactor = DEFAULT_OVERHEAD,
  kvCacheArchitecture,
}: MemoryEstimationInput): MemoryBreakdown {
  if (parameterCount < 0) {
    throw new Error('parameterCount must be non-negative');
  }

  const weightsGB = calculateWeightMemoryGB(
    parameterCount,
    weightPrecisionBits,
  );
  const activationsGB = calculateActivationMemoryGB(
    parameterCount,
    weightPrecisionBits,
    mode,
    activationMultiplierOverride,
  );
  const kvCacheGB =
    mode === 'inference'
      ? calculateKvCacheMemoryGB({
          sequenceLength,
          batchSize,
          numLayers,
          hiddenSize,
          precisionBits: kvCachePrecisionBits ?? weightPrecisionBits,
          numAttentionHeads,
          numKeyValueHeads,
          headDim,
          kvCacheArchitecture,
        })
      : 0;
  const kvCache =
    mode === 'inference'
      ? estimateKvCache({
          sequenceLength,
          batchSize,
          numLayers,
          hiddenSize,
          precisionBits: kvCachePrecisionBits ?? weightPrecisionBits,
          numAttentionHeads,
          numKeyValueHeads,
          headDim,
          kvCacheArchitecture,
        })
      : null;
  const stateCacheGB = kvCache?.stateCacheGB ?? 0;
  const optimizerGB =
    mode === 'training'
      ? calculateOptimizerMemoryGB(
          parameterCount,
          weightPrecisionBits,
          optimizer,
        )
      : 0;

  const baseTotalGB =
    weightsGB + activationsGB + kvCacheGB + stateCacheGB + optimizerGB;
  const overheadGB = baseTotalGB * (overheadFactor - 1);
  const totalGB = baseTotalGB + overheadGB;

  return {
    weightsGB,
    activationsGB,
    kvCacheGB,
    stateCacheGB,
    kvCacheBytesPerToken: kvCache?.bytesPerToken ?? 0,
    kvCacheTokens: kvCache?.totalTokens ?? 0,
    kvCacheMode: kvCache?.cacheMode ?? 'standard',
    kvCacheDescription:
      kvCache?.cacheDescription ?? 'No KV cache estimate is available.',
    kvAttentionLayers: kvCache?.attentionLayers ?? 0,
    optimizerGB,
    baseTotalGB,
    overheadGB,
    totalGB,
  };
}

export function estimateThroughput({
  parameterCount,
  gpuTFlops,
  precisionTFlops,
  efficiency = 0.3,
  memoryBandwidthGBs,
  weightPrecisionBits = 16,
  batchSize = 1,
  memoryEfficiency = 0.65,
  memoryTrafficMultiplier = 1,
}: ThroughputInput): ThroughputEstimate {
  const computeTFlops = precisionTFlops ?? gpuTFlops;
  const computeBoundTokensPerSecond =
    parameterCount > 0 && computeTFlops > 0 && efficiency > 0
      ? (computeTFlops * 10 ** 12 * efficiency) / (parameterCount * 2)
      : 0;
  const effectiveBandwidth =
    memoryBandwidthGBs && memoryBandwidthGBs > 0 && memoryEfficiency > 0
      ? memoryBandwidthGBs * 10 ** 9 * memoryEfficiency
      : 0;
  const weightBytesPerToken =
    parameterCount > 0 && batchSize > 0 && memoryTrafficMultiplier > 0
      ? (parameterCount *
          bitsToBytes(weightPrecisionBits) *
          memoryTrafficMultiplier) /
        batchSize
      : 0;
  const memoryBoundTokensPerSecond =
    effectiveBandwidth > 0 && weightBytesPerToken > 0
      ? effectiveBandwidth / weightBytesPerToken
      : 0;
  const availableBounds = [
    computeBoundTokensPerSecond,
    memoryBoundTokensPerSecond,
  ].filter((value) => value > 0);
  const tokensPerSecond = availableBounds.length
    ? Math.min(...availableBounds)
    : 0;
  const bottleneck: ThroughputEstimate['bottleneck'] =
    tokensPerSecond <= 0
      ? 'unavailable'
      : memoryBoundTokensPerSecond > 0 &&
          (computeBoundTokensPerSecond <= 0 ||
            memoryBoundTokensPerSecond <= computeBoundTokensPerSecond)
        ? 'memory'
        : 'compute';
  const millisecondsPerToken = tokensPerSecond ? 1000 / tokensPerSecond : 0;

  return {
    tokensPerSecond,
    millisecondsPerToken,
    computeBoundTokensPerSecond,
    memoryBoundTokensPerSecond,
    bottleneck,
  };
}

export function estimateCloudCost({
  hourlyRate,
  durationHours,
}: CloudInstanceCostInput): CloudCostEstimate {
  if (hourlyRate < 0 || durationHours < 0) {
    throw new Error('hourlyRate and durationHours must be non-negative');
  }

  return {
    hourlyRate,
    durationHours,
    totalCost: hourlyRate * durationHours,
  };
}

export function recommendGpus(
  requiredMemoryGB: number,
  maxResults = 3,
): RecommendedGpu[] {
  if (requiredMemoryGB <= 0) {
    return [];
  }

  const hardwareCatalog = gpus as unknown as HardwareLike[];

  return hardwareCatalog
    .map((gpu) => ({
      deviceCount: Math.max(1, gpu.device_count ?? 1),
      perDeviceMemoryGB:
        gpu.per_device_memory_gb ??
        gpu.memory_gb / Math.max(1, gpu.device_count ?? 1),
      name: gpu.name,
      memoryGB: gpu.memory_gb,
      fp32TFlops: gpu.fp32_tflops,
      memoryHeadroomGB: gpu.memory_gb - requiredMemoryGB,
      vendor: gpu.vendor,
      architecture: gpu.architecture,
      memoryType: gpu.memory_type,
      memoryModel: gpu.memory_model,
      requiredDevices: Math.max(
        1,
        Math.ceil(
          requiredMemoryGB /
            (gpu.per_device_memory_gb ??
              gpu.memory_gb / Math.max(1, gpu.device_count ?? 1)),
        ),
      ),
      aggregateFit: gpu.memory_gb >= requiredMemoryGB,
    }))
    .filter((gpu) => gpu.aggregateFit && gpu.requiredDevices <= gpu.deviceCount)
    .sort((a, b) => a.memoryHeadroomGB - b.memoryHeadroomGB)
    .slice(0, maxResults);
}

export function estimateHardwareFit(
  requiredMemoryGB: number,
  gpu: HardwareLike,
): HardwareFitEstimate {
  const deviceCount = Math.max(1, gpu.device_count ?? 1);
  const perDeviceMemoryGB =
    gpu.per_device_memory_gb ?? gpu.memory_gb / deviceCount;
  const requiredDevices =
    requiredMemoryGB > 0
      ? Math.max(1, Math.ceil(requiredMemoryGB / perDeviceMemoryGB))
      : 0;

  return {
    requiredMemoryGB,
    aggregateMemoryGB: gpu.memory_gb,
    perDeviceMemoryGB,
    deviceCount,
    requiredDevices,
    aggregateHeadroomGB: gpu.memory_gb - requiredMemoryGB,
    perDeviceHeadroomGB: perDeviceMemoryGB - requiredMemoryGB,
    fits: requiredMemoryGB <= gpu.memory_gb && requiredDevices <= deviceCount,
  };
}

export interface ModelFlopInput {
  numLayers: number;
  hiddenSize: number;
  sequenceLength: number;
  vocabSize: number;
  intermediateSize?: number;
  numAttentionHeads?: number;
  numKeyValueHeads?: number;
  headDim?: number;
  gatedMlp?: boolean;
}

export interface TransformerConfig {
  vocabSize: number;
  hiddenSize: number;
  numLayers: number;
  numAttentionHeads: number;
  intermediateSize?: number;
  numKeyValueHeads?: number;
  headDim?: number;
  numExperts?: number;
  numExpertsPerToken?: number;
  gatedMlp?: boolean;
  tieWordEmbeddings?: boolean;
  attentionBias?: boolean;
  mlpBias?: boolean;
}

export interface TransformerParameterBreakdown {
  embeddingParams: number;
  attentionParamsPerLayer: number;
  mlpParamsPerLayer: number;
  activeMlpParamsPerLayer: number;
  normalizationParamsPerLayer: number;
  lmHeadParams: number;
  totalParameters: number;
  activeParameters: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  numExperts: number;
  numExpertsPerToken: number;
}

export function estimateDecoderFlops({
  numLayers,
  hiddenSize,
  sequenceLength,
  vocabSize,
  intermediateSize,
  numAttentionHeads,
  numKeyValueHeads,
  headDim,
  gatedMlp,
}: ModelFlopInput): number {
  if (
    numLayers <= 0 ||
    hiddenSize <= 0 ||
    sequenceLength <= 0 ||
    vocabSize <= 0
  ) {
    return 0;
  }

  if (intermediateSize && numAttentionHeads && numKeyValueHeads && headDim) {
    const queryWidth = numAttentionHeads * headDim;
    const keyValueWidth = numKeyValueHeads * headDim;
    const projectionFlops =
      2 *
      sequenceLength *
      (hiddenSize * (queryWidth + keyValueWidth * 2 + hiddenSize));
    const attentionMixingFlops =
      4 * sequenceLength ** 2 * numAttentionHeads * headDim;
    const mlpProjectionCount = gatedMlp ? 3 : 2;
    const mlpFlops =
      2 * sequenceLength * hiddenSize * intermediateSize * mlpProjectionCount;
    return (
      numLayers * (projectionFlops + attentionMixingFlops + mlpFlops) +
      2 * sequenceLength * hiddenSize * vocabSize
    );
  }

  const attentionFlops = 4 * numLayers * sequenceLength * hiddenSize ** 2;
  const mlpFlops = 8 * numLayers * sequenceLength * hiddenSize ** 2;
  const projectionFlops = 2 * sequenceLength * hiddenSize * vocabSize;
  return attentionFlops + mlpFlops + projectionFlops;
}

export function estimateTransformerParameters({
  vocabSize,
  hiddenSize,
  numLayers,
  numAttentionHeads,
  intermediateSize,
  numKeyValueHeads,
  headDim,
  numExperts,
  numExpertsPerToken,
  gatedMlp = false,
  tieWordEmbeddings = true,
  attentionBias = false,
  mlpBias = false,
}: TransformerConfig): number {
  return estimateTransformerParameterBreakdown({
    vocabSize,
    hiddenSize,
    numLayers,
    numAttentionHeads,
    intermediateSize,
    numKeyValueHeads,
    headDim,
    numExperts,
    numExpertsPerToken,
    gatedMlp,
    tieWordEmbeddings,
    attentionBias,
    mlpBias,
  }).totalParameters;
}

export function estimateTransformerParameterBreakdown({
  vocabSize,
  hiddenSize,
  numLayers,
  numAttentionHeads,
  intermediateSize,
  numKeyValueHeads,
  headDim,
  numExperts,
  numExpertsPerToken,
  gatedMlp = false,
  tieWordEmbeddings = true,
  attentionBias = false,
  mlpBias = false,
}: TransformerConfig): TransformerParameterBreakdown {
  if (
    vocabSize <= 0 ||
    hiddenSize <= 0 ||
    numLayers <= 0 ||
    numAttentionHeads <= 0
  ) {
    return {
      embeddingParams: 0,
      attentionParamsPerLayer: 0,
      mlpParamsPerLayer: 0,
      activeMlpParamsPerLayer: 0,
      normalizationParamsPerLayer: 0,
      lmHeadParams: 0,
      totalParameters: 0,
      activeParameters: 0,
      numAttentionHeads: 0,
      numKeyValueHeads: 0,
      headDim: 0,
      numExperts: 0,
      numExpertsPerToken: 0,
    };
  }

  const resolvedKvHeads = Math.max(
    1,
    Math.min(
      numAttentionHeads,
      Math.floor(numKeyValueHeads ?? numAttentionHeads),
    ),
  );
  const resolvedHeadDim =
    headDim && headDim > 0 ? headDim : hiddenSize / numAttentionHeads;
  const embeddingParams = vocabSize * hiddenSize;
  const queryOutputSize = numAttentionHeads * resolvedHeadDim;
  const keyValueOutputSize = resolvedKvHeads * resolvedHeadDim;
  const attentionBiasParams = attentionBias
    ? queryOutputSize + keyValueOutputSize * 2 + hiddenSize
    : 0;
  const attnParamsPerLayer =
    hiddenSize * (queryOutputSize + keyValueOutputSize * 2 + hiddenSize) +
    attentionBiasParams;

  const effectiveIntermediate =
    intermediateSize && intermediateSize > 0
      ? intermediateSize
      : hiddenSize * 4;
  const expertCount = Math.max(1, Math.floor(numExperts ?? 1));
  const activeExpertCount = Math.min(
    expertCount,
    Math.max(1, Math.floor(numExpertsPerToken ?? expertCount)),
  );
  const mlpProjectionCount = gatedMlp ? 3 : 2;
  const mlpBiasParams = mlpBias
    ? effectiveIntermediate * (gatedMlp ? 3 : 2) + hiddenSize
    : 0;
  const mlpParamsPerExpert =
    hiddenSize * effectiveIntermediate * mlpProjectionCount + mlpBiasParams;
  const mlpParamsPerLayer = mlpParamsPerExpert * expertCount;
  const activeMlpParamsPerLayer = mlpParamsPerExpert * activeExpertCount;
  const normalizationParamsPerLayer = hiddenSize * 2;
  const lmHeadParams = tieWordEmbeddings ? 0 : vocabSize * hiddenSize;
  const totalParameters =
    embeddingParams +
    lmHeadParams +
    numLayers *
      (attnParamsPerLayer + mlpParamsPerLayer + normalizationParamsPerLayer);
  const activeParameters =
    embeddingParams +
    lmHeadParams +
    numLayers *
      (attnParamsPerLayer +
        activeMlpParamsPerLayer +
        normalizationParamsPerLayer);

  return {
    embeddingParams,
    attentionParamsPerLayer: attnParamsPerLayer,
    mlpParamsPerLayer,
    activeMlpParamsPerLayer,
    normalizationParamsPerLayer,
    lmHeadParams,
    totalParameters,
    activeParameters,
    numAttentionHeads,
    numKeyValueHeads: resolvedKvHeads,
    headDim: resolvedHeadDim,
    numExperts: expertCount,
    numExpertsPerToken: activeExpertCount,
  };
}

export function selectGpuComputeTFlops(
  gpu: HardwareLike,
  precisionBits: PrecisionBits,
  format?: WeightFormat,
): number {
  if (format === 'bf16') return gpu.bf16_tflops ?? 0;
  if (format === 'fp16') return gpu.fp16_tflops ?? 0;
  if (format === 'fp8') return gpu.fp8_tflops ?? 0;
  if (format === 'int8') return gpu.int8_tops ?? 0;
  if (format === 'nvfp4' || format === 'mxfp4') {
    return gpu.fp4_tflops ?? 0;
  }
  if (format === 'int4') return 0;
  if (precisionBits === 32) return gpu.fp32_tflops ?? 0;
  if (precisionBits === 16) return gpu.bf16_tflops ?? gpu.fp16_tflops ?? 0;
  if (precisionBits === 8) return gpu.fp8_tflops ?? gpu.int8_tops ?? 0;
  return gpu.fp4_tflops ?? 0;
}

export function calculateMemoryFromBillions(
  paramsInBillions: number,
  weightPrecisionBits: PrecisionBits,
): number {
  if (paramsInBillions <= 0) return 0;
  return calculateWeightMemoryGB(
    paramsInBillions * 10 ** 9,
    weightPrecisionBits,
  );
}
