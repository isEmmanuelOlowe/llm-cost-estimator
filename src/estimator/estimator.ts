import gpus from './gpus.json';

export type PrecisionBits = 4 | 8 | 16 | 32;

export type ExecutionMode = 'inference' | 'training';

export type OptimizerType = 'none' | 'adam' | 'adamw' | 'adafactor' | 'lamb';

export interface MemoryEstimationInput {
  parameterCount: number; // Raw parameter count
  weightPrecisionBits: PrecisionBits;
  mode: ExecutionMode;
  hiddenSize: number;
  numLayers: number;
  sequenceLength: number;
  batchSize: number;
  kvCachePrecisionBits?: PrecisionBits;
  activationMultiplierOverride?: number;
  optimizer?: OptimizerType;
  overheadFactor?: number;
}

export interface MemoryBreakdown {
  weightsGB: number;
  activationsGB: number;
  kvCacheGB: number;
  optimizerGB: number;
  baseTotalGB: number;
  overheadGB: number;
  totalGB: number;
}

export interface ThroughputInput {
  parameterCount: number;
  gpuTFlops: number;
  efficiency?: number; // 0-1 multiplier capturing kernels/framework efficiency
}

export interface ThroughputEstimate {
  tokensPerSecond: number;
  millisecondsPerToken: number;
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
}

export interface ArchitectureEstimate {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  intermediateSize: number;
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
  parameterCount: number
): ArchitectureEstimate {
  if (!Number.isFinite(parameterCount) || parameterCount <= 0) {
    return { hiddenSize: 0, numLayers: 0, numHeads: 0, intermediateSize: 0 };
  }

  const paramsInBillions = parameterCount / 10 ** 9;
  const archetype = LLAMA_STYLE_ARCHETYPES.find(
    (entry) => paramsInBillions <= entry.maxBillions
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
  weightPrecisionBits: PrecisionBits
): number {
  if (parameterCount <= 0) return 0;
  const bytes = parameterCount * bitsToBytes(weightPrecisionBits);
  return bytes / BYTES_PER_GB;
}

export function calculateActivationMemoryGB(
  parameterCount: number,
  weightPrecisionBits: PrecisionBits,
  mode: ExecutionMode,
  activationMultiplierOverride?: number
): number {
  if (parameterCount <= 0) return 0;
  const multiplier =
    activationMultiplierOverride ?? DEFAULT_ACTIVATION_MULTIPLIER[mode];
  const weightMemoryGB = calculateWeightMemoryGB(
    parameterCount,
    weightPrecisionBits
  );
  return weightMemoryGB * multiplier;
}

export function calculateKvCacheMemoryGB({
  sequenceLength,
  batchSize,
  numLayers,
  hiddenSize,
  precisionBits,
}: {
  sequenceLength: number;
  batchSize: number;
  numLayers: number;
  hiddenSize: number;
  precisionBits: PrecisionBits;
}): number {
  if (
    sequenceLength <= 0 ||
    batchSize <= 0 ||
    numLayers <= 0 ||
    hiddenSize <= 0
  ) {
    return 0;
  }

  const bytesPerElement = bitsToBytes(precisionBits);
  const kvPerToken = 2 * numLayers * hiddenSize * bytesPerElement;
  const totalBytes = kvPerToken * sequenceLength * batchSize;
  return totalBytes / BYTES_PER_GB;
}

export function calculateOptimizerMemoryGB(
  parameterCount: number,
  weightPrecisionBits: PrecisionBits,
  optimizer: OptimizerType
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
  sequenceLength,
  batchSize,
  kvCachePrecisionBits,
  activationMultiplierOverride,
  optimizer = mode === 'training' ? 'adamw' : 'none',
  overheadFactor = DEFAULT_OVERHEAD,
}: MemoryEstimationInput): MemoryBreakdown {
  if (parameterCount < 0) {
    throw new Error('parameterCount must be non-negative');
  }

  const weightsGB = calculateWeightMemoryGB(parameterCount, weightPrecisionBits);
  const activationsGB = calculateActivationMemoryGB(
    parameterCount,
    weightPrecisionBits,
    mode,
    activationMultiplierOverride
  );
  const kvCacheGB =
    mode === 'inference'
      ? calculateKvCacheMemoryGB({
          sequenceLength,
          batchSize,
          numLayers,
          hiddenSize,
          precisionBits: kvCachePrecisionBits ?? weightPrecisionBits,
        })
      : 0;
  const optimizerGB =
    mode === 'training'
      ? calculateOptimizerMemoryGB(
          parameterCount,
          weightPrecisionBits,
          optimizer
        )
      : 0;

  const baseTotalGB = weightsGB + activationsGB + kvCacheGB + optimizerGB;
  const overheadGB = baseTotalGB * (overheadFactor - 1);
  const totalGB = baseTotalGB + overheadGB;

  return {
    weightsGB,
    activationsGB,
    kvCacheGB,
    optimizerGB,
    baseTotalGB,
    overheadGB,
    totalGB,
  };
}

export function estimateThroughput({
  parameterCount,
  gpuTFlops,
  efficiency = 0.3,
}: ThroughputInput): ThroughputEstimate {
  if (parameterCount <= 0 || gpuTFlops <= 0 || efficiency <= 0) {
    return { tokensPerSecond: 0, millisecondsPerToken: 0 };
  }

  const flopsPerToken = parameterCount * 2; // Two matmuls per token approx
  const effectiveFlopsPerSecond = gpuTFlops * 10 ** 12 * efficiency;
  const tokensPerSecond = effectiveFlopsPerSecond / flopsPerToken;
  const millisecondsPerToken = tokensPerSecond
    ? (1000 / tokensPerSecond)
    : 0;

  return {
    tokensPerSecond,
    millisecondsPerToken,
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
  maxResults = 3
): RecommendedGpu[] {
  if (requiredMemoryGB <= 0) {
    return [];
  }

  return gpus
    .map((gpu) => ({
      name: gpu.name,
      memoryGB: gpu.memory_gb,
      fp32TFlops: gpu.fp32_tflops,
      memoryHeadroomGB: gpu.memory_gb - requiredMemoryGB,
    }))
    .filter((gpu) => gpu.memoryHeadroomGB >= 0)
    .sort((a, b) => a.memoryHeadroomGB - b.memoryHeadroomGB)
    .slice(0, maxResults);
}

export interface ModelFlopInput {
  numLayers: number;
  hiddenSize: number;
  sequenceLength: number;
  vocabSize: number;
}

export interface TransformerConfig {
  vocabSize: number;
  hiddenSize: number;
  numLayers: number;
  numAttentionHeads: number;
  intermediateSize?: number;
  numKeyValueHeads?: number;
}

export function estimateDecoderFlops({
  numLayers,
  hiddenSize,
  sequenceLength,
  vocabSize,
}: ModelFlopInput): number {
  if (
    numLayers <= 0 ||
    hiddenSize <= 0 ||
    sequenceLength <= 0 ||
    vocabSize <= 0
  ) {
    return 0;
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
}: TransformerConfig): number {
  if (vocabSize <= 0 || hiddenSize <= 0 || numLayers <= 0 || numAttentionHeads <= 0) {
    return 0;
  }

  const kvHeads = numKeyValueHeads ?? numAttentionHeads;
  const kvHeadDim = hiddenSize / kvHeads;

  const embeddingParams = vocabSize * hiddenSize;

  const qkvParams =
    hiddenSize * hiddenSize + // Query
    hiddenSize * kvHeadDim * kvHeads * 2; // Key + Value (supports GQA)
  const attentionOutputParams = hiddenSize * hiddenSize;
  const attnParamsPerLayer = qkvParams + attentionOutputParams;

  const effectiveIntermediate =
    intermediateSize && intermediateSize > 0
      ? intermediateSize
      : hiddenSize * 4;
  const mlpHidden = effectiveIntermediate;
  const mlpParamsPerLayer = hiddenSize * mlpHidden * 2;
  const normParamsPerLayer = hiddenSize * 2;

  const layerParams =
    attnParamsPerLayer * 2 + // weights and biases approximated
    mlpParamsPerLayer +
    normParamsPerLayer * 2;

  return embeddingParams + numLayers * layerParams;
}

export function calculateMemoryFromBillions(
  paramsInBillions: number,
  weightPrecisionBits: PrecisionBits
): number {
  if (paramsInBillions <= 0) return 0;
  return calculateWeightMemoryGB(paramsInBillions * 10 ** 9, weightPrecisionBits);
}

