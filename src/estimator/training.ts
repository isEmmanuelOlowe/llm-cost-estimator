import type { HardwareLike, OptimizerType, PrecisionBits } from './estimator';

export type TrainingMethod = 'full' | 'lora' | 'qlora';
export type TrainingDistribution = 'replicated' | 'fully-sharded';
export type LoraTargetCoverage = 'attention-qv' | 'all-linear';
export type TrainingComputeFormat = 'bf16' | 'fp16' | 'fp32';

export interface TrainingPlannerSettings {
  method: TrainingMethod;
  loraRank: number;
  targetCoverage: LoraTargetCoverage;
  sequenceLength: number;
  datasetTokens: number;
  epochs: number;
  globalBatchSize: number;
  microBatchSize: number;
  deviceCount: number;
  distribution: TrainingDistribution;
  optimizer: OptimizerType;
  gradientCheckpointing: boolean;
  efficiency: number;
  computeFormat: TrainingComputeFormat;
  optimizerPrecisionBits: 8 | 32;
}

export interface TrainingArchitectureInput {
  parameterCount: number;
  hiddenSize: number;
  numLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  intermediateSize: number;
}

export interface LoraParameterInput extends TrainingArchitectureInput {
  rank: number;
  targetCoverage: LoraTargetCoverage;
}

export interface TrainingMemoryInput extends TrainingArchitectureInput {
  method: TrainingMethod;
  loraRank: number;
  targetCoverage: LoraTargetCoverage;
  sequenceLength: number;
  microBatchSize: number;
  deviceCount: number;
  distribution: TrainingDistribution;
  optimizer: OptimizerType;
  gradientCheckpointing: boolean;
  overheadFactor?: number;
  trainingPrecisionBits?: PrecisionBits;
  optimizerPrecisionBits: 8 | 32;
}

export interface TrainingMemoryEstimate {
  method: TrainingMethod;
  trainableParameterCount: number;
  trainablePercent: number;
  baseWeightBits: number;
  baseWeightsGB: number;
  adapterWeightsGB: number;
  gradientsGB: number;
  optimizerGB: number;
  activationsGB: number;
  persistentPerDeviceGB: number;
  baseWeightsPerDeviceGB: number;
  adapterWeightsPerDeviceGB: number;
  gradientsPerDeviceGB: number;
  optimizerPerDeviceGB: number;
  overheadGB: number;
  perDeviceGB: number;
  aggregateGB: number;
  deviceCount: number;
  distribution: TrainingDistribution;
}

export interface TrainingRunInput {
  activeParameterCount: number;
  method: TrainingMethod;
  datasetTokens: number;
  epochs: number;
  deviceCount: number;
  tflopsPerDevice: number;
  efficiency: number;
  hourlyRatePerDevice?: number;
}

export interface TrainingRunEstimate {
  totalTrainingTokens: number;
  flopsPerToken: number;
  tokensPerSecond: number;
  durationHours: number;
  scalingEfficiency: number;
  clusterHourlyRate: number;
  totalCost: number;
}

export interface TrainingCloudRateLike {
  provider: string;
  name: string;
  gpu_catalog_names: readonly string[];
  hourly_rate: number;
  pricing_source_url: string;
  source_checked_at: string;
}

export interface TrainingPlanRecommendation {
  gpuName: string;
  provider: string;
  offeringName: string;
  deviceCount: number;
  memoryPerGpuGB: number;
  memoryHeadroomGB: number;
  memory: TrainingMemoryEstimate;
  run: TrainingRunEstimate;
  durationHours: number;
  totalCost: number;
  hourlyRatePerDevice: number;
  pricingSourceUrl: string;
  sourceCheckedAt: string;
  distribution: TrainingDistribution;
}

export interface RecommendTrainingPlansInput {
  memoryInput: TrainingMemoryInput;
  runInput: Omit<
    TrainingRunInput,
    'deviceCount' | 'tflopsPerDevice' | 'hourlyRatePerDevice'
  >;
  hardware: readonly HardwareLike[];
  cloudRates: readonly TrainingCloudRateLike[];
  computeFormat: TrainingComputeFormat;
  distributions?: readonly TrainingDistribution[];
  maxDevices?: number;
}

const BYTES_PER_GB = 1024 ** 3;
const DEFAULT_OVERHEAD = 1.15;
const QLORA_STORAGE_BITS = 4.5;

function optimizerBytesPerParameter(
  optimizer: OptimizerType,
  precisionBits: 8 | 32,
): number {
  if (optimizer === 'none') return 0;
  if (optimizer === 'adafactor') return precisionBits === 8 ? 5 : 6;
  return precisionBits === 8 ? 6 : 12;
}

function toGB(bytes: number): number {
  return bytes / BYTES_PER_GB;
}

function positive(value: number, fallback = 0): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export function estimateLoraTrainableParameters({
  parameterCount,
  hiddenSize,
  numLayers,
  numAttentionHeads,
  numKeyValueHeads,
  headDim,
  intermediateSize,
  rank,
  targetCoverage,
}: LoraParameterInput): number {
  const safeRank = Math.max(1, Math.floor(positive(rank, 1)));
  const safeLayers = Math.max(1, Math.floor(positive(numLayers, 1)));
  const hidden = positive(hiddenSize, 1);
  const queryWidth = positive(numAttentionHeads * headDim, hidden);
  const kvWidth = positive(numKeyValueHeads * headDim, queryWidth);
  const intermediate = positive(intermediateSize, hidden * 4);

  const qAndV = safeRank * (hidden + queryWidth + (hidden + kvWidth));
  const allAttention =
    safeRank *
    (hidden + queryWidth + (hidden + kvWidth) * 2 + (queryWidth + hidden));
  const gatedMlp = safeRank * 3 * (hidden + intermediate);
  const perLayer =
    targetCoverage === 'all-linear' ? allAttention + gatedMlp : qAndV;
  const estimate = Math.round(perLayer * safeLayers);

  return Math.min(Math.max(0, parameterCount), estimate);
}

export function estimateTrainingMemory(
  input: TrainingMemoryInput,
): TrainingMemoryEstimate {
  const parameterCount = positive(input.parameterCount);
  const deviceCount = Math.max(1, Math.floor(positive(input.deviceCount, 1)));
  const precisionBits = input.trainingPrecisionBits ?? 16;
  const method = input.method;
  const targetCoverage =
    method === 'qlora' ? 'all-linear' : input.targetCoverage;
  const trainableParameterCount =
    method === 'full'
      ? parameterCount
      : estimateLoraTrainableParameters({
          ...input,
          rank: input.loraRank,
          targetCoverage,
        });
  const baseWeightBits =
    method === 'qlora' ? QLORA_STORAGE_BITS : precisionBits;
  const baseWeightsGB = toGB(parameterCount * (baseWeightBits / 8));
  const adapterWeightsGB =
    method === 'full' ? 0 : toGB(trainableParameterCount * (precisionBits / 8));
  const gradientsGB = toGB(trainableParameterCount * (precisionBits / 8));
  const optimizerGB = toGB(
    trainableParameterCount *
      optimizerBytesPerParameter(input.optimizer, input.optimizerPrecisionBits),
  );
  const activationFactor = input.gradientCheckpointing ? 4 : 12;
  const quantizationFactor = method === 'qlora' ? 1.15 : 1;
  const activationsGB = toGB(
    positive(input.microBatchSize, 1) *
      positive(input.sequenceLength, 1) *
      positive(input.hiddenSize, 1) *
      positive(input.numLayers, 1) *
      (precisionBits / 8) *
      activationFactor *
      quantizationFactor,
  );
  const shardDivisor = input.distribution === 'fully-sharded' ? deviceCount : 1;
  const baseWeightsPerDeviceGB = baseWeightsGB / shardDivisor;
  const adapterWeightsPerDeviceGB = adapterWeightsGB / shardDivisor;
  const gradientsPerDeviceGB = gradientsGB / shardDivisor;
  const optimizerPerDeviceGB = optimizerGB / shardDivisor;
  const persistentPerDeviceGB =
    baseWeightsPerDeviceGB +
    adapterWeightsPerDeviceGB +
    gradientsPerDeviceGB +
    optimizerPerDeviceGB;
  const beforeOverheadGB = persistentPerDeviceGB + activationsGB;
  const overheadFactor = Math.max(1, input.overheadFactor ?? DEFAULT_OVERHEAD);
  const overheadGB = beforeOverheadGB * (overheadFactor - 1);
  const perDeviceGB = beforeOverheadGB + overheadGB;

  return {
    method,
    trainableParameterCount,
    trainablePercent:
      parameterCount > 0 ? (trainableParameterCount / parameterCount) * 100 : 0,
    baseWeightBits,
    baseWeightsGB,
    adapterWeightsGB,
    gradientsGB,
    optimizerGB,
    activationsGB,
    persistentPerDeviceGB,
    baseWeightsPerDeviceGB,
    adapterWeightsPerDeviceGB,
    gradientsPerDeviceGB,
    optimizerPerDeviceGB,
    overheadGB,
    perDeviceGB,
    aggregateGB: perDeviceGB * deviceCount,
    deviceCount,
    distribution: input.distribution,
  };
}

export function estimateTrainingRun({
  activeParameterCount,
  method,
  datasetTokens,
  epochs,
  deviceCount,
  tflopsPerDevice,
  efficiency,
  hourlyRatePerDevice = 0,
}: TrainingRunInput): TrainingRunEstimate {
  const devices = Math.max(1, Math.floor(positive(deviceCount, 1)));
  const totalTrainingTokens = positive(datasetTokens) * positive(epochs);
  const computeMultiplier = method === 'full' ? 6 : 4.5;
  const flopsPerToken = positive(activeParameterCount) * computeMultiplier;
  const scalingEfficiency =
    devices === 1 ? 1 : Math.max(0.55, 0.92 ** (devices - 1));
  const effectiveFlopsPerSecond =
    positive(tflopsPerDevice) *
    10 ** 12 *
    devices *
    Math.max(0, Math.min(1, efficiency)) *
    scalingEfficiency;
  const tokensPerSecond =
    flopsPerToken > 0 ? effectiveFlopsPerSecond / flopsPerToken : 0;
  const durationHours =
    tokensPerSecond > 0 ? totalTrainingTokens / tokensPerSecond / 3600 : 0;
  const clusterHourlyRate = Math.max(0, hourlyRatePerDevice) * devices;

  return {
    totalTrainingTokens,
    flopsPerToken,
    tokensPerSecond,
    durationHours,
    scalingEfficiency,
    clusterHourlyRate,
    totalCost: clusterHourlyRate * durationHours,
  };
}

function trainingTFlops(
  gpu: HardwareLike,
  computeFormat: TrainingComputeFormat,
): number {
  if (computeFormat === 'fp32') return positive(gpu.fp32_tflops);
  if (computeFormat === 'fp16') {
    return positive(gpu.fp16_tflops ?? 0) || positive(gpu.fp32_tflops);
  }
  return (
    positive(gpu.bf16_tflops ?? 0) ||
    positive(gpu.fp16_tflops ?? 0) ||
    positive(gpu.fp32_tflops)
  );
}

export function recommendTrainingPlans({
  memoryInput,
  runInput,
  hardware,
  cloudRates,
  computeFormat,
  distributions = [memoryInput.distribution],
  maxDevices = 8,
}: RecommendTrainingPlansInput): TrainingPlanRecommendation[] {
  const hardwareByName = new Map(hardware.map((gpu) => [gpu.name, gpu]));
  const plans: TrainingPlanRecommendation[] = [];

  for (const rate of cloudRates) {
    for (const gpuName of rate.gpu_catalog_names) {
      const gpu = hardwareByName.get(gpuName);
      if (!gpu) continue;
      const memoryPerGpuGB =
        gpu.per_device_memory_gb ??
        gpu.memory_gb / Math.max(1, gpu.device_count ?? 1);
      const tflops = trainingTFlops(gpu, computeFormat);
      if (memoryPerGpuGB <= 0 || tflops <= 0) continue;

      for (const distribution of distributions) {
        const firstDeviceCount = distribution === 'fully-sharded' ? 2 : 1;
        for (
          let deviceCount = firstDeviceCount;
          deviceCount <= maxDevices;
          deviceCount += 1
        ) {
          const memory = estimateTrainingMemory({
            ...memoryInput,
            deviceCount,
            distribution,
          });
          if (memory.perDeviceGB > memoryPerGpuGB) continue;
          const run = estimateTrainingRun({
            ...runInput,
            deviceCount,
            tflopsPerDevice: tflops,
            hourlyRatePerDevice: rate.hourly_rate,
          });
          plans.push({
            gpuName,
            provider: rate.provider,
            offeringName: rate.name,
            deviceCount,
            memoryPerGpuGB,
            memoryHeadroomGB: memoryPerGpuGB - memory.perDeviceGB,
            memory,
            run,
            durationHours: run.durationHours,
            totalCost: run.totalCost,
            hourlyRatePerDevice: rate.hourly_rate,
            pricingSourceUrl: rate.pricing_source_url,
            sourceCheckedAt: rate.source_checked_at,
            distribution,
          });
          break;
        }
      }
    }
  }

  return plans.sort(
    (left, right) =>
      left.totalCost - right.totalCost ||
      left.durationHours - right.durationHours,
  );
}
