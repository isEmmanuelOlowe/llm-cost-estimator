import Head from 'next/head';
import { useCallback, useMemo, useState } from 'react';

import {
  inspectHuggingFaceModel,
  type ModelInspection,
} from '@/lib/model-metadata';
import { fuzzySearchModels } from '@/lib/model-search';

import generatedModelPresets from '@/data/model-presets.generated.json';

import ThemeCycleButton from '@/components/layout/ThemeCycleButton';
import DeploymentDecisionPath from '@/components/model/DeploymentDecisionPath';
import KvCacheScalingCard from '@/components/model/KvCacheScalingCard';
import ModelArchitectureDiagram from '@/components/model/ModelArchitectureDiagram';
import TrainingPlannerCard from '@/components/model/TrainingPlannerCard';
import VramUsageBar from '@/components/model/VramUsageBar';
import Seo from '@/components/Seo';

import cloudInstances from '@/estimator/cloud-instances.json';
import {
  EstimateClassification,
  estimateCloudCost,
  estimateDecoderFlops,
  estimateHardwareFit,
  estimateLlamaStyleArchitecture,
  estimateMemory,
  estimateThroughput,
  estimateTransformerParameterBreakdown,
  ExecutionMode,
  type HardwareLike,
  MemoryBreakdown,
  PrecisionBits,
  recommendGpus,
  resolveEffectiveParameterCount,
  selectGpuComputeTFlops,
  WeightFormat,
} from '@/estimator/estimator';
import { groupGpus } from '@/estimator/gpu-groups';
import gpus from '@/estimator/gpus.json';
import {
  estimateTrainingMemory,
  estimateTrainingRun,
  recommendTrainingPlans,
  type TrainingMemoryInput,
  type TrainingPlannerSettings,
} from '@/estimator/training';

type CloudInstance = (typeof cloudInstances)[number];
type Gpu = (typeof gpus)[number];

type GeneratedModelPreset = (typeof generatedModelPresets)[number];
type ModelPreset = GeneratedModelPreset & {
  sourceModelId?: string;
  releaseSourceUrl?: string;
};

const DEFAULT_MODEL_ID = 'google/gemma-4-12B';
const gpuGroups = groupGpus(gpus);
const glmFiveTwo = generatedModelPresets.find(
  (preset) => preset.id === 'zai-org/GLM-5.2',
);
const glmFiveThree: ModelPreset | undefined = glmFiveTwo
  ? {
      ...glmFiveTwo,
      id: 'zai-org/GLM-5.3',
      label: 'GLM-5.3',
      sourceModelId: glmFiveTwo.id,
      releaseSourceUrl: 'https://z.ai/blog/glm-5.3',
      license: 'mit',
      sourceCheckedAt: '2026-08-24',
      summary:
        'GLM-5.3 coding-focused post-training release on the GLM-5.2 base; architecture follows the shared base checkpoint while public 5.3 weights are pending.',
    }
  : undefined;
const modelPresets: ModelPreset[] = [
  ...(glmFiveThree ? [glmFiveThree] : []),
  ...generatedModelPresets,
];
const featuredPresetIds = [
  'google/gemma-4-12B',
  'Qwen/Qwen3.8-27B',
  'zai-org/GLM-5.3',
  'moonshotai/Kimi-K3',
  'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
];
const featuredPresets = featuredPresetIds
  .map((id) => modelPresets.find((preset) => preset.id === id))
  .filter((preset): preset is ModelPreset => Boolean(preset));

const bitsOptions: PrecisionBits[] = [32, 16, 8, 4];
const weightFormatOptions: Array<{
  value: WeightFormat;
  label: string;
  bits: PrecisionBits;
}> = [
  { value: 'bf16', label: 'BF16', bits: 16 },
  { value: 'fp16', label: 'FP16', bits: 16 },
  { value: 'fp8', label: 'FP8', bits: 8 },
  { value: 'int8', label: 'INT8', bits: 8 },
  { value: 'int4', label: 'INT4', bits: 4 },
  { value: 'nvfp4', label: 'NVFP4', bits: 4 },
  { value: 'mxfp4', label: 'MXFP4', bits: 4 },
];
const weightFormatBits = Object.fromEntries(
  weightFormatOptions.map((option) => [option.value, option.bits]),
) as Record<WeightFormat, PrecisionBits>;

function formatModelLicense(license: string | null | undefined): string {
  if (!license) return 'License unknown';
  const normalized = license.toLowerCase();
  if (normalized === 'apache-2.0') return 'Apache 2.0';
  if (normalized === 'mit') return 'MIT';
  if (normalized === 'other') return 'Custom license';
  return license;
}

function formatFromArchitecture(
  dtype: string | undefined,
  bits: PrecisionBits,
): WeightFormat {
  const normalized = dtype?.toLowerCase() ?? '';
  if (normalized.includes('float8') || normalized === 'fp8') return 'fp8';
  if (normalized.includes('float16') && !normalized.includes('bfloat')) {
    return 'fp16';
  }
  if (normalized.includes('int4') || normalized.includes('nf4')) return 'int4';
  if (normalized.includes('int8')) return 'int8';
  if (bits === 8) return 'fp8';
  if (bits === 4) return 'int4';
  return 'bf16';
}

const learningResources = [
  {
    title: 'Transformers architecture docs',
    detail:
      'Configuration fields, model classes, attention, cache, and implementation references.',
    href: 'https://huggingface.co/docs/transformers/en/index',
  },
  {
    title: 'Hugging Face model hub',
    detail:
      'Compare model cards, revisions, files, safetensors metadata, and licenses.',
    href: 'https://huggingface.co/models',
  },
  {
    title: 'Model families',
    detail:
      'Browse current Gemma, Qwen, DeepSeek, GLM, and other open-weight releases.',
    href: 'https://huggingface.co/models?pipeline_tag=text-generation&sort=trending',
  },
  {
    title: 'KV cache & serving',
    detail:
      'Understand paged attention, continuous batching, quantization, and serving tradeoffs.',
    href: 'https://huggingface.co/docs/transformers/en/kv_cache',
  },
];

function classificationBadgeClass(
  classification: EstimateClassification,
): string {
  return classification === 'exact' ? 'badge-success' : 'badge-warning';
}

function formatNumber(value: number, fractionDigits = 2): string {
  if (!Number.isFinite(value)) return 'N/A';
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString('en-US', {
      maximumFractionDigits: fractionDigits,
      minimumFractionDigits: 0,
    });
  }
  return value.toFixed(fractionDigits);
}

function formatMemory(value: number): string {
  if (value <= 0) return '0 GB';
  return `${formatNumber(value)} GB`;
}

export default function HomePage() {
  const [modelId, setModelId] = useState<string>(DEFAULT_MODEL_ID);
  const [modelQuery, setModelQuery] = useState<string>(DEFAULT_MODEL_ID);
  const [isModelSearchOpen, setIsModelSearchOpen] = useState(false);
  const [parameterBillions, setParameterBillions] =
    useState<number>(11.959730224);
  const [sequenceLength, setSequenceLength] = useState<number>(131072);
  const [vocabSize, setVocabSize] = useState<number>(262144);
  const [mode, setMode] = useState<ExecutionMode>('inference');
  const [weightFormat, setWeightFormat] = useState<WeightFormat>('bf16');
  const weightBits = weightFormatBits[weightFormat];
  const [kvBits, setKvBits] = useState<PrecisionBits>(16);
  const [overheadFactor, setOverheadFactor] = useState<number>(1.15);
  const [trainingSettings, setTrainingSettings] =
    useState<TrainingPlannerSettings>({
      method: 'qlora',
      loraRank: 16,
      targetCoverage: 'attention-qv',
      sequenceLength: 2048,
      datasetTokens: 100_000_000,
      epochs: 1,
      globalBatchSize: 32,
      microBatchSize: 1,
      deviceCount: 1,
      distribution: 'replicated',
      optimizer: 'adamw',
      gradientCheckpointing: true,
      efficiency: 0.3,
      computeFormat: 'bf16',
      optimizerPrecisionBits: 32,
    });
  const [efficiency, setEfficiency] = useState<number>(0.3);
  const [memoryEfficiency, setMemoryEfficiency] = useState<number>(0.65);
  const [concurrentUsers, setConcurrentUsers] = useState<number>(1);
  const [architectureMode, setArchitectureMode] = useState<'auto' | 'manual'>(
    'manual',
  );
  const [manualHiddenSize, setManualHiddenSize] = useState<number>(3840);
  const [manualNumLayers, setManualNumLayers] = useState<number>(48);
  const [manualNumHeads, setManualNumHeads] = useState<number>(16);
  const [manualNumKeyValueHeads, setManualNumKeyValueHeads] =
    useState<number>(8);
  const [manualHeadDim, setManualHeadDim] = useState<number>(256);
  const [manualIntermediateSize, setManualIntermediateSize] =
    useState<number>(15360);
  const [manualExpertIntermediateSize, setManualExpertIntermediateSize] =
    useState<number>(0);
  const [
    manualSharedExpertIntermediateSize,
    setManualSharedExpertIntermediateSize,
  ] = useState<number>(0);
  const [manualNumSharedExperts, setManualNumSharedExperts] =
    useState<number>(0);
  const [manualIsEncoderDecoder, setManualIsEncoderDecoder] =
    useState<boolean>(false);
  const [manualModality, setManualModality] = useState<string>('multimodal');
  const [manualNumExperts, setManualNumExperts] = useState<number>(0);
  const [manualNumExpertsPerToken, setManualNumExpertsPerToken] =
    useState<number>(0);
  const [manualGatedMlp, setManualGatedMlp] = useState<boolean>(false);
  const [manualTieWordEmbeddings, setManualTieWordEmbeddings] =
    useState<boolean>(true);
  const [selectedGpuName, setSelectedGpuName] = useState<Gpu['name']>(
    gpus[0].name,
  );
  const [selectedInstanceName, setSelectedInstanceName] = useState<
    CloudInstance['name']
  >(cloudInstances[0].name);
  const [runtimeHours, setRuntimeHours] = useState<number>(1);
  const [customHourlyRate, setCustomHourlyRate] = useState<number | ''>('');
  const [modelError, setModelError] = useState<string | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(false);
  const [modelInspection, setModelInspection] =
    useState<ModelInspection | null>(null);
  const [parameterSource, setParameterSource] = useState<
    | 'manual'
    | 'catalog'
    | 'huggingface-safetensors'
    | 'huggingface-config'
    | 'estimated-from-architecture'
  >('catalog');
  const selectedPreset = useMemo(
    () =>
      modelQuery.trim() === modelId
        ? modelPresets.find((preset) => preset.id === modelId)
        : undefined,
    [modelId, modelQuery],
  );
  const modelSuggestions = useMemo(
    () => fuzzySearchModels(modelPresets, modelQuery, 8),
    [modelQuery],
  );
  const updateTrainingSettings = useCallback(
    (patch: Partial<TrainingPlannerSettings>) => {
      setTrainingSettings((current) => ({ ...current, ...patch }));
    },
    [],
  );

  const parameterCount = useMemo(
    () => Math.max(parameterBillions, 0) * 10 ** 9,
    [parameterBillions],
  );

  const autoArchitecture = useMemo(
    () => estimateLlamaStyleArchitecture(parameterCount),
    [parameterCount],
  );

  const enableManualOverrides = useCallback(() => {
    if (architectureMode === 'manual') return;
    if (autoArchitecture.hiddenSize) {
      setManualHiddenSize(autoArchitecture.hiddenSize);
    }
    if (autoArchitecture.numLayers) {
      setManualNumLayers(autoArchitecture.numLayers);
    }
    if (autoArchitecture.numHeads) {
      setManualNumHeads(autoArchitecture.numHeads);
      setManualNumKeyValueHeads(autoArchitecture.numHeads);
      setManualHeadDim(autoArchitecture.hiddenSize / autoArchitecture.numHeads);
    }
    if (autoArchitecture.intermediateSize) {
      setManualIntermediateSize(autoArchitecture.intermediateSize);
    }
    setArchitectureMode('manual');
  }, [
    architectureMode,
    autoArchitecture.hiddenSize,
    autoArchitecture.intermediateSize,
    autoArchitecture.numHeads,
    autoArchitecture.numLayers,
  ]);

  const enableAutoArchitecture = useCallback(() => {
    setArchitectureMode('auto');
  }, []);

  const effectiveHiddenSize = useMemo(() => {
    const autoValue = autoArchitecture.hiddenSize || manualHiddenSize;
    return architectureMode === 'auto'
      ? autoValue
      : manualHiddenSize || autoValue;
  }, [architectureMode, autoArchitecture.hiddenSize, manualHiddenSize]);

  const effectiveNumLayers = useMemo(() => {
    const autoValue = autoArchitecture.numLayers || manualNumLayers;
    return architectureMode === 'auto'
      ? autoValue
      : manualNumLayers || autoValue;
  }, [architectureMode, autoArchitecture.numLayers, manualNumLayers]);

  const effectiveNumHeads = useMemo(() => {
    const fallback = Math.max(1, Math.round((effectiveHiddenSize || 1) / 128));
    if (architectureMode === 'auto') {
      return autoArchitecture.numHeads || fallback;
    }
    return manualNumHeads || autoArchitecture.numHeads || fallback;
  }, [
    architectureMode,
    autoArchitecture.numHeads,
    effectiveHiddenSize,
    manualNumHeads,
  ]);

  const effectiveIntermediateSize = useMemo(() => {
    const autoValue =
      autoArchitecture.intermediateSize || manualIntermediateSize || 0;
    if (architectureMode === 'auto') {
      return autoValue;
    }
    if (manualIntermediateSize && manualIntermediateSize > 0) {
      return manualIntermediateSize;
    }
    return manualHiddenSize > 0 ? manualHiddenSize * 4 : autoValue;
  }, [
    architectureMode,
    autoArchitecture.intermediateSize,
    manualHiddenSize,
    manualIntermediateSize,
  ]);

  const effectiveNumKeyValueHeads = useMemo(() => {
    const fallback = effectiveNumHeads || 1;
    return architectureMode === 'auto'
      ? fallback
      : Math.max(1, manualNumKeyValueHeads || fallback);
  }, [architectureMode, effectiveNumHeads, manualNumKeyValueHeads]);

  const effectiveHeadDim = useMemo(() => {
    if (architectureMode === 'manual' && manualHeadDim > 0) {
      return manualHeadDim;
    }
    return effectiveNumHeads > 0 ? effectiveHiddenSize / effectiveNumHeads : 0;
  }, [architectureMode, effectiveHiddenSize, effectiveNumHeads, manualHeadDim]);

  const effectiveNumExperts = useMemo(
    () => (architectureMode === 'manual' ? Math.max(0, manualNumExperts) : 0),
    [architectureMode, manualNumExperts],
  );

  const effectiveNumExpertsPerToken = useMemo(
    () =>
      effectiveNumExperts > 0
        ? Math.min(
            effectiveNumExperts,
            Math.max(1, manualNumExpertsPerToken || effectiveNumExperts),
          )
        : 0,
    [effectiveNumExperts, manualNumExpertsPerToken],
  );

  const effectiveExpertIntermediateSize =
    architectureMode === 'manual'
      ? manualExpertIntermediateSize || effectiveIntermediateSize
      : effectiveIntermediateSize;
  const effectiveSharedExpertIntermediateSize =
    architectureMode === 'manual' ? manualSharedExpertIntermediateSize || 0 : 0;
  const effectiveNumSharedExperts =
    architectureMode === 'manual' ? manualNumSharedExperts || 0 : 0;
  const effectiveIsEncoderDecoder =
    architectureMode === 'manual' && manualIsEncoderDecoder;
  const effectiveModality =
    architectureMode === 'manual' ? manualModality : 'text';

  const effectiveGatedMlp = architectureMode === 'auto' ? true : manualGatedMlp;
  const effectiveTieWordEmbeddings =
    architectureMode === 'auto' ? true : manualTieWordEmbeddings;
  const effectiveKvCacheArchitecture =
    modelInspection?.architecture.kvCacheArchitecture ??
    selectedPreset?.kvCacheArchitecture ??
    undefined;

  const effectiveBatchSize = useMemo(() => {
    if (mode === 'inference') {
      return Math.max(1, concurrentUsers);
    }
    return Math.max(1, trainingSettings.globalBatchSize);
  }, [concurrentUsers, mode, trainingSettings.globalBatchSize]);

  const selectedGpu = useMemo(() => {
    return gpus.find((gpu) => gpu.name === selectedGpuName) ?? gpus[0];
  }, [selectedGpuName]);
  const selectedComputeTFlops = useMemo(
    () => selectGpuComputeTFlops(selectedGpu, weightBits, weightFormat),
    [selectedGpu, weightBits, weightFormat],
  );

  const matchingCloudInstances = useMemo(
    () =>
      cloudInstances.filter((instance) =>
        instance.gpu_catalog_names.includes(selectedGpu.name),
      ),
    [selectedGpu.name],
  );

  const selectedInstance = useMemo(() => {
    return (
      matchingCloudInstances.find(
        (instance) => instance.name === selectedInstanceName,
      ) ?? matchingCloudInstances[0]
    );
  }, [matchingCloudInstances, selectedInstanceName]);

  const effectiveActiveParameterCount = useMemo(
    () =>
      resolveEffectiveParameterCount(
        parameterCount,
        parameterSource === 'catalog'
          ? selectedPreset?.activeParameterCount
          : parameterSource === 'huggingface-safetensors' ||
              parameterSource === 'huggingface-config'
            ? modelInspection?.activeParameterCount
            : null,
      ),
    [parameterCount, parameterSource, selectedPreset, modelInspection],
  );

  const trainingMemoryInput = useMemo<TrainingMemoryInput>(
    () => ({
      parameterCount,
      hiddenSize: effectiveHiddenSize,
      numLayers: effectiveNumLayers,
      numAttentionHeads: effectiveNumHeads,
      numKeyValueHeads: effectiveNumKeyValueHeads,
      headDim: effectiveHeadDim,
      intermediateSize: effectiveIntermediateSize,
      method: trainingSettings.method,
      loraRank: trainingSettings.loraRank,
      targetCoverage: trainingSettings.targetCoverage,
      sequenceLength: trainingSettings.sequenceLength,
      microBatchSize: trainingSettings.microBatchSize,
      deviceCount: trainingSettings.deviceCount,
      distribution: trainingSettings.distribution,
      optimizer: trainingSettings.optimizer,
      gradientCheckpointing: trainingSettings.gradientCheckpointing,
      overheadFactor,
      trainingPrecisionBits:
        trainingSettings.computeFormat === 'fp32' ? 32 : 16,
      optimizerPrecisionBits: trainingSettings.optimizerPrecisionBits,
    }),
    [
      parameterCount,
      effectiveHiddenSize,
      effectiveNumLayers,
      effectiveNumHeads,
      effectiveNumKeyValueHeads,
      effectiveHeadDim,
      effectiveIntermediateSize,
      trainingSettings,
      overheadFactor,
    ],
  );
  const trainingMemory = useMemo(
    () => estimateTrainingMemory(trainingMemoryInput),
    [trainingMemoryInput],
  );
  const selectedTrainingTFlops = useMemo(() => {
    if (trainingSettings.computeFormat === 'fp32') {
      return selectedGpu.fp32_tflops;
    }
    if (trainingSettings.computeFormat === 'fp16') {
      return selectedGpu.fp16_tflops ?? selectedGpu.fp32_tflops;
    }
    return (
      selectedGpu.bf16_tflops ??
      selectedGpu.fp16_tflops ??
      selectedGpu.fp32_tflops
    );
  }, [selectedGpu, trainingSettings.computeFormat]);
  const selectedTrainingHourlyRate =
    typeof customHourlyRate === 'number' && customHourlyRate > 0
      ? customHourlyRate
      : (selectedInstance?.hourly_rate ?? 0);
  const trainingRun = useMemo(
    () =>
      estimateTrainingRun({
        activeParameterCount: effectiveActiveParameterCount,
        method: trainingSettings.method,
        datasetTokens: trainingSettings.datasetTokens,
        epochs: trainingSettings.epochs,
        deviceCount: trainingSettings.deviceCount,
        tflopsPerDevice: selectedTrainingTFlops,
        efficiency: trainingSettings.efficiency,
        hourlyRatePerDevice: selectedTrainingHourlyRate,
      }),
    [
      effectiveActiveParameterCount,
      trainingSettings,
      selectedTrainingTFlops,
      selectedTrainingHourlyRate,
    ],
  );
  const trainingRecommendations = useMemo(
    () =>
      recommendTrainingPlans({
        memoryInput: trainingMemoryInput,
        runInput: {
          activeParameterCount: effectiveActiveParameterCount,
          method: trainingSettings.method,
          datasetTokens: trainingSettings.datasetTokens,
          epochs: trainingSettings.epochs,
          efficiency: trainingSettings.efficiency,
        },
        hardware: gpus as unknown as HardwareLike[],
        cloudRates: cloudInstances,
        computeFormat: trainingSettings.computeFormat,
        distributions: ['replicated', 'fully-sharded'],
        maxDevices: 8,
      }),
    [
      trainingMemoryInput,
      effectiveActiveParameterCount,
      trainingSettings.method,
      trainingSettings.datasetTokens,
      trainingSettings.epochs,
      trainingSettings.efficiency,
      trainingSettings.computeFormat,
    ],
  );

  const flops = useMemo(() => {
    if (
      !effectiveNumLayers ||
      !effectiveHiddenSize ||
      !sequenceLength ||
      !vocabSize
    ) {
      return 0;
    }

    return estimateDecoderFlops({
      numLayers: effectiveNumLayers,
      hiddenSize: effectiveHiddenSize,
      sequenceLength:
        mode === 'training' ? trainingSettings.sequenceLength : sequenceLength,
      vocabSize,
      intermediateSize: effectiveIntermediateSize,
      numAttentionHeads: effectiveNumHeads,
      numKeyValueHeads: effectiveNumKeyValueHeads,
      headDim: effectiveHeadDim,
      gatedMlp: effectiveGatedMlp,
    });
  }, [
    effectiveHiddenSize,
    effectiveNumLayers,
    effectiveIntermediateSize,
    effectiveNumHeads,
    effectiveNumKeyValueHeads,
    effectiveHeadDim,
    effectiveGatedMlp,
    mode,
    sequenceLength,
    trainingSettings.sequenceLength,
    vocabSize,
  ]);

  const memoryBreakdown: MemoryBreakdown = useMemo(() => {
    if (!parameterCount || !effectiveHiddenSize || !effectiveNumLayers) {
      return {
        weightsGB: 0,
        activationsGB: 0,
        kvCacheGB: 0,
        stateCacheGB: 0,
        kvCacheBytesPerToken: 0,
        kvCacheTokens: 0,
        kvCacheMode: 'standard',
        kvCacheDescription: 'No KV cache estimate is available.',
        kvAttentionLayers: 0,
        optimizerGB: 0,
        baseTotalGB: 0,
        overheadGB: 0,
        totalGB: 0,
      };
    }

    return estimateMemory({
      parameterCount,
      weightPrecisionBits: weightBits,
      mode,
      hiddenSize: effectiveHiddenSize,
      numLayers: effectiveNumLayers,
      numAttentionHeads: effectiveNumHeads,
      numKeyValueHeads: effectiveNumKeyValueHeads,
      headDim: effectiveHeadDim,
      sequenceLength:
        mode === 'training' ? trainingSettings.sequenceLength : sequenceLength,
      batchSize: effectiveBatchSize,
      kvCachePrecisionBits: kvBits,
      optimizer: mode === 'training' ? trainingSettings.optimizer : 'none',
      overheadFactor,
      kvCacheArchitecture: effectiveKvCacheArchitecture,
    });
  }, [
    parameterCount,
    weightBits,
    mode,
    effectiveHiddenSize,
    effectiveNumLayers,
    effectiveNumHeads,
    effectiveNumKeyValueHeads,
    effectiveHeadDim,
    sequenceLength,
    effectiveBatchSize,
    kvBits,
    trainingSettings.optimizer,
    trainingSettings.sequenceLength,
    overheadFactor,
    effectiveKvCacheArchitecture,
  ]);

  const throughput = useMemo(() => {
    if (!parameterCount) {
      return {
        tokensPerSecond: 0,
        millisecondsPerToken: 0,
        computeBoundTokensPerSecond: 0,
        memoryBoundTokensPerSecond: 0,
        bottleneck: 'unavailable' as const,
      };
    }

    return estimateThroughput({
      parameterCount: effectiveActiveParameterCount,
      gpuTFlops: selectedGpu?.fp32_tflops ?? 0,
      precisionTFlops: selectedComputeTFlops,
      efficiency,
      memoryBandwidthGBs: selectedGpu?.memory_bandwidth_gb_s,
      weightPrecisionBits: weightBits,
      computeFormat: weightFormat,
      batchSize: effectiveBatchSize,
      memoryEfficiency,
    });
  }, [
    parameterCount,
    effectiveActiveParameterCount,
    selectedGpu,
    efficiency,
    memoryEfficiency,
    effectiveBatchSize,
    weightBits,
    weightFormat,
    selectedComputeTFlops,
  ]);

  const parameterBreakdown = useMemo(() => {
    if (
      effectiveNumExperts > 1 ||
      !effectiveHiddenSize ||
      !effectiveNumLayers ||
      !effectiveNumHeads ||
      !vocabSize
    ) {
      return undefined;
    }

    return estimateTransformerParameterBreakdown({
      vocabSize,
      hiddenSize: effectiveHiddenSize,
      numLayers: effectiveNumLayers,
      numAttentionHeads: effectiveNumHeads,
      numKeyValueHeads: effectiveNumKeyValueHeads,
      headDim: effectiveHeadDim,
      intermediateSize: effectiveIntermediateSize,
      numExperts: effectiveNumExperts || undefined,
      numExpertsPerToken: effectiveNumExpertsPerToken || undefined,
      gatedMlp: effectiveGatedMlp,
      tieWordEmbeddings: effectiveTieWordEmbeddings,
    });
  }, [
    effectiveHiddenSize,
    effectiveNumLayers,
    effectiveNumHeads,
    effectiveNumKeyValueHeads,
    effectiveHeadDim,
    effectiveIntermediateSize,
    effectiveNumExperts,
    effectiveNumExpertsPerToken,
    effectiveGatedMlp,
    effectiveTieWordEmbeddings,
    vocabSize,
  ]);

  const hourlyRate = useMemo(() => {
    if (typeof customHourlyRate === 'number' && customHourlyRate > 0) {
      return customHourlyRate;
    }
    return selectedInstance?.hourly_rate ?? 0;
  }, [customHourlyRate, selectedInstance?.hourly_rate]);

  const costEstimate = useMemo(() => {
    if (!runtimeHours || runtimeHours <= 0 || hourlyRate <= 0) {
      return { totalCost: 0, hourlyRate, durationHours: runtimeHours };
    }

    return estimateCloudCost({
      hourlyRate,
      durationHours: runtimeHours,
    });
  }, [hourlyRate, runtimeHours]);

  const recommendedGpuList = useMemo(() => {
    if (!memoryBreakdown.totalGB) return [];
    return recommendGpus(memoryBreakdown.totalGB, 5);
  }, [memoryBreakdown.totalGB]);

  const selectedHardwareFit = useMemo(
    () => estimateHardwareFit(memoryBreakdown.totalGB, selectedGpu),
    [memoryBreakdown.totalGB, selectedGpu],
  );
  const selectedGpuPerDeviceMemory =
    selectedGpu.per_device_memory_gb ??
    selectedGpu.memory_gb / Math.max(1, selectedGpu.device_count ?? 1);
  const trainingFitsSelected =
    trainingMemory.perDeviceGB <= selectedGpuPerDeviceMemory;
  const displayMemoryGB =
    mode === 'training' ? trainingMemory.perDeviceGB : memoryBreakdown.totalGB;
  const displayCapacityGB =
    mode === 'training' ? selectedGpuPerDeviceMemory : selectedGpu.memory_gb;
  const displayHeadroomGB =
    mode === 'training'
      ? selectedGpuPerDeviceMemory - trainingMemory.perDeviceGB
      : selectedHardwareFit.aggregateHeadroomGB;
  const displayFits =
    mode === 'training' ? trainingFitsSelected : selectedHardwareFit.fits;

  const applyModelMetadata = useCallback(
    (metadata: {
      parameterCount?: number;
      parameterSource?: ModelInspection['parameterSource'];
      architecture: ModelInspection['architecture'];
    }) => {
      if (metadata.parameterCount && metadata.parameterCount > 0) {
        setParameterBillions(metadata.parameterCount / 10 ** 9);
      }
      if (metadata.parameterSource === 'estimated-from-architecture') {
        setParameterSource('estimated-from-architecture');
      } else if (
        metadata.parameterSource === 'huggingface-safetensors' ||
        metadata.parameterSource === 'huggingface-config'
      ) {
        setParameterSource(metadata.parameterSource);
      }

      const architecture = metadata.architecture;
      const hasArchitecture = Boolean(
        (architecture.hiddenSize && architecture.hiddenSize > 0) ||
        (architecture.numLayers && architecture.numLayers > 0) ||
        (architecture.numAttentionHeads &&
          architecture.numAttentionHeads > 0) ||
        (architecture.intermediateSize && architecture.intermediateSize > 0),
      );

      if (hasArchitecture) setArchitectureMode('manual');
      const hiddenSize = architecture.hiddenSize ?? 0;
      const attentionHeads =
        architecture.numAttentionHeads ??
        (hiddenSize > 0 ? Math.max(1, Math.round(hiddenSize / 128)) : 0);
      setManualHiddenSize(hiddenSize);
      setManualNumLayers(architecture.numLayers ?? 0);
      setManualNumHeads(attentionHeads);
      setManualNumKeyValueHeads(
        architecture.numKeyValueHeads ?? attentionHeads,
      );
      setManualHeadDim(architecture.headDim ?? 0);
      setManualIntermediateSize(architecture.intermediateSize ?? 0);
      setManualExpertIntermediateSize(architecture.expertIntermediateSize ?? 0);
      setManualSharedExpertIntermediateSize(
        architecture.sharedExpertIntermediateSize ?? 0,
      );
      setManualNumSharedExperts(architecture.numSharedExperts ?? 0);
      setManualIsEncoderDecoder(architecture.isEncoderDecoder ?? false);
      setManualModality(
        architecture.modality ??
          (architecture.modelType?.includes('vision') ||
          architecture.modelType?.includes('audio')
            ? 'multimodal'
            : 'text'),
      );
      setManualNumExperts(architecture.numExperts ?? 0);
      setManualNumExpertsPerToken(architecture.numExpertsPerToken ?? 0);
      setManualGatedMlp(architecture.gatedMlp ?? true);
      setManualTieWordEmbeddings(architecture.tieWordEmbeddings ?? true);
      setSequenceLength(architecture.contextLength ?? 0);
      setVocabSize(architecture.vocabSize ?? 0);
      if (architecture.dtypeBits) {
        setWeightFormat(
          formatFromArchitecture(architecture.dtype, architecture.dtypeBits),
        );
      }
    },
    [],
  );

  const applyModelInspection = useCallback(
    (inspection: ModelInspection) => {
      setModelInspection(inspection);
      applyModelMetadata({
        parameterCount: inspection.parameterCount,
        parameterSource: inspection.parameterSource,
        architecture: {
          ...inspection.architecture,
          modality:
            inspection.pipelineTag?.includes('image') ||
            inspection.pipelineTag?.includes('audio') ||
            inspection.pipelineTag?.includes('video')
              ? 'multimodal'
              : inspection.architecture.modality,
        },
      });
    },
    [applyModelMetadata],
  );

  const applyModelPreset = useCallback(
    (preset: ModelPreset) => {
      setModelId(preset.id);
      setModelQuery(preset.id);
      setModelInspection(null);
      setModelError(null);
      setParameterSource('catalog');
      applyModelMetadata({
        parameterCount: preset.parameterCount,
        architecture: {
          modelType: preset.modelTypeTag,
          architectures: preset.architectures,
          modality: preset.modality,
          hiddenSize: preset.hiddenSize ?? undefined,
          numLayers: preset.numLayers ?? undefined,
          numAttentionHeads: preset.numHeads ?? undefined,
          numKeyValueHeads: preset.numKeyValueHeads ?? undefined,
          headDim: preset.headDim ?? undefined,
          intermediateSize: preset.intermediateSize ?? undefined,
          expertIntermediateSize: preset.expertIntermediateSize ?? undefined,
          sharedExpertIntermediateSize:
            preset.sharedExpertIntermediateSize ?? undefined,
          numSharedExperts: preset.numSharedExperts ?? undefined,
          isEncoderDecoder: preset.isEncoderDecoder ?? undefined,
          vocabSize: preset.vocabSize ?? undefined,
          contextLength: preset.contextLength,
          numExperts: preset.numExperts ?? undefined,
          numExpertsPerToken: preset.numExpertsPerToken ?? undefined,
          modalityArchitecture: preset.modalityArchitecture ?? undefined,
          kvCacheArchitecture: preset.kvCacheArchitecture ?? undefined,
          gatedMlp: preset.gatedMlp ?? undefined,
          tieWordEmbeddings: preset.tieWordEmbeddings ?? undefined,
          dtypeBits: (preset.dtypeBits as PrecisionBits | null) ?? undefined,
        },
      });
    },
    [applyModelMetadata],
  );

  const fetchModelConfig = useCallback(
    async (id: string, preserveSelection = false) => {
      setIsLoadingModel(true);
      setModelError(null);
      try {
        const inspection = await inspectHuggingFaceModel(id);
        if (!preserveSelection) {
          setModelId(inspection.id);
          setModelQuery(inspection.id);
        }
        applyModelInspection(inspection);
      } catch (error: unknown) {
        setModelError(
          error instanceof Error
            ? error.message
            : 'Unable to retrieve model configuration from Hugging Face.',
        );
        if (process.env.NODE_ENV !== 'production') {
          // eslint-disable-next-line no-console
          console.error(error);
        }
      } finally {
        setIsLoadingModel(false);
      }
    },
    [applyModelInspection],
  );

  return (
    <main className='photonic-shell min-h-screen bg-transparent text-base-content'>
      <Seo />
      <Head>
        <title>LLM Explorer</title>
      </Head>
      <div className='site-nav sticky top-0 z-40 border-b border-base-300/80 backdrop-blur'>
        <div className='mx-auto flex max-w-[1800px] items-center gap-4 px-5 py-3 sm:px-8 xl:px-12'>
          <a
            href='https://labiium.com'
            aria-label='Visit LABIIUM'
            className='flex shrink-0 items-baseline gap-2 tracking-tight text-base-content transition-colors hover:text-secondary'
          >
            <span className='font-display text-lg font-extrabold'>LABIIUM</span>
            <span className='hidden text-xs font-medium text-base-content/55 sm:inline'>
              / LLM Explorer
            </span>
          </a>
          <nav
            className='hidden min-w-0 flex-1 overflow-x-auto md:block'
            aria-label='Explorer sections'
          >
            <ul className='flex min-w-max items-center justify-end gap-1 text-xs font-medium text-base-content/70'>
              {[
                ['#inspect', 'Inspect'],
                ['#understand', 'Understand'],
                ['#estimate', 'Estimate'],
                ['#hardware', 'Hardware'],
                ...(mode === 'inference'
                  ? [
                      ['#performance', 'Performance'],
                      ['#cost', 'Cost'],
                    ]
                  : []),
              ].map(([href, label]) => (
                <li key={href}>
                  <a
                    href={href}
                    className='inline-flex rounded-lg px-3 py-2 transition hover:bg-base-100 hover:text-base-content focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary'
                  >
                    {label}
                  </a>
                </li>
              ))}
              <li>
                <details className='relative'>
                  <summary className='cursor-pointer list-none rounded-lg px-3 py-2 transition hover:bg-base-100 hover:text-base-content focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary'>
                    Learn
                  </summary>
                  <div className='absolute right-0 top-full z-50 mt-2 w-80 rounded-xl border border-base-300 bg-base-100 p-2 text-left shadow-2xl'>
                    <div className='px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-primary'>
                      Resources
                    </div>
                    {learningResources.map((resource) => (
                      <a
                        key={resource.title}
                        href={resource.href}
                        target='_blank'
                        rel='noreferrer'
                        className='block rounded-lg px-3 py-2 transition hover:bg-base-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary'
                      >
                        <span className='block text-xs font-semibold text-base-content'>
                          {resource.title} ↗
                        </span>
                        <span className='mt-1 block text-[11px] leading-relaxed text-base-content/60'>
                          {resource.detail}
                        </span>
                      </a>
                    ))}
                  </div>
                </details>
              </li>
            </ul>
          </nav>
          <ThemeCycleButton />
        </div>
      </div>
      <div
        id='top'
        className='mx-auto w-full max-w-[1800px] px-5 py-4 sm:px-8 xl:px-12'
      >
        <header className='flex flex-col gap-2 py-1 text-left md:flex-row md:items-center md:justify-between md:gap-8'>
          <div className='flex min-w-0 flex-wrap items-center gap-3'>
            <h1 className='text-3xl font-bold tracking-tight text-base-content'>
              LLM <span className='text-primary'>Explorer</span>
            </h1>
            <div className='hidden items-center gap-2 rounded-full border border-primary/25 bg-primary/10 px-3 py-1 text-[11px] font-semibold text-primary sm:flex'>
              <span className='size-1.5 rounded-full bg-primary' />
              Inspect → Understand → Deploy
            </div>
          </div>
          <p className='max-w-2xl text-xs leading-relaxed text-base-content/70 md:text-right'>
            From Hugging Face evidence to architecture, memory, hardware fit,
            and deployment cost.
          </p>
        </header>

        <section
          id='inspect'
          className='mt-4 scroll-mt-24 rounded-2xl border border-base-300 bg-base-100 p-6 shadow-lg shadow-black/10'
        >
          <div className='flex flex-col gap-4 md:flex-row md:items-end'>
            <div className='relative flex-1'>
              <label className='block'>
                <span className='label-text font-semibold'>
                  Hugging Face model ID or search
                </span>
                <input
                  className='input input-bordered mt-2 w-full'
                  placeholder='Search Kimi, DeepSeek, Gemma… or enter org/model'
                  value={modelQuery}
                  onFocus={() => setIsModelSearchOpen(true)}
                  onKeyDown={(event) => {
                    if (event.key === 'Escape') {
                      setIsModelSearchOpen(false);
                    } else if (event.key === 'Enter') {
                      setIsModelSearchOpen(false);
                      fetchModelConfig(modelQuery);
                    }
                  }}
                  onChange={(event) => {
                    setModelQuery(event.target.value);
                    setModelError(null);
                    setModelInspection(null);
                    setIsModelSearchOpen(true);
                  }}
                  aria-label='Hugging Face model ID or search'
                  aria-expanded={isModelSearchOpen}
                  aria-controls='model-search-results'
                  role='combobox'
                />
              </label>
              {isModelSearchOpen && modelSuggestions.length > 0 && (
                <div
                  id='model-search-results'
                  className='absolute left-0 right-0 z-30 mt-2 max-h-80 overflow-auto rounded-xl border border-base-300 bg-base-100 p-1 shadow-2xl'
                  role='listbox'
                >
                  {modelSuggestions.map((preset) => (
                    <button
                      key={preset.id}
                      type='button'
                      className='flex w-full items-start justify-between gap-3 rounded-lg px-3 py-2 text-left hover:bg-base-200 focus-visible:outline-2 focus-visible:outline-primary'
                      onClick={() => {
                        setIsModelSearchOpen(false);
                        applyModelPreset(preset);
                      }}
                      role='option'
                      aria-selected={selectedPreset?.id === preset.id}
                    >
                      <span className='min-w-0'>
                        <span className='block truncate text-sm font-semibold'>
                          {preset.label}
                        </span>
                        <span className='block truncate text-[11px] text-base-content/60'>
                          {preset.id}
                        </span>
                      </span>
                      <span className='badge badge-ghost badge-sm shrink-0'>
                        {preset.family}
                      </span>
                    </button>
                  ))}
                  <div className='border-t border-base-300 px-3 py-2 text-[11px] text-base-content/60'>
                    Choose a result to load its preset, or press Enter / Fetch
                    configuration to inspect any public Hugging Face ID.
                  </div>
                </div>
              )}
            </div>
            <button
              className='btn btn-primary w-full md:w-auto'
              onClick={() => fetchModelConfig(modelQuery)}
              disabled={isLoadingModel}
            >
              {isLoadingModel ? 'Loading…' : 'Fetch configuration'}
            </button>
          </div>
          {modelError && (
            <p className='mt-3 rounded-lg bg-error/10 px-4 py-2 text-sm text-error'>
              {modelError}
            </p>
          )}

          <div className='mt-6'>
            <div className='flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between'>
              <div>
                <h2 className='text-sm font-semibold text-base-content/80'>
                  Curated presets
                </h2>
                <p className='text-xs text-base-content/70'>
                  Start from modern open-weight models with known serving notes.
                </p>
              </div>
              <span className='badge badge-outline shrink-0 whitespace-nowrap'>
                {modelPresets.length} presets · {featuredPresets.length}{' '}
                featured
              </span>
            </div>
            <div className='mt-4 flex flex-col gap-3 lg:flex-row lg:items-center'>
              <div className='flex min-w-0 flex-wrap items-center gap-2'>
                <span className='text-xs font-semibold uppercase tracking-[0.12em] text-primary'>
                  Featured
                </span>
                {featuredPresets.map((preset) => (
                  <button
                    key={preset.id}
                    className={`btn btn-xs ${
                      selectedPreset?.id === preset.id
                        ? 'btn-primary'
                        : 'btn-outline'
                    }`}
                    type='button'
                    onClick={() => applyModelPreset(preset)}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {selectedPreset && (
            <div className='mt-5 flex flex-col gap-3 rounded-xl border border-base-300 bg-base-200 px-4 py-3 lg:flex-row lg:items-center lg:justify-between'>
              <div className='min-w-0'>
                <div className='flex flex-wrap items-center gap-2'>
                  <h2 className='truncate text-lg font-semibold'>
                    {selectedPreset.label}
                  </h2>
                  <span className='badge badge-primary badge-outline'>
                    {selectedPreset.family}
                  </span>
                  <span className='badge badge-success badge-outline'>
                    {formatModelLicense(selectedPreset.license)}
                  </span>
                  {selectedPreset.releaseSourceUrl && (
                    <a
                      className='link link-primary text-xs'
                      href={selectedPreset.releaseSourceUrl}
                      target='_blank'
                      rel='noreferrer'
                    >
                      Release notes ↗
                    </a>
                  )}
                </div>
                <p className='mt-1 line-clamp-2 text-xs text-base-content/70'>
                  {selectedPreset.summary}
                </p>
              </div>
              <dl className='grid shrink-0 grid-cols-3 gap-5 text-xs'>
                <div>
                  <dt className='text-base-content/55'>Parameters</dt>
                  <dd className='mt-1 font-bold'>
                    {formatNumber(parameterBillions, 2)}B
                  </dd>
                </div>
                <div>
                  <dt className='text-base-content/55'>Layers</dt>
                  <dd className='mt-1 font-bold'>
                    {effectiveNumLayers || '–'}
                  </dd>
                </div>
                <div>
                  <dt className='text-base-content/55'>Context</dt>
                  <dd className='mt-1 font-bold'>
                    {formatNumber(sequenceLength, 0)}
                  </dd>
                </div>
              </dl>
            </div>
          )}
        </section>

        <div className='mt-6'>
          <DeploymentDecisionPath
            eyebrow={mode === 'training' ? 'Training path' : 'Deployment path'}
            title={
              mode === 'training'
                ? 'What this training run needs—at a glance'
                : 'What this model needs—at a glance'
            }
            modelLabel={selectedPreset?.label ?? modelInspection?.id ?? modelId}
            parameterBillions={parameterBillions}
            layers={effectiveNumLayers}
            contextLength={
              mode === 'training'
                ? trainingSettings.sequenceLength
                : sequenceLength
            }
            weightFormat={
              mode === 'training'
                ? trainingSettings.method.toUpperCase()
                : weightFormat.toUpperCase()
            }
            totalMemoryGB={displayMemoryGB}
            memorySegments={
              mode === 'training'
                ? [
                    {
                      label: 'Base weights',
                      valueGB: trainingMemory.baseWeightsPerDeviceGB,
                      color: 'bg-memory-weights',
                    },
                    {
                      label: 'Trainable state',
                      valueGB:
                        trainingMemory.adapterWeightsPerDeviceGB +
                        trainingMemory.gradientsPerDeviceGB +
                        trainingMemory.optimizerPerDeviceGB,
                      color: 'bg-memory-kv',
                    },
                    {
                      label: 'Activations',
                      valueGB: trainingMemory.activationsGB,
                      color: 'bg-memory-activations',
                    },
                    {
                      label: 'Overhead',
                      valueGB: trainingMemory.overheadGB,
                      color: 'bg-memory-overhead',
                    },
                  ]
                : [
                    {
                      label: 'Weights',
                      valueGB: memoryBreakdown.weightsGB,
                      color: 'bg-memory-weights',
                    },
                    {
                      label: 'KV',
                      valueGB: memoryBreakdown.kvCacheGB,
                      color: 'bg-memory-kv',
                    },
                    {
                      label: 'Activations',
                      valueGB: memoryBreakdown.activationsGB,
                      color: 'bg-memory-activations',
                    },
                    {
                      label: 'Overhead',
                      valueGB: memoryBreakdown.overheadGB,
                      color: 'bg-memory-overhead',
                    },
                  ]
            }
            gpuName={
              mode === 'training'
                ? `${trainingSettings.deviceCount}× ${selectedGpu.name}`
                : selectedGpu.name
            }
            gpuCapacityGB={displayCapacityGB}
            fits={displayFits}
            headroomGB={displayHeadroomGB}
            tokensPerSecond={
              mode === 'training'
                ? trainingRun.tokensPerSecond
                : throughput.tokensPerSecond
            }
            performanceLabel={mode === 'training' ? 'Training rate' : 'Decode'}
            projectedCost={
              mode === 'training'
                ? selectedTrainingHourlyRate > 0
                  ? trainingRun.totalCost
                  : undefined
                : hourlyRate > 0
                  ? costEstimate.totalCost
                  : undefined
            }
            cloudCostLabel={
              mode === 'training'
                ? selectedTrainingHourlyRate > 0
                  ? `${
                      typeof customHourlyRate === 'number' &&
                      customHourlyRate > 0
                        ? 'Custom rate'
                        : (selectedInstance?.provider ?? 'Cloud rate')
                    } · ${formatNumber(trainingRun.durationHours, 1)}h`
                  : undefined
                : typeof customHourlyRate === 'number' && customHourlyRate > 0
                  ? `Custom rate · ${formatNumber(runtimeHours, 2)}h`
                  : selectedInstance
                    ? `${selectedInstance.provider} · ${formatNumber(runtimeHours, 2)}h`
                    : undefined
            }
          />
        </div>

        <section id='understand' className='mt-8 scroll-mt-24'>
          <ModelArchitectureDiagram
            modelType={
              modelInspection?.modelType ??
              selectedPreset?.modelTypeTag ??
              undefined
            }
            architectures={
              modelInspection?.architectures ?? selectedPreset?.architectures
            }
            sourceDirectoryUrl={
              modelInspection?.transformers?.directoryUrl ??
              selectedPreset?.sourceUrls.transformers ??
              undefined
            }
            sourceFiles={modelInspection?.transformers?.files}
            sourcePreview={modelInspection?.transformers?.preview}
            onLoadImplementation={() =>
              fetchModelConfig(
                selectedPreset?.sourceModelId ?? modelQuery ?? modelId,
                Boolean(selectedPreset?.sourceModelId),
              )
            }
            isLoadingImplementation={isLoadingModel}
            hiddenSize={effectiveHiddenSize}
            numLayers={effectiveNumLayers}
            numAttentionHeads={effectiveNumHeads}
            numKeyValueHeads={effectiveNumKeyValueHeads}
            headDim={effectiveHeadDim}
            intermediateSize={effectiveIntermediateSize}
            expertIntermediateSize={effectiveExpertIntermediateSize}
            sharedExpertIntermediateSize={effectiveSharedExpertIntermediateSize}
            numSharedExperts={effectiveNumSharedExperts || undefined}
            isEncoderDecoder={effectiveIsEncoderDecoder}
            modality={effectiveModality}
            modalityArchitecture={
              modelInspection?.architecture.modalityArchitecture ??
              selectedPreset?.modalityArchitecture ??
              undefined
            }
            vocabSize={vocabSize}
            numExperts={effectiveNumExperts || undefined}
            numExpertsPerToken={effectiveNumExpertsPerToken || undefined}
            parameterCount={parameterCount}
            parameterBreakdown={parameterBreakdown}
          />
        </section>

        <section className='mt-8 grid gap-6 lg:grid-cols-[1.05fr_0.95fr]'>
          <div className='space-y-6'>
            <div
              id='estimate'
              className='scroll-mt-24 rounded-2xl border border-base-300 bg-base-100 p-6 shadow-lg shadow-black/10'
            >
              <div className='flex flex-col gap-3 md:flex-row md:items-start md:justify-between'>
                <div>
                  <h2 className='text-xl font-semibold'>
                    {mode === 'training'
                      ? 'Training inputs'
                      : 'Quick estimator'}
                  </h2>
                  <p className='mt-1 text-sm text-base-content/70'>
                    {mode === 'training'
                      ? 'Set the model size here, then shape the complete run in the adjacent planner.'
                      : 'Size weights and KV cache instantly from core workload inputs.'}
                  </p>
                  <div className='mt-2 flex flex-wrap gap-2 text-xs'>
                    <span
                      className={`badge badge-sm ${classificationBadgeClass(
                        'exact',
                      )}`}
                    >
                      {mode === 'training'
                        ? 'Base/adapters: derived arithmetic'
                        : 'Weights/KV: exact arithmetic'}
                    </span>
                    <span
                      className={`badge badge-sm ${classificationBadgeClass(
                        'heuristic',
                      )}`}
                    >
                      {mode === 'training'
                        ? 'Runtime/activations: heuristic'
                        : 'Activations/overhead: heuristic'}
                    </span>
                  </div>
                </div>
                <div className='join self-start'>
                  <button
                    className={`btn btn-sm join-item ${
                      mode === 'inference' ? 'btn-primary' : 'btn-ghost'
                    }`}
                    type='button'
                    onClick={() => setMode('inference')}
                  >
                    Inference
                  </button>
                  <button
                    className={`btn btn-sm join-item ${
                      mode === 'training' ? 'btn-primary' : 'btn-ghost'
                    }`}
                    type='button'
                    onClick={() => setMode('training')}
                  >
                    Training
                  </button>
                </div>
              </div>

              <div className='mt-6 grid gap-4 md:grid-cols-2'>
                <label className='flex flex-col text-sm'>
                  Parameter count (billions)
                  <input
                    className='input input-bordered mt-1'
                    type='number'
                    lang='en-US'
                    min='0'
                    step='0.1'
                    value={parameterBillions}
                    onChange={(event) => {
                      setParameterSource('manual');
                      setModelInspection(null);
                      setParameterBillions(Number(event.target.value) || 0);
                    }}
                  />
                </label>
                {mode === 'inference' && (
                  <>
                    <label className='flex flex-col text-sm'>
                      Context length (tokens)
                      <input
                        className='input input-bordered mt-1'
                        type='number'
                        lang='en-US'
                        min='1'
                        value={sequenceLength}
                        onChange={(event) =>
                          setSequenceLength(Number(event.target.value) || 0)
                        }
                      />
                    </label>
                    <label className='flex flex-col text-sm'>
                      Weight format
                      <select
                        className='select select-bordered mt-1'
                        value={weightFormat}
                        onChange={(event) =>
                          setWeightFormat(event.target.value as WeightFormat)
                        }
                      >
                        {weightFormatOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className='flex flex-col text-sm'>
                      KV cache precision
                      <select
                        className='select select-bordered mt-1'
                        value={kvBits}
                        onChange={(event) =>
                          setKvBits(Number(event.target.value) as PrecisionBits)
                        }
                      >
                        {bitsOptions.map((bits) => (
                          <option key={bits} value={bits}>
                            {bits}-bit
                          </option>
                        ))}
                      </select>
                    </label>
                  </>
                )}
                <label className='flex flex-col text-sm'>
                  Overhead factor
                  <input
                    className='input input-bordered mt-1'
                    type='number'
                    lang='en-US'
                    min='1'
                    step='0.05'
                    value={overheadFactor}
                    onChange={(event) =>
                      setOverheadFactor(Number(event.target.value) || 1)
                    }
                  />
                </label>
              </div>

              {mode === 'inference' ? (
                <div className='mt-6 space-y-2 text-sm'>
                  <span className='font-semibold text-base-content/80'>
                    Concurrent users
                  </span>
                  <div className='flex items-center gap-3'>
                    <input
                      className='range range-primary flex-1'
                      type='range'
                      min='1'
                      max='64'
                      step='1'
                      value={concurrentUsers}
                      onChange={(event) =>
                        setConcurrentUsers(Number(event.target.value) || 1)
                      }
                    />
                    <span className='badge badge-outline'>
                      {concurrentUsers}
                    </span>
                  </div>
                  <p className='text-xs text-base-content/70'>
                    KV cache memory scales linearly with concurrent sequences
                    and context length.
                  </p>
                </div>
              ) : (
                <div className='mt-6 rounded-xl border border-secondary/25 bg-secondary/10 p-4 text-xs leading-relaxed text-base-content/75'>
                  Training uses BF16 compute by default; QLoRA stores the frozen
                  base in approximately 4-bit NF4. Workload, adapter, multi-GPU,
                  runtime, and cost controls are grouped in the training planner
                  beside these model inputs.
                </div>
              )}
            </div>

            {mode === 'inference' && (
              <div
                id='architecture-controls'
                className='scroll-mt-24 rounded-2xl border border-base-300 bg-base-100 p-6 shadow-lg shadow-black/10'
              >
                <div className='flex flex-col gap-3 md:flex-row md:items-start md:justify-between'>
                  <div>
                    <h2 className='text-xl font-semibold'>
                      Architecture assumptions
                    </h2>
                    <p className='mt-1 text-sm text-base-content/70'>
                      {architectureMode === 'auto'
                        ? 'Using LLaMA-style scaling heuristics derived from parameter count. Enable manual mode to match a specific checkpoint.'
                        : 'Provide exact architecture values to refine KV cache and activation estimates.'}
                    </p>
                    <div className='mt-2'>
                      <span
                        className={`badge badge-sm ${classificationBadgeClass(
                          architectureMode === 'manual' ? 'exact' : 'heuristic',
                        )}`}
                      >
                        {architectureMode === 'manual'
                          ? 'Manual architecture override'
                          : 'Heuristic architecture'}
                      </span>
                    </div>
                  </div>
                  <div className='join self-start'>
                    <button
                      className={`btn btn-sm join-item ${
                        architectureMode === 'auto'
                          ? 'btn-primary'
                          : 'btn-ghost'
                      }`}
                      type='button'
                      onClick={enableAutoArchitecture}
                    >
                      Auto
                    </button>
                    <button
                      className={`btn btn-sm join-item ${
                        architectureMode === 'manual'
                          ? 'btn-primary'
                          : 'btn-ghost'
                      }`}
                      type='button'
                      onClick={enableManualOverrides}
                    >
                      Manual
                    </button>
                  </div>
                </div>

                <dl className='mt-6 grid gap-4 text-sm sm:grid-cols-2'>
                  <div>
                    <dt className='font-semibold text-base-content/70'>
                      Hidden size
                    </dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveHiddenSize || '–'}
                    </dd>
                  </div>
                  <div>
                    <dt className='font-semibold text-base-content/70'>
                      Layers
                    </dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveNumLayers || '–'}
                    </dd>
                  </div>
                  <div>
                    <dt className='font-semibold text-base-content/70'>
                      Attention heads
                    </dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveNumHeads || '–'}
                    </dd>
                  </div>
                  <div>
                    <dt className='font-semibold text-base-content/70'>
                      KV heads
                    </dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveNumKeyValueHeads || '–'}
                    </dd>
                  </div>
                  <div>
                    <dt className='font-semibold text-base-content/70'>
                      Head dimension
                    </dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveHeadDim
                        ? formatNumber(effectiveHeadDim, 0)
                        : '–'}
                    </dd>
                  </div>
                  <div>
                    <dt className='font-semibold text-base-content/70'>
                      Feed-forward size
                    </dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveIntermediateSize || '–'}
                    </dd>
                  </div>
                </dl>

                {architectureMode === 'manual' && (
                  <details className='mt-5 rounded-xl border border-base-300 bg-base-200/60 p-4'>
                    <summary className='cursor-pointer text-sm font-semibold'>
                      Edit exact block dimensions
                    </summary>
                    <div className='mt-4 space-y-4'>
                      <div className='grid gap-4 md:grid-cols-2'>
                        <label className='flex flex-col text-sm'>
                          Hidden size
                          <input
                            className='input input-bordered mt-1'
                            type='number'
                            lang='en-US'
                            min='0'
                            value={manualHiddenSize}
                            onChange={(event) =>
                              setManualHiddenSize(
                                Number(event.target.value) || 0,
                              )
                            }
                          />
                        </label>
                        <label className='flex flex-col text-sm'>
                          Layers
                          <input
                            className='input input-bordered mt-1'
                            type='number'
                            lang='en-US'
                            min='0'
                            value={manualNumLayers}
                            onChange={(event) =>
                              setManualNumLayers(
                                Number(event.target.value) || 0,
                              )
                            }
                          />
                        </label>
                        <label className='flex flex-col text-sm'>
                          Attention heads
                          <input
                            className='input input-bordered mt-1'
                            type='number'
                            lang='en-US'
                            min='0'
                            value={manualNumHeads}
                            onChange={(event) =>
                              setManualNumHeads(Number(event.target.value) || 0)
                            }
                          />
                        </label>
                        <label className='flex flex-col text-sm'>
                          Key/value heads (GQA/MQA)
                          <input
                            className='input input-bordered mt-1'
                            type='number'
                            lang='en-US'
                            min='1'
                            value={manualNumKeyValueHeads}
                            onChange={(event) =>
                              setManualNumKeyValueHeads(
                                Number(event.target.value) || 1,
                              )
                            }
                          />
                        </label>
                        <label className='flex flex-col text-sm'>
                          Head dimension (optional)
                          <input
                            className='input input-bordered mt-1'
                            type='number'
                            lang='en-US'
                            min='1'
                            placeholder={
                              effectiveNumHeads
                                ? String(
                                    effectiveHiddenSize / effectiveNumHeads,
                                  )
                                : ''
                            }
                            value={manualHeadDim || ''}
                            onChange={(event) =>
                              setManualHeadDim(Number(event.target.value) || 0)
                            }
                          />
                        </label>
                        <label className='flex flex-col text-sm'>
                          Feed-forward size
                          <input
                            className='input input-bordered mt-1'
                            type='number'
                            lang='en-US'
                            min='0'
                            placeholder={
                              manualHiddenSize
                                ? String(manualHiddenSize * 4)
                                : ''
                            }
                            value={manualIntermediateSize || ''}
                            onChange={(event) =>
                              setManualIntermediateSize(
                                Number(event.target.value) || 0,
                              )
                            }
                          />
                        </label>
                        <label className='flex flex-col text-sm'>
                          Total experts (MoE, optional)
                          <input
                            className='input input-bordered mt-1'
                            type='number'
                            lang='en-US'
                            min='0'
                            value={manualNumExperts || ''}
                            onChange={(event) =>
                              setManualNumExperts(
                                Number(event.target.value) || 0,
                              )
                            }
                          />
                        </label>
                        <label className='flex flex-col text-sm'>
                          Experts per token
                          <input
                            className='input input-bordered mt-1'
                            type='number'
                            lang='en-US'
                            min='1'
                            value={manualNumExpertsPerToken || ''}
                            onChange={(event) =>
                              setManualNumExpertsPerToken(
                                Number(event.target.value) || 0,
                              )
                            }
                          />
                        </label>
                      </div>
                      <div className='flex flex-wrap gap-4 text-sm'>
                        <label className='flex items-center gap-2'>
                          <input
                            className='checkbox checkbox-primary'
                            type='checkbox'
                            checked={manualGatedMlp}
                            onChange={(event) =>
                              setManualGatedMlp(event.target.checked)
                            }
                          />
                          Gated MLP (SwiGLU/GEGLU)
                        </label>
                        <label className='flex items-center gap-2'>
                          <input
                            className='checkbox checkbox-primary'
                            type='checkbox'
                            checked={manualTieWordEmbeddings}
                            onChange={(event) =>
                              setManualTieWordEmbeddings(event.target.checked)
                            }
                          />
                          Tie input/output embeddings
                        </label>
                      </div>
                      <label className='flex flex-col text-sm'>
                        Vocabulary size (for FLOPs)
                        <input
                          className='input input-bordered mt-1'
                          type='number'
                          lang='en-US'
                          min='0'
                          value={vocabSize}
                          onChange={(event) =>
                            setVocabSize(Number(event.target.value) || 0)
                          }
                        />
                      </label>
                    </div>
                  </details>
                )}
                {architectureMode === 'auto' && (
                  <p className='mt-5 rounded-lg bg-base-200 px-3 py-2 text-xs text-base-content/70'>
                    Hidden size and depth follow public LLaMA scaling
                    heuristics. Switch to manual mode to match custom
                    architectures.
                  </p>
                )}
              </div>
            )}
          </div>

          <div className='space-y-6'>
            {mode === 'training' ? (
              <TrainingPlannerCard
                settings={trainingSettings}
                onChange={updateTrainingSettings}
                hardware={gpus as unknown as HardwareLike[]}
                selectedGpu={selectedGpu}
                onSelectedGpuChange={(name) =>
                  setSelectedGpuName(name as Gpu['name'])
                }
                memory={trainingMemory}
                run={trainingRun}
                selectedCloudRate={selectedInstance}
                customHourlyRate={customHourlyRate}
                onCustomHourlyRateChange={setCustomHourlyRate}
                recommendations={trainingRecommendations}
              />
            ) : (
              <div
                id='hardware'
                className='scroll-mt-24 rounded-2xl border border-base-300 bg-base-100 p-6 shadow-lg shadow-black/10'
              >
                <h2 className='text-xl font-semibold'>Memory &amp; hardware</h2>
                <p className='mt-1 text-sm text-base-content/70'>
                  {mode === 'inference'
                    ? `Assumes ${effectiveBatchSize} concurrent ${
                        effectiveBatchSize === 1 ? 'user' : 'users'
                      } at ${sequenceLength} tokens.`
                    : `Assumes a global batch size of ${effectiveBatchSize} sequences.`}
                </p>
                <p className='mt-2 text-xs text-base-content/70'>
                  Weight memory is exact arithmetic. Cache memory uses typed
                  model schedules when available (including compressed or
                  state-space layers); activations, total VRAM, fit checks, and
                  GPU recommendations remain estimates because runtime behavior
                  and placement assumptions vary.
                </p>
                <div className='mt-4 grid gap-3 text-sm'>
                  <div className='flex items-center justify-between'>
                    <span>Model weights</span>
                    <span className='font-semibold'>
                      {formatMemory(memoryBreakdown.weightsGB)}
                    </span>
                  </div>
                  <div className='flex items-center justify-between'>
                    <span>Activations</span>
                    <span className='font-semibold'>
                      {formatMemory(memoryBreakdown.activationsGB)}
                    </span>
                  </div>
                  <div className='flex items-center justify-between rounded-lg bg-primary/10 px-3 py-2'>
                    <span className='flex items-center gap-2'>
                      KV cache
                      <span className='badge badge-outline badge-sm'>
                        {kvBits}-bit
                      </span>
                    </span>
                    <span className='font-semibold'>
                      {formatMemory(memoryBreakdown.kvCacheGB)}
                    </span>
                  </div>
                  {memoryBreakdown.stateCacheGB > 0 && (
                    <div className='flex items-center justify-between rounded-lg bg-secondary/10 px-3 py-2'>
                      <span>Recurrent/state cache</span>
                      <span className='font-semibold'>
                        {formatMemory(memoryBreakdown.stateCacheGB)}
                      </span>
                    </div>
                  )}
                  <div className='flex items-center justify-between border-t border-base-300 pt-3'>
                    <span>Total before overhead</span>
                    <span className='font-semibold'>
                      {formatMemory(memoryBreakdown.baseTotalGB)}
                    </span>
                  </div>
                  <div className='flex items-center justify-between'>
                    <span>
                      Framework overhead ({formatNumber(overheadFactor, 2)}×)
                    </span>
                    <span className='font-semibold'>
                      {formatMemory(memoryBreakdown.overheadGB)}
                    </span>
                  </div>
                  <div className='flex items-center justify-between border-t border-base-300 pt-3 text-lg font-bold text-primary'>
                    <span>Total VRAM needed</span>
                    <span>{formatMemory(memoryBreakdown.totalGB)}</span>
                  </div>
                </div>

                <div className='mt-6 rounded-lg bg-base-200 p-4'>
                  <label className='text-sm font-semibold text-base-content/80'>
                    Compare against GPU
                  </label>
                  <select
                    className='select select-bordered mt-2 w-full'
                    value={selectedGpuName}
                    onChange={(event) => setSelectedGpuName(event.target.value)}
                    aria-label='Compare against GPU'
                  >
                    {gpuGroups.map((group) => (
                      <optgroup key={group.key} label={group.label}>
                        {group.gpus.map((gpu) => (
                          <option key={gpu.name} value={gpu.name}>
                            {gpu.name} ({gpu.memory_gb} GB)
                            {group.vendor === 'Apple'
                              ? ` · ${gpu.architecture}`
                              : ''}
                          </option>
                        ))}
                      </optgroup>
                    ))}
                  </select>

                  {selectedGpu && (
                    <>
                      <VramUsageBar
                        capacityGB={selectedGpu.memory_gb}
                        totalGB={memoryBreakdown.totalGB}
                        segments={[
                          {
                            label: 'Model weights',
                            valueGB: memoryBreakdown.weightsGB,
                            color: 'bg-memory-weights',
                          },
                          {
                            label: 'KV cache',
                            valueGB: memoryBreakdown.kvCacheGB,
                            color: 'bg-memory-kv',
                          },
                          ...(memoryBreakdown.activationsGB > 0
                            ? [
                                {
                                  label: 'Activations',
                                  valueGB: memoryBreakdown.activationsGB,
                                  color: 'bg-memory-activations',
                                },
                              ]
                            : []),
                          ...(memoryBreakdown.optimizerGB > 0
                            ? [
                                {
                                  label: 'Optimizer state',
                                  valueGB: memoryBreakdown.optimizerGB,
                                  color: 'bg-lab-sand',
                                },
                              ]
                            : []),
                          ...(memoryBreakdown.stateCacheGB > 0
                            ? [
                                {
                                  label: 'Recurrent/state cache',
                                  valueGB: memoryBreakdown.stateCacheGB,
                                  color: 'bg-lab-green',
                                },
                              ]
                            : []),
                          {
                            label: 'Framework overhead',
                            valueGB: memoryBreakdown.overheadGB,
                            color: 'bg-memory-overhead',
                          },
                        ]}
                        fits={selectedHardwareFit.fits}
                        requiredDevices={selectedHardwareFit.requiredDevices}
                        deviceCount={selectedHardwareFit.deviceCount}
                      />
                      <dl className='mt-4 grid gap-3 text-xs sm:grid-cols-2'>
                        <div>
                          <dt className='text-base-content/65'>
                            Vendor / architecture
                          </dt>
                          <dd className='font-semibold'>
                            {selectedGpu.vendor ?? 'Unknown'} ·{' '}
                            {selectedGpu.architecture ?? 'Unknown'}
                          </dd>
                        </div>
                        <div>
                          <dt className='text-base-content/65'>Memory</dt>
                          <dd className='font-semibold'>
                            {selectedGpu.memory_gb} GB aggregate ·{' '}
                            {selectedGpu.per_device_memory_gb} GB/device ·{' '}
                            {selectedGpu.memory_type}
                          </dd>
                        </div>
                        <div>
                          <dt className='text-base-content/65'>Bandwidth</dt>
                          <dd className='font-semibold'>
                            {selectedGpu.memory_bandwidth_gb_s
                              ? `${formatNumber(selectedGpu.memory_bandwidth_gb_s, 0)} GB/s`
                              : 'Not published'}
                          </dd>
                        </div>
                        <div>
                          <dt className='text-base-content/65'>Topology</dt>
                          <dd className='font-semibold'>
                            {selectedGpu.device_count > 1
                              ? `${selectedGpu.device_count} devices · ${selectedHardwareFit.requiredDevices} required`
                              : selectedGpu.memory_model === 'unified'
                                ? 'Unified local memory'
                                : 'Single device'}
                          </dd>
                        </div>
                      </dl>
                      <p
                        className={`mt-3 text-sm ${
                          selectedHardwareFit.fits
                            ? 'text-success'
                            : 'text-error'
                        }`}
                      >
                        {selectedHardwareFit.fits
                          ? `Fits within aggregate capacity with ${formatNumber(
                              selectedHardwareFit.aggregateHeadroomGB,
                              2,
                            )} GB headroom.`
                          : 'Model does not fit within the listed device/topology capacity.'}
                      </p>
                      {selectedGpu.notes && (
                        <p className='mt-2 text-xs text-base-content/65'>
                          {selectedGpu.notes}
                        </p>
                      )}
                      {selectedGpu.source_url && (
                        <a
                          className='link link-primary mt-2 inline-block text-xs'
                          href={selectedGpu.source_url}
                          target='_blank'
                          rel='noreferrer'
                        >
                          Verify official hardware specification ↗
                        </a>
                      )}
                    </>
                  )}
                </div>

                {recommendedGpuList.length > 0 && (
                  <div className='mt-4 text-sm'>
                    <p className='font-semibold text-base-content/70'>
                      Closest matching GPUs
                    </p>
                    <ul className='mt-2 space-y-1'>
                      {recommendedGpuList.map((gpu) => (
                        <li key={gpu.name} className='flex justify-between'>
                          <span>
                            {gpu.name}
                            {gpu.requiredDevices > 1
                              ? ` · ${gpu.requiredDevices}/${gpu.deviceCount} devices`
                              : ''}
                          </span>
                          <span className='text-base-content/70'>
                            {formatNumber(gpu.memoryHeadroomGB, 2)} GB spare
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {mode === 'inference' && (
                  <KvCacheScalingCard
                    kvCacheGB={memoryBreakdown.kvCacheGB}
                    stateCacheGB={memoryBreakdown.stateCacheGB}
                    cacheMode={memoryBreakdown.kvCacheMode}
                    cacheDescription={memoryBreakdown.kvCacheDescription}
                    attentionLayers={memoryBreakdown.kvAttentionLayers}
                    bytesPerToken={memoryBreakdown.kvCacheBytesPerToken}
                    totalTokens={memoryBreakdown.kvCacheTokens}
                    sequenceLength={sequenceLength}
                    batchSize={effectiveBatchSize}
                    precisionBits={kvBits}
                    numLayers={effectiveNumLayers}
                    numAttentionHeads={effectiveNumHeads}
                    numKeyValueHeads={effectiveNumKeyValueHeads}
                    headDim={effectiveHeadDim}
                  />
                )}
              </div>
            )}

            {mode === 'inference' && (
              <>
                <div
                  id='performance'
                  className='scroll-mt-24 rounded-2xl border border-base-300 bg-base-100 p-6 shadow-lg shadow-black/10'
                >
                  <h2 className='text-xl font-semibold'>Performance</h2>
                  <p className='mt-2 text-xs text-base-content/70'>
                    Performance outputs are heuristic decode estimates. TPS uses
                    the lower of compute and weight-memory bandwidth ceilings
                    and does not represent prefill/TTFT or a measured serving
                    run; it
                    {selectedPreset?.activeParameterCount
                      ? ' prefers active parameters for the selected MoE-style preset.'
                      : ' defaults to total parameters when no active-parameter metadata is available.'}
                  </p>
                  <div className='mt-5 grid gap-3 text-sm sm:grid-cols-2'>
                    <label className='rounded-xl border border-base-300 bg-base-200 p-4'>
                      <span className='font-semibold'>Compute efficiency</span>
                      <input
                        className='input input-bordered mt-2 w-full'
                        type='number'
                        lang='en-US'
                        min='0.05'
                        max='1'
                        step='0.05'
                        value={efficiency}
                        onChange={(event) =>
                          setEfficiency(Number(event.target.value) || 0.3)
                        }
                      />
                      <span className='mt-2 block text-xs text-base-content/60'>
                        Kernel/framework utilization of the published compute
                        ceiling.
                      </span>
                    </label>
                    <label className='rounded-xl border border-base-300 bg-base-200 p-4'>
                      <span className='font-semibold'>
                        Memory-bandwidth efficiency
                      </span>
                      <input
                        className='input input-bordered mt-2 w-full'
                        type='number'
                        lang='en-US'
                        min='0.1'
                        max='1'
                        step='0.05'
                        value={memoryEfficiency}
                        onChange={(event) =>
                          setMemoryEfficiency(
                            Number(event.target.value) || 0.65,
                          )
                        }
                      />
                      <span className='mt-2 block text-xs text-base-content/60'>
                        Effective bandwidth after runtime, placement, and kernel
                        overhead.
                      </span>
                    </label>
                  </div>

                  <div className='mt-4 rounded-xl border border-base-300 bg-base-200 p-4'>
                    <div className='flex flex-wrap items-start justify-between gap-3'>
                      <div>
                        <h3 className='font-semibold'>
                          Selected hardware path
                        </h3>
                        <p className='mt-1 text-xs text-base-content/65'>
                          {selectedGpu?.name ?? 'No hardware selected'} ·{' '}
                          {weightFormat.toUpperCase()} weights ·{' '}
                          {effectiveBatchSize} sequence(s)
                        </p>
                      </div>
                      {throughput.bottleneck !== 'unavailable' && (
                        <span className='badge badge-primary'>
                          {throughput.bottleneck}-bound
                        </span>
                      )}
                    </div>
                    <dl className='mt-4 grid gap-3 text-xs sm:grid-cols-3'>
                      <div>
                        <dt className='text-base-content/60'>
                          Compute ceiling
                        </dt>
                        <dd className='mt-1 text-sm font-semibold'>
                          {selectedComputeTFlops > 0
                            ? `${formatNumber(selectedComputeTFlops, 1)} ${weightFormat.toUpperCase()} TFLOP/s`
                            : 'Unavailable for this format'}
                        </dd>
                      </div>
                      <div>
                        <dt className='text-base-content/60'>
                          Memory bandwidth
                        </dt>
                        <dd className='mt-1 text-sm font-semibold'>
                          {selectedGpu?.memory_bandwidth_gb_s
                            ? `${formatNumber(selectedGpu.memory_bandwidth_gb_s, 0)} GB/s`
                            : 'Unavailable'}
                        </dd>
                      </div>
                      <div>
                        <dt className='text-base-content/60'>
                          Published AI figure
                        </dt>
                        <dd className='mt-1 text-sm font-semibold'>
                          {selectedGpu?.ai_tops
                            ? `${formatNumber(selectedGpu.ai_tops, 0)} TOPS`
                            : 'Not published'}
                        </dd>
                      </div>
                    </dl>
                  </div>

                  <div className='mt-4 grid gap-3 sm:grid-cols-3'>
                    <div className='rounded-xl border border-primary/30 bg-primary/10 p-4'>
                      <div className='text-xs font-semibold uppercase tracking-wide text-primary'>
                        Decode TPS
                      </div>
                      <div className='mt-2 text-3xl font-bold tabular-nums'>
                        {throughput.tokensPerSecond
                          ? formatNumber(throughput.tokensPerSecond, 2)
                          : 'N/A'}
                      </div>
                      <div className='mt-1 text-xs text-base-content/65'>
                        tokens / second
                      </div>
                    </div>
                    <div className='rounded-xl border border-base-300 bg-base-200 p-4'>
                      <div className='text-xs font-semibold uppercase tracking-wide text-base-content/60'>
                        Compute ceiling
                      </div>
                      <div className='mt-2 text-2xl font-bold tabular-nums'>
                        {throughput.computeBoundTokensPerSecond > 0
                          ? formatNumber(
                              throughput.computeBoundTokensPerSecond,
                              2,
                            )
                          : '—'}
                      </div>
                      <div className='mt-1 text-xs text-base-content/65'>
                        tokens / second
                      </div>
                    </div>
                    <div className='rounded-xl border border-base-300 bg-base-200 p-4'>
                      <div className='text-xs font-semibold uppercase tracking-wide text-base-content/60'>
                        Memory ceiling
                      </div>
                      <div className='mt-2 text-2xl font-bold tabular-nums'>
                        {throughput.memoryBoundTokensPerSecond > 0
                          ? formatNumber(
                              throughput.memoryBoundTokensPerSecond,
                              2,
                            )
                          : '—'}
                      </div>
                      <div className='mt-1 text-xs text-base-content/65'>
                        tokens / second
                      </div>
                    </div>
                  </div>

                  <div className='mt-3 grid gap-3 text-sm sm:grid-cols-2'>
                    <div className='rounded-xl border border-base-300 p-4'>
                      <span className='font-semibold'>Latency estimate</span>
                      <div className='mt-1 text-xl font-bold tabular-nums'>
                        {throughput.millisecondsPerToken
                          ? `${formatNumber(throughput.millisecondsPerToken, 2)} ms/token`
                          : 'N/A'}
                      </div>
                    </div>
                    <div className='rounded-xl border border-base-300 p-4'>
                      <span className='font-semibold'>FLOPs / sequence</span>
                      <div className='mt-1 text-xl font-bold tabular-nums'>
                        {flops
                          ? `${formatNumber(flops / 10 ** 12, 2)} TFLOPs`
                          : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>

                <div
                  id='cost'
                  className='scroll-mt-24 rounded-2xl border border-base-300 bg-base-100 p-6 shadow-lg shadow-black/10'
                >
                  <h2 className='text-xl font-semibold'>
                    Cloud cost projection
                  </h2>
                  <p className='mt-2 text-xs text-base-content/70'>
                    Verified on-demand offerings matching {selectedGpu.name}.
                    Cost is hourly rate × runtime and excludes storage,
                    networking, taxes, and unavailable capacity.
                  </p>
                  <div className='mt-4 grid gap-4 text-sm md:grid-cols-2'>
                    <label className='flex flex-col'>
                      Cloud instance
                      <select
                        className='select select-bordered mt-1'
                        value={selectedInstance?.name ?? ''}
                        onChange={(event) =>
                          setSelectedInstanceName(event.target.value)
                        }
                      >
                        {matchingCloudInstances.length === 0 && (
                          <option value='' disabled>
                            No verified offering for this GPU
                          </option>
                        )}
                        {matchingCloudInstances.map((instance) => (
                          <option key={instance.name} value={instance.name}>
                            {instance.provider} · {instance.name} · $
                            {formatNumber(instance.hourly_rate, 2)}/hr
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className='flex flex-col'>
                      Runtime (hours)
                      <input
                        className='input input-bordered mt-1'
                        type='number'
                        lang='en-US'
                        min='0'
                        step='0.25'
                        value={runtimeHours}
                        onChange={(event) =>
                          setRuntimeHours(Number(event.target.value) || 0)
                        }
                      />
                    </label>
                    <label className='flex flex-col'>
                      Custom hourly rate (optional)
                      <input
                        className='input input-bordered mt-1'
                        type='number'
                        lang='en-US'
                        min='0'
                        step='0.01'
                        value={customHourlyRate === '' ? '' : customHourlyRate}
                        onChange={(event) => {
                          const value = event.target.value;
                          setCustomHourlyRate(value ? Number(value) : '');
                        }}
                      />
                    </label>
                    <div className='rounded-lg bg-base-200 p-3'>
                      <p>
                        <span className='font-semibold'>
                          Effective hourly rate:
                        </span>{' '}
                        {hourlyRate > 0
                          ? `$${formatNumber(hourlyRate, 2)}`
                          : 'N/A'}
                      </p>
                      <p>
                        <span className='font-semibold'>Projected cost:</span>{' '}
                        {hourlyRate > 0
                          ? `$${formatNumber(costEstimate.totalCost, 2)}`
                          : 'N/A'}
                      </p>
                      {selectedInstance ? (
                        <>
                          <p className='mt-2 text-xs text-base-content/65'>
                            {selectedInstance.pricing_basis} Checked{' '}
                            {selectedInstance.source_checked_at}; billed{' '}
                            {selectedInstance.billing_increment}.
                          </p>
                          <a
                            className='link link-primary mt-1 inline-block text-xs'
                            href={selectedInstance.pricing_source_url}
                            target='_blank'
                            rel='noreferrer'
                          >
                            Verify current provider pricing ↗
                          </a>
                        </>
                      ) : (
                        <p className='mt-2 text-xs text-warning'>
                          No fixed, provider-verified on-demand rate is
                          available for this exact hardware. Enter a custom rate
                          only if you have a current quote.
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
