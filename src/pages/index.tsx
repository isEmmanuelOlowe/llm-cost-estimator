import axios from 'axios';
import Head from 'next/head';
import { useCallback, useMemo, useState } from 'react';

import Seo from '@/components/Seo';

import cloudInstances from '@/estimator/cloud-instances.json';
import {
  estimateCloudCost,
  estimateDecoderFlops,
  estimateLlamaStyleArchitecture,
  estimateMemory,
  estimateThroughput,
  estimateTransformerParameters,
  ExecutionMode,
  MemoryBreakdown,
  PrecisionBits,
  recommendGpus,
} from '@/estimator/estimator';
import gpus from '@/estimator/gpus.json';

type CloudInstance = (typeof cloudInstances)[number];
type Gpu = (typeof gpus)[number];

interface ModelMetadata {
  parameterCount: number;
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  intermediateSize?: number;
  sequenceLength?: number;
  vocabSize?: number;
  dtypeBits?: PrecisionBits;
}

const DEFAULT_MODEL_ID = 'meta-llama/Llama-2-7b-hf';

const bitsOptions: PrecisionBits[] = [32, 16, 8, 4];

function safeNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  return undefined;
}

function formatNumber(value: number, fractionDigits = 2): string {
  if (!Number.isFinite(value)) return 'N/A';
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, {
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
  const [parameterBillions, setParameterBillions] = useState<number>(7);
  const [sequenceLength, setSequenceLength] = useState<number>(4096);
  const [vocabSize, setVocabSize] = useState<number>(32000);
  const [mode, setMode] = useState<ExecutionMode>('inference');
  const [weightBits, setWeightBits] = useState<PrecisionBits>(16);
  const [kvBits, setKvBits] = useState<PrecisionBits>(16);
  const [overheadFactor, setOverheadFactor] = useState<number>(1.15);
  const [optimizer, setOptimizer] = useState<
    'adamw' | 'adam' | 'adafactor' | 'lamb' | 'none'
  >('adamw');
  const [efficiency, setEfficiency] = useState<number>(0.3);
  const [concurrentUsers, setConcurrentUsers] = useState<number>(1);
  const [trainingBatchSize, setTrainingBatchSize] = useState<number>(32);
  const [architectureMode, setArchitectureMode] = useState<'auto' | 'manual'>(
    'auto'
  );
  const [manualHiddenSize, setManualHiddenSize] = useState<number>(4096);
  const [manualNumLayers, setManualNumLayers] = useState<number>(32);
  const [manualNumHeads, setManualNumHeads] = useState<number>(32);
  const [manualIntermediateSize, setManualIntermediateSize] =
    useState<number>(0);
  const [selectedGpuName, setSelectedGpuName] = useState<Gpu['name']>(
    gpus[0].name
  );
  const [selectedInstanceName, setSelectedInstanceName] = useState<
    CloudInstance['name']
  >(cloudInstances[0].name);
  const [runtimeHours, setRuntimeHours] = useState<number>(1);
  const [customHourlyRate, setCustomHourlyRate] = useState<number | ''>('');
  const [modelError, setModelError] = useState<string | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(false);

  const parameterCount = useMemo(
    () => Math.max(parameterBillions, 0) * 10 ** 9,
    [parameterBillions]
  );

  const autoArchitecture = useMemo(
    () => estimateLlamaStyleArchitecture(parameterCount),
    [parameterCount]
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
    return architectureMode === 'auto' ? autoValue : manualHiddenSize || autoValue;
  }, [architectureMode, autoArchitecture.hiddenSize, manualHiddenSize]);

  const effectiveNumLayers = useMemo(() => {
    const autoValue = autoArchitecture.numLayers || manualNumLayers;
    return architectureMode === 'auto' ? autoValue : manualNumLayers || autoValue;
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

  const effectiveBatchSize = useMemo(() => {
    if (mode === 'inference') {
      return Math.max(1, concurrentUsers);
    }
    return Math.max(1, trainingBatchSize);
  }, [concurrentUsers, mode, trainingBatchSize]);

  const selectedGpu = useMemo(() => {
    return gpus.find((gpu) => gpu.name === selectedGpuName) ?? gpus[0];
  }, [selectedGpuName]);

  const selectedInstance = useMemo(() => {
    return (
      cloudInstances.find((instance) => instance.name === selectedInstanceName) ??
      cloudInstances[0]
    );
  }, [selectedInstanceName]);

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
      sequenceLength,
      vocabSize,
    });
  }, [
    effectiveHiddenSize,
    effectiveNumLayers,
    sequenceLength,
    vocabSize,
  ]);

  const memoryBreakdown: MemoryBreakdown = useMemo(() => {
    if (!parameterCount || !effectiveHiddenSize || !effectiveNumLayers) {
      return {
        weightsGB: 0,
        activationsGB: 0,
        kvCacheGB: 0,
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
      sequenceLength,
      batchSize: effectiveBatchSize,
      kvCachePrecisionBits: kvBits,
      optimizer: mode === 'training' ? optimizer : 'none',
      overheadFactor,
    });
  }, [
    parameterCount,
    weightBits,
    mode,
    effectiveHiddenSize,
    effectiveNumLayers,
      sequenceLength,
    effectiveBatchSize,
    kvBits,
    optimizer,
    overheadFactor,
  ]);

  const throughput = useMemo(() => {
    if (!parameterCount) {
      return { tokensPerSecond: 0, millisecondsPerToken: 0 };
    }

    return estimateThroughput({
      parameterCount,
      gpuTFlops: selectedGpu?.fp32_tflops ?? 0,
      efficiency,
    });
  }, [parameterCount, selectedGpu, efficiency]);

  const hourlyRate = useMemo(() => {
    if (typeof customHourlyRate === 'number' && customHourlyRate > 0) {
      return customHourlyRate;
    }
    return selectedInstance.hourly_rate;
  }, [customHourlyRate, selectedInstance.hourly_rate]);

  const costEstimate = useMemo(() => {
    if (!runtimeHours || runtimeHours <= 0) {
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

  const applyModelMetadata = useCallback((metadata: ModelMetadata) => {
    if (metadata.parameterCount && metadata.parameterCount > 0) {
      setParameterBillions(metadata.parameterCount / 10 ** 9);
    }

    const hasArchitecture = Boolean(
      (metadata.hiddenSize && metadata.hiddenSize > 0) ||
        (metadata.numLayers && metadata.numLayers > 0) ||
        (metadata.numHeads && metadata.numHeads > 0) ||
        (metadata.intermediateSize && metadata.intermediateSize > 0)
    );

    if (hasArchitecture) {
      setArchitectureMode('manual');
    }

    if (metadata.hiddenSize && metadata.hiddenSize > 0) {
      setManualHiddenSize(metadata.hiddenSize);
    }
    if (metadata.numLayers && metadata.numLayers > 0) {
      setManualNumLayers(metadata.numLayers);
    }
    if (metadata.numHeads && metadata.numHeads > 0) {
      setManualNumHeads(metadata.numHeads);
    }
    if (metadata.intermediateSize && metadata.intermediateSize > 0) {
      setManualIntermediateSize(metadata.intermediateSize);
    }
    if (metadata.sequenceLength && metadata.sequenceLength > 0) {
      setSequenceLength(metadata.sequenceLength);
    }
    if (metadata.vocabSize && metadata.vocabSize > 0) {
      setVocabSize(metadata.vocabSize);
    }
    if (metadata.dtypeBits) setWeightBits(metadata.dtypeBits);
  }, []);

  const fetchModelConfig = useCallback(
    async (id: string) => {
      const trimmedId = id.trim();
      if (!trimmedId) {
        setModelError('Please enter a Hugging Face model ID.');
        return;
      }

      setIsLoadingModel(true);
      setModelError(null);
      try {
        const response = await axios.get(
          `https://huggingface.co/${trimmedId}/raw/main/config.json`,
          {
            timeout: 10000,
          }
        );
        const metadata = parseModelConfig(response.data);
        applyModelMetadata(metadata);
      } catch (error: unknown) {
        setModelError(
          'Unable to retrieve model configuration from Hugging Face.'
        );
        if (process.env.NODE_ENV !== 'production') {
          // eslint-disable-next-line no-console
          console.error(error);
        }
      } finally {
        setIsLoadingModel(false);
      }
    },
    [applyModelMetadata]
  );

  const memoryHeadroom = useMemo(() => {
    if (!selectedGpu) return null;
    return selectedGpu.memory_gb - memoryBreakdown.totalGB;
  }, [selectedGpu, memoryBreakdown.totalGB]);

  return (
    <main data-theme='dark' className='min-h-screen bg-base-200 text-base-content'>
      <Seo />
      <Head>
        <title>LLM Cost &amp; Resource Estimator</title>
      </Head>
      <div className='mx-auto max-w-6xl px-4 py-10'>
        <header className='space-y-4 text-center'>
          <h1 className='text-3xl font-bold text-primary'>
            LLM Resource Usage &amp; Cost Calculator
          </h1>
          <p className='text-base-content/80'>
            Analyse memory requirements, throughput and cloud spend for open-source language models in seconds.
          </p>
        </header>

        <section className='mt-10 rounded-xl border border-base-300 bg-base-100 p-6 shadow-lg'>
          <div className='flex flex-col gap-4 md:flex-row md:items-end'>
            <label className='flex-1'>
              <span className='label-text font-semibold'>Hugging Face model ID</span>
              <input
                className='input input-bordered mt-2 w-full'
                placeholder='meta-llama/Llama-2-7b-hf'
                value={modelId}
                onChange={(event) => setModelId(event.target.value)}
                aria-label='Hugging Face model identifier'
              />
            </label>
            <button
              className='btn btn-primary w-full md:w-auto'
              onClick={() => fetchModelConfig(modelId)}
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

          <dl className='mt-6 grid gap-4 text-sm md:grid-cols-3'>
            <div>
              <dt className='font-semibold text-base-content/70'>Parameters</dt>
              <dd className='text-lg font-bold'>
                {formatNumber(parameterBillions, 3)} B
              </dd>
            </div>
            <div>
              <dt className='font-semibold text-base-content/70'>Hidden size</dt>
              <dd className='text-lg font-bold'>{effectiveHiddenSize || '–'}</dd>
            </div>
            <div>
              <dt className='font-semibold text-base-content/70'>Layers</dt>
              <dd className='text-lg font-bold'>{effectiveNumLayers || '–'}</dd>
            </div>
          </dl>
        </section>

        <section className='mt-10 grid gap-8 lg:grid-cols-[1.15fr_0.85fr]'>
          <div className='space-y-6'>
            <div className='rounded-xl border border-base-300 bg-base-100 p-6 shadow-lg'>
              <div className='flex flex-col gap-3 md:flex-row md:items-start md:justify-between'>
                <div>
                  <h2 className='text-xl font-semibold'>Quick estimator</h2>
                  <p className='mt-1 text-sm text-base-content/70'>
                    Size weights and KV cache instantly from core workload inputs.
                  </p>
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

              <div className='mt-6 grid gap-3 sm:grid-cols-3'>
                <div className='rounded-lg border border-base-300 bg-base-200 p-4 text-sm'>
                  <p className='text-xs font-semibold uppercase tracking-wide text-base-content/60'>
                    Model weights
                  </p>
                  <p className='mt-2 text-lg font-bold text-primary'>
                    {formatMemory(memoryBreakdown.weightsGB)}
                  </p>
                  <p className='text-xs text-base-content/60'>
                    {weightBits}-bit across {formatNumber(parameterBillions, 2)}B params
                  </p>
                </div>
                <div className='rounded-lg border border-base-300 bg-base-200 p-4 text-sm'>
                  <p className='text-xs font-semibold uppercase tracking-wide text-base-content/60'>
                    {mode === 'inference' ? 'KV cache' : 'Activations'}
                  </p>
                  <p className='mt-2 text-lg font-bold text-primary'>
                    {mode === 'inference'
                      ? formatMemory(memoryBreakdown.kvCacheGB)
                      : formatMemory(memoryBreakdown.activationsGB)}
                  </p>
                  <p className='text-xs text-base-content/60'>
                    {mode === 'inference'
                      ? `${sequenceLength} tokens × ${effectiveBatchSize} streams @ ${kvBits}-bit`
                      : 'Backprop activations using default heuristics'}
                  </p>
                </div>
                <div className='rounded-lg border border-base-300 bg-base-200 p-4 text-sm'>
                  <p className='text-xs font-semibold uppercase tracking-wide text-base-content/60'>
                    Total VRAM (incl. overhead)
                  </p>
                  <p className='mt-2 text-lg font-bold text-primary'>
                    {formatMemory(memoryBreakdown.totalGB)}
                  </p>
                  <p className='text-xs text-base-content/60'>
                    {formatNumber(overheadFactor, 2)}× framework headroom applied
                  </p>
                </div>
              </div>

              <div className='mt-6 grid gap-4 md:grid-cols-2'>
                <label className='flex flex-col text-sm'>
                  Parameter count (billions)
                  <input
                    className='input input-bordered mt-1'
                    type='number'
                    min='0'
                    step='0.1'
                    value={parameterBillions}
                    onChange={(event) =>
                      setParameterBillions(Number(event.target.value) || 0)
                    }
                  />
                </label>
                <label className='flex flex-col text-sm'>
                  Context length (tokens)
                  <input
                    className='input input-bordered mt-1'
                    type='number'
                    min='1'
                    value={sequenceLength}
                    onChange={(event) =>
                      setSequenceLength(Number(event.target.value) || 0)
                    }
                  />
                </label>
                <label className='flex flex-col text-sm'>
                  Weight precision
                  <select
                    className='select select-bordered mt-1'
                    value={weightBits}
                    onChange={(event) =>
                      setWeightBits(Number(event.target.value) as PrecisionBits)
                    }
                  >
                    {bitsOptions.map((bits) => (
                      <option key={bits} value={bits}>
                        {bits}-bit
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
                <label className='flex flex-col text-sm'>
                  Overhead factor
                  <input
                    className='input input-bordered mt-1'
                    type='number'
                    min='1'
                    step='0.05'
                    value={overheadFactor}
                    onChange={(event) =>
                      setOverheadFactor(Number(event.target.value) || 1)
                    }
                  />
                </label>
                {mode === 'training' && (
                  <label className='flex flex-col text-sm'>
                    Optimiser
                    <select
                      className='select select-bordered mt-1'
                      value={optimizer}
                      onChange={(event) =>
                        setOptimizer(event.target.value as typeof optimizer)
                      }
                    >
                      <option value='adamw'>AdamW</option>
                      <option value='adam'>Adam</option>
                      <option value='adafactor'>Adafactor</option>
                      <option value='lamb'>LAMB</option>
                      <option value='none'>None</option>
                    </select>
                  </label>
                )}
              </div>

              {mode === 'inference' ? (
                <div className='mt-6 space-y-2 text-sm'>
                  <span className='font-semibold text-base-content/80'>
                    Concurrent streams
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
                    <span className='badge badge-outline'>{concurrentUsers}</span>
                  </div>
                  <p className='text-xs text-base-content/70'>
                    KV cache grows linearly with concurrent sequences and context length.
                  </p>
                </div>
              ) : (
                <label className='mt-6 flex flex-col text-sm'>
                  Global batch size
                  <input
                    className='input input-bordered mt-1'
                    type='number'
                    min='1'
                    value={trainingBatchSize}
                    onChange={(event) =>
                      setTrainingBatchSize(Number(event.target.value) || 1)
                    }
                  />
                </label>
              )}
            </div>

            <div className='rounded-xl border border-base-300 bg-base-100 p-6 shadow-lg'>
              <div className='flex flex-col gap-3 md:flex-row md:items-start md:justify-between'>
                <div>
                  <h2 className='text-xl font-semibold'>Architecture extras</h2>
                  <p className='mt-1 text-sm text-base-content/70'>
                    {architectureMode === 'auto'
                      ? 'Hidden size, layer count and heads follow LLaMA-style scaling heuristics. Switch to manual to tailor them to a specific checkpoint.'
                      : 'You are overriding the heuristics. Adjust these numbers to mirror the model you plan to deploy.'}
                  </p>
                </div>
                <div className='join self-start'>
                  <button
                    className={`btn btn-sm join-item ${
                      architectureMode === 'auto' ? 'btn-primary' : 'btn-ghost'
                    }`}
                    type='button'
                    onClick={enableAutoArchitecture}
                  >
                    Auto
                  </button>
                  <button
                    className={`btn btn-sm join-item ${
                      architectureMode === 'manual' ? 'btn-primary' : 'btn-ghost'
                    }`}
                    type='button'
                    onClick={enableManualOverrides}
                  >
                    Manual
                  </button>
                </div>
              </div>

              <div className='mt-6 space-y-4 rounded-lg border border-dashed border-base-300 bg-base-200/60 p-4'>
                <dl className='grid gap-4 text-sm sm:grid-cols-2'>
                  <div>
                    <dt className='font-semibold text-base-content/70'>Hidden size</dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveHiddenSize || '–'}
                    </dd>
                  </div>
                  <div>
                    <dt className='font-semibold text-base-content/70'>Layers</dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveNumLayers || '–'}
                    </dd>
                  </div>
                  <div>
                    <dt className='font-semibold text-base-content/70'>Attention heads</dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveNumHeads || '–'}
                    </dd>
                  </div>
                  <div>
                    <dt className='font-semibold text-base-content/70'>Feed-forward size</dt>
                    <dd className='text-lg font-semibold'>
                      {effectiveIntermediateSize || '–'}
                    </dd>
                  </div>
                </dl>

                {architectureMode === 'manual' && (
                  <div className='grid gap-4 pt-2 text-sm md:grid-cols-2'>
                    <label className='flex flex-col'>
                      Hidden size
                      <input
                        className='input input-bordered mt-1'
                        type='number'
                        min='0'
                        value={manualHiddenSize}
                        onChange={(event) =>
                          setManualHiddenSize(Number(event.target.value) || 0)
                        }
                      />
                    </label>
                    <label className='flex flex-col'>
                      Layers
                      <input
                        className='input input-bordered mt-1'
                        type='number'
                        min='0'
                        value={manualNumLayers}
                        onChange={(event) =>
                          setManualNumLayers(Number(event.target.value) || 0)
                        }
                      />
                    </label>
                    <label className='flex flex-col'>
                      Attention heads
                      <input
                        className='input input-bordered mt-1'
                        type='number'
                        min='0'
                        value={manualNumHeads}
                        onChange={(event) =>
                          setManualNumHeads(Number(event.target.value) || 0)
                        }
                      />
                    </label>
                    <label className='flex flex-col'>
                      Feed-forward size
                      <input
                        className='input input-bordered mt-1'
                        type='number'
                        min='0'
                        placeholder={
                          manualHiddenSize ? String(manualHiddenSize * 4) : ''
                        }
                        value={manualIntermediateSize || ''}
                        onChange={(event) =>
                          setManualIntermediateSize(Number(event.target.value) || 0)
                        }
                      />
                    </label>
                    <label className='flex flex-col md:col-span-2'>
                      Vocabulary size (for FLOPs)
                      <input
                        className='input input-bordered mt-1'
                        type='number'
                        min='0'
                        value={vocabSize}
                        onChange={(event) =>
                          setVocabSize(Number(event.target.value) || 0)
                        }
                      />
                    </label>
                  </div>
                )}
              </div>
              {architectureMode === 'auto' && (
                <p className='mt-4 rounded-lg bg-base-200 px-3 py-2 text-xs text-base-content/70'>
                  Hidden size, depth and heads are prefilled from LLaMA scaling tables. Override them when using a custom architecture.
                </p>
              )}
            </div>
          </div>

          <div className='space-y-6'>
            <div className='rounded-xl border border-base-300 bg-base-100 p-6 shadow-lg'>
              <h2 className='text-xl font-semibold'>Memory &amp; hardware</h2>
              <p className='mt-1 text-sm text-base-content/70'>
                {mode === 'inference'
                  ? `Assumes ${effectiveBatchSize} concurrent ${
                      effectiveBatchSize === 1 ? 'stream' : 'streams'
                    } at ${sequenceLength} tokens.`
                  : `Assumes a global batch size of ${effectiveBatchSize} sequences.`}
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
                {mode === 'training' && memoryBreakdown.optimizerGB > 0 && (
                  <div className='flex items-center justify-between'>
                    <span>Optimizer state</span>
                    <span className='font-semibold'>
                      {formatMemory(memoryBreakdown.optimizerGB)}
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
                  <span>Framework overhead ({formatNumber(overheadFactor, 2)}×)</span>
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
                >
                  {gpus.map((gpu) => (
                    <option key={gpu.name} value={gpu.name}>
                      {gpu.name} ({gpu.memory_gb} GB)
                    </option>
                  ))}
                </select>

                {selectedGpu && (
                  <p
                    className={`mt-3 text-sm ${
                      memoryHeadroom !== null && memoryHeadroom >= 0
                        ? 'text-success'
                        : 'text-error'
                    }`}
                  >
                    {memoryHeadroom !== null && memoryHeadroom >= 0
                      ? `Fits with ${formatNumber(memoryHeadroom, 2)} GB headroom.`
                      : 'Model does not fit on the selected GPU.'}
                  </p>
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
                        <span>{gpu.name}</span>
                        <span className='text-base-content/70'>
                          {formatNumber(gpu.memoryHeadroomGB, 2)} GB spare
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className='rounded-xl border border-base-300 bg-base-100 p-6 shadow-lg'>
              <h2 className='text-xl font-semibold'>Performance</h2>
              <div className='mt-4 grid gap-4 text-sm md:grid-cols-2'>
                <label className='flex flex-col'>
                  Kernel efficiency
                  <input
                    className='input input-bordered mt-1'
                    type='number'
                    min='0.05'
                    max='1'
                    step='0.05'
                    value={efficiency}
                    onChange={(event) =>
                      setEfficiency(Number(event.target.value) || 0.3)
                    }
                  />
                </label>
                <div className='rounded-lg bg-base-200 p-3'>
                  <p>
                    <span className='font-semibold'>GPU FP32 throughput:</span>{' '}
                    {selectedGpu?.fp32_tflops
                      ? `${formatNumber(selectedGpu.fp32_tflops, 1)} TFLOPs`
                      : 'N/A'}
                  </p>
                  <p className='text-xs text-base-content/70'>
                    Adjust the efficiency multiplier to reflect framework and kernel
                    optimisations.
                  </p>
                </div>
              </div>

              <div className='mt-4 space-y-3 text-sm'>
                <p>
                  <span className='font-semibold'>Estimated FLOPs / sequence:</span>{' '}
                  {flops ? `${formatNumber(flops / 10 ** 12, 2)} TFLOPs` : 'N/A'}
                </p>
                <p>
                  <span className='font-semibold'>Tokens per second:</span>{' '}
                  {throughput.tokensPerSecond
                    ? formatNumber(throughput.tokensPerSecond, 2)
                    : 'N/A'}
                </p>
                <p>
                  <span className='font-semibold'>Milliseconds per token:</span>{' '}
                  {throughput.millisecondsPerToken
                    ? formatNumber(throughput.millisecondsPerToken, 2)
                    : 'N/A'}
                </p>
              </div>
            </div>

            <div className='rounded-xl border border-base-300 bg-base-100 p-6 shadow-lg'>
              <h2 className='text-xl font-semibold'>Cloud cost projection</h2>
              <div className='mt-4 grid gap-4 text-sm md:grid-cols-2'>
                <label className='flex flex-col'>
                  Cloud instance
                  <select
                    className='select select-bordered mt-1'
                    value={selectedInstanceName}
                    onChange={(event) =>
                      setSelectedInstanceName(event.target.value)
                    }
                  >
                    {cloudInstances.map((instance) => (
                      <option key={instance.name} value={instance.name}>
                        {instance.provider} {instance.name} ({instance.gpu})
                      </option>
                    ))}
                  </select>
                </label>
                <label className='flex flex-col'>
                  Runtime (hours)
                  <input
                    className='input input-bordered mt-1'
                    type='number'
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
                    <span className='font-semibold'>Effective hourly rate:</span>{' '}
                    ${formatNumber(hourlyRate, 2)}
                  </p>
                  <p>
                    <span className='font-semibold'>Estimated cost:</span>{' '}
                    ${formatNumber(costEstimate.totalCost, 2)}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

function parseModelConfig(config: Record<string, unknown>): ModelMetadata {
  const directParamKeys = [
    'num_parameters',
    'n_parameters',
    'n_params',
    'num_params',
    'total_params',
    'model_size',
  ];

  let parameterCount = 0;
  for (const key of directParamKeys) {
    const value = safeNumber(config[key]);
    if (value && value > 0) {
      parameterCount = value;
      break;
    }
  }

  if (!parameterCount && typeof config.model === 'object' && config.model !== null) {
    const nested = config.model as Record<string, unknown>;
    const nestedCount = safeNumber(nested.num_parameters);
    if (nestedCount && nestedCount > 0) {
      parameterCount = nestedCount;
    }
  }

  const hiddenSize =
    safeNumber(config.hidden_size) ?? safeNumber(config.d_model) ?? 0;
  const numLayers =
    safeNumber(config.num_hidden_layers) ??
    safeNumber(config.num_layers) ??
    safeNumber(config.n_layer) ??
    0;
  const numHeads =
    safeNumber(config.num_attention_heads) ??
    safeNumber(config.num_heads) ??
    safeNumber(config.n_head) ??
    0;
  const intermediateSize =
    safeNumber(config.intermediate_size) ??
    safeNumber(config.ffn_dim) ??
    safeNumber(config.d_ff);
  const sequenceLength =
    safeNumber(config.max_position_embeddings) ??
    safeNumber(config.seq_length) ??
    safeNumber(config.n_positions);
  const vocabSize = safeNumber(config.vocab_size);
  const kvHeads =
    safeNumber(config.num_key_value_heads) ?? safeNumber(config.n_head_kv);

  if (!parameterCount && hiddenSize && numLayers && numHeads && vocabSize) {
    parameterCount = estimateTransformerParameters({
      vocabSize,
      hiddenSize,
      numLayers,
      numAttentionHeads: numHeads,
      intermediateSize,
      numKeyValueHeads: kvHeads ?? undefined,
    });
  }

  const dtypeRaw = config.torch_dtype as string | undefined;
  const dtypeBits: PrecisionBits | undefined = (() => {
    switch (dtypeRaw) {
      case 'float32':
      case 'torch.float32':
        return 32;
      case 'float16':
      case 'torch.float16':
      case 'bfloat16':
      case 'torch.bfloat16':
        return 16;
      case 'int8':
        return 8;
      case 'int4':
        return 4;
      default:
        return undefined;
    }
  })();

  return {
    parameterCount,
    hiddenSize: hiddenSize ?? 0,
    numLayers: numLayers ?? 0,
    numHeads: numHeads ?? 0,
    intermediateSize: intermediateSize ?? undefined,
    sequenceLength: sequenceLength ?? undefined,
    vocabSize: vocabSize ?? undefined,
    dtypeBits,
  };
}

