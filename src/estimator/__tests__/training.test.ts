import {
  estimateLoraTrainableParameters,
  estimateTrainingMemory,
  estimateTrainingRun,
  recommendTrainingPlans,
} from '../training';

const architecture = {
  parameterCount: 12_000_000_000,
  hiddenSize: 3840,
  numLayers: 48,
  numAttentionHeads: 16,
  numKeyValueHeads: 8,
  headDim: 256,
  intermediateSize: 15360,
};

const baseMemoryInput = {
  ...architecture,
  method: 'qlora' as const,
  loraRank: 16,
  targetCoverage: 'all-linear' as const,
  sequenceLength: 2048,
  microBatchSize: 1,
  deviceCount: 1,
  distribution: 'replicated' as const,
  optimizer: 'adamw' as const,
  gradientCheckpointing: true,
  overheadFactor: 1.15,
  trainingPrecisionBits: 16 as const,
  optimizerPrecisionBits: 32 as const,
};

describe('training estimator', () => {
  it('derives a bounded adapter parameter count from architecture dimensions', () => {
    const qv = estimateLoraTrainableParameters({
      ...architecture,
      rank: 16,
      targetCoverage: 'attention-qv',
    });
    const allLinear = estimateLoraTrainableParameters({
      ...architecture,
      rank: 16,
      targetCoverage: 'all-linear',
    });

    expect(qv).toBeGreaterThan(0);
    expect(allLinear).toBeGreaterThan(qv);
    expect(allLinear).toBeLessThan(architecture.parameterCount * 0.02);
  });

  it('keeps QLoRA below LoRA memory by quantizing the frozen base', () => {
    const qlora = estimateTrainingMemory(baseMemoryInput);
    const lora = estimateTrainingMemory({
      ...baseMemoryInput,
      method: 'lora',
    });

    expect(qlora.baseWeightsGB).toBeLessThan(lora.baseWeightsGB);
    expect(qlora.trainableParameterCount).toBe(lora.trainableParameterCount);
    expect(qlora.perDeviceGB).toBeLessThan(lora.perDeviceGB);
  });

  it('distinguishes replicated training from fully sharded state', () => {
    const replicatedOne = estimateTrainingMemory({
      ...baseMemoryInput,
      method: 'full',
    });
    const replicatedFour = estimateTrainingMemory({
      ...baseMemoryInput,
      method: 'full',
      deviceCount: 4,
    });
    const shardedFour = estimateTrainingMemory({
      ...baseMemoryInput,
      method: 'full',
      deviceCount: 4,
      distribution: 'fully-sharded',
    });

    expect(replicatedFour.perDeviceGB).toBeCloseTo(
      replicatedOne.perDeviceGB,
      6,
    );
    expect(shardedFour.perDeviceGB).toBeLessThan(replicatedFour.perDeviceGB);
    expect(shardedFour.activationsGB).toBeCloseTo(
      replicatedFour.activationsGB,
      6,
    );
  });

  it('accounts for model and optimizer precision in full fine-tuning', () => {
    const bf16 = estimateTrainingMemory({
      ...baseMemoryInput,
      method: 'full',
      trainingPrecisionBits: 16,
      optimizerPrecisionBits: 32,
    });
    const fp32 = estimateTrainingMemory({
      ...baseMemoryInput,
      method: 'full',
      trainingPrecisionBits: 32,
      optimizerPrecisionBits: 32,
    });
    const eightBitOptimizer = estimateTrainingMemory({
      ...baseMemoryInput,
      method: 'full',
      trainingPrecisionBits: 16,
      optimizerPrecisionBits: 8,
    });

    expect(fp32.baseWeightsGB).toBeGreaterThan(bf16.baseWeightsGB);
    expect(fp32.gradientsGB).toBeGreaterThan(bf16.gradientsGB);
    expect(eightBitOptimizer.optimizerGB).toBeLessThan(bf16.optimizerGB);
    expect(eightBitOptimizer.perDeviceGB).toBeLessThan(bf16.perDeviceGB);
  });

  it('turns dataset size and aggregate compute into runtime and cost', () => {
    const one = estimateTrainingRun({
      activeParameterCount: 12_000_000_000,
      method: 'qlora',
      datasetTokens: 100_000_000,
      epochs: 1,
      deviceCount: 1,
      tflopsPerDevice: 100,
      efficiency: 0.3,
      hourlyRatePerDevice: 0.99,
    });
    const four = estimateTrainingRun({
      activeParameterCount: 12_000_000_000,
      method: 'qlora',
      datasetTokens: 100_000_000,
      epochs: 1,
      deviceCount: 4,
      tflopsPerDevice: 100,
      efficiency: 0.3,
      hourlyRatePerDevice: 0.99,
    });

    expect(one.durationHours).toBeGreaterThan(0);
    expect(one.totalCost).toBeGreaterThan(0);
    expect(four.durationHours).toBeLessThan(one.durationHours);
    expect(four.scalingEfficiency).toBeLessThan(1);
  });

  it('recommends the lowest-cost verified plan independently of selection', () => {
    const plans = recommendTrainingPlans({
      memoryInput: baseMemoryInput,
      runInput: {
        activeParameterCount: 12_000_000_000,
        method: 'qlora',
        datasetTokens: 100_000_000,
        epochs: 1,
        efficiency: 0.3,
      },
      computeFormat: 'bf16',
      hardware: [
        {
          name: 'GPU A',
          memory_gb: 24,
          per_device_memory_gb: 24,
          device_count: 1,
          fp32_tflops: 40,
        },
        {
          name: 'GPU B',
          memory_gb: 80,
          per_device_memory_gb: 80,
          device_count: 1,
          fp32_tflops: 100,
          bf16_tflops: 200,
        },
      ],
      cloudRates: [
        {
          provider: 'Provider',
          name: 'A',
          gpu_catalog_names: ['GPU A'],
          hourly_rate: 0.5,
          pricing_source_url: 'https://example.com/a',
          source_checked_at: '2026-08-24',
        },
        {
          provider: 'Provider',
          name: 'B',
          gpu_catalog_names: ['GPU B'],
          hourly_rate: 2,
          pricing_source_url: 'https://example.com/b',
          source_checked_at: '2026-08-24',
        },
      ],
      maxDevices: 4,
    });

    expect(plans.length).toBeGreaterThan(0);
    expect(
      plans.every((plan) => plan.memory.perDeviceGB <= plan.memoryPerGpuGB),
    ).toBe(true);
    expect(plans[0].totalCost).toBeLessThanOrEqual(
      plans.at(-1)?.totalCost ?? Number.POSITIVE_INFINITY,
    );
  });

  it('can recommend a sharded multi-GPU plan when selected DDP cannot fit', () => {
    const plans = recommendTrainingPlans({
      memoryInput: {
        ...baseMemoryInput,
        method: 'full',
        distribution: 'replicated',
      },
      runInput: {
        activeParameterCount: 12_000_000_000,
        method: 'full',
        datasetTokens: 10_000_000,
        epochs: 1,
        efficiency: 0.3,
      },
      computeFormat: 'bf16',
      distributions: ['replicated', 'fully-sharded'],
      hardware: [
        {
          name: '80GB GPU',
          memory_gb: 80,
          per_device_memory_gb: 80,
          device_count: 1,
          fp32_tflops: 50,
          bf16_tflops: 100,
        },
      ],
      cloudRates: [
        {
          provider: 'Provider',
          name: '80GB GPU cloud',
          gpu_catalog_names: ['80GB GPU'],
          hourly_rate: 2,
          pricing_source_url: 'https://example.com',
          source_checked_at: '2026-08-24',
        },
      ],
      maxDevices: 8,
    });

    expect(plans.length).toBeGreaterThan(0);
    expect(plans[0].distribution).toBe('fully-sharded');
    expect(plans[0].deviceCount).toBeGreaterThan(1);
  });
});
