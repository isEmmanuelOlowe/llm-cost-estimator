import {
  bitsToBytes,
  calculateActivationMemoryGB,
  calculateKvCacheMemoryGB,
  calculateMemoryFromBillions,
  calculateOptimizerMemoryGB,
  calculateWeightMemoryGB,
  estimateCloudCost,
  estimateDecoderFlops,
  estimateHardwareFit,
  estimateKvCache,
  estimateLlamaStyleArchitecture,
  estimateMemory,
  estimateThroughput,
  estimateTransformerParameterBreakdown,
  estimateTransformerParameters,
  recommendGpus,
  resolveEffectiveParameterCount,
  selectGpuComputeTFlops,
} from '../estimator';

describe('Estimator utilities', () => {
  it('converts bits to bytes', () => {
    expect(bitsToBytes(32)).toBe(4);
    expect(bitsToBytes(16)).toBe(2);
    expect(bitsToBytes(4)).toBe(0.5);
  });

  it('computes weight memory in GB', () => {
    const paramCount = 7 * 10 ** 9;
    const fp16 = calculateWeightMemoryGB(paramCount, 16);
    expect(fp16).toBeCloseTo((paramCount * 2) / 1024 ** 3);
  });

  it('scales activation memory by mode', () => {
    const params = 1 * 10 ** 9;
    const inferenceActivations = calculateActivationMemoryGB(
      params,
      16,
      'inference',
    );
    const trainingActivations = calculateActivationMemoryGB(
      params,
      16,
      'training',
    );

    expect(trainingActivations).toBeGreaterThan(inferenceActivations);
    expect(trainingActivations / inferenceActivations).toBeCloseTo(10);
  });

  it('computes kv cache memory', () => {
    const kv = calculateKvCacheMemoryGB({
      sequenceLength: 4096,
      batchSize: 1,
      numLayers: 32,
      hiddenSize: 4096,
      precisionBits: 16,
    });
    expect(kv).toBeGreaterThan(1);
    expect(kv).toBeLessThan(3);
    expect(kv).toBeCloseTo(2, 5);
  });

  it('scales KV cache with key/value heads rather than query heads', () => {
    const mha = estimateKvCache({
      sequenceLength: 4096,
      batchSize: 1,
      numLayers: 80,
      hiddenSize: 8192,
      numAttentionHeads: 64,
      numKeyValueHeads: 64,
      precisionBits: 16,
    });
    const gqa = estimateKvCache({
      sequenceLength: 4096,
      batchSize: 1,
      numLayers: 80,
      hiddenSize: 8192,
      numAttentionHeads: 64,
      numKeyValueHeads: 8,
      precisionBits: 16,
    });

    expect(gqa.bytesPerToken).toBe(mha.bytesPerToken / 8);
    expect(gqa.totalGB).toBeCloseTo(mha.totalGB / 8, 8);
    expect(gqa.headDim).toBe(128);
  });

  it('accounts for DeepSeek V4 compressed cache layers instead of treating every layer as full KV', () => {
    const standard = estimateKvCache({
      sequenceLength: 1_048_576,
      batchSize: 1,
      numLayers: 61,
      hiddenSize: 7168,
      numAttentionHeads: 128,
      numKeyValueHeads: 1,
      headDim: 512,
      precisionBits: 16,
    });
    const optimized = estimateKvCache({
      sequenceLength: 1_048_576,
      batchSize: 1,
      numLayers: 61,
      hiddenSize: 7168,
      numAttentionHeads: 128,
      numKeyValueHeads: 1,
      headDim: 512,
      precisionBits: 16,
      kvCacheArchitecture: {
        mode: 'deepseek-v4',
        keyValueShared: true,
        localAttentionLayers: 61,
        fullAttentionLayers: 0,
        slidingWindow: 128,
        compressedSparseLayers: 30,
        heavilyCompressedLayers: 31,
        compressedSparseRate: 4,
        heavilyCompressedRate: 128,
        indexHeadDim: 128,
      },
    });

    expect(optimized.cacheMode).toBe('deepseek-v4');
    expect(optimized.totalGB).toBeLessThan(standard.totalGB / 5);
    expect(optimized.cacheDescription).toContain('compressed-sparse');
  });

  it('caps local attention layers and excludes state-space layers from KV storage', () => {
    const inkling = estimateKvCache({
      sequenceLength: 1_048_576,
      batchSize: 1,
      numLayers: 42,
      hiddenSize: 4096,
      numAttentionHeads: 32,
      numKeyValueHeads: 8,
      headDim: 128,
      precisionBits: 16,
      kvCacheArchitecture: {
        mode: 'hybrid-sliding-window',
        slidingWindow: 512,
        localAttentionLayers: 35,
        fullAttentionLayers: 7,
      },
    });
    const nemotron = estimateKvCache({
      sequenceLength: 262_144,
      batchSize: 1,
      numLayers: 52,
      hiddenSize: 2688,
      numAttentionHeads: 32,
      numKeyValueHeads: 2,
      headDim: 128,
      precisionBits: 16,
      kvCacheArchitecture: {
        mode: 'hybrid-state-space',
        fullAttentionLayers: 6,
        noAttentionLayers: 46,
        recurrentStateLayers: 23,
        recurrentStateBytesPerLayer: 2_195_456,
      },
    });

    expect(inkling.totalGB).toBeLessThan(50);
    expect(nemotron.attentionLayers).toBe(6);
    expect(nemotron.stateCacheGB).toBeGreaterThan(0);
    expect(nemotron.totalGB).toBeGreaterThan(nemotron.stateCacheGB);
  });

  it('uses latent KV rank for MLA-style Kimi attention', () => {
    const kimi = estimateKvCache({
      sequenceLength: 262_144,
      batchSize: 1,
      numLayers: 61,
      hiddenSize: 7168,
      numAttentionHeads: 64,
      numKeyValueHeads: 64,
      headDim: 128,
      precisionBits: 16,
      kvCacheArchitecture: {
        mode: 'latent-attention',
        keyValueShared: true,
        fullAttentionLayers: 61,
        latentKvRank: 512,
        latentRopeDim: 64,
      },
    });

    expect(kimi.cacheMode).toBe('latent-attention');
    expect(kimi.totalGB).toBeCloseTo(17.15625, 5);
    expect(kimi.totalGB).toBeLessThan(100);
  });

  it('computes optimizer memory for adam', () => {
    const params = 3 * 10 ** 9;
    const optimizerGb = calculateOptimizerMemoryGB(params, 16, 'adamw');
    const weightGb = calculateWeightMemoryGB(params, 16);
    expect(optimizerGb).toBeCloseTo(weightGb * 4);
  });

  it('provides a full memory breakdown', () => {
    const breakdown = estimateMemory({
      parameterCount: 13 * 10 ** 9,
      weightPrecisionBits: 16,
      mode: 'inference',
      hiddenSize: 5120,
      numLayers: 40,
      sequenceLength: 4096,
      batchSize: 1,
    });

    expect(breakdown.weightsGB).toBeGreaterThan(20);
    expect(breakdown.kvCacheGB).toBeGreaterThan(3);
    expect(breakdown.totalGB).toBeGreaterThan(breakdown.baseTotalGB);
  });

  it('estimates throughput with efficiency factor', () => {
    const throughput = estimateThroughput({
      parameterCount: 7 * 10 ** 9,
      gpuTFlops: 40,
    });

    expect(throughput.tokensPerSecond).toBeGreaterThan(0);
    expect(throughput.millisecondsPerToken).toBeGreaterThan(0);
    expect(throughput.tokensPerSecond).toBeCloseTo(857.1428571428571, 8);
    expect(throughput.bottleneck).toBe('compute');
  });

  it('uses memory bandwidth when compute throughput is unavailable', () => {
    const throughput = estimateThroughput({
      parameterCount: 7 * 10 ** 9,
      gpuTFlops: 0,
      memoryBandwidthGBs: 546,
      weightPrecisionBits: 16,
      memoryEfficiency: 1,
    });

    expect(throughput.computeBoundTokensPerSecond).toBe(0);
    expect(throughput.memoryBoundTokensPerSecond).toBeCloseTo(
      (546 * 10 ** 9) / (7 * 10 ** 9 * 2),
      8,
    );
    expect(throughput.tokensPerSecond).toBe(
      throughput.memoryBoundTokensPerSecond,
    );
    expect(throughput.bottleneck).toBe('memory');
  });

  it('estimates cloud cost', () => {
    const cost = estimateCloudCost({ hourlyRate: 3.06, durationHours: 2 });
    expect(cost.totalCost).toBeCloseTo(6.12);
  });

  it('recommends GPUs with enough memory headroom', () => {
    const results = recommendGpus(10, 5);
    expect(results.length).toBeGreaterThan(0);
    expect(results.every((gpu) => gpu.memoryHeadroomGB >= 0)).toBe(true);
    expect(results.every((gpu) => gpu.requiredDevices <= gpu.deviceCount)).toBe(
      true,
    );
  });

  it('calculates aggregate and per-device hardware fit', () => {
    const fit = estimateHardwareFit(200, {
      name: 'Test 2-GPU system',
      memory_gb: 256,
      per_device_memory_gb: 128,
      device_count: 2,
      fp32_tflops: 1,
    });

    expect(fit.fits).toBe(true);
    expect(fit.requiredDevices).toBe(2);
    expect(fit.aggregateHeadroomGB).toBe(56);
    expect(fit.perDeviceHeadroomGB).toBe(-72);
  });

  it('estimates decoder flops', () => {
    const flops = estimateDecoderFlops({
      numLayers: 32,
      hiddenSize: 4096,
      sequenceLength: 2048,
      vocabSize: 32000,
    });
    expect(flops).toBeGreaterThan(0);
    expect(flops / 10 ** 12).toBeCloseTo(13.731010445312, 8);
  });

  it('accounts for architecture dimensions in detailed FLOPs estimates', () => {
    const mha = estimateDecoderFlops({
      numLayers: 32,
      hiddenSize: 4096,
      sequenceLength: 2048,
      vocabSize: 32000,
      intermediateSize: 11008,
      numAttentionHeads: 32,
      numKeyValueHeads: 32,
      headDim: 128,
      gatedMlp: true,
    });
    const gqa = estimateDecoderFlops({
      numLayers: 32,
      hiddenSize: 4096,
      sequenceLength: 2048,
      vocabSize: 32000,
      intermediateSize: 11008,
      numAttentionHeads: 32,
      numKeyValueHeads: 8,
      headDim: 128,
      gatedMlp: true,
    });

    expect(mha).toBeGreaterThan(gqa);
    expect(gqa).toBeGreaterThan(0);
  });

  it('estimates transformer parameters when not explicitly provided', () => {
    const params = estimateTransformerParameters({
      vocabSize: 32000,
      hiddenSize: 4096,
      numLayers: 32,
      numAttentionHeads: 32,
    });

    expect(params).toBeGreaterThan(0);
  });

  it('matches scratch-validated numeric checks for exact arithmetic paths', () => {
    expect(calculateWeightMemoryGB(7 * 10 ** 9, 16)).toBeCloseTo(
      13.0385160446167,
      8,
    );
    expect(
      calculateKvCacheMemoryGB({
        sequenceLength: 4096,
        batchSize: 1,
        numLayers: 40,
        hiddenSize: 5120,
        precisionBits: 16,
      }),
    ).toBeCloseTo(3.125, 8);
    expect(calculateOptimizerMemoryGB(3 * 10 ** 9, 16, 'adamw')).toBeCloseTo(
      22.351741790771484,
      8,
    );
  });

  it('treats fallback parameter estimation as heuristic rather than exact', () => {
    const sevenBLike = estimateTransformerParameters({
      vocabSize: 32000,
      hiddenSize: 4096,
      numLayers: 32,
      numAttentionHeads: 32,
      intermediateSize: 11008,
    });

    expect(sevenBLike / 10 ** 9).toBeCloseTo(5.164498944, 8);
    expect(sevenBLike).toBeGreaterThan(5 * 10 ** 9);
  });

  it('exposes parameter components and active MoE parameters', () => {
    const breakdown = estimateTransformerParameterBreakdown({
      vocabSize: 10000,
      hiddenSize: 2048,
      numLayers: 24,
      numAttentionHeads: 16,
      numKeyValueHeads: 4,
      intermediateSize: 5632,
      numExperts: 8,
      numExpertsPerToken: 2,
      gatedMlp: true,
      tieWordEmbeddings: false,
    });

    expect(breakdown.totalParameters).toBeGreaterThan(
      breakdown.activeParameters,
    );
    expect(breakdown.numKeyValueHeads).toBe(4);
    expect(breakdown.numExpertsPerToken).toBe(2);
  });

  it('selects precision-aware hardware throughput when published', () => {
    const hardware = {
      name: 'Test accelerator',
      memory_gb: 32,
      fp32_tflops: 10,
      fp16_tflops: 40,
      bf16_tflops: 38,
      fp8_tflops: 80,
    };

    expect(selectGpuComputeTFlops(hardware, 32)).toBe(10);
    expect(selectGpuComputeTFlops(hardware, 16)).toBe(38);
    expect(selectGpuComputeTFlops(hardware, 8)).toBe(80);
    expect(selectGpuComputeTFlops(hardware, 16, 'bf16')).toBe(38);
    expect(selectGpuComputeTFlops(hardware, 16, 'fp16')).toBe(40);
    expect(
      selectGpuComputeTFlops(
        { name: 'FP32-only', memory_gb: 1, fp32_tflops: 10 },
        16,
      ),
    ).toBe(0);
  });

  it('prefers active parameters for MoE-aware throughput inputs', () => {
    expect(resolveEffectiveParameterCount(80 * 10 ** 9, 3 * 10 ** 9)).toBe(
      3 * 10 ** 9,
    );
    expect(resolveEffectiveParameterCount(27 * 10 ** 9, null)).toBe(
      27 * 10 ** 9,
    );
  });

  it('derives a llama-style architecture from parameter counts', () => {
    const sevenB = estimateLlamaStyleArchitecture(7 * 10 ** 9);
    expect(sevenB.hiddenSize).toBeGreaterThan(3000);
    expect(sevenB.numLayers).toBeGreaterThan(20);

    const seventyB = estimateLlamaStyleArchitecture(70 * 10 ** 9);
    expect(seventyB.hiddenSize).toBeGreaterThan(sevenB.hiddenSize);
    expect(seventyB.numLayers).toBeGreaterThan(sevenB.numLayers);
  });

  it('estimates memory from billions helper', () => {
    const gb = calculateMemoryFromBillions(7, 16);
    expect(gb).toBeCloseTo(calculateWeightMemoryGB(7 * 10 ** 9, 16));
  });
});
