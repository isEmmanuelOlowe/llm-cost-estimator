import {
  bitsToBytes,
  calculateActivationMemoryGB,
  calculateKvCacheMemoryGB,
  calculateMemoryFromBillions,
  calculateOptimizerMemoryGB,
  calculateWeightMemoryGB,
  estimateCloudCost,
  estimateDecoderFlops,
  estimateLlamaStyleArchitecture,
  estimateMemory,
  estimateThroughput,
  estimateTransformerParameters,
  recommendGpus,
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
      'inference'
    );
    const trainingActivations = calculateActivationMemoryGB(
      params,
      16,
      'training'
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
  });

  it('estimates cloud cost', () => {
    const cost = estimateCloudCost({ hourlyRate: 3.06, durationHours: 2 });
    expect(cost.totalCost).toBeCloseTo(6.12);
  });

  it('recommends GPUs with enough memory headroom', () => {
    const results = recommendGpus(10, 5);
    expect(results.length).toBeGreaterThan(0);
    expect(results.every((gpu) => gpu.memoryHeadroomGB >= 0)).toBe(true);
  });

  it('estimates decoder flops', () => {
    const flops = estimateDecoderFlops({
      numLayers: 32,
      hiddenSize: 4096,
      sequenceLength: 2048,
      vocabSize: 32000,
    });
    expect(flops).toBeGreaterThan(0);
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

