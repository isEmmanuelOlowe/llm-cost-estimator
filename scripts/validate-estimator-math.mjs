const BYTES_PER_GB = 1024 ** 3;

function bitsToBytes(bits) {
  return bits / 8;
}

function weightGB(parameterCount, bits) {
  return (parameterCount * bitsToBytes(bits)) / BYTES_PER_GB;
}

function kvCacheGB({ sequenceLength, batchSize, numLayers, hiddenSize, bits }) {
  return (
    (2 *
      numLayers *
      hiddenSize *
      bitsToBytes(bits) *
      sequenceLength *
      batchSize) /
    BYTES_PER_GB
  );
}

function optimizerGB(parameterCount, bits, multiplier) {
  return (parameterCount * bitsToBytes(bits) * multiplier) / BYTES_PER_GB;
}

function throughput({ parameterCount, gpuTFlops, efficiency = 0.3 }) {
  const flopsPerToken = parameterCount * 2;
  const effectiveFlopsPerSecond = gpuTFlops * 10 ** 12 * efficiency;
  return effectiveFlopsPerSecond / flopsPerToken;
}

function decoderFlops({ numLayers, hiddenSize, sequenceLength, vocabSize }) {
  return (
    4 * numLayers * sequenceLength * hiddenSize ** 2 +
    8 * numLayers * sequenceLength * hiddenSize ** 2 +
    2 * sequenceLength * hiddenSize * vocabSize
  );
}

function assertClose(name, actual, expected, tolerance = 1e-9) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${name} mismatch: expected ${expected}, got ${actual}`);
  }
}

const cases = [
  ['7B fp16 weights (GiB)', weightGB(7e9, 16), 13.0385160446167],
  [
    '13B kv cache (GiB)',
    kvCacheGB({
      sequenceLength: 4096,
      batchSize: 1,
      numLayers: 40,
      hiddenSize: 5120,
      bits: 16,
    }),
    3.125,
  ],
  [
    '3B AdamW optimizer state (GiB)',
    optimizerGB(3e9, 16, 4),
    22.351741790771484,
  ],
  [
    '7B throughput @40TF 30% (tok/s)',
    throughput({ parameterCount: 7e9, gpuTFlops: 40, efficiency: 0.3 }),
    857.1428571428571,
  ],
  [
    'Decoder FLOPs example (TFLOPs)',
    decoderFlops({
      numLayers: 32,
      hiddenSize: 4096,
      sequenceLength: 2048,
      vocabSize: 32000,
    }) / 1e12,
    13.731010445312,
  ],
];

for (const [name, actual, expected] of cases) {
  assertClose(name, actual, expected);
  console.log(`${name}: ${actual}`);
}

console.log('Estimator math validation passed.');
