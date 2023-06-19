export function calculateMemory(numParams: number, bytes: number): number {
  const bytesPerParam = bytes / 8; // Assuming each parameter is a 32-bit float.
  const paramsInGB = (numParams * bytesPerParam) / 1024 ** 3; // Convert to GB.
  return paramsInGB;
}

export function calculateNumParams(
  numLayers: number,
  seqLength: number,
  hiddenDims: number
): number {
  const selfAttentionParams = 4 * numLayers * hiddenDims * hiddenDims;
  const ffNetworkParams = 2 * numLayers * hiddenDims * hiddenDims * 4;
  const positionalEncodingsParams = seqLength * hiddenDims;
  const totalParams =
    selfAttentionParams + ffNetworkParams + positionalEncodingsParams;
  return totalParams;
}

interface Config {
  vocab_size: number;
  d_model: number;
  n_head: number;
  num_layers: number;
  n_head_kv: number;
  d_ff: number;
  n_positions: number;
  num_decoder_layers: number;
}

export function estimateModelSizeT5(config: Config): number {
  let totalParams = 0;

  // Count the number of parameters in the encoder layers
  const encoderLayersParams =
    config.d_model * config.vocab_size * 4 * config.d_model + // self-attention params
    config.d_model * config.vocab_size * config.d_ff + // feed-forward layer params
    config.d_model * config.vocab_size * 2; // layer normalization params
  totalParams += config.num_layers * encoderLayersParams;

  // Count the number of parameters in the decoder layers
  const decoderLayersParams =
    config.d_model * config.vocab_size * 4 * config.d_model + // self-attention params
    config.d_model * config.vocab_size * config.d_ff * 2 + // encoder-decoder attention + feed-forward layer params
    config.d_model * config.vocab_size * 3; // layer normalization params
  totalParams += config.num_decoder_layers * decoderLayersParams;

  // Count the number of parameters in the embedding layers
  const embeddingParams =
    config.vocab_size * config.d_model + // token embeddings
    config.n_positions * config.d_model; // position embeddings
  totalParams += embeddingParams;

  return totalParams;
}

export function calculateFlops(
  numLayers: number,
  seqLength: number,
  hiddenDims: number,
  numHeads: number
): number {
  // Matrix multiplication in self-attention.
  const selfAttentionFlops = numLayers * seqLength * hiddenDims * numHeads;
  // Matrix multiplication in feed-forward networks.
  const ffNetworkFlops = numLayers * seqLength * hiddenDims * hiddenDims;
  const totalFlops = selfAttentionFlops + ffNetworkFlops;
  return totalFlops;
}

export function estimateModelSize(
  vocab_size: number,
  hidden_size: number,
  n_head: number,
  n_layer: number,
  n_head_kv: number
): number {
  // Word Embeddings
  let total_params = vocab_size * hidden_size;

  // Self-attention Layers
  total_params +=
    n_layer * (n_head * hidden_size * 3 + n_head_kv * hidden_size * 2) * 2;

  // Position-wise Feed-forward Networks
  total_params += n_layer * hidden_size * 4;

  // LayerNorm Layers
  total_params += n_layer * hidden_size * 2;

  return total_params;
}

export function estimateInferenceTime(
  flops: number,
  gpu_tflops: number,
  num_params: number,
  memory_bandwidth: number,
  overhead_factor: number
): number {
  // Convert GPU TFLOPs to FLOPs and divide by 2 because of FMA.
  const gpu_flops: number = (gpu_tflops * 1e12) / 2;

  // Estimate the compute time based on FLOPs.
  const compute_time: number = flops / gpu_flops;

  // Estimate the memory time.
  // Convert memory bandwidth to bytes per second.
  const memory_bandwidth_bytes: number = memory_bandwidth * 1024 ** 3;
  // Calculate bytes per parameter.
  const bytes_per_param = 4;
  const memory_time: number =
    (num_params * bytes_per_param) / memory_bandwidth_bytes;

  // The total inference time is the maximum of the compute time and the memory time,
  // because the GPU can often do computation and memory operations simultaneously.
  // Multiply by the overhead factor to account for software overhead.
  const inference_time: number =
    Math.max(compute_time, memory_time) * overhead_factor;

  return inference_time;
}

export function estimateTrainingCost(
  flops: number,
  gpu_tflops: number,
  num_params: number,
  memory_bandwidth: number,
  overhead_factor: number,
  gpu_hourly_cost: number,
  num_epochs: number,
  dataset_size: number
): number {
  // Estimate the time required to process one item.
  const time_per_item: number = estimateInferenceTime(
    flops,
    gpu_tflops,
    num_params,
    memory_bandwidth,
    overhead_factor
  );

  // Estimate the total training time.
  const total_time: number = time_per_item * dataset_size * num_epochs;

  // Convert time to hours.
  const total_time_hours: number = total_time / 3600;

  // Calculate the total cost.
  const total_cost: number = total_time_hours * gpu_hourly_cost;

  return total_cost;
}

export function recommendGPU(memoryInGB: number, flops: number): string {
  // These thresholds and recommendations are highly simplified.
  // Actual GPU requirements may vary based on many factors.
  if (memoryInGB <= 8 && flops <= 1e9) {
    return 'Minimum recommendation: NVIDIA GTX 1060 6GB';
  } else if (memoryInGB <= 11 && flops <= 2e9)
    return 'Minimum recommendation: NVIDIA GTX 1080 Ti';
  else if (memoryInGB <= 24 && flops <= 5e9) {
    return 'Minimum recommendation: NVIDIA RTX 2080 Ti';
  } else {
    return 'Minimum recommendation: NVIDIA A100 or higher';
  }
}
