export function calculateMemory(numParams: number, bytes: number): number {
  const bytesPerParam = bytes / 8;
  const paramsInGB = (numParams * bytesPerParam) / 1024 ** 3;
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

  const encoderLayersParams =
    config.d_model * config.vocab_size * 4 * config.d_model +
    config.d_model * config.vocab_size * config.d_ff +
    config.d_model * config.vocab_size * 2;
  totalParams += config.num_layers * encoderLayersParams;

  const decoderLayersParams =
    config.d_model * config.vocab_size * 4 * config.d_model +
    config.d_model * config.vocab_size * config.d_ff * 2 +
    config.d_model * config.vocab_size * 3;
  totalParams += config.num_decoder_layers * decoderLayersParams;

  const embeddingParams =
    config.vocab_size * config.d_model +
    config.n_positions * config.d_model;
  totalParams += embeddingParams;

  return totalParams;
}

export function calculateFlops(
  numLayers: number,
  seqLength: number,
  hiddenDims: number,
  vocabSize: number
): number {
  const attentionMlpFlops =
    10 * numLayers * seqLength * hiddenDims * hiddenDims;
  const outputProjectionFlops = 2 * seqLength * hiddenDims * vocabSize;
  const totalFlops = attentionMlpFlops + outputProjectionFlops;
  return totalFlops;
}

export function estimateModelSize(
  vocab_size: number,
  hidden_size: number,
  n_head: number,
  n_layer: number,
  n_head_kv?: number | null
): number {
  let total_params = vocab_size * hidden_size;
  const effective_n_head_kv = n_head_kv ?? n_head;
  total_params +=
    n_layer *
    (n_head * hidden_size * 3 + effective_n_head_kv * hidden_size * 2) *
    2;
  total_params += n_layer * hidden_size * 4;
  total_params += n_layer * hidden_size * 2;
  return total_params;
}

export function estimateInferenceTime(
  flops: number,
  gpu_tflops: number,
  num_params: number,
  memory_bandwidth: number,
  overhead_factor: number,
  bytes_per_param: number
): number {
  const gpu_flops_val: number = gpu_tflops * 10 ** 12;
  const compute_time: number = (flops * 10 ** 9) / gpu_flops_val;
  const memory_bandwidth_bytes: number = memory_bandwidth * 1024 ** 3;
  const memory_time: number =
    (num_params * bytes_per_param) / memory_bandwidth_bytes;
  const inference_time: number =
    Math.max(compute_time, memory_time) * overhead_factor;
  return inference_time;
}

export function estimateTrainingCost(
  inference_gflops_per_sequence: number,
  gpu_tflops: number,
  num_params: number,
  memory_bandwidth: number,
  overhead_factor: number,
  gpu_hourly_cost: number,
  num_epochs: number,
  dataset_size: number,
  model_precision_bits: number
): number {
  const training_gflops_per_sequence = inference_gflops_per_sequence * 3;
  const model_precision_bytes = model_precision_bits / 8;
  const gradient_precision_bytes = model_precision_bits / 8;
  const optimizer_states_bytes = 8;
  const bytes_per_param_training_effective =
    model_precision_bytes + gradient_precision_bytes + optimizer_states_bytes;

  const time_per_item: number = estimateInferenceTime(
    training_gflops_per_sequence,
    gpu_tflops,
    num_params,
    memory_bandwidth,
    overhead_factor,
    bytes_per_param_training_effective
  );

  const total_time: number = time_per_item * dataset_size * num_epochs;
  const total_time_hours: number = total_time / 3600;
  const total_cost: number = total_time_hours * gpu_hourly_cost;
  return total_cost;
}

export function recommendGPU(memoryInGB: number, flops: number): string {
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
