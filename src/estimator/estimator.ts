export function calculateMemory(numParams: number, bytes: number): number {
  const bytesPerParam = bytes / 8; // Assuming each parameter is a 32-bit float.
  const paramsInGB = (numParams * bytesPerParam) / 1024 ** 3; // Convert to GB.
  return paramsInGB;
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