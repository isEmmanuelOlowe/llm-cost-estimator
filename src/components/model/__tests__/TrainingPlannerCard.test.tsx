import { fireEvent, render, screen } from '@testing-library/react';

import TrainingPlannerCard from '../TrainingPlannerCard';

const settings = {
  method: 'qlora' as const,
  loraRank: 16,
  targetCoverage: 'attention-qv' as const,
  sequenceLength: 2048,
  datasetTokens: 100_000_000,
  epochs: 1,
  globalBatchSize: 32,
  microBatchSize: 1,
  deviceCount: 1,
  distribution: 'replicated' as const,
  optimizer: 'adamw' as const,
  gradientCheckpointing: true,
  efficiency: 0.3,
  computeFormat: 'bf16' as const,
  optimizerPrecisionBits: 32 as const,
};

const memory = {
  method: 'qlora' as const,
  trainableParameterCount: 20_000_000,
  trainablePercent: 0.16,
  baseWeightBits: 4.5,
  baseWeightsGB: 6.3,
  adapterWeightsGB: 0.04,
  gradientsGB: 0.04,
  optimizerGB: 0.22,
  activationsGB: 3,
  persistentPerDeviceGB: 6.6,
  baseWeightsPerDeviceGB: 6.3,
  adapterWeightsPerDeviceGB: 0.04,
  gradientsPerDeviceGB: 0.04,
  optimizerPerDeviceGB: 0.22,
  overheadGB: 1.4,
  perDeviceGB: 11,
  aggregateGB: 11,
  deviceCount: 1,
  distribution: 'replicated' as const,
};

const run = {
  totalTrainingTokens: 100_000_000,
  flopsPerToken: 54_000_000_000,
  tokensPerSecond: 500,
  durationHours: 55,
  scalingEfficiency: 1,
  clusterHourlyRate: 0.99,
  totalCost: 54.45,
};

const gpu = {
  name: 'GPU A',
  memory_gb: 24,
  per_device_memory_gb: 24,
  device_count: 1,
  fp32_tflops: 40,
};

describe('TrainingPlannerCard', () => {
  it('keeps primary planning choices visible and updates the method', () => {
    const onChange = jest.fn();
    render(
      <TrainingPlannerCard
        settings={settings}
        onChange={onChange}
        hardware={[gpu]}
        selectedGpu={gpu}
        onSelectedGpuChange={jest.fn()}
        memory={memory}
        run={run}
        customHourlyRate=''
        onCustomHourlyRateChange={jest.fn()}
        recommendations={[]}
      />,
    );

    expect(screen.getByText('Plan the complete run')).toBeInTheDocument();
    expect(screen.getByText('VRAM / GPU')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /^LoRA/ }));
    expect(onChange).toHaveBeenCalledWith({ method: 'lora' });
    fireEvent.change(
      screen.getByRole('combobox', { name: 'QLoRA compute format' }),
      { target: { value: 'fp32' } },
    );
    expect(onChange).toHaveBeenCalledWith({ computeFormat: 'fp32' });
    expect(screen.getByText(/KV cache:/)).toBeInTheDocument();
  });

  it('shows a recommendation independent of selected hardware', () => {
    render(
      <TrainingPlannerCard
        settings={settings}
        onChange={jest.fn()}
        hardware={[gpu]}
        selectedGpu={gpu}
        onSelectedGpuChange={jest.fn()}
        memory={memory}
        run={run}
        customHourlyRate=''
        onCustomHourlyRateChange={jest.fn()}
        recommendations={[
          {
            gpuName: 'GPU B',
            provider: 'Provider',
            offeringName: 'Cloud GPU B',
            deviceCount: 2,
            memoryPerGpuGB: 24,
            memoryHeadroomGB: 13,
            memory,
            run,
            durationHours: 55,
            totalCost: 54.45,
            hourlyRatePerDevice: 0.5,
            pricingSourceUrl: 'https://example.com',
            sourceCheckedAt: '2026-08-24',
            distribution: 'fully-sharded',
          },
        ]}
      />,
    );

    expect(screen.getByText(/independent of selection/i)).toBeInTheDocument();
    expect(screen.getByText('2× GPU B')).toBeInTheDocument();
  });
});
