import { render, screen } from '@testing-library/react';

import DeploymentDecisionPath from '../DeploymentDecisionPath';

describe('DeploymentDecisionPath', () => {
  it('puts the model, memory requirement, and hardware decision in one visual path', () => {
    render(
      <DeploymentDecisionPath
        modelLabel='Gemma 4 12B'
        parameterBillions={12}
        layers={48}
        contextLength={131072}
        weightFormat='BF16'
        totalMemoryGB={31.17}
        memorySegments={[
          { label: 'Weights', valueGB: 22.28, color: 'bg-sky-500' },
          { label: 'KV cache', valueGB: 0.38, color: 'bg-violet-500' },
        ]}
        gpuName='RTX 5090'
        gpuCapacityGB={32}
        fits
        headroomGB={0.83}
        tokensPerSecond={20.5}
        projectedCost={4.25}
        cloudCostLabel='RunPod · 1h'
      />,
    );

    expect(
      screen.getByText('What this model needs—at a glance'),
    ).toBeInTheDocument();
    expect(screen.getByText('31.17 GB')).toBeInTheDocument();
    expect(screen.getByText('0.83 GB headroom')).toBeInTheDocument();
    expect(screen.getByText('Cloud cost')).toBeInTheDocument();
    expect(screen.queryByText('Cloud run')).not.toBeInTheDocument();
    expect(screen.getByText('RunPod · 1h')).toBeInTheDocument();
    expect(
      screen.getByRole('img', {
        name: /requires 31.17 GB and fits on RTX 5090/i,
      }),
    ).toBeInTheDocument();
  });

  it('does not invent a cloud price when no verified GPU offering exists', () => {
    render(
      <DeploymentDecisionPath
        modelLabel='Model'
        parameterBillions={7}
        layers={32}
        contextLength={8192}
        weightFormat='BF16'
        totalMemoryGB={14}
        memorySegments={[]}
        gpuName='Local-only GPU'
        gpuCapacityGB={16}
        fits
        headroomGB={2}
      />,
    );

    expect(screen.getByText('Cloud cost')).toBeInTheDocument();
    expect(screen.getAllByText('N/A').length).toBeGreaterThan(0);
    expect(screen.getByText('No verified rate')).toBeInTheDocument();
  });
});
