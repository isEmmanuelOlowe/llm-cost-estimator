import { render, screen } from '@testing-library/react';

import VramUsageBar from '../VramUsageBar';

describe('VramUsageBar', () => {
  it('shows segment percentages and an aggregate overflow message', () => {
    render(
      <VramUsageBar
        capacityGB={32}
        totalGB={85.94}
        segments={[
          { label: 'Model weights', valueGB: 22.28, color: 'bg-sky-500' },
          { label: 'KV cache', valueGB: 48, color: 'bg-violet-500' },
          { label: 'Activations', valueGB: 4.46, color: 'bg-amber-500' },
          {
            label: 'Framework overhead',
            valueGB: 11.21,
            color: 'bg-slate-400',
          },
        ]}
        fits={false}
        requiredDevices={1}
        deviceCount={1}
      />,
    );

    expect(screen.getByText('Overflow')).toBeInTheDocument();
    expect(
      screen.getByText(/85\.94 GB required exceeds aggregate capacity by/),
    ).toBeInTheDocument();
    expect(screen.getByText(/150%/)).toBeInTheDocument();
  });

  it('explains headroom for a fitting topology', () => {
    render(
      <VramUsageBar
        capacityGB={96}
        totalGB={64}
        segments={[
          { label: 'Model weights', valueGB: 64, color: 'bg-sky-500' },
        ]}
        fits
        requiredDevices={1}
        deviceCount={1}
      />,
    );

    expect(screen.getByText('Fits')).toBeInTheDocument();
    expect(screen.getByText(/32 GB headroom/)).toBeInTheDocument();
  });
});
