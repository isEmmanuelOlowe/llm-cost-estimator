import { groupGpus } from '../gpu-groups';

describe('GPU picker grouping', () => {
  it('groups by vendor and architecture and keeps current hardware first', () => {
    const groups = groupGpus([
      {
        name: 'Legacy B',
        vendor: 'NVIDIA',
        architecture: 'Blackwell',
        status: 'legacy',
      },
      {
        name: 'Current B',
        vendor: 'NVIDIA',
        architecture: 'Blackwell',
        status: 'current',
      },
      {
        name: 'MI300X',
        vendor: 'AMD',
        architecture: 'CDNA 3',
        status: 'current',
      },
    ]);

    expect(groups.map((group) => group.label)).toEqual([
      'NVIDIA · Blackwell',
      'AMD · CDNA 3',
    ]);
    expect(groups[0].gpus.map((gpu) => gpu.name)).toEqual([
      'Current B',
      'Legacy B',
    ]);
  });

  it('uses an explicit fallback group for incomplete catalog records', () => {
    const [group] = groupGpus([
      { name: 'Unknown accelerator', vendor: null, architecture: null },
    ]);

    expect(group.label).toBe('Other vendors · Other architectures');
  });

  it('keeps Apple unified-memory devices in one group while retaining chip names', () => {
    const groups = groupGpus([
      {
        name: 'Mac mini M4',
        vendor: 'Apple',
        architecture: 'Apple M4',
      },
      {
        name: 'MacBook Pro M5 Max',
        vendor: 'Apple',
        architecture: 'Apple M5 Max',
      },
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0].label).toBe('Apple · Unified memory');
    expect(groups[0].gpus.map((gpu) => gpu.architecture)).toEqual([
      'Apple M4',
      'Apple M5 Max',
    ]);
  });
});
