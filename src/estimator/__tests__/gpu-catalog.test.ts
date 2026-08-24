import gpus from '../gpus.json';

describe('hardware catalog', () => {
  it('contains source-backed multi-vendor hardware with topology fields', () => {
    expect(new Set(gpus.map((gpu) => gpu.vendor))).toEqual(
      new Set(['NVIDIA', 'AMD', 'Intel', 'Apple']),
    );
    expect(
      gpus.every(
        (gpu) =>
          gpu.memory_gb > 0 &&
          gpu.device_count >= 1 &&
          gpu.per_device_memory_gb > 0 &&
          gpu.source_url &&
          gpu.source_checked_at,
      ),
    ).toBe(true);
  });

  it('keeps aggregate system memory distinct from per-device memory', () => {
    const dgxSpark = gpus.find((gpu) => gpu.name === 'NVIDIA DGX Spark');
    const dgxB200 = gpus.find(
      (gpu) => gpu.name === 'NVIDIA DGX B200 (8x B200)',
    );

    expect(dgxSpark).toMatchObject({
      memory_gb: 128,
      per_device_memory_gb: 128,
      memory_model: 'unified',
    });
    expect(dgxB200).toMatchObject({
      memory_gb: 1440,
      per_device_memory_gb: 180,
      device_count: 8,
    });
    expect(gpus.find((gpu) => gpu.name === 'Intel Arc Pro B70')).toMatchObject({
      memory_gb: 32,
      memory_bandwidth_gb_s: 608,
      ai_tops: 367,
      vendor: 'Intel',
    });
    expect(
      gpus.find((gpu) => gpu.name === 'Apple Mac Studio M3 Ultra (256GB)'),
    ).toMatchObject({
      memory_gb: 256,
      memory_bandwidth_gb_s: 819,
      memory_model: 'unified',
      vendor: 'Apple',
    });
    expect(
      gpus.find((gpu) => gpu.name === 'Apple MacBook Pro M5 Max (128GB)'),
    ).toMatchObject({
      memory_gb: 128,
      memory_bandwidth_gb_s: 614,
      memory_model: 'unified',
      vendor: 'Apple',
    });
    expect(
      gpus.find((gpu) => gpu.name === 'Apple MacBook Pro M4 Max (128GB)'),
    ).toMatchObject({
      memory_gb: 128,
      memory_bandwidth_gb_s: 546,
      memory_model: 'unified',
      vendor: 'Apple',
    });
  });

  it('covers the current Blackwell and Blackwell Ultra system families', () => {
    const names = new Set(gpus.map((gpu) => gpu.name));
    expect([...names]).toEqual(
      expect.arrayContaining([
        'NVIDIA B200 SXM 180GB',
        'NVIDIA B100 SXM 192GB',
        'NVIDIA B300 SXM 288GB',
        'NVIDIA HGX B300 (8x B300)',
        'NVIDIA DGX B300 (8x B300)',
        'NVIDIA GB200 NVL72 (72x B200)',
        'NVIDIA GB300 NVL72 (72x B300)',
        'NVIDIA DGX Station GB300',
        'NVIDIA GeForce RTX 5080',
        'NVIDIA GeForce RTX 5070 Ti',
        'NVIDIA GeForce RTX 5070',
        'Intel Arc Pro B70',
        'NVIDIA GeForce RTX 5060 Ti 16GB',
        'NVIDIA GeForce RTX 5060',
        'NVIDIA GeForce RTX 5050',
      ]),
    );
  });
});
