import cloudInstances from '../cloud-instances.json';
import gpus from '../gpus.json';

describe('cloud price catalog', () => {
  it('labels reference rates with provider verification links', () => {
    expect(
      cloudInstances.every(
        (instance) =>
          instance.hourly_rate > 0 &&
          instance.pricing_source_url &&
          instance.pricing_basis &&
          instance.source_checked_at &&
          instance.billing_increment === 'per second',
      ),
    ).toBe(true);
  });

  it('maps every displayed rate to an exact hardware catalog entry', () => {
    const gpuByName = new Map(gpus.map((gpu) => [gpu.name, gpu]));

    for (const instance of cloudInstances) {
      expect(instance.gpu_catalog_names.length).toBeGreaterThan(0);
      for (const gpuName of instance.gpu_catalog_names) {
        const gpu = gpuByName.get(gpuName);
        expect(gpu).toBeDefined();
        expect(gpu?.memory_gb).toBe(instance.vram_per_gpu_gb * instance.gpus);
      }
    }
  });

  it('uses the current verified RTX 5090 Secure Cloud rate', () => {
    const instance = cloudInstances.find((entry) =>
      entry.gpu_catalog_names.includes('NVIDIA GeForce RTX 5090'),
    );

    expect(instance).toMatchObject({
      provider: 'RunPod',
      hourly_rate: 0.99,
      source_checked_at: '2026-08-24',
    });
  });
});
