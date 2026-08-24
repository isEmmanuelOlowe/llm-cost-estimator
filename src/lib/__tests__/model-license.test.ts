import { resolveModelLicense } from '../model-license';

describe('model license resolver', () => {
  it('names the Nemotron OpenMDW license instead of other', () => {
    const info = resolveModelLicense({
      modelId: 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
      license: 'other',
      modelUrl:
        'https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
    });

    expect(info.id).toBe('openmdw-1.1');
    expect(info.label).toBe('OpenMDW 1.1');
    expect(info.custom).toBe(false);
  });

  it('covers common permissive and community licenses', () => {
    expect(
      resolveModelLicense({
        modelId: 'model',
        license: 'apache-2.0',
        modelUrl: 'https://huggingface.co/model',
      }).label,
    ).toBe('Apache 2.0');
    expect(
      resolveModelLicense({
        modelId: 'model',
        license: 'llama3.1',
        modelUrl: 'https://huggingface.co/model',
      }).label,
    ).toBe('Llama Community');
  });

  it('names Kimi model-specific terms', () => {
    expect(
      resolveModelLicense({
        modelId: 'moonshotai/Kimi-K3',
        license: 'other',
        modelUrl: 'https://huggingface.co/moonshotai/Kimi-K3',
      }).label,
    ).toBe('Kimi K3 License');
  });
});
