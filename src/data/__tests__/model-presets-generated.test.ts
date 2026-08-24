import modelPresets from '@/data/model-presets.generated.json';

describe('generated model presets', () => {
  it('contains the required modern target families', () => {
    const ids = modelPresets.map((preset) => preset.id);

    expect(ids).toEqual(
      expect.arrayContaining([
        'Qwen/Qwen3.5-27B',
        'Qwen/Qwen3.5-35B-A3B',
        'Qwen/Qwen3-Coder-Next',
        'openai/gpt-oss-20b',
        'zai-org/GLM-4.7-Flash',
        'moonshotai/Kimi-K2.5',
        'google/gemma-4-12B',
        'google/gemma-4-26B-A4B',
        'google/gemma-3-4b-it',
        'Qwen/Qwen3.8-27B',
        'Qwen/Qwen3.8-2.4T-A95B',
        'Qwen/Qwen3-32B',
        'deepseek-ai/DeepSeek-V3.2',
        'openai/gpt-oss-120b',
        'meta-models/Muse-Glimmer-30B',
        'thinkingmachines/Inkling',
        'deepseek-ai/DeepSeek-V4-Pro',
        'deepseek-ai/DeepSeek-V4-Flash',
        'thinkingmachines/Inkling-Small',
        'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16',
        'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
        'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16',
        'moonshotai/Kimi-K3',
        'moonshotai/Kimi-K2.6',
        'moonshotai/Kimi-K2.7-Code',
        'MiniMaxAI/MiniMax-M3',
        'zai-org/GLM-5.2',
      ]),
    );
  });

  it('provides engine support metadata for every preset', () => {
    expect(
      modelPresets.every((preset) => preset.engineSupport.length > 0),
    ).toBe(true);
  });

  it('records immutable source and weight provenance for every preset', () => {
    expect(
      modelPresets.every(
        (preset) =>
          preset.revision &&
          preset.sourceCheckedAt &&
          preset.sourceUrls.api &&
          preset.parameterCount > 0,
      ),
    ).toBe(true);
  });

  it('puts the current default stack first', () => {
    expect(modelPresets.slice(0, 5).map((preset) => preset.id)).toEqual([
      'google/gemma-4-12B',
      'Qwen/Qwen3.8-27B',
      'zai-org/GLM-5',
      'moonshotai/Kimi-K3',
      'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
    ]);
  });

  it('keeps non-featured presets contiguous by family', () => {
    const families = modelPresets.slice(5).map((preset) => preset.family);
    const seen = new Set<string>();
    let previous: string | undefined;

    families.forEach((family) => {
      if (family !== previous) {
        expect(seen.has(family)).toBe(false);
        seen.add(family);
        previous = family;
      }
    });
  });

  it('keeps typed multimodal evidence for the unified Gemma default', () => {
    const preset = modelPresets.find(
      (entry) => entry.id === 'google/gemma-4-12B',
    );

    expect(preset?.modalityArchitecture).toMatchObject({
      family: 'gemma4_unified',
      evidence: 'config',
      vision: { encoderFree: true, pooledPatchSize: 48 },
      video: true,
      audio: { encoderFree: true, samplesPerToken: 640 },
    });
  });

  it('keeps architecture-aware cache metadata for optimized families', () => {
    const deepseek = modelPresets.find(
      (entry) => entry.id === 'deepseek-ai/DeepSeek-V4-Pro',
    );
    const inkling = modelPresets.find(
      (entry) => entry.id === 'thinkingmachines/Inkling-Small',
    );
    const nemotron = modelPresets.find(
      (entry) => entry.id === 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    );
    const kimiK3 = modelPresets.find(
      (entry) => entry.id === 'moonshotai/Kimi-K3',
    );
    const kimiK26 = modelPresets.find(
      (entry) => entry.id === 'moonshotai/Kimi-K2.6',
    );

    expect(deepseek?.kvCacheArchitecture).toMatchObject({
      mode: 'deepseek-v4',
      keyValueShared: true,
      compressedSparseLayers: 30,
      heavilyCompressedLayers: 31,
    });
    expect(inkling?.kvCacheArchitecture).toMatchObject({
      mode: 'hybrid-sliding-window',
      localAttentionLayers: 35,
      fullAttentionLayers: 7,
      slidingWindow: 512,
    });
    expect(nemotron?.kvCacheArchitecture).toMatchObject({
      mode: 'hybrid-state-space',
      fullAttentionLayers: 6,
      recurrentStateLayers: 23,
    });
    expect(kimiK3?.kvCacheArchitecture).toMatchObject({
      mode: 'hybrid-latent-state',
      fullAttentionLayers: 24,
      latentKvRank: 512,
      latentRopeDim: 64,
    });
    expect(kimiK26?.kvCacheArchitecture).toMatchObject({
      mode: 'latent-attention',
      fullAttentionLayers: 61,
      latentKvRank: 512,
      latentRopeDim: 64,
    });
  });
});
