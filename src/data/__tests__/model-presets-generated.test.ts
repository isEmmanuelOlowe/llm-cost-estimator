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
      ]),
    );
  });

  it('provides engine support metadata for every preset', () => {
    expect(
      modelPresets.every((preset) => preset.engineSupport.length > 0),
    ).toBe(true);
  });
});
