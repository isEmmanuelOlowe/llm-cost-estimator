import {
  buildArchitectureFlow,
  buildArchitectureOverview,
} from '../model-architecture';

const baseInput = {
  modelType: 'llama',
  architectures: ['LlamaForCausalLM'],
  hiddenSize: 4096,
  numLayers: 32,
  numAttentionHeads: 32,
  numKeyValueHeads: 8,
  headDim: 128,
  intermediateSize: 11008,
  vocabSize: 32000,
  sourceDirectoryUrl:
    'https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama',
  sourceFiles: [
    {
      name: 'modeling_llama.py',
      url: 'https://github.com/huggingface/transformers/blob/main/modeling_llama.py',
    },
  ],
};

describe('architecture flow model', () => {
  it('exposes GQA shapes and implementation links', () => {
    const flow = buildArchitectureFlow(baseInput);
    const qkv = flow.find((node) => node.id === 'qkv');

    expect(qkv?.shape).toContain('Q [B,S,32,128]');
    expect(qkv?.shape).toContain('K/V [B,S,8,128]');
    expect(qkv?.sourceFile).toBe('modeling_llama.py');
    expect(qkv?.sourceUrl).toContain('modeling_llama.py');
  });

  it('adds router and expert stages for MoE architectures', () => {
    const flow = buildArchitectureFlow({
      ...baseInput,
      modelType: 'qwen3_moe',
      numExperts: 128,
      numExpertsPerToken: 8,
    });

    expect(flow.map((node) => node.id)).toEqual(
      expect.arrayContaining(['router', 'experts']),
    );
    expect(flow.find((node) => node.id === 'router')?.detail).toContain(
      '8 of 128',
    );
  });

  it('keeps attention and feed-forward sublayers in sequential block order', () => {
    const ids = buildArchitectureFlow(baseInput).map((node) => node.id);

    expect(ids.indexOf('attention-residual')).toBeLessThan(
      ids.indexOf('mlp-norm'),
    );
    expect(ids.indexOf('mlp-residual')).toBeLessThan(ids.indexOf('repeat'));
  });

  it('compresses the same structure into an overview', () => {
    expect(buildArchitectureOverview(baseInput).map((node) => node.id)).toEqual(
      ['input', 'embedding', 'repeat', 'final-norm', 'lm-head'],
    );
  });

  it('models typed image, video, audio, and fusion stages instead of a generic projector', () => {
    const flow = buildArchitectureFlow({
      ...baseInput,
      modelType: 'gemma4_unified',
      modality: 'multimodal',
      modalityArchitecture: {
        family: 'gemma4_unified',
        evidence: 'config',
        vision: {
          encoderFree: true,
          patchSize: 16,
          pooledPatchSize: 48,
          rawChannels: 3,
          embedDim: 3840,
          outputDim: 3840,
          softTokens: 280,
        },
        video: true,
        audio: {
          encoderFree: true,
          featureDim: 640,
          samplesPerToken: 640,
          outputDim: 3840,
        },
      },
    });
    const ids = flow.map((node) => node.id);

    expect(ids).toEqual(
      expect.arrayContaining([
        'image-input',
        'vision-patch-embed',
        'vision-position',
        'vision-projector',
        'video-input',
        'audio-input',
        'audio-projector',
        'token-fusion',
      ]),
    );
    expect(
      flow.find((node) => node.id === 'vision-patch-embed')?.detail,
    ).toContain('LayerNorm');
  });
});
