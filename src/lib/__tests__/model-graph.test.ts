import { buildModelGraph } from '../model-graph';

const input = {
  modelType: 'gemma4',
  modality: 'multimodal',
  hiddenSize: 3840,
  numLayers: 48,
  numAttentionHeads: 16,
  numKeyValueHeads: 8,
  headDim: 256,
  intermediateSize: 15360,
  vocabSize: 262144,
};

describe('model graph layout', () => {
  it('groups the transformer internals into a compact repeated block', () => {
    const graph = buildModelGraph(input, 'block');
    const byId = new Map(graph.nodes.map((node) => [node.id, node]));

    expect(byId.get('input')?.y).toBeLessThan(byId.get('embedding')?.y ?? 0);
    expect(byId.get('qkv')?.y).toBe(byId.get('attention-norm')?.y);
    expect(byId.get('mlp-norm')?.y).toBeGreaterThan(
      byId.get('attention-residual')?.y ?? 0,
    );
    expect(byId.get('lm-head')?.y).toBeGreaterThan(
      byId.get('mlp-residual')?.y ?? 0,
    );
    expect(graph.edges).toContainEqual(
      expect.objectContaining({
        from: 'token-fusion',
        to: 'attention-norm',
        kind: 'flow',
      }),
    );
    expect(byId.has('repeat')).toBe(false);
    expect(graph.blockBounds?.height).toBeLessThanOrEqual(520);
    expect(graph.height).toBeLessThan(1_300);
  });

  it('connects residual and data-flow edges separately', () => {
    const graph = buildModelGraph(input, 'block');

    expect(graph.edges.some((edge) => edge.kind === 'flow')).toBe(true);
    expect(graph.edges.some((edge) => edge.kind === 'residual')).toBe(true);
    expect(graph.edges).toContainEqual(
      expect.objectContaining({
        from: 'token-fusion',
        to: 'attention-residual',
        kind: 'residual',
      }),
    );
    expect(graph.blockBounds?.height).toBeGreaterThan(350);
  });

  it('renders multimodal branches and converges them before the decoder block', () => {
    const graph = buildModelGraph(
      {
        ...input,
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
      },
      'block',
    );
    const ids = new Set(graph.nodes.map((node) => node.id));

    expect(ids.has('image-input')).toBe(true);
    expect(ids.has('vision-patch-embed')).toBe(true);
    expect(ids.has('audio-projector')).toBe(true);
    expect(graph.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          from: 'vision-projector',
          to: 'token-fusion',
        }),
        expect.objectContaining({
          from: 'audio-projector',
          to: 'token-fusion',
        }),
        expect.objectContaining({
          from: 'token-fusion',
          to: 'attention-norm',
        }),
      ]),
    );
  });
});
