import { fuzzySearchModels } from '../model-search';

const models = [
  {
    id: 'moonshotai/Kimi-K2.7-Code',
    label: 'Kimi K2.7 Code',
    family: 'Kimi',
    summary: 'Coding model',
  },
  {
    id: 'google/gemma-4-12B',
    label: 'Gemma 4 12B Unified',
    family: 'Gemma 4',
    summary: 'Multimodal model',
  },
  {
    id: 'deepseek-ai/DeepSeek-V4-Flash',
    label: 'DeepSeek V4 Flash',
    family: 'DeepSeek V4',
    summary: 'Sparse model',
  },
];

describe('fuzzy model search', () => {
  it('matches IDs, labels, family names, and typo-like subsequences', () => {
    expect(fuzzySearchModels(models, 'kimi 2.7 code')[0].id).toBe(
      'moonshotai/Kimi-K2.7-Code',
    );
    expect(fuzzySearchModels(models, 'deep v4 flsh')[0].id).toBe(
      'deepseek-ai/DeepSeek-V4-Flash',
    );
  });

  it('returns the catalog prefix for an empty query', () => {
    expect(fuzzySearchModels(models, '', 2).map((model) => model.id)).toEqual([
      'moonshotai/Kimi-K2.7-Code',
      'google/gemma-4-12B',
    ]);
  });
});
