import { TextDecoder as NodeTextDecoder } from 'node:util';

import {
  inspectHuggingFaceModel,
  normalizeModelId,
  parseModelConfig,
  parseNumericValue,
  parseSafetensorsFileHeader,
} from '../model-metadata';

if (!globalThis.TextDecoder) {
  globalThis.TextDecoder = NodeTextDecoder as typeof TextDecoder;
}

describe('model metadata normalization', () => {
  it('normalizes model IDs and Hugging Face URLs', () => {
    expect(normalizeModelId('meta-llama/Llama-2-7b-hf')).toBe(
      'meta-llama/Llama-2-7b-hf',
    );
    expect(
      normalizeModelId('https://huggingface.co/Qwen/Qwen2.5-7B/tree/main'),
    ).toBe('Qwen/Qwen2.5-7B');
  });

  it('parses human-readable parameter counts', () => {
    expect(parseNumericValue('7.2B')).toBe(7.2e9);
    expect(parseNumericValue('1.5 trillion parameters')).toBe(1.5e12);
    expect(parseNumericValue('not a number')).toBeUndefined();
  });

  it('normalizes architecture aliases and detects GQA/MoE metadata', () => {
    const parsed = parseModelConfig({
      model_type: 'llama',
      architectures: ['LlamaForCausalLM'],
      hidden_size: 4096,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      intermediate_size: 11008,
      vocab_size: 32000,
      max_position_embeddings: 131072,
      hidden_act: 'silu',
      torch_dtype: 'bfloat16',
      tie_word_embeddings: false,
    });

    expect(parsed.architecture.modelType).toBe('llama');
    expect(parsed.architecture.numKeyValueHeads).toBe(8);
    expect(parsed.architecture.contextLength).toBe(131072);
    expect(parsed.architecture.gatedMlp).toBe(true);
    expect(parsed.architecture.dtypeBits).toBe(16);
    expect(parsed.parameterSource).toBe('estimated-from-architecture');
    expect(parsed.warnings).toEqual(
      expect.arrayContaining([expect.stringContaining('Grouped-query')]),
    );
  });

  it('preserves typed multimodal sub-config evidence for the architecture graph', () => {
    const parsed = parseModelConfig({
      model_type: 'gemma4_unified',
      text_config: {
        hidden_size: 3840,
        num_hidden_layers: 48,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        intermediate_size: 15360,
        vocab_size: 262144,
      },
      vision_config: {
        patch_size: 16,
        pooling_kernel_size: 3,
        mm_embed_dim: 3840,
        output_proj_dims: 3840,
        num_soft_tokens: 280,
      },
      audio_config: {
        audio_embed_dim: 640,
        audio_samples_per_token: 640,
      },
      video_token_id: 258884,
    });

    expect(parsed.architecture.modality).toBe('multimodal');
    expect(parsed.architecture.modalityArchitecture).toMatchObject({
      family: 'gemma4_unified',
      evidence: 'config',
      vision: {
        encoderFree: true,
        patchSize: 16,
        pooledPatchSize: 48,
        softTokens: 280,
      },
      video: true,
      audio: {
        encoderFree: true,
        featureDim: 640,
        samplesPerToken: 640,
      },
    });
  });

  it('detects DeepSeek V4 compressed and shared-KV cache layers', () => {
    const parsed = parseModelConfig({
      model_type: 'deepseek_v4',
      num_hidden_layers: 6,
      num_attention_heads: 64,
      num_key_value_heads: 1,
      head_dim: 512,
      compress_ratios: [128, 128, 4, 4, 128, 4],
      sliding_window: 128,
      index_head_dim: 128,
    });

    expect(parsed.architecture.kvCacheArchitecture).toMatchObject({
      mode: 'deepseek-v4',
      keyValueShared: true,
      localAttentionLayers: 6,
      slidingWindow: 128,
      compressedSparseLayers: 3,
      heavilyCompressedLayers: 3,
      indexHeadDim: 128,
    });
  });

  it('detects local/global and state-space cache schedules', () => {
    const inkling = parseModelConfig({
      model_type: 'inkling_mm_model',
      text_config: {
        num_hidden_layers: 4,
        local_layer_ids: [0, 1, 3],
        sliding_window_size: 512,
      },
    });
    const nemotron = parseModelConfig({
      model_type: 'nemotron_h',
      num_hidden_layers: 4,
      hybrid_override_pattern: 'M*M*',
      mamba_num_heads: 8,
      mamba_head_dim: 64,
      ssm_state_size: 16,
      n_groups: 2,
      conv_kernel: 4,
      mamba_ssm_cache_dtype: 'float32',
    });

    expect(inkling.architecture.kvCacheArchitecture).toMatchObject({
      mode: 'hybrid-sliding-window',
      localAttentionLayers: 3,
      fullAttentionLayers: 1,
      slidingWindow: 512,
    });
    expect(nemotron.architecture.kvCacheArchitecture).toMatchObject({
      mode: 'hybrid-state-space',
      fullAttentionLayers: 2,
      recurrentStateLayers: 2,
      noAttentionLayers: 2,
    });
    expect(
      nemotron.architecture.kvCacheArchitecture?.recurrentStateBytesPerLayer,
    ).toBeGreaterThan(0);
  });

  it('does not present a false dense parameter decomposition for MoE configs', () => {
    const parsed = parseModelConfig({
      hidden_size: 2048,
      num_hidden_layers: 48,
      num_attention_heads: 16,
      vocab_size: 151936,
      intermediate_size: 5120,
      moe_intermediate_size: 512,
      num_experts: 512,
      num_experts_per_tok: 10,
    });

    expect(parsed.parameterBreakdown).toBeUndefined();
    expect(parsed.parameterCount).toBeUndefined();
    expect(parsed.warnings).toEqual(
      expect.arrayContaining([
        expect.stringContaining('MoE parameter composition'),
      ]),
    );
  });

  it('prefers explicit config/API counts over heuristic architecture math', () => {
    const parsed = parseModelConfig(
      {
        d_model: 2048,
        n_layer: 24,
        n_head: 16,
        n_vocab: 50000,
      },
      [{ num_parameters: '3.1B' }],
    );

    expect(parsed.parameterCount).toBe(3.1e9);
    expect(parsed.parameterSource).toBe('huggingface-config');
  });

  it('reads quantization metadata without treating it as executable code', () => {
    const parsed = parseModelConfig({
      hidden_size: 1024,
      num_hidden_layers: 12,
      num_attention_heads: 16,
      vocab_size: 32000,
      quantization_config: {
        quant_method: 'gptq',
        bits: 4,
        group_size: 128,
      },
    });

    expect(parsed.quantization).toMatchObject({
      method: 'gptq',
      bits: 4,
      groupSize: 128,
    });
    expect(parsed.architecture.dtypeBits).toBe(4);
  });

  it('counts tensors from a bounded safetensors header range', async () => {
    const header = JSON.stringify({
      embedding: {
        dtype: 'BF16',
        shape: [4, 8],
        data_offsets: [0, 64],
      },
      linear: {
        dtype: 'F32',
        shape: [8, 2],
        data_offsets: [64, 128],
      },
      __metadata__: { format: 'pt' },
    });
    const headerBytes = Buffer.from(header, 'utf8');
    const bytes = new Uint8Array(8 + headerBytes.length);
    new DataView(bytes.buffer).setBigUint64(
      0,
      BigInt(headerBytes.length),
      true,
    );
    bytes.set(headerBytes, 8);
    const originalFetch = globalThis.fetch;

    globalThis.fetch = jest.fn(async (_input, init) => {
      const headers = init?.headers as Record<string, string> | undefined;
      const range = headers?.Range ?? headers?.range ?? '';
      const match = range.match(/bytes=(\d+)-(\d+)/);
      const start = match ? Number(match[1]) : 0;
      const end = match ? Number(match[2]) : bytes.length - 1;
      const body = bytes.slice(start, end + 1);
      return {
        ok: true,
        status: 206,
        headers: {
          get: (name: string) =>
            name.toLowerCase() === 'content-length'
              ? String(body.byteLength)
              : `bytes ${start}-${end}/${bytes.length}`,
        },
        arrayBuffer: async () => body.buffer,
      } as unknown as Response;
    }) as typeof fetch;

    try {
      await expect(
        parseSafetensorsFileHeader('https://example.test/model.safetensors'),
      ).resolves.toEqual({
        parameterCount: 48,
        parameterCountByDtype: { BF16: 32, F32: 16 },
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('builds a pinned inspection from Hub metadata and Transformers source', async () => {
    const originalFetch = globalThis.fetch;
    const api = {
      id: 'org/model',
      sha: 'abcdef1234567890abcdef1234567890abcdef12',
      lastModified: '2026-08-22T00:00:00.000Z',
      author: 'org',
      pipeline_tag: 'text-generation',
      library_name: 'transformers',
      model_type: 'llama',
      tags: ['transformers', 'license:apache-2.0'],
      siblings: [
        { rfilename: 'config.json' },
        { rfilename: 'README.md' },
        { rfilename: 'model.safetensors' },
      ],
      safetensors: { total: 1234567890, parameters: { BF16: 1234567890 } },
    };
    const config = {
      model_type: 'llama',
      architectures: ['LlamaForCausalLM'],
      hidden_size: 4096,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      intermediate_size: 11008,
      vocab_size: 32000,
      max_position_embeddings: 8192,
      hidden_act: 'silu',
      transformers_version: '4.50.0',
    };

    globalThis.fetch = jest.fn(async (input) => {
      const url = String(input);
      if (url.includes('/api/models/org/model')) {
        return {
          ok: true,
          status: 200,
          json: async () => api,
        } as unknown as Response;
      }
      if (
        url.includes(
          '/resolve/abcdef1234567890abcdef1234567890abcdef12/config.json',
        )
      ) {
        return {
          ok: true,
          status: 200,
          json: async () => config,
        } as unknown as Response;
      }
      if (url.endsWith('/README.md')) {
        return {
          ok: true,
          status: 200,
          headers: { get: () => '20' },
          text: async () => '# Model card',
        } as unknown as Response;
      }
      if (url.includes('api.github.com/repos')) {
        return {
          ok: true,
          status: 200,
          json: async () => [
            {
              type: 'file',
              name: 'modeling_llama.py',
              html_url:
                'https://github.com/huggingface/transformers/blob/main/modeling_llama.py',
              download_url:
                'https://raw.githubusercontent.com/modeling_llama.py',
            },
          ],
        } as unknown as Response;
      }
      return {
        ok: true,
        status: 200,
        headers: { get: () => '10' },
        text: async () =>
          'class LlamaModel: pass\npast_key_values.update(key_states, value_states)',
      } as unknown as Response;
    }) as typeof fetch;

    try {
      const inspection = await inspectHuggingFaceModel('org/model');
      expect(inspection).toMatchObject({
        id: 'org/model',
        revision: 'abcdef1234567890abcdef1234567890abcdef12',
        parameterCount: 1234567890,
        parameterSource: 'huggingface-safetensors',
        modelType: 'llama',
      });
      expect(inspection.architecture.numKeyValueHeads).toBe(8);
      expect(inspection.transformers?.preview?.name).toBe('modeling_llama.py');
      expect(inspection.transformers?.cacheAnalysis?.signals).toEqual(
        expect.arrayContaining([
          'source updates a Transformers Cache implementation',
        ]),
      );
      expect(inspection.evidence.map((item) => item.kind)).toEqual(
        expect.arrayContaining([
          'hub-api',
          'config',
          'weights',
          'transformers',
        ]),
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
