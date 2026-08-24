export interface ArchitectureSourceFile {
  name: string;
  url: string;
}

export interface ModalityArchitecture {
  family: string;
  evidence: string;
  vision?: {
    encoderFree?: boolean | null;
    patchSize?: number | null;
    pooledPatchSize?: number | null;
    rawChannels?: number | null;
    embedDim?: number | null;
    outputDim?: number | null;
    softTokens?: number | null;
  } | null;
  video?: boolean;
  audio?: {
    encoderFree?: boolean | null;
    featureDim?: number | null;
    samplesPerToken?: number | null;
    outputDim?: number | null;
  } | null;
}

export interface ArchitectureFlowInput {
  modelType?: string;
  architectures?: string[];
  hiddenSize: number;
  numLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  intermediateSize: number;
  expertIntermediateSize?: number;
  sharedExpertIntermediateSize?: number;
  numSharedExperts?: number;
  isEncoderDecoder?: boolean;
  modality?: string;
  vocabSize: number;
  numExperts?: number;
  numExpertsPerToken?: number;
  modalityArchitecture?: ModalityArchitecture;
  sourceDirectoryUrl?: string;
  sourceFiles?: ArchitectureSourceFile[];
}

export interface ArchitectureFlowNode {
  id: string;
  label: string;
  kind:
    | 'input'
    | 'normalization'
    | 'attention'
    | 'routing'
    | 'mlp'
    | 'output'
    | 'residual';
  detail: string;
  shape: string;
  sourceFile?: string;
  sourceUrl?: string;
}

function format(value: number): string {
  return Number.isFinite(value) && value > 0
    ? value.toLocaleString('en-US', { maximumFractionDigits: 0 })
    : 'unknown';
}

function pickSourceFile(
  files: ArchitectureSourceFile[] | undefined,
  prefix: string,
  fallbackDirectory?: string,
  modelType?: string,
): ArchitectureSourceFile | undefined {
  const file = files?.find((entry) => entry.name.startsWith(prefix));
  if (file) return file;
  if (!fallbackDirectory) return undefined;
  const aliases: Record<string, string> = {
    gemma3_text: 'gemma3',
    gemma4_unified: 'gemma4_unified',
    qwen3_5_moe_text: 'qwen3_5',
    minimax_m3_vl: 'minimax_m3_vl',
  };
  const directoryName =
    aliases[modelType?.toLowerCase() ?? ''] ?? modelType ?? 'model';
  const suffix = `${prefix}${directoryName}.py`;
  return {
    name: suffix,
    url: `${fallbackDirectory}/${suffix}`,
  };
}

export function buildArchitectureFlow(
  input: ArchitectureFlowInput,
): ArchitectureFlowNode[] {
  const modeling =
    input.sourceFiles?.find((entry) => entry.name.startsWith('modular_')) ??
    pickSourceFile(
      input.sourceFiles,
      'modeling_',
      input.sourceDirectoryUrl,
      input.modelType,
    );
  const configuration = pickSourceFile(
    input.sourceFiles,
    'configuration_',
    input.sourceDirectoryUrl,
    input.modelType,
  );
  const expertDetail =
    input.numExperts && input.numExperts > 1
      ? `${format(input.numExpertsPerToken ?? 1)} of ${format(input.numExperts)} experts per token; router dispatches tokens before expert MLPs.${input.numSharedExperts ? ` ${format(input.numSharedExperts)} shared expert path(s) also contribute.` : ''}`
      : 'Dense feed-forward projection shared by every token.';
  const mlpWidth =
    input.numExperts && input.numExperts > 1
      ? (input.expertIntermediateSize ?? input.intermediateSize)
      : input.intermediateSize;
  const attentionRatio =
    input.numAttentionHeads > 0
      ? `${Math.round((input.numKeyValueHeads / input.numAttentionHeads) * 100)}% of query-head count`
      : 'unknown KV ratio';
  const modality = input.modalityArchitecture;
  const isMultimodal = Boolean(input.modality && input.modality !== 'text');
  const visionPatchSize = modality?.vision?.pooledPatchSize ?? 0;
  const visionPatchDim =
    visionPatchSize > 0
      ? visionPatchSize * visionPatchSize * (modality?.vision?.rawChannels ?? 3)
      : 0;

  return [
    {
      id: 'input',
      label: isMultimodal
        ? 'Text tokens / modality placeholders'
        : 'Tokens / text inputs',
      kind: 'input',
      detail:
        'Token IDs reserve placeholder positions for media features before the language-model embedding and fusion steps.',
      shape: '[B, S] token IDs',
      sourceFile: configuration?.name,
      sourceUrl: configuration?.url,
    },
    ...(isMultimodal && modality?.vision
      ? [
          {
            id: 'image-input',
            label: 'Image pixels / merged patches',
            kind: 'input' as const,
            detail: modality.vision.encoderFree
              ? `Images are merged into ${format(visionPatchSize)}×${format(visionPatchSize)} patches before the encoder-free projection path.`
              : 'Image pixels are converted into patches and passed to the model-defined vision encoder.',
            shape:
              visionPatchDim > 0
                ? `[B, P, ${format(visionPatchDim)}] pixel patch values`
                : '[B, pixels] image values',
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
          ...(modality.vision.encoderFree
            ? [
                {
                  id: 'vision-patch-embed',
                  label: 'Image patch LN → Dense → LN',
                  kind: 'normalization' as const,
                  detail:
                    'The unified implementation has no vision tower: LayerNorm normalizes merged pixel patches, a Dense layer projects them, then a second LayerNorm stabilizes the patch states.',
                  shape:
                    visionPatchDim > 0
                      ? `[B, P, ${format(visionPatchDim)}] → [B, P, ${format(modality.vision.embedDim ?? input.hiddenSize)}]`
                      : `[B, pixels] → [B, P, H=${format(modality.vision.embedDim ?? input.hiddenSize)}]`,
                  sourceFile: modeling?.name,
                  sourceUrl: modeling?.url,
                },
                {
                  id: 'vision-position',
                  label: 'Factorized 2D position + LN',
                  kind: 'normalization' as const,
                  detail:
                    'Adds factorized two-axis positional embeddings for image patches and applies the positional LayerNorm before language-space projection.',
                  shape: `[B, P, H=${format(modality.vision.embedDim ?? input.hiddenSize)}] + position → [B, P, H]`,
                  sourceFile: modeling?.name,
                  sourceUrl: modeling?.url,
                },
              ]
            : [
                {
                  id: 'vision-encoder',
                  label: 'Vision encoder',
                  kind: 'attention' as const,
                  detail:
                    'Runs the model-defined image encoder and produces visual token features before language-space projection.',
                  shape:
                    visionPatchDim > 0
                      ? `[B, P, ${format(visionPatchDim)}] → [B, P, ${format(modality.vision.embedDim ?? input.hiddenSize)}]`
                      : `[B, pixels] → [B, P, H=${format(modality.vision.embedDim ?? input.hiddenSize)}]`,
                  sourceFile: modeling?.name,
                  sourceUrl: modeling?.url,
                },
              ]),
          {
            id: 'vision-projector',
            label: 'Vision → language projection',
            kind: 'mlp' as const,
            detail: `Multimodal normalization and linear projection map visual features into the language residual width${modality.vision.softTokens ? ` (${format(modality.vision.softTokens)} soft tokens per image)` : ''}.`,
            shape: `[B, P, ${format(modality.vision.embedDim ?? input.hiddenSize)}] → [B, P, H=${format(modality.vision.outputDim ?? input.hiddenSize)}]`,
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
        ]
      : isMultimodal
        ? [
            {
              id: 'modality-input',
              label: 'Non-text modality input',
              kind: 'input' as const,
              detail:
                'The checkpoint is tagged as multimodal, but its config did not expose a typed vision/audio sub-config.',
              shape: '[B, features, encoder_dim] raw modality features',
              sourceFile: modeling?.name,
              sourceUrl: modeling?.url,
            },
            {
              id: 'modality-projector',
              label: 'Modality encoder / projector',
              kind: 'input' as const,
              detail:
                'Projects non-text features into the language-model residual width; exact structure remains unverified until the implementation is loaded.',
              shape: `[B, features, encoder_dim] → [B, S, H=${format(input.hiddenSize)}]`,
              sourceFile: modeling?.name,
              sourceUrl: modeling?.url,
            },
          ]
        : []),
    ...(isMultimodal && modality?.audio
      ? [
          {
            id: 'audio-input',
            label: 'Audio frames',
            kind: 'input' as const,
            detail: modality.audio.encoderFree
              ? `Raw audio is chunked into fixed-size frames of ${format(modality.audio.samplesPerToken ?? modality.audio.featureDim ?? 0)} samples per soft token.`
              : 'Audio features are produced by the model-defined audio encoder or feature extractor.',
            shape: `[B, T, F=${format(modality.audio.featureDim ?? 0)}]`,
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
          {
            id: 'audio-projector',
            label: modality.audio.encoderFree
              ? 'Audio feature projection'
              : 'Audio encoder / projection',
            kind: 'mlp' as const,
            detail: modality.audio.encoderFree
              ? 'RMSNorm → Linear projects each audio frame directly into the language residual width; there is no conformer tower in this unified path.'
              : 'Maps audio encoder features into the language-model residual width.',
            shape: `[B, T, F=${format(modality.audio.featureDim ?? 0)}] → [B, T, H=${format(modality.audio.outputDim ?? input.hiddenSize)}]`,
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
        ]
      : []),
    ...(isMultimodal && modality?.video
      ? [
          {
            id: 'video-input',
            label: 'Video frames / patches',
            kind: 'input' as const,
            detail:
              'Video frames reuse the vision feature path; frame and patch positions are flattened before feature scattering.',
            shape:
              visionPatchDim > 0
                ? `[B, F, P, ${format(visionPatchDim)}]`
                : '[B, F, pixels]',
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
        ]
      : []),
    ...(input.isEncoderDecoder
      ? [
          {
            id: 'encoder',
            label: 'Encoder stack',
            kind: 'attention' as const,
            detail:
              'Encoder layers build bidirectional representations that decoder cross-attention can read.',
            shape: `[B, S_enc, H=${format(input.hiddenSize)}]`,
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
        ]
      : []),
    {
      id: 'embedding',
      label: 'Text token embedding',
      kind: 'input',
      detail: `Vocabulary lookup projects ${format(input.vocabSize)} token IDs into the ${format(input.hiddenSize)}-wide residual stream.`,
      shape: `[B, S, vocab=${format(input.vocabSize)}] → [B, S, H=${format(input.hiddenSize)}]`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    ...(isMultimodal
      ? [
          {
            id: 'token-fusion',
            label: 'Multimodal token fusion',
            kind: 'residual' as const,
            detail:
              'Media features replace matching placeholder embeddings through a masked scatter operation; the fused sequence is then passed to the language model.',
            shape: `[B, S, H=${format(input.hiddenSize)}] + media features → [B, S, H]`,
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
        ]
      : []),
    {
      id: 'attention-norm',
      label: 'Pre-attention norm',
      kind: 'normalization',
      detail:
        'Normalizes the residual stream before attention; exact norm class depends on the model implementation.',
      shape: `[B, S, H=${format(input.hiddenSize)}]`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    {
      id: 'qkv',
      label: 'Q / K / V projections',
      kind: 'attention',
      detail: `Query heads: ${format(input.numAttentionHeads)}. Key/value heads: ${format(input.numKeyValueHeads)} (${attentionRatio}). Head dimension: ${format(input.headDim)}.`,
      shape: `Q [B,S,${format(input.numAttentionHeads)},${format(input.headDim)}] · K/V [B,S,${format(input.numKeyValueHeads)},${format(input.headDim)}]`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    {
      id: 'position',
      label: 'Position / RoPE transform',
      kind: 'attention',
      detail:
        'Applies the model-defined position encoding or rotary transform to query/key states.',
      shape: `[B, heads, S, D=${format(input.headDim)}]`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    {
      id: 'attention',
      label: 'Attention scores + value mixing',
      kind: 'attention',
      detail: `Computes causal attention across ${format(input.numAttentionHeads)} query heads. KV-cache growth is governed by ${format(input.numKeyValueHeads)} K/V heads.`,
      shape: `[B, Q=${format(input.numAttentionHeads)}, S, S] → [B, S, H]`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    {
      id: 'attention-residual',
      label: 'Attention output + residual',
      kind: 'residual',
      detail:
        'Projects attention output back into the residual width and adds the skip connection.',
      shape: `[B, S, H=${format(input.hiddenSize)}] + residual`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    {
      id: 'mlp-norm',
      label: 'Pre-MLP norm',
      kind: 'normalization',
      detail:
        'Normalizes the post-attention residual before the feed-forward or expert path.',
      shape: `[B, S, H=${format(input.hiddenSize)}]`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    ...(input.numExperts && input.numExperts > 1
      ? [
          {
            id: 'router',
            label: 'MoE router / top-k gate',
            kind: 'routing' as const,
            detail: expertDetail,
            shape: `[B, S, H] → expert scores [B, S, E=${format(input.numExperts)}]`,
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
          {
            id: 'experts',
            label: 'Selected expert MLPs',
            kind: 'mlp' as const,
            detail: `Each selected expert expands to ${format(mlpWidth)} channels, applies its activation/gating, then projects back to H.`,
            shape: `[B, S, H=${format(input.hiddenSize)}] → [B, S, I=${format(mlpWidth)}] → [B, S, H]`,
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
        ]
      : [
          {
            id: 'mlp',
            label: 'Feed-forward / MLP',
            kind: 'mlp' as const,
            detail: expertDetail,
            shape: `[B, S, H=${format(input.hiddenSize)}] → [B, S, I=${format(mlpWidth)}] → [B, S, H]`,
            sourceFile: modeling?.name,
            sourceUrl: modeling?.url,
          },
        ]),
    {
      id: 'mlp-residual',
      label: 'MLP output + residual',
      kind: 'residual',
      detail:
        'Adds the feed-forward or routed expert output back to the residual stream.',
      shape: `[B, S, H=${format(input.hiddenSize)}] + residual`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    {
      id: 'repeat',
      label: `Repeat decoder block × ${format(input.numLayers)}`,
      kind: 'residual',
      detail:
        'The attention, residual, norm, and MLP sequence is repeated for every decoder layer.',
      shape: `[B, S, H=${format(input.hiddenSize)}]`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    {
      id: 'final-norm',
      label: 'Final norm',
      kind: 'normalization',
      detail:
        'Normalizes the final residual stream before the language-model head.',
      shape: `[B, S, H=${format(input.hiddenSize)}]`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
    {
      id: 'lm-head',
      label: 'Language-model head / logits',
      kind: 'output',
      detail: `Projects hidden states back to the ${format(input.vocabSize)}-token vocabulary; weights may be tied to embeddings.`,
      shape: `[B, S, H=${format(input.hiddenSize)}] → [B, S, vocab=${format(input.vocabSize)}]`,
      sourceFile: modeling?.name,
      sourceUrl: modeling?.url,
    },
  ];
}

export function buildArchitectureOverview(
  input: ArchitectureFlowInput,
): ArchitectureFlowNode[] {
  const detailed = buildArchitectureFlow(input);
  const find = (id: string) => detailed.find((node) => node.id === id);
  const ids = [
    'input',
    'embedding',
    ...(input.modality && input.modality !== 'text' ? ['token-fusion'] : []),
    'repeat',
    'final-norm',
    'lm-head',
  ];
  return [...ids.map(find)].filter((node): node is ArchitectureFlowNode =>
    Boolean(node),
  );
}
