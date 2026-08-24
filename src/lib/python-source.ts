const NODE_PATTERNS: Record<string, string[]> = {
  qkv: ['TextAttention', 'Attention', 'q_proj', 'query'],
  position: ['TextRotary', 'Rotary', 'rope', 'position'],
  attention: [
    'TextAttention',
    'Attention',
    'attention_forward',
    'scaled_dot_product_attention',
  ],
  'attention-norm': ['RMSNorm', 'LayerNorm'],
  'attention-residual': ['DecoderLayer', 'Attention'],
  'mlp-norm': ['RMSNorm', 'LayerNorm'],
  mlp: ['MLP', 'FeedForward'],
  router: ['Router', 'gate'],
  experts: ['MoE', 'Expert'],
  'shared-experts': ['MoE', 'Expert'],
  embedding: ['Embedding', 'embed_tokens'],
  'lm-head': ['ForCausalLM', 'lm_head'],
  'vision-encoder': ['Vision', 'Encoder'],
  'vision-projector': ['Projector', 'MultiModalProjector'],
  'audio-projector': ['Audio', 'Projector'],
  'token-fusion': ['masked_scatter', 'merge'],
};

export interface ImportedPythonSource {
  name: string;
  url: string;
  rawUrl: string;
}

function patternsForNode(nodeId: string, nodeLabel: string): string[] {
  return [
    ...(NODE_PATTERNS[nodeId] ?? []),
    ...nodeLabel.split(/[^A-Za-z0-9_]+/).filter((word) => word.length > 3),
  ];
}

export function findImportedPythonSource(
  content: string,
  nodeId: string,
  nodeLabel: string,
  sourceUrl: string,
): ImportedPythonSource | undefined {
  const lines = content.split(/\r?\n/);
  const patterns = patternsForNode(nodeId, nodeLabel).map((pattern) =>
    pattern.toLowerCase(),
  );

  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index].match(/^\s*from\s+([.\w]+)\s+import\s*(.*)$/);
    if (!match) continue;
    let importBlock = match[2];
    let end = index;
    while (importBlock.includes('(') && !importBlock.includes(')')) {
      end += 1;
      if (end >= lines.length) break;
      importBlock += ` ${lines[end]}`;
    }
    const normalizedBlock = importBlock.toLowerCase();
    if (!patterns.some((pattern) => normalizedBlock.includes(pattern))) {
      index = end;
      continue;
    }

    const urlMatch = sourceUrl.match(
      /^(https:\/\/github\.com\/[^/]+\/[^/]+)\/(blob|tree)\/([^/]+)\/(.+)\/[^/]+\.py$/,
    );
    if (!urlMatch) return undefined;
    const [, repository, , revision, currentDirectory] = urlMatch;
    const modulePath = match[1];
    const relativePrefix = modulePath.match(/^\.+/)?.[0] ?? '';
    const moduleName = modulePath.slice(relativePrefix.length);
    const directoryParts = currentDirectory.split('/');
    const ascents = Math.max(0, relativePrefix.length - 1);
    directoryParts.splice(Math.max(0, directoryParts.length - ascents));
    const targetParts = [
      ...directoryParts,
      ...moduleName.split('.').filter(Boolean),
    ];
    const name = `${targetParts.at(-1)}.py`;
    const path = `${targetParts.join('/')}.py`;
    const url = `${repository}/blob/${revision}/${path}`;
    const rawUrl = url
      .replace('https://github.com/', 'https://raw.githubusercontent.com/')
      .replace(`/blob/${revision}/`, `/${revision}/`);
    return { name, url, rawUrl };
  }

  return undefined;
}

export interface PythonExcerpt {
  content: string;
  startLine: number;
  endLine: number;
  matchedPattern?: string;
}

function indentation(line: string): number {
  return line.match(/^\s*/)?.[0].replaceAll('\t', '    ').length ?? 0;
}

export function extractPythonExcerpt(
  content: string,
  nodeId: string,
  nodeLabel: string,
  maxLines = 140,
): PythonExcerpt {
  const lines = content.split(/\r?\n/);

  const patterns = patternsForNode(nodeId, nodeLabel);
  let matchIndex = -1;
  let matchedPattern: string | undefined;
  for (const pattern of patterns) {
    const normalized = pattern.toLowerCase();
    const index = lines.findIndex(
      (line) =>
        /^\s*(class|def)\s+/.test(line) &&
        line.toLowerCase().includes(normalized),
    );
    if (index >= 0) {
      matchIndex = index;
      matchedPattern = pattern;
      break;
    }
  }
  if (matchIndex < 0) {
    for (const pattern of patterns) {
      const normalized = pattern.toLowerCase();
      const index = lines.findIndex((line) =>
        line.toLowerCase().includes(normalized),
      );
      if (index >= 0) {
        matchIndex = index;
        matchedPattern = pattern;
        break;
      }
    }
  }

  if (matchIndex < 0) {
    const firstDefinition = lines.findIndex((line) =>
      /^\s*(class|def)\s+/.test(line),
    );
    const start = firstDefinition >= 0 ? firstDefinition : 0;
    const end = Math.min(lines.length, start + maxLines);
    return {
      content: lines.slice(start, end).join('\n'),
      startLine: start + 1,
      endLine: end,
    };
  }

  let start = matchIndex;
  for (
    let index = matchIndex;
    index >= Math.max(0, matchIndex - 24);
    index -= 1
  ) {
    if (/^\s*(class|def)\s+/.test(lines[index])) {
      start = index;
      break;
    }
  }

  const startIndent = indentation(lines[start]);
  let end = Math.min(lines.length, start + maxLines);
  for (let index = start + 1; index < end; index += 1) {
    const line = lines[index];
    if (!line.trim()) continue;
    if (
      index > matchIndex + 4 &&
      indentation(line) <= startIndent &&
      /^\s*(class|def)\s+/.test(line)
    ) {
      end = index;
      break;
    }
  }

  return {
    content: lines.slice(start, end).join('\n'),
    startLine: start + 1,
    endLine: end,
    matchedPattern,
  };
}
