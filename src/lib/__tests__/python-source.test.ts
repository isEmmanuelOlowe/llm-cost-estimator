import {
  extractPythonExcerpt,
  findImportedPythonSource,
} from '../python-source';

const source = `${Array.from({ length: 20 }, (_, index) => `# header ${index}`).join('\n')}
from helpers import FlashAttentionKwargs
class LlamaAttention(nn.Module):
    def __init__(self):
        self.q_proj = Linear()

    def forward(self, hidden_states):
        return self.q_proj(hidden_states)

class LlamaMLP(nn.Module):
    def forward(self, hidden_states):
        return hidden_states
${Array.from({ length: 180 }, (_, index) => `# footer ${index}`).join('\n')}`;

describe('python source excerpts', () => {
  it('focuses the excerpt on the implementation matching the selected node', () => {
    const excerpt = extractPythonExcerpt(
      source,
      'attention',
      'Attention scores + value mixing',
      40,
    );

    expect(excerpt.content).toContain('class LlamaAttention');
    expect(excerpt.content).not.toContain('FlashAttentionKwargs');
    expect(excerpt.content).toContain('def forward');
    expect(excerpt.content).not.toContain('class LlamaMLP');
    expect(excerpt.startLine).toBeGreaterThan(1);
  });

  it('follows a relative Python import to the smaller implementation module', () => {
    const imported = findImportedPythonSource(
      'from ..gemma4.modeling_gemma4 import (\n    Gemma4TextAttention,\n)',
      'attention',
      'Attention scores',
      'https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4_unified/modular_gemma4_unified.py',
    );

    expect(imported).toEqual({
      name: 'modeling_gemma4.py',
      url: 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py',
      rawUrl:
        'https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py',
    });
  });

  it('prefers the language-model attention class over an earlier audio class', () => {
    const content = `class Gemma4AudioAttention:\n    pass\n\nclass Gemma4TextAttention:\n    def forward(self):\n        return self.q_proj`;
    const excerpt = extractPythonExcerpt(
      content,
      'attention',
      'Attention scores + value mixing',
    );

    expect(excerpt.content).toContain('class Gemma4TextAttention');
    expect(excerpt.content).not.toContain('class Gemma4AudioAttention');
  });
});
