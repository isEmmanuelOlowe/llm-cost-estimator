import { fireEvent, render, screen } from '@testing-library/react';

import PythonSourceModal from '../PythonSourceModal';

const source = {
  name: 'modeling_llama.py',
  url: 'https://github.com/huggingface/transformers/modeling_llama.py',
  content:
    'class LlamaAttention(nn.Module):\n    def forward(self, hidden_states):\n        # mix values\n        return hidden_states',
};

describe('PythonSourceModal', () => {
  it('shows highlighted, dismissible component source', () => {
    const onClose = jest.fn();
    render(
      <PythonSourceModal
        open
        nodeId='attention'
        nodeLabel='Attention scores + value mixing'
        source={source}
        onClose={onClose}
      />,
    );

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('class')).toHaveClass('text-lab-aqua');
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('closes from the explicit button', () => {
    const onClose = jest.fn();
    render(
      <PythonSourceModal
        open
        nodeId='attention'
        nodeLabel='Attention'
        source={source}
        onClose={onClose}
      />,
    );

    fireEvent.click(
      screen.getByRole('button', { name: 'Close implementation source' }),
    );
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
