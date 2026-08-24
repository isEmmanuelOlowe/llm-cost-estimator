import { fireEvent, render, screen } from '@testing-library/react';

import ArchitectureFlowExplorer from '../ArchitectureFlowExplorer';

const props = {
  modelType: 'llama',
  hiddenSize: 4096,
  numLayers: 32,
  numAttentionHeads: 32,
  numKeyValueHeads: 8,
  headDim: 128,
  intermediateSize: 11008,
  vocabSize: 32000,
  sourceDirectoryUrl:
    'https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama',
};

describe('ArchitectureFlowExplorer', () => {
  it('supports overview/block switching, zoom controls, and node inspection', () => {
    render(<ArchitectureFlowExplorer {...props} />);

    expect(
      screen.getByTestId('architecture-flow-explorer'),
    ).toBeInTheDocument();
    expect(screen.getAllByText('Q / K / V projections').length).toBeGreaterThan(
      0,
    );
    expect(screen.getByText('× 32 layers')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getByText('110%')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Overview' }));
    expect(screen.queryByText('Q / K / V projections')).not.toBeInTheDocument();
    expect(
      screen.getAllByText('Repeat decoder block × 32').length,
    ).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole('button', { name: 'Inside one block' }));
    fireEvent.click(
      screen.getByRole('button', { name: /Q \/ K \/ V projections/ }),
    );
    expect(screen.getByText(/Key\/value heads: 8/)).toBeInTheDocument();
  });

  it('keeps ordinary wheel input for page scrolling and reveals internals with deliberate zoom', () => {
    render(<ArchitectureFlowExplorer {...props} />);

    const graph = screen.getByLabelText('Interactive model architecture graph');
    fireEvent.wheel(graph, { deltaY: -100 });
    expect(screen.getByText('100%')).toBeInTheDocument();

    fireEvent.wheel(graph, { ctrlKey: true, deltaY: -100 });
    expect(screen.getByText('105%')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getAllByText('query').length).toBeGreaterThan(0);
    expect(
      screen.getByText(/ordinary scroll moves the page/i),
    ).toBeInTheDocument();
  });

  it('shows implementation source inline when available', () => {
    render(
      <ArchitectureFlowExplorer
        {...props}
        sourcePreview={{
          name: 'modeling_llama.py',
          url: 'https://github.com/huggingface/transformers/blob/main/modeling_llama.py',
          content: 'class LlamaAttention:\n    def forward(self): ...',
        }}
      />,
    );

    expect(screen.getByText(/class LlamaAttention/)).toBeInTheDocument();
  });
});
