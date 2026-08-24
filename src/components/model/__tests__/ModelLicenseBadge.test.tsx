import { fireEvent, render, screen } from '@testing-library/react';

import ModelLicenseBadge from '../ModelLicenseBadge';

describe('ModelLicenseBadge', () => {
  it('opens named license guidance and links to the authoritative text', () => {
    render(
      <ModelLicenseBadge
        modelId='nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16'
        license='other'
        modelUrl='https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16'
      />,
    );

    fireEvent.click(screen.getByText('OpenMDW 1.1 · details'));
    expect(screen.getByText('Model license')).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: /Read the full license text/ }),
    ).toHaveAttribute(
      'href',
      'https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/LICENSE',
    );
  });
});
