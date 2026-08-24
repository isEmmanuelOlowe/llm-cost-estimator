import { render, screen } from '@testing-library/react';

import KvCacheScalingCard from '../KvCacheScalingCard';

describe('KvCacheScalingCard', () => {
  it('keeps the cache value badge on one line', () => {
    render(
      <KvCacheScalingCard
        kvCacheGB={0.38}
        bytesPerToken={3072}
        totalTokens={131072}
        sequenceLength={131072}
        batchSize={1}
        precisionBits={16}
        numLayers={48}
        numAttentionHeads={16}
        numKeyValueHeads={8}
        headDim={256}
      />,
    );

    expect(screen.getByText('0.38 GB')).toHaveClass(
      'shrink-0',
      'whitespace-nowrap',
    );
  });
});
