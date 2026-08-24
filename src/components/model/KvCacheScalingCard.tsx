interface KvCacheScalingCardProps {
  kvCacheGB: number;
  stateCacheGB?: number;
  cacheMode?: string;
  cacheDescription?: string;
  attentionLayers?: number;
  bytesPerToken: number;
  totalTokens: number;
  sequenceLength: number;
  batchSize: number;
  precisionBits: number;
  numLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
}

function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return 'N/A';
  return value.toLocaleString('en-US', { maximumFractionDigits: digits });
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  if (value >= 1024 ** 3) return `${formatNumber(value / 1024 ** 3)} GB`;
  if (value >= 1024 ** 2) return `${formatNumber(value / 1024 ** 2)} MB`;
  if (value >= 1024) return `${formatNumber(value / 1024)} KB`;
  return `${formatNumber(value, 0)} B`;
}

export default function KvCacheScalingCard({
  kvCacheGB,
  stateCacheGB = 0,
  cacheMode = 'standard',
  cacheDescription,
  attentionLayers,
  bytesPerToken,
  totalTokens,
  sequenceLength,
  batchSize,
  precisionBits,
  numLayers,
  numAttentionHeads,
  numKeyValueHeads,
  headDim,
}: KvCacheScalingCardProps) {
  const ratio =
    numAttentionHeads > 0 ? numKeyValueHeads / numAttentionHeads : 1;
  return (
    <div className='mt-5 rounded-xl border border-primary/30 bg-primary/5 p-4'>
      <div className='flex flex-col gap-2 md:flex-row md:items-start md:justify-between'>
        <div>
          <h3 className='font-semibold'>KV-cache scaling</h3>
          <p className='mt-1 text-xs text-base-content/70'>
            KV memory follows the selected model&apos;s cache architecture.
            GQA/MQA, sliding windows, compressed attention, and state-space
            layers can reduce or change the scaling beyond the standard formula;
            the details below identify which schedule was used.
          </p>
        </div>
        <span className='badge badge-primary h-auto min-h-6 shrink-0 whitespace-nowrap px-3 py-1 tabular-nums'>
          {formatNumber(kvCacheGB)} GB
        </span>
      </div>

      {stateCacheGB > 0 && (
        <div className='mt-3 rounded-lg bg-secondary/10 p-3 text-xs'>
          <span className='font-semibold'>Separate recurrent state:</span>{' '}
          {formatNumber(stateCacheGB)} GB is included in total memory but is not
          KV tensor storage.
        </div>
      )}

      {cacheMode !== 'standard' && (
        <div className='mt-3 rounded-lg bg-base-100 p-3 text-xs'>
          <div className='font-semibold'>
            Architecture-aware cache: {cacheMode}
          </div>
          {attentionLayers !== undefined && attentionLayers > 0 && (
            <div className='mt-1 text-base-content/65'>
              Attention layers retaining KV state:{' '}
              {formatNumber(attentionLayers, 0)}
            </div>
          )}
          {cacheDescription && (
            <div className='mt-1 text-base-content/65'>{cacheDescription}</div>
          )}
        </div>
      )}

      <div className='mt-4 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4'>
        <div className='rounded-lg bg-base-100 p-3'>
          <div className='text-base-content/65'>KV bytes / token</div>
          <div className='mt-1 font-semibold'>{formatBytes(bytesPerToken)}</div>
        </div>
        <div className='rounded-lg bg-base-100 p-3'>
          <div className='text-base-content/65'>Tokens resident</div>
          <div className='mt-1 font-semibold'>
            {formatNumber(totalTokens, 0)}
          </div>
        </div>
        <div className='rounded-lg bg-base-100 p-3'>
          <div className='text-base-content/65'>Attention → KV heads</div>
          <div className='mt-1 font-semibold'>
            {formatNumber(numAttentionHeads, 0)} →{' '}
            {formatNumber(numKeyValueHeads, 0)} ({formatNumber(ratio * 100, 0)}
            %)
          </div>
        </div>
        <div className='rounded-lg bg-base-100 p-3'>
          <div className='text-base-content/65'>Shape inputs</div>
          <div className='mt-1 font-semibold'>
            {formatNumber(numLayers, 0)} layers · {formatNumber(headDim, 0)} dim
            · {precisionBits}-bit
          </div>
        </div>
      </div>

      <div className='mt-4 rounded-lg bg-base-100 p-3 text-xs text-base-content/75'>
        {cacheMode === 'standard' ? (
          <>
            <span className='font-semibold'>Formula:</span> 2 (K + V) ×{' '}
            {formatNumber(numLayers, 0)} layers ×{' '}
            {formatNumber(numKeyValueHeads, 0)} KV heads ×{' '}
            {formatNumber(headDim, 0)} head dim × {precisionBits / 8} bytes ×{' '}
            {formatNumber(sequenceLength, 0)} context ×{' '}
            {formatNumber(batchSize, 0)} sequence(s).
          </>
        ) : (
          <>
            <span className='font-semibold'>Accounting:</span>{' '}
            {cacheDescription} Effective KV storage is{' '}
            {formatBytes(bytesPerToken)} per resident token at the current
            context; local windows and compressed entries are counted by layer.
          </>
        )}
      </div>
    </div>
  );
}
