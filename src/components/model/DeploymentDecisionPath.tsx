interface DeploymentDecisionPathProps {
  modelLabel: string;
  parameterBillions: number;
  layers: number;
  contextLength: number;
  weightFormat: string;
  totalMemoryGB: number;
  memorySegments: Array<{
    label: string;
    valueGB: number;
    color: string;
  }>;
  gpuName: string;
  gpuCapacityGB: number;
  fits: boolean;
  headroomGB: number;
  tokensPerSecond?: number;
  projectedCost?: number;
  cloudCostLabel?: string;
}

function format(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '—';
  return value.toLocaleString('en-US', { maximumFractionDigits: digits });
}

export default function DeploymentDecisionPath({
  modelLabel,
  parameterBillions,
  layers,
  contextLength,
  weightFormat,
  totalMemoryGB,
  memorySegments,
  gpuName,
  gpuCapacityGB,
  fits,
  headroomGB,
  tokensPerSecond,
  projectedCost,
  cloudCostLabel,
}: DeploymentDecisionPathProps) {
  const memoryScale = Math.max(totalMemoryGB, gpuCapacityGB, 1);

  return (
    <section
      aria-labelledby='deployment-path-title'
      className='rounded-2xl border border-primary/25 bg-gradient-to-br from-base-100 to-primary/5 p-5 shadow-lg shadow-black/10 sm:p-6'
    >
      <div className='flex flex-col gap-3 md:flex-row md:items-end md:justify-between'>
        <div>
          <div className='text-[10px] font-semibold uppercase tracking-[0.2em] text-primary'>
            Deployment path
          </div>
          <h2
            id='deployment-path-title'
            className='mt-1 text-2xl font-semibold'
          >
            What this model needs—at a glance
          </h2>
        </div>
        <div className='flex flex-wrap gap-2 text-xs'>
          <a className='btn btn-primary btn-xs' href='#understand'>
            Explore architecture
          </a>
          <a className='btn btn-outline btn-xs' href='#estimate'>
            Tune assumptions
          </a>
        </div>
      </div>

      <div
        className='mt-5 grid gap-3 lg:grid-cols-[1fr_auto_1.2fr_auto_1fr] lg:items-stretch'
        role='img'
        aria-label={`${modelLabel} requires ${format(totalMemoryGB, 2)} GB and ${fits ? 'fits' : 'does not fit'} on ${gpuName}`}
      >
        <div className='rounded-xl border border-base-300 bg-base-200/70 p-4'>
          <div className='flex items-start justify-between gap-3'>
            <div className='min-w-0'>
              <div className='text-[10px] font-semibold uppercase tracking-[0.16em] text-base-content/55'>
                1 · Model
              </div>
              <div
                className='mt-2 truncate text-lg font-bold'
                title={modelLabel}
              >
                {modelLabel}
              </div>
            </div>
            <span className='badge badge-primary badge-outline'>
              {weightFormat}
            </span>
          </div>
          <dl className='mt-4 grid grid-cols-3 gap-2 text-xs'>
            <div>
              <dt className='text-base-content/55'>Parameters</dt>
              <dd className='mt-1 text-base font-bold'>
                {format(parameterBillions, 2)}B
              </dd>
            </div>
            <div>
              <dt className='text-base-content/55'>Layers</dt>
              <dd className='mt-1 text-base font-bold'>{format(layers, 0)}</dd>
            </div>
            <div>
              <dt className='text-base-content/55'>Context</dt>
              <dd className='mt-1 text-base font-bold'>
                {format(contextLength, 0)}
              </dd>
            </div>
          </dl>
        </div>

        <div
          className='hidden items-center text-2xl text-primary lg:flex'
          aria-hidden='true'
        >
          →
        </div>

        <div className='rounded-xl border border-base-300 bg-base-200/70 p-4'>
          <div className='flex items-end justify-between gap-3'>
            <div>
              <div className='text-[10px] font-semibold uppercase tracking-[0.16em] text-base-content/55'>
                2 · Memory
              </div>
              <div className='mt-2 text-2xl font-bold tabular-nums'>
                {format(totalMemoryGB, 2)} GB
              </div>
            </div>
            <div className='text-right text-xs text-base-content/55'>
              of {format(gpuCapacityGB, 0)} GB selected
            </div>
          </div>
          <div className='relative mt-4 flex h-8 overflow-hidden rounded-lg bg-base-300'>
            {memorySegments
              .filter((segment) => segment.valueGB > 0)
              .map((segment) => (
                <div
                  key={segment.label}
                  className={segment.color}
                  style={{ width: `${(segment.valueGB / memoryScale) * 100}%` }}
                  title={`${segment.label}: ${format(segment.valueGB, 2)} GB`}
                />
              ))}
            <span
              className='absolute inset-y-0 w-0.5 bg-base-content'
              style={{
                left: `${Math.min(100, (gpuCapacityGB / memoryScale) * 100)}%`,
              }}
            />
          </div>
          <div className='mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-base-content/65'>
            {memorySegments
              .filter((segment) => segment.valueGB > 0)
              .map((segment) => (
                <span key={segment.label} className='flex items-center gap-1.5'>
                  <span className={`size-2 rounded-sm ${segment.color}`} />
                  {segment.label} {format(segment.valueGB, 2)} GB
                </span>
              ))}
          </div>
        </div>

        <div
          className='hidden items-center text-2xl text-primary lg:flex'
          aria-hidden='true'
        >
          →
        </div>

        <div
          className={`rounded-xl border p-4 ${
            fits
              ? 'border-success/35 bg-success/5'
              : 'border-error/35 bg-error/5'
          }`}
        >
          <div className='flex items-start justify-between gap-3'>
            <div className='min-w-0'>
              <div className='text-[10px] font-semibold uppercase tracking-[0.16em] text-base-content/55'>
                3 · Hardware
              </div>
              <div className='mt-2 line-clamp-2 text-lg font-bold'>
                {gpuName}
              </div>
            </div>
            <span className={`badge ${fits ? 'badge-success' : 'badge-error'}`}>
              {fits ? 'Fits' : 'Overflow'}
            </span>
          </div>
          <div
            className={`mt-4 text-sm font-semibold ${fits ? 'text-success' : 'text-error'}`}
          >
            {fits
              ? `${format(headroomGB, 2)} GB headroom`
              : `${format(Math.abs(headroomGB), 2)} GB over capacity`}
          </div>
          <div className='mt-3 grid grid-cols-2 gap-2 border-t border-base-300/70 pt-3 text-xs'>
            <div>
              <div className='text-base-content/55'>Decode</div>
              <div className='mt-1 font-bold'>
                {tokensPerSecond
                  ? `${format(tokensPerSecond, 2)} tok/s`
                  : 'N/A'}
              </div>
            </div>
            <div>
              <div className='text-base-content/55'>Cloud cost</div>
              <div className='mt-1 font-bold'>
                {projectedCost !== undefined
                  ? `$${format(projectedCost, 2)}`
                  : 'N/A'}
              </div>
              <div className='mt-0.5 line-clamp-1 text-[10px] text-base-content/50'>
                {cloudCostLabel ?? 'No verified rate'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
