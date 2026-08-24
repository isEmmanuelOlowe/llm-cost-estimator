interface VramUsageSegment {
  label: string;
  valueGB: number;
  color: string;
}

interface VramUsageBarProps {
  capacityGB: number;
  totalGB: number;
  segments: VramUsageSegment[];
  fits: boolean;
  requiredDevices: number;
  deviceCount: number;
}

function formatGB(value: number): string {
  return `${value.toLocaleString('en-US', { maximumFractionDigits: 2 })} GB`;
}

function formatPercent(value: number): string {
  return `${value.toLocaleString('en-US', { maximumFractionDigits: 1 })}%`;
}

export default function VramUsageBar({
  capacityGB,
  totalGB,
  segments,
  fits,
  requiredDevices,
  deviceCount,
}: VramUsageBarProps) {
  const scaleGB = Math.max(1, capacityGB, totalGB);
  const capacityPosition = Math.min(100, (capacityGB / scaleGB) * 100);
  const positionedSegments = segments.reduce<
    Array<{ segment: VramUsageSegment; leftGB: number }>
  >((positions, segment) => {
    const leftGB = positions.at(-1)
      ? positions[positions.length - 1].leftGB +
        positions[positions.length - 1].segment.valueGB
      : 0;
    return [...positions, { segment, leftGB }];
  }, []);
  const aggregateOverflowGB = Math.max(0, totalGB - capacityGB);
  const topologyOverflow = requiredDevices > deviceCount;

  return (
    <div className='mt-5 rounded-xl border border-base-300 bg-base-100 p-4'>
      <div className='flex flex-wrap items-start justify-between gap-2'>
        <div>
          <h3 className='text-sm font-semibold'>VRAM allocation</h3>
          <p className='mt-1 text-xs text-base-content/65'>
            Selected topology capacity: {formatGB(capacityGB)} aggregate. Each
            color segment is measured against that capacity.
          </p>
        </div>
        <span
          className={`badge badge-sm ${fits ? 'badge-success' : 'badge-error'}`}
        >
          {fits ? 'Fits' : 'Overflow'}
        </span>
      </div>

      <div
        className='relative mt-4 h-9 overflow-hidden rounded-lg bg-base-300'
        aria-label={`VRAM usage: ${formatGB(totalGB)} required of ${formatGB(capacityGB)} capacity`}
        role='img'
      >
        {positionedSegments.map(({ segment, leftGB }) => {
          const left = (leftGB / scaleGB) * 100;
          const width = (segment.valueGB / scaleGB) * 100;
          return (
            <div
              key={segment.label}
              className={`absolute inset-y-0 ${segment.color}`}
              style={{ left: `${left}%`, width: `${width}%` }}
              title={`${segment.label}: ${formatGB(segment.valueGB)}`}
            />
          );
        })}
        <div
          className='absolute inset-y-0 z-10 w-0.5 bg-base-content'
          style={{ left: `calc(${capacityPosition}% - 1px)` }}
          title={`Capacity boundary: ${formatGB(capacityGB)}`}
        />
      </div>
      <div className='mt-1 flex justify-between text-[10px] text-base-content/60'>
        <span>0 GB</span>
        <span>Capacity {formatGB(capacityGB)}</span>
        {totalGB > capacityGB && (
          <span>Overflow shown to {formatGB(totalGB)}</span>
        )}
      </div>

      <div className='mt-4 grid gap-2 text-xs sm:grid-cols-2'>
        {segments.map((segment) => (
          <div
            key={segment.label}
            className='flex items-center justify-between gap-3'
          >
            <span className='flex min-w-0 items-center gap-2'>
              <span
                className={`h-2.5 w-2.5 shrink-0 rounded-sm ${segment.color}`}
              />
              <span className='truncate'>{segment.label}</span>
            </span>
            <span className='shrink-0 font-semibold tabular-nums'>
              {formatGB(segment.valueGB)} ·{' '}
              {formatPercent((segment.valueGB / capacityGB) * 100)}
            </span>
          </div>
        ))}
      </div>

      <div
        className={`mt-4 rounded-lg px-3 py-2 text-xs ${
          fits ? 'bg-success/10 text-success' : 'bg-error/10 text-error'
        }`}
      >
        {fits
          ? `${formatGB(totalGB)} uses ${formatPercent((totalGB / capacityGB) * 100)} of selected capacity with ${formatGB(capacityGB - totalGB)} headroom.`
          : aggregateOverflowGB > 0
            ? `${formatGB(totalGB)} required exceeds aggregate capacity by ${formatGB(aggregateOverflowGB)}.`
            : `Aggregate capacity is sufficient, but the topology needs ${requiredDevices} device(s) and only exposes ${deviceCount}.`}
        {topologyOverflow && aggregateOverflowGB > 0 && (
          <span>
            {' '}
            The per-device topology also cannot satisfy the placement.
          </span>
        )}
      </div>
    </div>
  );
}
