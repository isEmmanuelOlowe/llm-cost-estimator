import type { HardwareLike } from '@/estimator/estimator';
import type {
  TrainingCloudRateLike,
  TrainingMemoryEstimate,
  TrainingPlannerSettings,
  TrainingPlanRecommendation,
  TrainingRunEstimate,
} from '@/estimator/training';

interface TrainingPlannerCardProps {
  settings: TrainingPlannerSettings;
  onChange: (patch: Partial<TrainingPlannerSettings>) => void;
  hardware: readonly HardwareLike[];
  selectedGpu: HardwareLike;
  onSelectedGpuChange: (name: string) => void;
  memory: TrainingMemoryEstimate;
  run: TrainingRunEstimate;
  selectedCloudRate?: TrainingCloudRateLike;
  customHourlyRate: number | '';
  onCustomHourlyRateChange: (rate: number | '') => void;
  recommendations: TrainingPlanRecommendation[];
}

const METHOD_COPY = {
  full: {
    label: 'Full tune',
    detail: 'Train every parameter. Highest memory and optimizer cost.',
  },
  lora: {
    label: 'LoRA',
    detail: 'Freeze the base and train low-rank attention adapters.',
  },
  qlora: {
    label: 'QLoRA',
    detail: '4-bit frozen base plus all-linear low-rank adapters.',
  },
} as const;

function format(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return 'N/A';
  return value.toLocaleString('en-US', { maximumFractionDigits: digits });
}

function formatDuration(hours: number): string {
  if (!Number.isFinite(hours) || hours <= 0) return 'N/A';
  if (hours < 1) return `${format(hours * 60, 0)} min`;
  if (hours < 48) return `${format(hours, 1)} hr`;
  return `${format(hours / 24, 1)} days`;
}

export default function TrainingPlannerCard({
  settings,
  onChange,
  hardware,
  selectedGpu,
  onSelectedGpuChange,
  memory,
  run,
  selectedCloudRate,
  customHourlyRate,
  onCustomHourlyRateChange,
  recommendations,
}: TrainingPlannerCardProps) {
  const memoryPerGpu =
    selectedGpu.per_device_memory_gb ??
    selectedGpu.memory_gb / Math.max(1, selectedGpu.device_count ?? 1);
  const fits = memory.perDeviceGB <= memoryPerGpu;
  const best = recommendations[0];
  const gradientAccumulation = Math.max(
    1,
    Math.ceil(
      settings.globalBatchSize /
        (settings.microBatchSize * settings.deviceCount),
    ),
  );

  return (
    <section
      id='hardware'
      className='scroll-mt-24 rounded-2xl border border-base-300 bg-base-100 p-5 shadow-lg shadow-black/10 sm:p-6'
    >
      <div className='flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between'>
        <div>
          <div className='text-[10px] font-semibold uppercase tracking-[0.18em] text-secondary'>
            Training planner
          </div>
          <h2 className='mt-1 text-xl font-semibold'>Plan the complete run</h2>
          <p className='mt-1 max-w-2xl text-xs leading-relaxed text-base-content/65'>
            Choose the adaptation method and workload. Memory is estimated per
            GPU; runtime and cost are heuristic roofline projections.
          </p>
        </div>
        <span
          className={`badge shrink-0 ${fits ? 'badge-success' : 'badge-error'}`}
        >
          {fits ? 'Selected setup fits' : 'Selected setup overflows'}
        </span>
      </div>

      <div className='mt-5 grid gap-2 sm:grid-cols-3'>
        {(Object.keys(METHOD_COPY) as TrainingPlannerSettings['method'][]).map(
          (method) => (
            <button
              key={method}
              type='button'
              aria-pressed={settings.method === method}
              className={`rounded-xl border p-3 text-left transition ${
                settings.method === method
                  ? 'border-secondary bg-secondary/10'
                  : 'border-base-300 bg-base-200 hover:border-secondary/50'
              }`}
              onClick={() => onChange({ method })}
            >
              <span className='block text-sm font-bold'>
                {METHOD_COPY[method].label}
              </span>
              <span className='mt-1 block text-[11px] leading-relaxed text-base-content/60'>
                {METHOD_COPY[method].detail}
              </span>
            </button>
          ),
        )}
      </div>

      <div className='mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3'>
        <label className='text-xs font-semibold'>
          Training sequence length
          <input
            className='input input-bordered mt-1 w-full'
            type='number'
            min='128'
            step='128'
            value={settings.sequenceLength}
            onChange={(event) =>
              onChange({ sequenceLength: Number(event.target.value) || 128 })
            }
          />
        </label>
        <label className='text-xs font-semibold'>
          Dataset size (million tokens)
          <input
            className='input input-bordered mt-1 w-full'
            type='number'
            min='0.1'
            step='1'
            value={settings.datasetTokens / 1_000_000}
            onChange={(event) =>
              onChange({
                datasetTokens: (Number(event.target.value) || 0.1) * 1_000_000,
              })
            }
          />
        </label>
        <label className='text-xs font-semibold'>
          Epochs
          <input
            className='input input-bordered mt-1 w-full'
            type='number'
            min='0.1'
            step='0.5'
            value={settings.epochs}
            onChange={(event) =>
              onChange({ epochs: Number(event.target.value) || 1 })
            }
          />
        </label>
        <label className='text-xs font-semibold sm:col-span-2 xl:col-span-1'>
          GPU model
          <select
            className='select select-bordered mt-1 w-full'
            value={selectedGpu.name}
            onChange={(event) => onSelectedGpuChange(event.target.value)}
          >
            {hardware.map((gpu) => (
              <option key={gpu.name} value={gpu.name}>
                {gpu.name} · {gpu.per_device_memory_gb ?? gpu.memory_gb} GB/GPU
              </option>
            ))}
          </select>
        </label>
        <label className='text-xs font-semibold'>
          GPU count
          <select
            className='select select-bordered mt-1 w-full'
            value={settings.deviceCount}
            onChange={(event) =>
              onChange({ deviceCount: Number(event.target.value) })
            }
          >
            {[1, 2, 3, 4, 5, 6, 7, 8].map((count) => (
              <option key={count} value={count}>
                {count} GPU{count === 1 ? '' : 's'}
              </option>
            ))}
          </select>
        </label>
        <label className='text-xs font-semibold'>
          State placement
          <select
            className='select select-bordered mt-1 w-full'
            value={settings.distribution}
            onChange={(event) =>
              onChange({
                distribution: event.target
                  .value as TrainingPlannerSettings['distribution'],
              })
            }
          >
            <option value='replicated'>Replicated (DDP)</option>
            <option value='fully-sharded'>Fully sharded (FSDP / ZeRO-3)</option>
          </select>
        </label>
        <label className='text-xs font-semibold'>
          {settings.method === 'qlora'
            ? 'QLoRA compute format'
            : 'Weight / compute format'}
          <select
            className='select select-bordered mt-1 w-full'
            value={settings.computeFormat}
            onChange={(event) =>
              onChange({
                computeFormat: event.target
                  .value as TrainingPlannerSettings['computeFormat'],
              })
            }
          >
            <option value='bf16'>BF16</option>
            <option value='fp16'>FP16</option>
            <option value='fp32'>FP32</option>
          </select>
        </label>
      </div>

      <div className='mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4'>
        <div className='rounded-xl border border-primary/30 bg-primary/10 p-4'>
          <div className='text-[10px] font-semibold uppercase tracking-wide text-primary'>
            VRAM / GPU
          </div>
          <div className='mt-2 text-2xl font-bold tabular-nums'>
            {format(memory.perDeviceGB)} GB
          </div>
          <div className='mt-1 text-[11px] text-base-content/60'>
            {format(memoryPerGpu - memory.perDeviceGB)} GB headroom
          </div>
        </div>
        <div className='rounded-xl border border-base-300 bg-base-200 p-4'>
          <div className='text-[10px] font-semibold uppercase tracking-wide text-base-content/55'>
            Trainable
          </div>
          <div className='mt-2 text-2xl font-bold tabular-nums'>
            {format(memory.trainableParameterCount / 1e6, 1)}M
          </div>
          <div className='mt-1 text-[11px] text-base-content/60'>
            {format(memory.trainablePercent, 3)}% of parameters
          </div>
        </div>
        <div className='rounded-xl border border-base-300 bg-base-200 p-4'>
          <div className='text-[10px] font-semibold uppercase tracking-wide text-base-content/55'>
            Estimated time
          </div>
          <div className='mt-2 text-2xl font-bold tabular-nums'>
            {formatDuration(run.durationHours)}
          </div>
          <div className='mt-1 text-[11px] text-base-content/60'>
            {format(run.tokensPerSecond, 0)} training tok/s
          </div>
        </div>
        <div className='rounded-xl border border-base-300 bg-base-200 p-4'>
          <div className='text-[10px] font-semibold uppercase tracking-wide text-base-content/55'>
            Selected cost
          </div>
          <div className='mt-2 text-2xl font-bold tabular-nums'>
            {run.clusterHourlyRate > 0 ? `$${format(run.totalCost)}` : 'N/A'}
          </div>
          <div className='mt-1 text-[11px] text-base-content/60'>
            {typeof customHourlyRate === 'number' && customHourlyRate > 0
              ? `Custom · $${format(customHourlyRate)}/GPU/hr`
              : selectedCloudRate
                ? `${selectedCloudRate.provider} · $${format(selectedCloudRate.hourly_rate)}/GPU/hr`
                : 'No exact verified cloud rate'}
          </div>
        </div>
      </div>

      <div className='mt-4 grid gap-3 rounded-xl border border-base-300 bg-base-200 p-4 text-xs sm:grid-cols-2 lg:grid-cols-5'>
        {[
          ['Base weights / GPU', memory.baseWeightsPerDeviceGB],
          [
            settings.method === 'full'
              ? 'Gradients / GPU'
              : 'Adapters + gradients',
            memory.adapterWeightsPerDeviceGB + memory.gradientsPerDeviceGB,
          ],
          ['Optimizer / GPU', memory.optimizerPerDeviceGB],
          ['Activations / GPU', memory.activationsGB],
          ['Overhead / GPU', memory.overheadGB],
        ].map(([label, value]) => (
          <div key={String(label)}>
            <div className='text-base-content/55'>{label}</div>
            <div className='mt-1 font-bold'>{format(Number(value))} GB</div>
          </div>
        ))}
      </div>

      {best ? (
        <div className='mt-5 rounded-xl border border-success/30 bg-success/10 p-4'>
          <div className='flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between'>
            <div>
              <div className='text-[10px] font-semibold uppercase tracking-[0.16em] text-success'>
                Best verified cloud estimate · independent of selection
              </div>
              <h3 className='mt-1 text-base font-bold'>
                {best.deviceCount}× {best.gpuName}
              </h3>
              <p className='mt-1 text-xs text-base-content/65'>
                {best.distribution === 'fully-sharded'
                  ? 'FSDP / ZeRO-3'
                  : 'Replicated DDP'}{' '}
                · {format(best.memory.perDeviceGB)} GB/GPU ·{' '}
                {formatDuration(best.durationHours)} · ${format(best.totalCost)}{' '}
                estimated total
              </p>
            </div>
            <a
              className='link link-primary shrink-0 text-xs'
              href={best.pricingSourceUrl}
              target='_blank'
              rel='noreferrer'
            >
              Verify {best.provider} rate ↗
            </a>
          </div>
          {recommendations.length > 1 && (
            <details className='mt-3 border-t border-success/20 pt-3'>
              <summary className='cursor-pointer text-xs font-semibold'>
                Compare {Math.min(3, recommendations.length - 1)} alternatives
              </summary>
              <ul className='mt-2 space-y-1 text-xs text-base-content/70'>
                {recommendations.slice(1, 4).map((plan) => (
                  <li
                    key={`${plan.offeringName}-${plan.deviceCount}`}
                    className='flex flex-wrap justify-between gap-2'
                  >
                    <span>
                      {plan.deviceCount}× {plan.gpuName} ·{' '}
                      {plan.distribution === 'fully-sharded' ? 'FSDP' : 'DDP'}
                    </span>
                    <span>
                      {formatDuration(plan.durationHours)} · $
                      {format(plan.totalCost)}
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      ) : (
        <p className='mt-5 rounded-xl bg-warning/10 p-4 text-xs text-warning'>
          No provider-verified cloud hardware in the catalog fits this plan.
          Reduce sequence/micro-batch size, enable full sharding, or use a
          custom quote.
        </p>
      )}

      <details className='mt-4 rounded-xl border border-base-300 bg-base-200 p-4'>
        <summary className='cursor-pointer text-sm font-semibold'>
          Advanced training assumptions
        </summary>
        <div className='mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3'>
          {settings.method !== 'full' && (
            <label className='text-xs font-semibold'>
              LoRA rank
              <input
                className='input input-bordered mt-1 w-full'
                type='number'
                min='1'
                max='256'
                value={settings.loraRank}
                onChange={(event) =>
                  onChange({ loraRank: Number(event.target.value) || 1 })
                }
              />
            </label>
          )}
          {settings.method === 'lora' && (
            <label className='text-xs font-semibold'>
              Adapter coverage
              <select
                className='select select-bordered mt-1 w-full'
                value={settings.targetCoverage}
                onChange={(event) =>
                  onChange({
                    targetCoverage: event.target
                      .value as TrainingPlannerSettings['targetCoverage'],
                  })
                }
              >
                <option value='attention-qv'>Attention Q/V</option>
                <option value='all-linear'>All linear layers</option>
              </select>
            </label>
          )}
          <label className='text-xs font-semibold'>
            Global batch
            <input
              className='input input-bordered mt-1 w-full'
              type='number'
              min='1'
              value={settings.globalBatchSize}
              onChange={(event) =>
                onChange({ globalBatchSize: Number(event.target.value) || 1 })
              }
            />
          </label>
          <label className='text-xs font-semibold'>
            Micro-batch / GPU
            <input
              className='input input-bordered mt-1 w-full'
              type='number'
              min='1'
              value={settings.microBatchSize}
              onChange={(event) =>
                onChange({ microBatchSize: Number(event.target.value) || 1 })
              }
            />
          </label>
          <label className='text-xs font-semibold'>
            Optimizer
            <select
              className='select select-bordered mt-1 w-full'
              value={settings.optimizer}
              onChange={(event) =>
                onChange({
                  optimizer: event.target
                    .value as TrainingPlannerSettings['optimizer'],
                })
              }
            >
              <option value='adamw'>AdamW</option>
              <option value='adam'>Adam</option>
              <option value='adafactor'>Adafactor</option>
              <option value='lamb'>LAMB</option>
              <option value='none'>None</option>
            </select>
          </label>
          <label className='text-xs font-semibold'>
            Optimizer-state precision
            <select
              className='select select-bordered mt-1 w-full'
              value={settings.optimizerPrecisionBits}
              disabled={settings.optimizer === 'none'}
              onChange={(event) =>
                onChange({
                  optimizerPrecisionBits: Number(event.target.value) as 8 | 32,
                })
              }
            >
              <option value='32'>32-bit states</option>
              <option value='8'>8-bit states</option>
            </select>
          </label>
          <label className='text-xs font-semibold'>
            Compute efficiency
            <input
              className='input input-bordered mt-1 w-full'
              type='number'
              min='0.05'
              max='1'
              step='0.05'
              value={settings.efficiency}
              onChange={(event) =>
                onChange({ efficiency: Number(event.target.value) || 0.3 })
              }
            />
          </label>
          <label className='text-xs font-semibold'>
            Custom $ / GPU / hour
            <input
              className='input input-bordered mt-1 w-full'
              type='number'
              min='0'
              step='0.01'
              placeholder={
                selectedCloudRate
                  ? String(selectedCloudRate.hourly_rate)
                  : 'Current quote'
              }
              value={customHourlyRate}
              onChange={(event) =>
                onCustomHourlyRateChange(
                  event.target.value ? Number(event.target.value) : '',
                )
              }
            />
          </label>
          <label className='flex items-center gap-2 self-end rounded-xl border border-base-300 px-3 py-3 text-xs font-semibold'>
            <input
              className='checkbox checkbox-primary'
              type='checkbox'
              checked={settings.gradientCheckpointing}
              onChange={(event) =>
                onChange({ gradientCheckpointing: event.target.checked })
              }
            />
            Gradient checkpointing
          </label>
        </div>
        <p className='mt-4 text-xs text-base-content/60'>
          Effective batch uses approximately {gradientAccumulation} accumulation
          step{gradientAccumulation === 1 ? '' : 's'}. DDP replicates model and
          optimizer state on every GPU; full sharding divides persistent state
          but adds communication and peak-gather uncertainty.
        </p>
        <p className='mt-2 rounded-lg border border-base-300 bg-base-100 px-3 py-2 text-xs text-base-content/60'>
          <span className='font-semibold text-base-content'>KV cache:</span> not
          allocated during standard teacher-forced fine-tuning. Attention
          working tensors use the selected{' '}
          {settings.computeFormat.toUpperCase()} compute precision instead;
          KV-cache precision remains an inference control.
        </p>
      </details>

      <p className='mt-4 text-[11px] leading-relaxed text-base-content/55'>
        LoRA freezes the base and trains low-rank matrices; QLoRA uses a frozen
        4-bit NF4-style base with LoRA adapters. Runtime uses a 6×
        parameter-token rule for full tuning and 4.5× for adapter tuning, then
        applies user efficiency and multi-GPU scaling. Validate with a short
        measured run before purchasing capacity.
      </p>
      <div className='mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]'>
        <a
          className='link link-primary'
          href='https://arxiv.org/abs/2106.09685'
          target='_blank'
          rel='noreferrer'
        >
          LoRA paper ↗
        </a>
        <a
          className='link link-primary'
          href='https://arxiv.org/abs/2305.14314'
          target='_blank'
          rel='noreferrer'
        >
          QLoRA paper ↗
        </a>
        <a
          className='link link-primary'
          href='https://docs.pytorch.org/docs/stable/fsdp.html'
          target='_blank'
          rel='noreferrer'
        >
          PyTorch FSDP ↗
        </a>
      </div>
    </section>
  );
}
