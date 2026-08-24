import type { ModelInspection } from '@/lib/model-metadata';

interface ModelEvidencePanelProps {
  inspection: ModelInspection | null;
}

function formatCount(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '—';
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  return value.toLocaleString('en-US');
}

export default function ModelEvidencePanel({
  inspection,
}: ModelEvidencePanelProps) {
  if (!inspection) {
    return (
      <div className='mt-5 rounded-lg border border-dashed border-base-300 bg-base-200/60 p-4 text-sm text-base-content/70'>
        Fetch a public Hugging Face model to see pinned revisions, safetensors
        parameter evidence, model files, and the matching Transformers source.
      </div>
    );
  }

  const sourcePreview = inspection.transformers?.preview;

  return (
    <div className='mt-5 rounded-xl border border-base-300 bg-base-200 p-4'>
      <div className='flex flex-col gap-3 md:flex-row md:items-start md:justify-between'>
        <div>
          <h3 className='font-semibold'>Evidence &amp; implementation map</h3>
          <p className='mt-1 text-xs text-base-content/70'>
            Public metadata is read at a pinned Hub revision. Remote Python is
            never imported or executed in the browser.
          </p>
        </div>
        <div className='flex flex-wrap gap-2 text-xs'>
          <span className='badge badge-outline'>
            revision: {inspection.revision.slice(0, 10)}
          </span>
          <span className='badge badge-outline'>
            {inspection.files.length} repository files
          </span>
          {inspection.gated && (
            <span className='badge badge-warning'>gated</span>
          )}
        </div>
      </div>

      <dl className='mt-4 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4'>
        <div>
          <dt className='text-base-content/70'>Parameter evidence</dt>
          <dd className='font-semibold'>
            {formatCount(inspection.parameterCount)} ·{' '}
            {inspection.parameterSource}
          </dd>
        </div>
        <div>
          <dt className='text-base-content/70'>Safetensors</dt>
          <dd className='font-semibold'>
            {inspection.safetensors
              ? `${inspection.safetensors.files.length} file${
                  inspection.safetensors.files.length === 1 ? '' : 's'
                } · ${inspection.safetensors.source}`
              : 'Not reported'}
          </dd>
        </div>
        <div>
          <dt className='text-base-content/70'>Model type</dt>
          <dd className='font-semibold'>
            {inspection.modelType ?? 'Unknown'}
            {inspection.architectures.length > 0
              ? ` · ${inspection.architectures[0]}`
              : ''}
          </dd>
        </div>
        <div>
          <dt className='text-base-content/70'>Last Hub update</dt>
          <dd className='font-semibold'>
            {inspection.lastModified
              ? new Date(inspection.lastModified).toLocaleDateString()
              : 'Unknown'}
          </dd>
        </div>
      </dl>

      <div className='mt-4 flex flex-wrap gap-2'>
        {inspection.evidence.map((evidence) => (
          <a
            key={`${evidence.kind}-${evidence.url}`}
            className={`badge badge-sm badge-outline ${
              evidence.confidence === 'unavailable' ? 'badge-error' : ''
            }`}
            href={evidence.url}
            target='_blank'
            rel='noreferrer'
            title={evidence.detail}
          >
            {evidence.label}
          </a>
        ))}
      </div>

      <div className='mt-4 flex flex-wrap gap-2 text-xs'>
        {inspection.architecture.dtype && (
          <span className='badge badge-sm badge-outline'>
            dtype: {inspection.architecture.dtype}
          </span>
        )}
        {inspection.quantization && (
          <span className='badge badge-sm badge-warning badge-outline'>
            quantization: {inspection.quantization.method ?? 'configured'}
            {inspection.quantization.bits
              ? ` · ${inspection.quantization.bits}-bit`
              : ''}
          </span>
        )}
        {Object.entries(inspection.parameterCountByDtype).map(
          ([dtype, count]) => (
            <span key={dtype} className='badge badge-sm badge-outline'>
              {dtype}: {formatCount(count)}
            </span>
          ),
        )}
      </div>

      {inspection.transformers && (
        <div className='mt-4 rounded-lg border border-base-300 bg-base-100 p-3 text-xs'>
          <div className='flex flex-col gap-2 md:flex-row md:items-center md:justify-between'>
            <div>
              <span className='font-semibold'>Transformers source:</span>{' '}
              {inspection.transformers.files.length > 0
                ? inspection.transformers.files
                    .map((file) => file.name)
                    .join(', ')
                : 'directory lookup unavailable'}
              {inspection.transformers.transformersVersion
                ? ` · checkpoint metadata: ${inspection.transformers.transformersVersion}`
                : ''}
            </div>
            <a
              className='link link-primary whitespace-nowrap'
              href={inspection.transformers.directoryUrl}
              target='_blank'
              rel='noreferrer'
            >
              Open implementation directory ↗
            </a>
          </div>
        </div>
      )}

      {inspection.remoteCodeFiles.length > 0 && (
        <div className='mt-4 rounded-lg border border-warning/40 bg-warning/10 p-3 text-xs text-warning-content'>
          <div className='font-semibold'>Remote-code review required</div>
          <p className='mt-1'>
            Detected: {inspection.remoteCodeFiles.join(', ')}. Review the pinned
            repository files before loading this checkpoint in a trusted local
            runtime; the browser app never executes them.
          </p>
          <a
            className='link mt-2 inline-block'
            href={`https://huggingface.co/${inspection.id}/tree/${inspection.revision}`}
            target='_blank'
            rel='noreferrer'
          >
            Review pinned repository ↗
          </a>
        </div>
      )}

      {sourcePreview && (
        <details className='mt-4 rounded-lg border border-base-300 bg-base-100 p-3 text-xs'>
          <summary className='cursor-pointer font-semibold'>
            Preview {sourcePreview.name} (read-only)
          </summary>
          <a
            className='link link-primary mt-2 inline-block'
            href={sourcePreview.url}
            target='_blank'
            rel='noreferrer'
          >
            Open full source ↗
          </a>
          <pre className='mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-neutral p-3 text-[11px] leading-relaxed text-neutral-content'>
            {sourcePreview.content}
          </pre>
        </details>
      )}

      {inspection.cardExcerpt && (
        <details className='mt-3 rounded-lg border border-base-300 bg-base-100 p-3 text-xs'>
          <summary className='cursor-pointer font-semibold'>
            Pinned model-card excerpt
          </summary>
          <pre className='mt-3 max-h-48 overflow-auto whitespace-pre-wrap text-[11px] leading-relaxed text-base-content/75'>
            {inspection.cardExcerpt}
          </pre>
        </details>
      )}

      {(inspection.generationConfig || inspection.tokenizerConfig) && (
        <details className='mt-3 rounded-lg border border-base-300 bg-base-100 p-3 text-xs'>
          <summary className='cursor-pointer font-semibold'>
            Optional runtime metadata
          </summary>
          {inspection.generationConfig && (
            <div className='mt-3'>
              <div className='font-semibold'>generation_config.json</div>
              <pre className='mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-[11px] text-base-content/75'>
                {JSON.stringify(inspection.generationConfig, null, 2)}
              </pre>
            </div>
          )}
          {inspection.tokenizerConfig && (
            <div className='mt-3'>
              <div className='font-semibold'>tokenizer_config.json</div>
              <pre className='mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-[11px] text-base-content/75'>
                {JSON.stringify(inspection.tokenizerConfig, null, 2)}
              </pre>
            </div>
          )}
        </details>
      )}

      {inspection.warnings.length > 0 && (
        <ul className='mt-4 space-y-1 rounded-lg bg-warning/10 p-3 text-xs text-warning-content'>
          {inspection.warnings.map((warning) => (
            <li key={warning}>• {warning}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
