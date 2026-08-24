import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { resolveModelLicense } from '@/lib/model-license';

interface ModelLicenseBadgeProps {
  modelId: string;
  license?: string | null;
  modelUrl: string;
}

export default function ModelLicenseBadge({
  modelId,
  license,
  modelUrl,
}: ModelLicenseBadgeProps) {
  const [open, setOpen] = useState(false);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const info = useMemo(
    () => resolveModelLicense({ modelId, license, modelUrl }),
    [license, modelId, modelUrl],
  );

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('keydown', handleKeyDown);
    const frame = window.requestAnimationFrame(() => closeRef.current?.focus());
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      window.cancelAnimationFrame(frame);
    };
  }, [open]);

  return (
    <>
      <button
        type='button'
        className='badge badge-success badge-outline cursor-pointer whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-secondary'
        aria-haspopup='dialog'
        onClick={() => setOpen(true)}
      >
        {info.label} · details
      </button>
      {open &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            className='fixed inset-0 z-[110] grid place-items-center bg-black/65 p-4 backdrop-blur-sm'
            role='presentation'
            onMouseDown={(event) => {
              if (event.target === event.currentTarget) setOpen(false);
            }}
          >
            <section
              role='dialog'
              aria-modal='true'
              aria-labelledby='model-license-title'
              className='w-full max-w-lg rounded-2xl border border-base-300 bg-[var(--graph-block)] p-5 text-left shadow-2xl'
            >
              <div className='flex items-start justify-between gap-4'>
                <div>
                  <div className='text-[10px] font-semibold uppercase tracking-[0.16em] text-secondary'>
                    Model license
                  </div>
                  <h3
                    id='model-license-title'
                    className='mt-1 text-xl font-bold'
                  >
                    {info.label}
                  </h3>
                </div>
                <button
                  ref={closeRef}
                  type='button'
                  className='btn btn-sm btn-outline'
                  aria-label='Close model license guidance'
                  onClick={() => setOpen(false)}
                >
                  Close ×
                </button>
              </div>
              <p className='mt-3 text-sm leading-relaxed text-base-content/70'>
                {info.summary}
              </p>
              <p className='mt-4 rounded-xl bg-base-200 p-4 text-sm leading-relaxed text-base-content/80'>
                {info.usage}
              </p>
              <div className='mt-4 flex flex-wrap items-center justify-between gap-3'>
                <span className='text-[11px] text-base-content/50'>
                  Practical summary only—not legal advice.
                </span>
                <a
                  className='link link-primary text-sm'
                  href={info.sourceUrl}
                  target='_blank'
                  rel='noreferrer'
                >
                  Read the full license text ↗
                </a>
              </div>
            </section>
          </div>,
          document.body,
        )}
    </>
  );
}
