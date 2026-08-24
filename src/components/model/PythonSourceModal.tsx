import { useEffect, useMemo, useRef, useState } from 'react';

import {
  extractPythonExcerpt,
  findImportedPythonSource,
} from '@/lib/python-source';

interface PythonSourceModalProps {
  open: boolean;
  nodeId: string;
  nodeLabel: string;
  source: {
    name: string;
    url: string;
    content: string;
  };
  onClose: () => void;
}

const TOKEN_PATTERN =
  /(#.*$|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b(?:class|def|return|if|else|elif|for|while|in|import|from|as|with|yield|raise|try|except|finally|None|True|False|self|super|async|await)\b|\b\d+(?:\.\d+)?\b)/g;
const TOKEN_EXACT_PATTERN =
  /^(#.*|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b(?:class|def|return|if|else|elif|for|while|in|import|from|as|with|yield|raise|try|except|finally|None|True|False|self|super|async|await)\b|\b\d+(?:\.\d+)?\b)$/;

function tokenClass(token: string): string {
  if (token.startsWith('#')) return 'text-base-content/45 italic';
  if (token.startsWith('"') || token.startsWith("'")) return 'text-lab-green';
  if (/^\d/.test(token)) return 'text-lab-amber';
  return 'text-lab-aqua';
}

function HighlightedLine({ line }: { line: string }) {
  const parts = line.split(TOKEN_PATTERN);
  return (
    <>
      {parts.map((part, index) => {
        return TOKEN_EXACT_PATTERN.test(part) ? (
          <span key={`${index}-${part}`} className={tokenClass(part)}>
            {part}
          </span>
        ) : (
          <span key={`${index}-${part}`}>{part}</span>
        );
      })}
    </>
  );
}

export default function PythonSourceModal({
  open,
  nodeId,
  nodeLabel,
  source,
  onClose,
}: PythonSourceModalProps) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const sourceKey = `${source.url}:${nodeId}`;
  const importedSource = useMemo(
    () =>
      findImportedPythonSource(source.content, nodeId, nodeLabel, source.url),
    [nodeId, nodeLabel, source.content, source.url],
  );
  const [resolvedSource, setResolvedSource] = useState<{
    key: string;
    source: PythonSourceModalProps['source'];
  }>();
  const displayedSource =
    resolvedSource?.key === sourceKey ? resolvedSource.source : source;
  const excerpt = useMemo(
    () => extractPythonExcerpt(displayedSource.content, nodeId, nodeLabel),
    [displayedSource.content, nodeId, nodeLabel],
  );

  useEffect(() => {
    if (!open || !importedSource || resolvedSource?.key === sourceKey) return;
    let cancelled = false;
    fetch(importedSource.rawUrl)
      .then((response) => {
        if (!response.ok)
          throw new Error(`Source request failed: ${response.status}`);
        return response.text();
      })
      .then((content) => {
        if (cancelled) return;
        setResolvedSource({
          key: sourceKey,
          source: {
            name: importedSource.name,
            url: importedSource.url,
            content,
          },
        });
      })
      .catch(() => {
        // Keep the already loaded module excerpt when the imported file is unavailable.
      });
    return () => {
      cancelled = true;
    };
  }, [importedSource, open, resolvedSource?.key, sourceKey]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    const frame = window.requestAnimationFrame(() => closeRef.current?.focus());
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      window.cancelAnimationFrame(frame);
    };
  }, [onClose, open]);

  if (!open) return null;
  const lines = excerpt.content.split('\n');

  return (
    <div
      className='fixed inset-0 z-[100] grid place-items-center bg-black/70 p-3 backdrop-blur-sm sm:p-6'
      role='presentation'
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        role='dialog'
        aria-modal='true'
        aria-labelledby='python-source-title'
        className='flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-base-300 bg-base-100 shadow-2xl'
      >
        <header className='flex items-start justify-between gap-4 border-b border-base-300 px-4 py-3 sm:px-5'>
          <div className='min-w-0'>
            <div className='text-[10px] font-semibold uppercase tracking-[0.18em] text-secondary'>
              Python implementation · lines {excerpt.startLine}–
              {excerpt.endLine}
            </div>
            <h3
              id='python-source-title'
              className='mt-1 truncate text-lg font-bold'
            >
              {nodeLabel}
            </h3>
            <p className='mt-1 truncate text-xs text-base-content/55'>
              {displayedSource.name}
              {excerpt.matchedPattern
                ? ` · focused near “${excerpt.matchedPattern}”`
                : ' · nearest available excerpt'}
            </p>
          </div>
          <button
            ref={closeRef}
            type='button'
            className='btn btn-sm btn-outline shrink-0'
            aria-label='Close implementation source'
            onClick={onClose}
          >
            Close ×
          </button>
        </header>

        <div className='min-h-0 flex-1 overflow-auto bg-[var(--graph-canvas)]'>
          <pre className='min-w-max p-4 text-[11px] leading-5 text-[var(--graph-text)] sm:p-5 sm:text-xs'>
            <code>
              {lines.map((line, index) => (
                <span
                  key={`${excerpt.startLine + index}-${line}`}
                  className='block'
                >
                  <span className='mr-4 inline-block w-10 select-none text-right text-[var(--graph-muted)]/55'>
                    {excerpt.startLine + index}
                  </span>
                  <HighlightedLine line={line || ' '} />
                </span>
              ))}
            </code>
          </pre>
        </div>

        <footer className='flex flex-wrap items-center justify-between gap-3 border-t border-base-300 px-4 py-3 text-xs text-base-content/60 sm:px-5'>
          <span>
            Read-only excerpt · press Escape or click outside to close
          </span>
          <a
            className='link link-primary'
            href={displayedSource.url}
            target='_blank'
            rel='noreferrer'
          >
            Open full upstream file ↗
          </a>
        </footer>
      </section>
    </div>
  );
}
