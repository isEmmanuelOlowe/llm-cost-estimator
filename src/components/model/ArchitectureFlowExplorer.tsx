import { useMemo, useState } from 'react';

import {
  type ArchitectureFlowInput,
  type ArchitectureFlowNode,
  buildArchitectureFlow,
  buildArchitectureOverview,
} from '@/lib/model-architecture';

import ArchitectureGraphCanvas from './ArchitectureGraphCanvas';

interface ArchitectureFlowExplorerProps extends ArchitectureFlowInput {
  architectureLabel?: string;
  sourcePreview?: {
    name: string;
    url: string;
    content: string;
  };
  onLoadImplementation?: () => void;
  isLoadingImplementation?: boolean;
}

function nodeKey(node: ArchitectureFlowNode): string {
  return `${node.id}:${node.label}:${node.shape}:${node.sourceFile ?? ''}`;
}

export default function ArchitectureFlowExplorer({
  architectureLabel,
  sourcePreview,
  onLoadImplementation,
  isLoadingImplementation = false,
  ...input
}: ArchitectureFlowExplorerProps) {
  const [view, setView] = useState<'overview' | 'block'>('block');
  const [zoom, setZoom] = useState(1);
  const [selectedNodeId, setSelectedNodeId] = useState('attention');
  const [focusNodeId, setFocusNodeId] = useState<string | undefined>(
    'attention',
  );
  const nodes = useMemo(() => {
    if (view === 'overview') return buildArchitectureOverview(input);
    return buildArchitectureFlow(input).filter((node) => node.id !== 'repeat');
  }, [input, view]);
  const activeNodeId = nodes.some((node) => node.id === selectedNodeId)
    ? selectedNodeId
    : (nodes[0]?.id ?? '');
  const selectedNode = nodes.find((node) => node.id === activeNodeId);

  return (
    <div className='mt-5' data-testid='architecture-flow-explorer'>
      <div className='flex flex-col gap-4 rounded-xl border border-base-300 bg-base-200 p-4 xl:flex-row xl:items-center xl:justify-between'>
        <div className='min-w-0'>
          <div className='flex flex-wrap items-center gap-2'>
            <h3 className='text-lg font-semibold'>Architecture workspace</h3>
            {architectureLabel && (
              <span className='badge badge-outline max-w-full truncate text-xs'>
                {architectureLabel}
              </span>
            )}
            {input.modalityArchitecture && (
              <span className='badge badge-success badge-outline text-xs'>
                Media path verified
              </span>
            )}
          </div>
          <p className='mt-1 text-xs text-base-content/65'>
            Overview first, then inspect one repeated block. At 110% zoom,
            components reveal their internal operations.
          </p>
        </div>

        <div className='flex flex-wrap items-center gap-2'>
          <div className='join' aria-label='Architecture graph view'>
            <button
              type='button'
              className={`btn btn-xs join-item ${view === 'overview' ? 'btn-primary' : 'btn-ghost'}`}
              aria-pressed={view === 'overview'}
              onClick={() => setView('overview')}
            >
              Overview
            </button>
            <button
              type='button'
              className={`btn btn-xs join-item ${view === 'block' ? 'btn-primary' : 'btn-ghost'}`}
              aria-pressed={view === 'block'}
              onClick={() => setView('block')}
            >
              Inside one block
            </button>
          </div>
          <button
            type='button'
            className='btn btn-xs btn-outline'
            aria-label='Zoom out'
            onClick={() => setZoom((value) => Math.max(0.5, value - 0.1))}
          >
            −
          </button>
          <input
            className='range range-primary range-xs w-24'
            type='range'
            min='0.5'
            max='2'
            step='0.05'
            value={zoom}
            aria-label='Architecture graph zoom'
            onChange={(event) => setZoom(Number(event.target.value))}
          />
          <button
            type='button'
            className='btn btn-xs btn-outline'
            aria-label='Zoom in'
            onClick={() => setZoom((value) => Math.min(2, value + 0.1))}
          >
            +
          </button>
          <span className='w-10 text-right text-xs tabular-nums text-base-content/65'>
            {Math.round(zoom * 100)}%
          </span>
          <label className='sr-only' htmlFor='architecture-component-select'>
            Focus graph component
          </label>
          <select
            id='architecture-component-select'
            className='select select-bordered select-xs max-w-52'
            value={activeNodeId}
            onChange={(event) => {
              setSelectedNodeId(event.target.value);
              setFocusNodeId(event.target.value);
            }}
          >
            {nodes.map((node) => (
              <option key={nodeKey(node)} value={node.id}>
                {node.label}
              </option>
            ))}
          </select>
          <button
            type='button'
            className='btn btn-xs btn-outline'
            onClick={() => setFocusNodeId(activeNodeId)}
          >
            Center
          </button>
        </div>
      </div>

      <div className='mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_20rem]'>
        <ArchitectureGraphCanvas
          input={input}
          view={view}
          zoom={zoom}
          onZoomChange={setZoom}
          selectedNodeId={activeNodeId}
          onSelectNode={(id) => {
            setSelectedNodeId(id);
            setFocusNodeId(id);
          }}
          focusNodeId={focusNodeId}
          onFocusComplete={() => setFocusNodeId(undefined)}
        />

        <aside className='rounded-xl border border-base-300 bg-base-200 p-4 xl:min-h-[38rem]'>
          <div className='flex flex-wrap items-center justify-between gap-2'>
            <h4 className='font-semibold'>Component detail</h4>
            <span className='text-[10px] uppercase tracking-wide text-base-content/50'>
              selected
            </span>
          </div>
          {selectedNode ? (
            <>
              <div className='mt-4 text-base font-semibold'>
                {selectedNode.label}
              </div>
              <div className='mt-3 rounded-lg border border-base-300 bg-base-100 p-3 font-mono text-[11px] leading-relaxed'>
                {selectedNode.shape}
              </div>
              <p className='mt-3 text-xs leading-relaxed text-base-content/75'>
                {selectedNode.detail}
              </p>
              {selectedNode.sourceUrl && (
                <a
                  className='link link-primary mt-4 inline-block text-xs'
                  href={selectedNode.sourceUrl}
                  target='_blank'
                  rel='noreferrer'
                >
                  Open {selectedNode.sourceFile ?? 'implementation'} ↗
                </a>
              )}
              {!sourcePreview && onLoadImplementation && (
                <button
                  type='button'
                  className='btn btn-primary btn-xs mt-3 block'
                  disabled={isLoadingImplementation}
                  onClick={onLoadImplementation}
                >
                  {isLoadingImplementation
                    ? 'Loading implementation…'
                    : 'Load implementation source'}
                </button>
              )}
              {sourcePreview &&
                selectedNode.sourceFile === sourcePreview.name && (
                  <details className='mt-4 rounded-lg border border-base-300 bg-base-100 p-3'>
                    <summary className='cursor-pointer text-xs font-semibold'>
                      Code: {sourcePreview.name}
                    </summary>
                    <pre className='mt-3 max-h-64 overflow-auto whitespace-pre-wrap break-words text-[10px] leading-relaxed text-base-content/75'>
                      {sourcePreview.content}
                    </pre>
                  </details>
                )}
            </>
          ) : (
            <p className='mt-3 text-xs text-base-content/70'>
              Select a node to inspect it.
            </p>
          )}
          <div className='mt-5 border-t border-base-300 pt-4 text-[11px] leading-relaxed text-base-content/60'>
            <div className='flex items-center gap-2'>
              <span className='h-0 w-6 border-t-2 border-slate-400' /> Data flow
            </div>
            <div className='mt-2 flex items-center gap-2'>
              <span className='h-0 w-6 border-t-2 border-dashed border-amber-500' />
              Residual path
            </div>
            <div className='mt-2 flex items-center gap-2'>
              <span className='h-0 w-6 border-t-2 border-dashed border-violet-400' />
              Shared expert path
            </div>
            <p className='mt-4'>
              B=batch · S=sequence · H=hidden · D=head dimension ·
              I=feed-forward width
            </p>
          </div>
        </aside>
      </div>

      <details className='mt-3 rounded-xl border border-base-300 bg-base-200 p-4'>
        <summary className='cursor-pointer text-sm font-semibold'>
          Component inventory and implementation links
        </summary>
        <div className='mt-3 overflow-x-auto'>
          <table className='table table-zebra table-sm text-xs'>
            <thead>
              <tr>
                <th>Component</th>
                <th>Tensor shape / role</th>
                <th>Implementation</th>
              </tr>
            </thead>
            <tbody>
              {nodes.map((node) => (
                <tr key={`table-${nodeKey(node)}`}>
                  <td className='whitespace-nowrap font-semibold'>
                    {node.label}
                  </td>
                  <td className='min-w-56 font-mono text-[11px] text-base-content/70'>
                    {node.shape}
                  </td>
                  <td>
                    {node.sourceUrl ? (
                      <a
                        className='link link-primary'
                        href={node.sourceUrl}
                        target='_blank'
                        rel='noreferrer'
                      >
                        {node.sourceFile ?? 'Source'} ↗
                      </a>
                    ) : (
                      'Config-derived'
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
