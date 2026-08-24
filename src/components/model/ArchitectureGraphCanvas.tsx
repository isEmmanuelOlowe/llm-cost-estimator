import { useMemo, useRef, useState } from 'react';

import type { ArchitectureFlowInput } from '@/lib/model-architecture';
import {
  buildModelGraph,
  type ModelGraphEdge,
  type ModelGraphNode,
} from '@/lib/model-graph';

interface ArchitectureGraphCanvasProps {
  input: ArchitectureFlowInput;
  view: 'overview' | 'block';
  zoom: number;
  onZoomChange: (zoom: number) => void;
  selectedNodeId: string;
  onSelectNode: (id: string) => void;
  focusNodeId?: string;
  onFocusComplete?: () => void;
}

const colors: Record<
  ModelGraphNode['kind'],
  { fill: string; stroke: string; badge: string }
> = {
  input: {
    fill: 'var(--graph-input-fill)',
    stroke: 'var(--graph-input-stroke)',
    badge: 'var(--graph-input-stroke)',
  },
  normalization: {
    fill: 'var(--graph-normalization-fill)',
    stroke: 'var(--graph-normalization-stroke)',
    badge: 'var(--graph-normalization-stroke)',
  },
  attention: {
    fill: 'var(--graph-attention-fill)',
    stroke: 'var(--graph-attention-stroke)',
    badge: 'var(--graph-attention-stroke)',
  },
  routing: {
    fill: 'var(--graph-routing-fill)',
    stroke: 'var(--graph-routing-stroke)',
    badge: 'var(--graph-routing-stroke)',
  },
  mlp: {
    fill: 'var(--graph-mlp-fill)',
    stroke: 'var(--graph-mlp-stroke)',
    badge: 'var(--graph-mlp-stroke)',
  },
  residual: {
    fill: 'var(--graph-residual-fill)',
    stroke: 'var(--graph-residual-stroke)',
    badge: 'var(--graph-residual-stroke)',
  },
  output: {
    fill: 'var(--graph-output-fill)',
    stroke: 'var(--graph-output-stroke)',
    badge: 'var(--graph-output-stroke)',
  },
};

const graphViewportHeight = 820;

function buildingBlocks(node: ModelGraphNode): string[] {
  if (node.id === 'qkv') return ['query', 'key', 'value'];
  if (node.id === 'position') return ['rotate Q', 'rotate K'];
  if (node.id === 'attention') return ['scores', 'mask', 'softmax', 'mix'];
  if (node.id === 'router') return ['score', 'top-k', 'dispatch'];
  if (node.id === 'experts' || node.id === 'shared-experts') {
    return ['dispatch', 'expert MLP', 'merge'];
  }
  if (node.id === 'mlp') return ['gate', 'up', 'activate', 'down'];
  if (node.kind === 'normalization') return ['normalize', 'scale'];
  if (node.kind === 'residual') return ['project', 'add skip'];
  if (node.id === 'embedding') return ['lookup', 'residual'];
  if (node.kind === 'output') return ['normalize', 'project', 'logits'];
  if (node.kind === 'input') return ['encode', 'reshape'];
  return [];
}

function wrap(text: string, maxChars: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = '';
  for (const word of words) {
    if (`${line} ${word}`.trim().length > maxChars && line) {
      lines.push(line);
      line = word;
    } else {
      line = `${line} ${word}`.trim();
    }
  }
  if (line) lines.push(line);
  return lines.slice(0, 3);
}

function edgePath(
  from: ModelGraphNode,
  to: ModelGraphNode,
  kind: ModelGraphEdge['kind'],
): string {
  const fromCenter = {
    x: from.x + from.width / 2,
    y: from.y + from.height / 2,
  };
  const toCenter = { x: to.x + to.width / 2, y: to.y + to.height / 2 };
  const sameRow = Math.abs(fromCenter.y - toCenter.y) < 4;
  if (kind === 'residual' && sameRow) {
    const start = { x: from.x + from.width, y: fromCenter.y };
    const end = { x: toCenter.x, y: to.y };
    const railY = Math.min(from.y, to.y) - 50;
    return `M ${start.x} ${start.y} C ${start.x + 24} ${start.y}, ${start.x + 24} ${railY}, ${start.x + 48} ${railY} L ${end.x - 32} ${railY} C ${end.x - 8} ${railY}, ${end.x} ${railY}, ${end.x} ${end.y}`;
  }
  if (kind === 'residual') {
    const start = { x: fromCenter.x, y: from.y + from.height };
    const end = { x: toCenter.x, y: to.y };
    const railY = end.y - 38;
    return `M ${start.x} ${start.y} C ${start.x} ${railY}, ${start.x + 20} ${railY}, ${start.x + 44} ${railY} L ${end.x - 36} ${railY} C ${end.x - 12} ${railY}, ${end.x} ${railY}, ${end.x} ${end.y}`;
  }
  if (kind === 'shared') {
    const start = { x: from.x + from.width, y: fromCenter.y };
    const end = { x: to.x + to.width, y: toCenter.y };
    const railX = Math.max(start.x, end.x) + 48;
    return `M ${start.x} ${start.y} C ${railX} ${start.y}, ${railX} ${start.y}, ${railX} ${start.y + 24} L ${railX} ${end.y - 24} C ${railX} ${end.y}, ${railX} ${end.y}, ${end.x} ${end.y}`;
  }
  if (sameRow) {
    const rightward = toCenter.x >= fromCenter.x;
    const start = {
      x: rightward ? from.x + from.width : from.x,
      y: fromCenter.y,
    };
    const end = {
      x: rightward ? to.x : to.x + to.width,
      y: toCenter.y,
    };
    const bend = start.x + (end.x - start.x) / 2;
    return `M ${start.x} ${start.y} C ${bend} ${start.y}, ${bend} ${end.y}, ${end.x} ${end.y}`;
  }
  const downward = toCenter.y >= fromCenter.y;
  const start = {
    x: fromCenter.x,
    y: downward ? from.y + from.height : from.y,
  };
  const end = {
    x: toCenter.x,
    y: downward ? to.y : to.y + to.height,
  };
  if (Math.abs(end.x - start.x) < 4) {
    const middleY = start.y + (end.y - start.y) / 2;
    return `M ${start.x} ${start.y} C ${start.x} ${middleY}, ${end.x} ${middleY}, ${end.x} ${end.y}`;
  }
  const direction = downward ? 1 : -1;
  const railY =
    kind === 'flow' && to.id === 'attention-norm'
      ? to.y - 24
      : start.y + ((end.y - start.y) / 2 || 50 * direction);
  const corner = 24;
  return `M ${start.x} ${start.y} C ${start.x} ${start.y + corner * direction}, ${start.x} ${railY}, ${start.x + (end.x >= start.x ? corner : -corner)} ${railY} L ${end.x + (end.x >= start.x ? -corner : corner)} ${railY} C ${end.x} ${railY}, ${end.x} ${end.y - corner * direction}, ${end.x} ${end.y}`;
}

function GraphNode({
  node,
  selected,
  position,
  onSelect,
  onPointerDown,
  showInternals,
}: {
  node: ModelGraphNode;
  selected: boolean;
  position: { x: number; y: number };
  onSelect: () => void;
  onPointerDown: (event: React.PointerEvent<SVGGElement>) => void;
  showInternals: boolean;
}) {
  const color = colors[node.kind];
  const titleLines = wrap(node.label, 20).slice(0, 2);
  const shapeLines = wrap(node.shape, 31);
  const internalStages = buildingBlocks(node);
  return (
    <g
      role='button'
      tabIndex={0}
      aria-label={`Select ${node.label}`}
      transform={`translate(${position.x} ${position.y})`}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') onSelect();
      }}
      onPointerDown={onPointerDown}
      style={{ cursor: 'grab' }}
    >
      <rect
        width={node.width}
        height={node.height}
        rx='16'
        fill={color.fill}
        stroke={selected ? 'var(--graph-selection)' : color.stroke}
        strokeWidth={selected ? 4 : 2}
      />
      <rect width={node.width} height='6' rx='3' fill={color.badge} />
      {titleLines.map((line, index) => (
        <text
          key={`title-${line}`}
          x='16'
          y={30 + index * 19}
          fill='var(--graph-text)'
          fontSize='15'
          fontWeight='700'
        >
          {line}
        </text>
      ))}
      {showInternals && internalStages.length > 0 ? (
        <g aria-label={`${node.label} internal building blocks`}>
          {internalStages.map((stage, index) => {
            const gap = 5;
            const innerWidth = node.width - 32;
            const width =
              (innerWidth - gap * (internalStages.length - 1)) /
              internalStages.length;
            return (
              <g
                key={`${node.id}-${stage}`}
                transform={`translate(${16 + index * (width + gap)} 78)`}
              >
                <rect
                  width={width}
                  height='28'
                  rx='7'
                  fill='var(--graph-inner)'
                  stroke={color.stroke}
                  strokeOpacity='0.75'
                />
                <text
                  x={width / 2}
                  y='18'
                  textAnchor='middle'
                  fill='var(--graph-text)'
                  fontSize='8.5'
                  fontWeight='600'
                >
                  {stage}
                </text>
              </g>
            );
          })}
        </g>
      ) : (
        shapeLines.map((line, index) => (
          <text
            key={`shape-${line}`}
            x='16'
            y={72 + index * 14}
            fill='var(--graph-muted)'
            fontSize='10'
            fontFamily='ui-monospace, monospace'
          >
            {line}
          </text>
        ))
      )}
    </g>
  );
}

export default function ArchitectureGraphCanvas({
  input,
  view,
  zoom,
  onZoomChange,
  selectedNodeId,
  onSelectNode,
  focusNodeId,
  onFocusComplete,
}: ArchitectureGraphCanvasProps) {
  const graphKey = useMemo(
    () => JSON.stringify({ input, view }),
    [input, view],
  );
  const graph = useMemo(() => buildModelGraph(input, view), [input, view]);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const defaultPositions = useMemo(
    () =>
      Object.fromEntries(
        graph.nodes.map((node) => [node.id, { x: node.x, y: node.y }]),
      ),
    [graph],
  );
  const [positionState, setPositionState] = useState<{
    key: string;
    value: Record<string, { x: number; y: number }>;
  }>({ key: '', value: {} });
  const positions =
    positionState.key === graphKey ? positionState.value : defaultPositions;
  const [panState, setPanState] = useState<{
    key: string;
    value: { x: number; y: number };
  }>({ key: '', value: { x: 0, y: 0 } });
  const storedPan = panState.key === graphKey ? panState.value : { x: 0, y: 0 };
  const viewWidth = graph.width / zoom;
  const viewHeight = graphViewportHeight / zoom;
  const maxPanX = Math.max(0, graph.width - viewWidth);
  const maxPanY = Math.max(0, graph.height - viewHeight);
  const interaction = useRef<
    | {
        type: 'pan';
        startX: number;
        startY: number;
        panX: number;
        panY: number;
      }
    | {
        type: 'node';
        id: string;
        startX: number;
        startY: number;
        x: number;
        y: number;
      }
    | null
  >(null);

  const getPosition = (node: ModelGraphNode) =>
    positions[node.id] ?? { x: node.x, y: node.y };
  const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
  const focusedNode = focusNodeId
    ? graph.nodes.find((node) => node.id === focusNodeId)
    : undefined;
  const focusPan = focusedNode
    ? {
        x: Math.max(
          0,
          Math.min(
            maxPanX,
            getPosition(focusedNode).x + focusedNode.width / 2 - viewWidth / 2,
          ),
        ),
        y: Math.max(
          0,
          Math.min(
            maxPanY,
            getPosition(focusedNode).y +
              focusedNode.height / 2 -
              viewHeight / 2,
          ),
        ),
      }
    : undefined;
  const pan = focusPan ?? storedPan;
  const commitFocus = () => {
    if (!focusPan) return;
    setPanState({ key: graphKey, value: focusPan });
    onFocusComplete?.();
  };
  const clampPan = (candidate: { x: number; y: number }) => ({
    x: Math.max(0, Math.min(maxPanX, candidate.x)),
    y: Math.max(0, Math.min(maxPanY, candidate.y)),
  });

  const zoomAt = (
    nextZoom: number,
    anchor?: { clientX: number; clientY: number },
  ) => {
    const boundedZoom = Math.max(0.5, Math.min(2, nextZoom));
    const svg = svgRef.current;
    if (!svg || !anchor) {
      onZoomChange(boundedZoom);
      return;
    }
    const rect = svg.getBoundingClientRect();
    const ratioX = Math.max(
      0,
      Math.min(1, (anchor.clientX - rect.left) / rect.width),
    );
    const ratioY = Math.max(
      0,
      Math.min(1, (anchor.clientY - rect.top) / rect.height),
    );
    const graphPoint = {
      x: pan.x + ratioX * viewWidth,
      y: pan.y + ratioY * viewHeight,
    };
    const nextViewWidth = graph.width / boundedZoom;
    const nextViewHeight = graphViewportHeight / boundedZoom;
    setPanState({
      key: graphKey,
      value: {
        x: Math.max(
          0,
          Math.min(
            Math.max(0, graph.width - nextViewWidth),
            graphPoint.x - ratioX * nextViewWidth,
          ),
        ),
        y: Math.max(
          0,
          Math.min(
            Math.max(0, graph.height - nextViewHeight),
            graphPoint.y - ratioY * nextViewHeight,
          ),
        ),
      },
    });
    onZoomChange(boundedZoom);
  };

  const toGraphPoint = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const rect = svg.getBoundingClientRect();
    return {
      x: pan.x + ((clientX - rect.left) / rect.width) * viewWidth,
      y: pan.y + ((clientY - rect.top) / rect.height) * viewHeight,
    };
  };

  const handlePointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const current = interaction.current;
    if (!current) return;
    if (current.type === 'pan') {
      const point = toGraphPoint(event.clientX, event.clientY);
      const start = toGraphPoint(current.startX, current.startY);
      setPanState({
        key: graphKey,
        value: clampPan({
          x: current.panX - (point.x - start.x),
          y: current.panY - (point.y - start.y),
        }),
      });
      return;
    }
    const point = toGraphPoint(event.clientX, event.clientY);
    setPositionState((previous) => ({
      key: graphKey,
      value: {
        ...(previous.key === graphKey ? previous.value : defaultPositions),
        [current.id]: {
          x:
            current.x +
            point.x -
            toGraphPoint(current.startX, current.startY).x,
          y:
            current.y +
            point.y -
            toGraphPoint(current.startX, current.startY).y,
        },
      },
    }));
  };

  const handlePointerUp = () => {
    interaction.current = null;
  };

  return (
    <div
      className='overflow-hidden rounded-2xl border border-base-300 bg-[var(--graph-canvas)]'
      data-testid='architecture-graph-canvas'
    >
      <div className='flex flex-wrap items-center justify-between gap-3 border-b border-base-300 px-4 py-3 text-xs text-base-content/70'>
        <span>
          Drag to pan · Ctrl/⌘ + wheel to zoom · ordinary scroll moves the page
        </span>
        <button
          type='button'
          className='btn btn-ghost btn-xs'
          onClick={() => {
            setPanState({ key: graphKey, value: { x: 0, y: 0 } });
            onZoomChange(1);
            setPositionState({ key: graphKey, value: defaultPositions });
            onFocusComplete?.();
          }}
        >
          Reset graph
        </button>
      </div>
      <div className='relative'>
        <div className='overflow-auto'>
          <svg
            ref={svgRef}
            className='block h-[38rem] min-w-[860px] w-full touch-none select-none'
            viewBox={`${pan.x} ${pan.y} ${viewWidth} ${viewHeight}`}
            preserveAspectRatio='xMidYMin meet'
            role='application'
            aria-label='Interactive model architecture graph'
            onWheel={(event) => {
              if (!event.ctrlKey && !event.metaKey) return;
              event.preventDefault();
              zoomAt(zoom + (event.deltaY < 0 ? 0.05 : -0.05), event);
            }}
            onPointerDown={(event) => {
              commitFocus();
              interaction.current = {
                type: 'pan',
                startX: event.clientX,
                startY: event.clientY,
                panX: pan.x,
                panY: pan.y,
              };
            }}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onPointerLeave={handlePointerUp}
          >
            <rect
              width={graph.width}
              height={graph.height}
              fill='var(--graph-canvas)'
            />
            {graph.blockBounds && (
              <>
                <rect
                  x={graph.blockBounds.x}
                  y={graph.blockBounds.y}
                  width={graph.blockBounds.width}
                  height={graph.blockBounds.height}
                  rx='24'
                  fill='var(--graph-block)'
                  stroke='var(--graph-block-stroke)'
                  strokeDasharray='8 8'
                />
                <text
                  x={graph.blockBounds.x + 24}
                  y={graph.blockBounds.y + 32}
                  fill='var(--graph-text)'
                  fontSize='17'
                  fontWeight='700'
                >
                  One transformer block
                </text>
                <g
                  transform={`translate(${graph.blockBounds.x + graph.blockBounds.width - 142} ${graph.blockBounds.y - 22})`}
                >
                  <rect
                    width='126'
                    height='38'
                    rx='19'
                    fill='var(--color-primary)'
                    stroke='var(--color-secondary)'
                  />
                  <text
                    x='63'
                    y='25'
                    fill='var(--color-primary-content)'
                    fontSize='15'
                    fontWeight='800'
                    textAnchor='middle'
                  >
                    × {input.numLayers} layers
                  </text>
                </g>
              </>
            )}
            {graph.edges.map((edge) => {
              const from = nodeMap.get(edge.from);
              const to = nodeMap.get(edge.to);
              if (!from || !to) return null;
              const positionedFrom = { ...from, ...getPosition(from) };
              const positionedTo = { ...to, ...getPosition(to) };
              const stroke =
                edge.kind === 'residual'
                  ? 'var(--graph-residual)'
                  : edge.kind === 'shared'
                    ? 'var(--graph-shared)'
                    : 'var(--graph-flow)';
              return (
                <g key={edge.id} pointerEvents='none'>
                  <path
                    d={edgePath(positionedFrom, positionedTo, edge.kind)}
                    fill='none'
                    stroke={stroke}
                    strokeWidth={edge.kind === 'flow' ? 3 : 2}
                    strokeDasharray={edge.kind === 'flow' ? undefined : '7 6'}
                    markerEnd={`url(#arrow-${edge.kind})`}
                  />
                </g>
              );
            })}
            <defs>
              <marker
                id='arrow-flow'
                markerWidth='12'
                markerHeight='12'
                refX='10'
                refY='6'
                orient='auto'
                markerUnits='userSpaceOnUse'
              >
                <path d='M 0 0 L 12 6 L 0 12 z' fill='var(--graph-flow)' />
              </marker>
              <marker
                id='arrow-residual'
                markerWidth='12'
                markerHeight='12'
                refX='10'
                refY='6'
                orient='auto'
                markerUnits='userSpaceOnUse'
              >
                <path d='M 0 0 L 12 6 L 0 12 z' fill='var(--graph-residual)' />
              </marker>
              <marker
                id='arrow-shared'
                markerWidth='12'
                markerHeight='12'
                refX='10'
                refY='6'
                orient='auto'
                markerUnits='userSpaceOnUse'
              >
                <path d='M 0 0 L 12 6 L 0 12 z' fill='var(--graph-shared)' />
              </marker>
            </defs>
            {graph.nodes.map((node) => (
              <GraphNode
                key={node.id}
                node={node}
                position={getPosition(node)}
                selected={selectedNodeId === node.id}
                onSelect={() => onSelectNode(node.id)}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  commitFocus();
                  const position = getPosition(node);
                  interaction.current = {
                    type: 'node',
                    id: node.id,
                    startX: event.clientX,
                    startY: event.clientY,
                    x: position.x,
                    y: position.y,
                  };
                  event.currentTarget.setPointerCapture?.(event.pointerId);
                }}
                showInternals={zoom >= 1.1}
              />
            ))}
          </svg>
        </div>
        <div
          className='absolute bottom-3 right-3 top-3 z-10 flex w-8 flex-col items-center gap-2 rounded-full border border-base-300 bg-base-200/90 px-1.5 py-2 shadow-lg backdrop-blur'
          title='Scroll through the graph or drag the canvas'
        >
          <span aria-hidden='true' className='text-xs font-bold text-secondary'>
            ↕
          </span>
          <label className='sr-only' htmlFor='architecture-vertical-position'>
            Graph vertical position
          </label>
          <input
            id='architecture-vertical-position'
            className='graph-scrollbar min-h-24 flex-1'
            type='range'
            min='0'
            max={Math.max(1, maxPanY)}
            step='1'
            value={Math.min(maxPanY, pan.y)}
            disabled={maxPanY <= 0}
            onPointerDown={commitFocus}
            onChange={(event) => {
              setPanState({
                key: graphKey,
                value: clampPan({ x: pan.x, y: Number(event.target.value) }),
              });
              onFocusComplete?.();
            }}
          />
        </div>
      </div>
    </div>
  );
}
