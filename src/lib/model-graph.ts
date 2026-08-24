import {
  type ArchitectureFlowInput,
  type ArchitectureFlowNode,
  buildArchitectureFlow,
  buildArchitectureOverview,
} from './model-architecture';

export type GraphEdgeKind = 'flow' | 'residual' | 'shared';

export interface ModelGraphNode extends ArchitectureFlowNode {
  x: number;
  y: number;
  width: number;
  height: number;
  annotation?: boolean;
}

export interface ModelGraphEdge {
  id: string;
  from: string;
  to: string;
  kind: GraphEdgeKind;
  label?: string;
}

export interface ModelGraph {
  nodes: ModelGraphNode[];
  edges: ModelGraphEdge[];
  width: number;
  height: number;
  blockBounds?: { x: number; y: number; width: number; height: number };
}

const GRAPH_WIDTH = 1500;
const NODE_WIDTH = 236;
const NODE_HEIGHT = 126;
const VERTICAL_GAP = 20;

function findNode(
  flow: ArchitectureFlowNode[],
  id: string,
): ArchitectureFlowNode | undefined {
  return flow.find((node) => node.id === id);
}

function placeNode(
  flow: ArchitectureFlowNode[],
  id: string,
  x: number,
  y: number,
  options: Partial<
    Pick<ModelGraphNode, 'width' | 'height' | 'annotation'>
  > = {},
): ModelGraphNode | undefined {
  const node = findNode(flow, id);
  if (!node) return undefined;
  return {
    ...node,
    x,
    y,
    width: options.width ?? NODE_WIDTH,
    height: options.height ?? NODE_HEIGHT,
    annotation: options.annotation,
  };
}

function addSequentialEdges(edges: ModelGraphEdge[], ids: string[]) {
  ids.forEach((id, index) => {
    const next = ids[index + 1];
    if (next) {
      edges.push({
        id: `flow-${id}-${next}`,
        from: id,
        to: next,
        kind: 'flow',
      });
    }
  });
}

export function buildModelGraph(
  input: ArchitectureFlowInput,
  view: 'overview' | 'block',
): ModelGraph {
  const flow = buildArchitectureFlow(input);
  const centerX = (GRAPH_WIDTH - NODE_WIDTH) / 2;

  if (view === 'overview') {
    const overview = buildArchitectureOverview(input);
    const nodes = overview.map((node, index) => ({
      ...node,
      x: centerX,
      y: 60 + index * (NODE_HEIGHT + VERTICAL_GAP),
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
    }));
    const edges: ModelGraphEdge[] = [];
    addSequentialEdges(
      edges,
      overview.map((node) => node.id),
    );
    return {
      nodes,
      edges,
      width: GRAPH_WIDTH,
      height: Math.max(
        700,
        120 + overview.length * (NODE_HEIGHT + VERTICAL_GAP),
      ),
    };
  }

  const nodes: ModelGraphNode[] = [];
  const edges: ModelGraphEdge[] = [];
  const topStart = 32;
  const columnX = {
    left: 40,
    center: centerX,
    right: GRAPH_WIDTH - NODE_WIDTH - 40,
  };
  const addNode = (id: string, x: number, y: number) => {
    const node = placeNode(flow, id, x, y);
    if (node) nodes.push(node);
    return node;
  };
  const addEdge = (
    from: string,
    to: string,
    kind: GraphEdgeKind = 'flow',
    label?: string,
  ) => {
    edges.push({
      id: `${kind}-${from}-${to}`,
      from,
      to,
      kind,
      ...(label ? { label } : {}),
    });
  };
  const isMultimodal = Boolean(input.modality && input.modality !== 'text');
  let flowIntoBlock = 'embedding';
  let topBottom = topStart + NODE_HEIGHT;

  if (isMultimodal) {
    addNode('input', columnX.left, topStart);
    addNode('embedding', columnX.left, topStart + NODE_HEIGHT + VERTICAL_GAP);
    addEdge('input', 'embedding');

    const modalityArchitecture = input.modalityArchitecture;
    if (modalityArchitecture?.vision) {
      const imageInputY = topStart;
      const hasVideo = modalityArchitecture.video;
      const visionStageIds = modalityArchitecture.vision.encoderFree
        ? ['vision-patch-embed', 'vision-position', 'vision-projector']
        : ['vision-encoder', 'vision-projector'];
      addNode(
        'image-input',
        hasVideo ? columnX.center - 190 : columnX.center,
        imageInputY,
      );
      visionStageIds.forEach((id, index) => {
        addNode(
          id,
          columnX.center,
          topStart + (index + 1) * (NODE_HEIGHT + VERTICAL_GAP),
        );
        if (index > 0) addEdge(visionStageIds[index - 1], id);
      });
      addEdge('image-input', visionStageIds[0]);

      if (hasVideo) {
        addNode('video-input', columnX.center + 170, imageInputY);
        addEdge('video-input', visionStageIds[0]);
      }
      const visionOutput = visionStageIds.at(-1);
      const visionOutputNode = visionOutput
        ? nodes.find((node) => node.id === visionOutput)
        : undefined;
      topBottom = visionOutputNode
        ? visionOutputNode.y + NODE_HEIGHT
        : topStart + NODE_HEIGHT;
    } else if (findNode(flow, 'modality-input')) {
      const modalityInputY = topStart;
      const modalityProjectorY = topStart + NODE_HEIGHT + VERTICAL_GAP;
      addNode('modality-input', columnX.center, modalityInputY);
      addNode('modality-projector', columnX.center, modalityProjectorY);
      addEdge('modality-input', 'modality-projector');
      topBottom = modalityProjectorY + NODE_HEIGHT;
    }

    if (modalityArchitecture?.audio) {
      const audioInputY = topStart;
      const audioProjectorY = topStart + NODE_HEIGHT + VERTICAL_GAP;
      addNode('audio-input', columnX.right, audioInputY);
      addNode('audio-projector', columnX.right, audioProjectorY);
      addEdge('audio-input', 'audio-projector');
      topBottom = Math.max(topBottom, audioProjectorY + NODE_HEIGHT);
    }

    const mediaOutputs = [
      modalityArchitecture?.vision ? 'vision-projector' : undefined,
      modalityArchitecture?.audio ? 'audio-projector' : undefined,
      !modalityArchitecture?.vision && findNode(flow, 'modality-projector')
        ? 'modality-projector'
        : undefined,
    ].filter((id): id is string => Boolean(id));
    const fusionY = topBottom + 32;
    addNode('token-fusion', centerX, fusionY);
    addEdge('embedding', 'token-fusion');
    mediaOutputs.forEach((id) => addEdge(id, 'token-fusion'));
    flowIntoBlock = 'token-fusion';
    topBottom = fusionY + NODE_HEIGHT;
    if (input.isEncoderDecoder && findNode(flow, 'encoder')) {
      const encoderY = topBottom + VERTICAL_GAP;
      addNode('encoder', centerX, encoderY);
      addEdge('token-fusion', 'encoder');
      flowIntoBlock = 'encoder';
      topBottom = encoderY + NODE_HEIGHT;
    }
  } else {
    const topIds = [
      'input',
      'embedding',
      ...(input.isEncoderDecoder ? ['encoder'] : []),
    ].filter((id) => findNode(flow, id));
    topIds.forEach((id, index) => {
      addNode(id, centerX, topStart + index * (NODE_HEIGHT + VERTICAL_GAP));
    });
    addSequentialEdges(edges, topIds);
    flowIntoBlock = topIds.at(-1) ?? 'embedding';
    topBottom =
      topStart + topIds.length * (NODE_HEIGHT + VERTICAL_GAP) - VERTICAL_GAP;
  }

  const attentionBlockIds = [
    'attention-norm',
    'qkv',
    'position',
    'attention',
    'attention-residual',
  ].filter((id) => findNode(flow, id));
  const mlpBlockIds = [
    'mlp-norm',
    ...(input.numExperts && input.numExperts > 1
      ? ['router', 'experts']
      : ['mlp']),
    'mlp-residual',
  ].filter((id) => findNode(flow, id));
  const mainBlockIds = [...attentionBlockIds, ...mlpBlockIds];
  const blockTop = topBottom + 56;
  const blockX = 60;
  const blockWidth = GRAPH_WIDTH - blockX * 2;
  const rowStartX = blockX + 36;
  const rowEndX = blockX + blockWidth - NODE_WIDTH - 36;
  const attentionY = blockTop + 92;
  const mlpY = attentionY + NODE_HEIGHT + 74;
  const hasSharedExperts = Boolean(
    input.numSharedExperts && input.numSharedExperts > 0,
  );
  const blockHeight = hasSharedExperts ? 590 : 438;
  const placeRow = (ids: string[], y: number) => {
    ids.forEach((id, index) => {
      const progress = ids.length > 1 ? index / (ids.length - 1) : 0.5;
      const node = placeNode(
        flow,
        id,
        rowStartX + (rowEndX - rowStartX) * progress,
        y,
      );
      if (node) nodes.push(node);
    });
  };

  placeRow(attentionBlockIds, attentionY);
  placeRow(mlpBlockIds, mlpY);
  addSequentialEdges(edges, mainBlockIds);
  if (mainBlockIds[0]) {
    addEdge(flowIntoBlock, mainBlockIds[0]);
  }

  if (hasSharedExperts) {
    const shared = placeNode(flow, 'experts', rowEndX, mlpY + NODE_HEIGHT + 46);
    if (shared) {
      shared.id = 'shared-experts';
      shared.label = 'Shared expert path';
      shared.detail = `${input.numSharedExperts} shared expert path(s) stay active for every token alongside routed experts.`;
      shared.kind = 'mlp';
      nodes.push(shared);
      if (
        mainBlockIds.includes('mlp-norm') &&
        mainBlockIds.includes('mlp-residual')
      ) {
        edges.push({
          id: 'shared-expert-input',
          from: 'mlp-norm',
          to: 'shared-experts',
          kind: 'shared',
          label: 'shared',
        });
        edges.push({
          id: 'shared-expert-output',
          from: 'shared-experts',
          to: 'mlp-residual',
          kind: 'shared',
          label: 'merge',
        });
      }
    }
  }
  if (
    mainBlockIds.includes('attention-norm') &&
    mainBlockIds.includes('attention-residual')
  ) {
    edges.push({
      id: 'attention-residual-bypass',
      from: flowIntoBlock,
      to: 'attention-residual',
      kind: 'residual',
      label: 'residual',
    });
  }
  if (
    mainBlockIds.includes('mlp-norm') &&
    mainBlockIds.includes('mlp-residual')
  ) {
    edges.push({
      id: 'mlp-residual-bypass',
      from: 'mlp-norm',
      to: 'mlp-residual',
      kind: 'residual',
      label: 'residual',
    });
  }

  const tailTop = blockTop + blockHeight + 70;
  const tailIds = ['final-norm', 'lm-head'].filter((id) => findNode(flow, id));
  tailIds.forEach((id, index) => {
    const node = placeNode(
      flow,
      id,
      centerX + (index - (tailIds.length - 1) / 2) * (NODE_WIDTH + 70),
      tailTop,
    );
    if (node) nodes.push(node);
  });
  addSequentialEdges(edges, tailIds);
  if (mainBlockIds.includes('mlp-residual') && tailIds[0]) {
    edges.push({
      id: 'exit-block',
      from: 'mlp-residual',
      to: tailIds[0],
      kind: 'flow',
    });
  }

  return {
    nodes,
    edges,
    width: GRAPH_WIDTH,
    height: tailTop + NODE_HEIGHT + 70,
    blockBounds: {
      x: blockX,
      y: blockTop,
      width: blockWidth,
      height: blockHeight,
    },
  };
}
