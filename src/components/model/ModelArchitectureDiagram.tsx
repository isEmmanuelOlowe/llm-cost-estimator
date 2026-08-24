import type { ModalityArchitecture } from '@/lib/model-architecture';

import type { TransformerParameterBreakdown } from '@/estimator/estimator';

import ArchitectureFlowExplorer from './ArchitectureFlowExplorer';

interface ModelArchitectureDiagramProps {
  modelType?: string;
  architectures?: string[];
  sourceDirectoryUrl?: string;
  sourceFiles?: Array<{ name: string; url: string }>;
  sourcePreview?: { name: string; url: string; content: string };
  onLoadImplementation?: () => void;
  isLoadingImplementation?: boolean;
  hiddenSize: number;
  numLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  intermediateSize: number;
  expertIntermediateSize?: number;
  sharedExpertIntermediateSize?: number;
  numSharedExperts?: number;
  isEncoderDecoder?: boolean;
  modality?: string;
  modalityArchitecture?: ModalityArchitecture;
  vocabSize: number;
  numExperts?: number;
  numExpertsPerToken?: number;
  parameterCount: number;
  parameterBreakdown?: TransformerParameterBreakdown;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '—';
  return value.toLocaleString('en-US', { maximumFractionDigits: 0 });
}

function formatParams(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '—';
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  return `${(value / 1e3).toFixed(1)}K`;
}

function Meter({
  label,
  value,
  total,
}: {
  label: string;
  value: number;
  total: number;
}) {
  const width =
    total > 0 ? Math.max(2, Math.min(100, (value / total) * 100)) : 0;
  return (
    <div>
      <div className='flex justify-between gap-3 text-xs'>
        <span>{label}</span>
        <span className='font-semibold'>{formatParams(value)}</span>
      </div>
      <div className='mt-1 h-2 rounded-full bg-base-300'>
        <div
          className='h-2 rounded-full bg-primary'
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

export default function ModelArchitectureDiagram({
  modelType,
  architectures,
  sourceDirectoryUrl,
  sourceFiles,
  sourcePreview,
  onLoadImplementation,
  isLoadingImplementation,
  hiddenSize,
  numLayers,
  numAttentionHeads,
  numKeyValueHeads,
  headDim,
  intermediateSize,
  expertIntermediateSize,
  sharedExpertIntermediateSize,
  numSharedExperts,
  isEncoderDecoder,
  modality,
  modalityArchitecture,
  vocabSize,
  numExperts,
  numExpertsPerToken,
  parameterCount,
  parameterBreakdown,
}: ModelArchitectureDiagramProps) {
  const totalBreakdown = parameterBreakdown?.totalParameters ?? parameterCount;
  return (
    <div className='rounded-2xl border border-base-300 bg-base-100 p-5 shadow-lg shadow-black/10 sm:p-6'>
      <div className='flex flex-col gap-2 md:flex-row md:items-start md:justify-between'>
        <div>
          <h2 className='text-2xl font-semibold'>See how the model works</h2>
          <p className='mt-1 text-sm text-base-content/70'>
            Follow the model from inputs to logits, then zoom into one repeated
            block to reveal its building blocks.
          </p>
        </div>
        <span className='badge badge-primary badge-outline whitespace-nowrap'>
          {formatParams(parameterCount)} parameters
        </span>
      </div>

      <ArchitectureFlowExplorer
        modelType={modelType}
        architectures={architectures}
        sourceDirectoryUrl={sourceDirectoryUrl}
        sourceFiles={sourceFiles}
        sourcePreview={sourcePreview}
        onLoadImplementation={onLoadImplementation}
        isLoadingImplementation={isLoadingImplementation}
        hiddenSize={hiddenSize}
        numLayers={numLayers}
        numAttentionHeads={numAttentionHeads}
        numKeyValueHeads={numKeyValueHeads}
        headDim={headDim}
        intermediateSize={intermediateSize}
        expertIntermediateSize={expertIntermediateSize}
        sharedExpertIntermediateSize={sharedExpertIntermediateSize}
        numSharedExperts={numSharedExperts}
        isEncoderDecoder={isEncoderDecoder}
        modality={modality}
        modalityArchitecture={modalityArchitecture}
        vocabSize={vocabSize}
        numExperts={numExperts}
        numExpertsPerToken={numExpertsPerToken}
        architectureLabel={
          modelType ??
          architectures?.[0] ??
          'Normalized transformer architecture'
        }
      />

      {parameterBreakdown && (
        <details className='mt-4 rounded-xl border border-base-300 bg-base-200 p-4'>
          <summary className='cursor-pointer text-sm font-semibold'>
            Parameter composition
          </summary>
          <div className='mt-4'>
            <div className='flex items-center justify-between gap-3 text-sm'>
              <h3 className='font-semibold'>Estimated parameter composition</h3>
              <span className='text-xs text-base-content/70'>
                {formatParams(totalBreakdown)} total
              </span>
            </div>
            <div className='mt-4 space-y-3'>
              <Meter
                label='Embeddings'
                value={parameterBreakdown.embeddingParams}
                total={totalBreakdown}
              />
              <Meter
                label={`Attention × ${formatNumber(numLayers)}`}
                value={parameterBreakdown.attentionParamsPerLayer * numLayers}
                total={totalBreakdown}
              />
              <Meter
                label={`MLP / experts × ${formatNumber(numLayers)}`}
                value={parameterBreakdown.mlpParamsPerLayer * numLayers}
                total={totalBreakdown}
              />
              <Meter
                label='Output head'
                value={parameterBreakdown.lmHeadParams}
                total={totalBreakdown}
              />
            </div>
            <p className='mt-3 text-xs text-base-content/65'>
              This composition is analytical. When safetensors metadata is
              available, the fetched total takes precedence over this model-form
              estimate.
            </p>
          </div>
        </details>
      )}
    </div>
  );
}
