import * as React from 'react';
import { useState } from 'react';

import Seo from '@/components/Seo';

import {
  calculateFlops,
  calculateMemory,
  calculateNumParams,
  estimateInferenceTime,
  estimateTrainingCost,
} from '@/estimator/estimator';
// import gpus from '@/estimator/gpus.json';

/**
 * SVGR Support
 * Caveat: No React Props Type.
 *
 * You can override the next-env if the type is important to you
 * @see https://stackoverflow.com/questions/68103844/how-to-override-next-js-svg-module-declaration
 */

export default function HomePage() {
  // Estiamting the number of parameters
  const [numParams, setNumParams] = useState<number>(0);
  const [bytes, setBytes] = useState<number>(32);
  //Estimating the number of flops
  const [numLayers, setNumLayers] = useState<number>(0);
  const [seqLength, setSeqLength] = useState<number>(0);
  const [hiddenSize, setHiddenSize] = useState<number>(0);
  const [numHeads, setNumHeads] = useState<number>(0);

  //Estimating the inference time
  const [flops, setFlops] = useState<number>(0);
  const [gpuTFlops, setGpuTFlops] = useState<number>(0);
  const [memoryBandwith, setMemoryBaandwith] = useState<number>(0);
  const [overheadFactor, setOverheadFactor] = useState<number>(1.2);

  //Estimating cloud training cost
  const [gpuHourlyCost, setGpuHourlyCost] = useState<number>(0);
  const [numEpochs, setNumEpochs] = useState<number>(0);
  const [datasetSize, setDatasetSize] = useState<number>(0);

  const formatGB = (num: number): string => {
    if (num < 1 || num === undefined) {
      return '1<';
    } else {
      return num.toFixed(2).toString();
    }
  };

  return (
    <main data-theme='dark' className='bg-full'>
      <Seo />
      <h1 className='mh-auto text-primary mt-5 text-center'>
        AI Project Prophet
      </h1>
      <p className='text-center'>
        Estimate Hardware and Costs of running LLMs and Transformer projects
      </p>
      <div className='grid md:grid-cols-2'>
        <div className='m-5 rounded-lg border-2 border-black p-5'>
          <h2>Memory Estimator</h2>
          <p>
            Estimate the amount of GPU memory required to load a models weights
          </p>
          <div className='md:flex md:justify-evenly'>
            <div className='form-control w-full max-w-xs'>
              <label className='label'>
                <span className='label-text'>Parameter Size</span>
                <span className='label-text-alt'>billions</span>
              </label>
              <input
                className='input input-bordered input-primary w-full max-w-xs'
                placeholder='Number of Model Parameters'
                type='number'
                onChange={(e) => {
                  setNumParams(parseInt(e.target.value));
                }}
                value={numParams}
              />{' '}
            </div>
            <div className='form-control w-full max-w-xs'>
              <label className='label'>
                <span className='label-text'>Float Size</span>
                <span className='label-text-alt'>Bits</span>
              </label>
              <select
                onChange={(e) => {
                  setBytes(parseInt(e.target.value));
                }}
                className='select select-bordered'
              >
                <option value={32}>32-bit Float</option>
                <option value={16}>16-bit Float</option>
                <option value={8}>8-Bit Float</option>
                <option value={4}>4-Bit Float</option>
              </select>
            </div>
          </div>
          <p>
            Estimated memory:{' '}
            {formatGB(calculateMemory(numParams * 10 ** 9, bytes))} GB
          </p>
        </div>

        <div className='m-5 rounded-lg border-2 border-black p-5'>
          <h2>FLOPs Estimator</h2>
          <p>Estimate the total FLOPs for a inference a transformer model.</p>
          <div className='grid gap-1 md:grid-cols-2'>
            <div className='max-w-s'>
              <label className='label'>
                <span className='label-text'>Layers</span>
              </label>
              <input
                className='input input-bordered input-primary w-full max-w-xs'
                placeholder='Number of Layers'
                type='number'
                onChange={(e) => {
                  setNumLayers(parseFloat(e.target.value));
                }}
                value={numLayers}
              />
            </div>
            <div>
              <label className='label'>
                <span className='label-text'>Maximum Sequence Length</span>
              </label>
              <input
                className='input input-bordered input-primary w-full max-w-xs'
                placeholder='Maximum Sequence Length'
                type='number'
                onChange={(e) => {
                  setSeqLength(parseFloat(e.target.value));
                }}
                value={seqLength}
              />
            </div>
            <div>
              <label className='label'>
                <span className='label-text'>Hidden Layer Dimensions</span>
              </label>
              <input
                className='input input-bordered input-primary w-full max-w-xs'
                placeholder='Dimenonality of Hidden Layer'
                type='number'
                onChange={(e) => {
                  setHiddenSize(parseFloat(e.target.value));
                }}
                value={hiddenSize}
              />
            </div>
            <div>
              <label className='label'>
                <span className='label-text'>Attention Heads</span>
              </label>
              <input
                className='input input-bordered input-primary w-full max-w-xs'
                placeholder='Number of Attention Heads'
                type='number'
                onChange={(e) => {
                  setNumHeads(parseFloat(e.target.value));
                }}
                value={numHeads}
              />
            </div>
          </div>
          <p>
            Estimated model parametrs:{' '}
            {calculateNumParams(numLayers, seqLength, hiddenSize) / 10 ** 9} in
            Billions{' '}
          </p>
          <p>
            Estimated Number of Floating Point Operations:{' '}
            {calculateFlops(numLayers, seqLength, hiddenSize, numHeads) /
              10 ** 10}{' '}
            GFLOPs{' '}
          </p>
        </div>

        <div className='m-5 rounded-lg border-2 border-black p-5'>
          <h2>Inference Time Estimator</h2>
          <p>Estimate the time it takes to run inference on a model.</p>
          <div className='flex flex-col'>
            <label className='label'>
              <span className='label-text'>FLOPS of inference</span>
              <span className='label-text-alt'>GFLOP</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='Number of Floating Point Operations of '
              type='number'
              onChange={(e) => {
                setFlops(parseFloat(e.target.value));
              }}
              value={flops}
            />
            <label className='label'>
              <span className='label-text'>GPU TFlops</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='GPU TFlops'
              type='number'
              onChange={(e) => {
                setGpuTFlops(parseFloat(e.target.value));
              }}
              value={gpuTFlops}
            />
            <label className='label'>
              <span className='label-text'>GPU Memory Bandwith</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='GPU Memory Bandwith'
              type='number'
              onChange={(e) => {
                setMemoryBaandwith(parseFloat(e.target.value));
              }}
              value={memoryBandwith}
            />
            <label className='label'>
              <span className='label-text'>Model Parameters</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='number of Model Parameters'
              type='number'
              onChange={(e) => {
                setNumParams(parseFloat(e.target.value));
              }}
              value={numParams}
            />
            <label className='label'>
              <span className='label-text'>Software Overhead Factor</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='Overhead Factor'
              type='number'
              onChange={(e) => {
                setOverheadFactor(parseFloat(e.target.value));
              }}
              value={overheadFactor}
            />
          </div>
          <p>
            Estimated Inference Time:{' '}
            {estimateInferenceTime(
              flops,
              gpuTFlops,
              memoryBandwith,
              numParams,
              overheadFactor
            )}{' '}
            in seconds
          </p>
        </div>

        <div className='m-5 rounded-lg border-2 border-black p-5'>
          <h2>Cloud Training Estimator</h2>
          <div className='flex flex-col'>
            <p>Estimate the cost of training a model on Cloud Services</p>
            <label className='label'>
              <span className='label-text'>FLOPS of inference</span>
              <span className='label-text-alt'>GFLOP</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='Number of Floating Point Operations of '
              type='number'
              onChange={(e) => {
                setFlops(parseFloat(e.target.value));
              }}
              value={flops}
            />
            <label className='label'>
              <span className='label-text'>GPU TFlops</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='GPU TFlops'
              type='number'
              onChange={(e) => {
                setGpuTFlops(parseFloat(e.target.value));
              }}
              value={gpuTFlops}
            />
            <label className='label'>
              <span className='label-text'>Parameter Size</span>
              <span className='label-text-alt'>billions</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='Number of Model Parameters'
              type='number'
              onChange={(e) => {
                setNumParams(parseInt(e.target.value));
              }}
              value={numParams}
            />
            <label className='label'>
              <span className='label-text'>GPU Hourly Cost</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='GPU Hourly Cost'
              type='number'
              onChange={(e) => {
                setGpuHourlyCost(parseFloat(e.target.value));
              }}
              value={gpuHourlyCost}
            />
            <label className='label'>
              <span className='label-text'>Number of Epochs</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='Number of Epochs'
              type='number'
              onChange={(e) => {
                setNumEpochs(parseFloat(e.target.value));
              }}
              value={numEpochs}
            />
            <label className='label'>
              <span className='label-text'>Datakset Size</span>
            </label>
            <input
              className='input input-bordered input-primary w-full max-w-xs'
              placeholder='Dataset Size'
              type='number'
              onChange={(e) => {
                setDatasetSize(parseFloat(e.target.value));
              }}
              value={datasetSize}
            />
          </div>
          <p>
            Estimated Cloud Training Cost: $
            {estimateTrainingCost(
              flops,
              gpuTFlops,
              numParams,
              memoryBandwith,
              overheadFactor,
              gpuHourlyCost,
              numEpochs,
              datasetSize
            )}{' '}
          </p>
        </div>
      </div>
    </main>
  );
}
