import axios from 'axios';
import * as React from 'react';
import { useState } from 'react';

import Seo from '@/components/Seo';

import {
  calculateFlops,
  calculateMemory,
  estimateInferenceTime,
  estimateTrainingCost,
  estimateModelSize,
  estimateModelSizeT5,
  recommendGPU,
} from '@/estimator/estimator';
import gpus from '@/estimator/gpus.json';
/**
 * SVGR Support
 * Caveat: No React Props Type.
 *
 * You can override the next-env if the type is important to you
 * @see https://stackoverflow.com/questions/68103844/how-to-override-next-js-svg-module-declaration
 */

export default function HomePage() {
  const [modelId, setModelId] = useState<string>('google/flan-t5-xl');
  const [modelError, setModelError] = useState<boolean>(false);
  // Estiamting the number of parameters
  const [numParams, setNumParams] = useState<number>(0);
  const [bytes, setBytes] = useState<number>(32);
  //Estimating the number of flops
  const [numLayers, setNumLayers] = useState<number>(0);
  const [hiddenSize, setHiddenSize] = useState<number>(0);
  const [numHeads, setNumHeads] = useState<number>(0); // numHeads is kept for other potential uses or if UI still has it
  const [vocabSize, setVocabSize] = useState<number>(0);
  const [seqLength, setSeqLength] = useState<number>(512); // Added seqLength state
  // const [head_kv, setHead_kv] = useState<number>(0);

  //Estimating the inference time
  const [calculatedGFlops, setCalculatedGFlops] = useState<number>(0); // For FLOPs estimator output
  const [selectedGpuName, setSelectedGpuName] = useState<string>(gpus[0].name);
  const [gpuTFlops, setGpuTFlops] = useState<number>(
    parseFloat(gpus[0].flops.split(' ')[0])
  );
  const [memoryBandwidth, setMemoryBandwidth] = useState<number>(
    parseFloat(gpus[0].memory.split(' ')[0])
  );
  const [overheadFactor, setOverheadFactor] = useState<number>(1.2);
  // numParams for inference estimator is taken from the general numParams state

  //Estimating cloud training cost
  const [selectedTrainingGpuName, setSelectedTrainingGpuName] = useState<string>(gpus[0].name);
  const [trainingGpuTFlops, setTrainingGpuTFlops] = useState<number>(
    parseFloat(gpus[0].flops.split(' ')[0])
  );
  const [trainingMemoryBandwidth, setTrainingMemoryBandwidth] = useState<number>(
    parseFloat(gpus[0].memory.split(' ')[0])
  );
  const [gpuHourlyCost, setGpuHourlyCost] = useState<number>(0);
  const [numEpochs, setNumEpochs] = useState<number>(0);
  const [datasetSize, setDatasetSize] = useState<number>(0);
  const [recommendedGpuMessage, setRecommendedGpuMessage] = useState<string>('');

  React.useEffect(() => {
    const memoryInGB = calculateMemory(numParams * 10 ** 9, bytes);
    const flopsForRecommendation = calculatedGFlops * 10 ** 9; // Convert GFLOPs to FLOPs
    if (numParams > 0 && bytes > 0 && calculatedGFlops > 0) { // ensure valid inputs before recommending
      setRecommendedGpuMessage(recommendGPU(memoryInGB, flopsForRecommendation));
    } else {
      setRecommendedGpuMessage('Please provide model parameters and FLOPs for a recommendation.');
    }
  }, [numParams, bytes, calculatedGFlops]);

  const formatInferenceTime = (seconds: number): string => {
    if (seconds < 0.001 && seconds > 0) {
      return `${(seconds * 1000).toFixed(2)} ms`;
    } else if (seconds < 1) {
      return `${seconds.toFixed(4)} s`;
    } else if (seconds < 60) {
      return `${seconds.toFixed(2)} s`;
    } else {
      return `${(seconds / 60).toFixed(2)} min`;
    }
  };

  const formatGB = (num: number): string => {
    if (num === undefined || num <= 0) { // Changed condition to handle undefined or zero/negative
      return 'N/A'; // Or some other placeholder
    }
    if (num < 1) { // Kept this for small values that are positive
      return '<1';
    } else {
      return num.toFixed(2).toString();
    }
  };

  const setModel = async (modelId: string) => {
    if (modelId.length > 0) {
      try {
        const res = await axios.get(
          `https://huggingface.co/${modelId}/raw/main/config.json`
        );
        const model = res.data;
        // console.log(model);
        if (model) {
          setModelError(false); // Reset error state at the beginning

          // Extract num_params
          let params: number | undefined = model.num_params; // HuggingFace might provide it directly

          if (params) {
            setNumParams(params / 10 ** 9); // Assuming params are in raw numbers, convert to billions
          } else {
            // Try to calculate num_params if not directly available
            if (model.architectures?.includes('T5ForConditionalGeneration')) {
              // Ensure all necessary T5 config fields are present
              if (model.d_model && model.vocab_size && model.num_layers && model.num_decoder_layers && model.d_ff && model.n_positions) {
                params = estimateModelSizeT5({
                  d_model: model.d_model,
                  vocab_size: model.vocab_size,
                  n_head: model.num_heads || model.n_head, // Handle variations
                  num_layers: model.num_layers,
                  n_head_kv: model.num_heads_kv || model.n_head_kv || model.num_attention_heads, // Handle variations
                  d_ff: model.d_ff,
                  n_positions: model.n_positions,
                  num_decoder_layers: model.num_decoder_layers,
                });
                setNumParams(params / 10 ** 9);
              } else {
                console.error('Missing parameters for T5 model size estimation');
                setModelError(true);
              }
            } else {
              // Use generic estimator for other architectures
              // Ensure all necessary generic config fields are present
              const hiddenSizeVal = model.hidden_size || model.d_model;
              const numLayersVal = model.num_layers || model.n_layer || model.num_hidden_layers;
              const numHeadsVal = model.num_heads || model.n_head || model.num_attention_heads;
              const vocabSizeVal = model.vocab_size;
              // n_head_kv might not always be present, provide a default or handle absence
              const numHeadsKvVal = model.n_head_kv || model.num_key_value_heads;


              if (hiddenSizeVal && numLayersVal && numHeadsVal && vocabSizeVal) {
                params = estimateModelSize(
                  vocabSizeVal,
                  hiddenSizeVal,
                  numHeadsVal,
                  numLayersVal,
                  numHeadsKvVal // This could be undefined, estimateModelSize should handle it or be adjusted
                );
                setNumParams(params / 10 ** 9);
              } else {
                console.error('Missing parameters for generic model size estimation from config:', model);
                setModelError(true);
              }
            }
          }

          if (!params) {
            console.error('Could not determine num_params');
            setModelError(true);
            // Potentially set a specific error message for the UI here
          }

          // Extract other parameters with robust handling for different names
          const numLayersValue = model.num_layers || model.n_layer || model.num_hidden_layers;
          if (numLayersValue !== undefined) setNumLayers(numLayersValue);
          else {
            console.error('num_layers not found in config');
            setModelError(true);
          }

          const hiddenSizeValue = model.hidden_size || model.d_model;
          if (hiddenSizeValue !== undefined) setHiddenSize(hiddenSizeValue);
          else {
            console.error('hidden_size or d_model not found in config');
            setModelError(true);
          }

          const numHeadsValue = model.num_attention_heads || model.num_heads || model.n_head;
          if (numHeadsValue !== undefined) setNumHeads(numHeadsValue);
          else {
            console.error('num_heads not found in config');
            setModelError(true);
          }

          if (model.vocab_size !== undefined) setVocabSize(model.vocab_size);
          else {
            console.error('vocab_size not found in config');
            setModelError(true);
          }

          if (model.torch_dtype) {
            const dtype = model.torch_dtype;
            if (dtype === 'float32' || dtype === 'torch.float32') setBytes(32);
            else if (dtype === 'float16' || dtype === 'torch.float16') setBytes(16);
            else if (dtype === 'bfloat16' || dtype === 'torch.bfloat16') setBytes(16); // bfloat16 is also 2 bytes
            else if (dtype === 'int8') setBytes(8);
            else {
              console.warn(`Unsupported torch_dtype: ${dtype}, defaulting to 32 bits`);
              setBytes(32); // Default or handle as an error
            }
          } else {
            console.warn('torch_dtype not found in config, defaulting to 32 bits');
            setBytes(32); // Default if not specified
          }

        } else {
          //This case should ideally not be reached if axios throws an error for non-200 responses
          setModelError(true);
        }
      } catch (e) {
        console.error('Failed to fetch or process model config:', e);
        setModelError(true);
      }
    } else {
      // Handle empty modelId input if necessary, though the button might be disabled
      setModelError(true);
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
      <hr className='mx-auto my-6 w-3/4 border-2 border-black' />

      <div className='m-auto max-w-xs rounded-lg border-2 border-black p-5'>
        <label className='label'>
          <span className='label-text'>Huggingface Model ID</span>
        </label>
        <input
          className={
            modelError
              ? 'input input-bordered input-error w-full max-w-xs'
              : 'input input-bordered input-primary w-full max-w-xs'
          }
          placeholder='e.g. bert-base-uncased'
          type='text'
          onChange={(e) => {
            setModelId(e.target.value);
          }}
          value={modelId}
        />
        <button
          className='btn btn-primary mt-2 w-full'
          onClick={() => setModel(modelId)}
        >
          {' '}
          Fetch Model{' '}
        </button>
      </div>
      <div className='grid md:grid-cols-2'>
        <div className='m-5 rounded-lg border-2 border-black p-5'>
          <h2>Memory Estimator</h2>
          <p>
            Estimate the amount of GPU memory required to load a models weights
          </p>
          <div className='md:flex md:justify-evenly md:space-x-4'>
            <div className='form-control w-full max-w-xs mb-4 md:mb-0'>
              <label className='label'>
                <span className='label-text'>Parameter Size</span>
                <span className='label-text-alt'>billions</span>
                <span className='tooltip tooltip-right' data-tip="Total number of parameters in the model, in billions. E.g., for a 7B model, enter 7.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Number of Model Parameters'
                type='number'
                onChange={(e) => {
                  setNumParams(parseFloat(e.target.value) || 0);
                }}
                value={numParams}
              />
            </div>
            <div className='form-control w-full max-w-xs'>
              <label className='label'>
                <span className='label-text'>Float Size</span>
                <span className='label-text-alt'>Bits</span>
                <span className='tooltip tooltip-right' data-tip="Precision of model weights. Common values are 32 (FP32), 16 (FP16/BF16), 8 (INT8), 4 (NF4).">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <select
                onChange={(e) => {
                  setBytes(parseInt(e.target.value));
                }}
                className='select select-bordered w-full'
                value={bytes}
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
            <span className='font-bold text-lg'>{formatGB(calculateMemory(numParams * 10 ** 9, bytes))} GB</span>
          </p>
          <div className='mt-2'>
            <p className='font-semibold'>GPU Recommendation: <span className='font-normal'>{recommendedGpuMessage}</span></p>
          </div>
        </div>

        <div className='m-5 rounded-lg border-2 border-black p-5'>
          <h2>FLOPs Estimator</h2>
          <p className='mb-4'>Estimate the total FLOPs for a inference a transformer model.</p>
          <div className='grid gap-4 md:grid-cols-2'>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Layers</span>
                <span className='tooltip tooltip-right' data-tip="Number of transformer layers in the model.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Number of Layers'
                type='number'
                onChange={(e) => {
                  setNumLayers(parseFloat(e.target.value) || 0);
                }}
                value={numLayers}
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Sequence Length</span>
                <span className='tooltip tooltip-right' data-tip="The number of tokens in an input sequence for the model.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Sequence Length'
                type='number'
                onChange={(e) => {
                  setSeqLength(parseFloat(e.target.value) || 0);
                }}
                value={seqLength}
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Vocab Size</span>
                <span className='tooltip tooltip-right' data-tip="The number of unique tokens in the model's vocabulary.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Vocab Size'
                type='number'
                onChange={(e) => {
                  setVocabSize(parseFloat(e.target.value) || 0);
                }}
                value={vocabSize}
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Hidden Layer Dimensions</span>
                <span className='tooltip tooltip-right' data-tip="The dimensionality of the model's hidden states (d_model).">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Dimenonality of Hidden Layer'
                type='number'
                onChange={(e) => {
                  setHiddenSize(parseFloat(e.target.value));
                }}
                value={hiddenSize}
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Attention Heads</span>
                <span className='tooltip tooltip-right' data-tip="Number of attention heads in the model. (Currently not used in FLOPs calculation but kept for completeness).">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Number of Attention Heads'
                type='number'
                onChange={(e) => {
                  setNumHeads(parseFloat(e.target.value) || 0);
                }}
                value={numHeads}
              />
            </div>
          </div>
          <p className='mt-4'>
            Estimated Number of Floating Point Operations:{' '}
            <span className='font-bold text-lg'>
            {
              (() => {
                const currentFlops = calculateFlops(numLayers, seqLength, hiddenSize, vocabSize) / 10**9;
                if (currentFlops !== calculatedGFlops && !isNaN(currentFlops)) {
                  setTimeout(() => setCalculatedGFlops(currentFlops), 0);
                }
                return currentFlops.toFixed(2);
              })()
            }
            </span>{' '}
            GFLOPs{' '}
          </p>
        </div>

        <div className='m-5 rounded-lg border-2 border-black p-5'>
          <h2>Inference Time Estimator</h2>
          <p className='mb-4'>Estimate the time it takes to run inference on a model.</p>
          <div className='flex flex-col space-y-4'> {/* Added space-y-4 */}
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Calculated GFLOPs (from above)</span>
              </label>
              <input
                className='input input-bordered input-primary w-full bg-gray-100 dark:bg-gray-700' // Made read-only distinct
                type='number'
                value={calculatedGFlops.toFixed(2)}
                readOnly
              />
            </div>

            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Select GPU</span>
                <span className='tooltip tooltip-right' data-tip="Select a GPU to estimate inference time. TFLOPs and Memory Bandwidth will be auto-filled.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <select
                className='select select-bordered w-full'
                value={selectedGpuName}
                onChange={(e) => {
                  const selectedName = e.target.value;
                  const selectedGpu = gpus.find(gpu => gpu.name === selectedName);
                  if (selectedGpu) {
                    setSelectedGpuName(selectedName);
                    setGpuTFlops(parseFloat(selectedGpu.flops.split(' ')[0]));
                    setMemoryBandwidth(parseFloat(selectedGpu.memory.split(' ')[0]));
                  }
                }}
              >
                {gpus.map((gpu) => (
                  <option key={gpu.name} value={gpu.name}>
                    {gpu.name} ({gpu.flops}, {gpu.memory})
                  </option>
                ))}
              </select>
            </div>

            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>GPU TFlops (Selected)</span>
              </label>
              <input
                className='input input-bordered input-primary w-full bg-gray-100 dark:bg-gray-700'
                type='number'
                value={gpuTFlops}
                readOnly
              />
            </div>

            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>GPU Memory Bandwidth (GB/s - Selected)</span>
              </label>
              <input
                className='input input-bordered input-primary w-full bg-gray-100 dark:bg-gray-700'
                type='number'
                value={memoryBandwidth}
                readOnly
              />
            </div>

            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Model Parameters (Billions)</span>
                <span className='tooltip tooltip-right' data-tip="Should match the 'Parameter Size' from the Memory Estimator.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Number of Model Parameters (Billions)'
                type='number'
                onChange={(e) => {
                  setNumParams(parseFloat(e.target.value) || 0);
                }}
                value={numParams}
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Software Overhead Factor</span>
                <span className='tooltip tooltip-right' data-tip="Accounts for inefficiencies or variations in software stack and implementation. Typically between 1.1 (very optimized) and 1.5 (less optimized).">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Overhead Factor'
                type='number'
                onChange={(e) => {
                  setOverheadFactor(parseFloat(e.target.value) || 1.0);
                }}
                value={overheadFactor}
              />
            </div>
          </div>
          <p className='mt-4'>
            Estimated Inference Time:{' '}
            <span className='font-bold text-lg'>
              {formatInferenceTime(
                estimateInferenceTime(
                  calculatedGFlops,      // GFLOPs from FLOPs estimator
                  gpuTFlops,             // TFLOPs from selected GPU
                  numParams * 10 ** 9,   // Actual number of parameters
                  memoryBandwidth,       // GB/s from selected GPU
                  overheadFactor,
                  bytes / 8              // bytes_per_param from model precision
              ))}
            </span>
            {/* Removed "in seconds" as formatInferenceTime now includes units */}
          </p>
        </div>

        <div className='m-5 rounded-lg border-2 border-black p-5'>
          <h2>Cloud Training Estimator</h2>
          <p className='mb-4'>Estimate the cost of training a model on Cloud Services</p>
          <div className='flex flex-col space-y-4'> {/* Added space-y-4 */}
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Calculated GFLOPs (from FLOPs Estimator)</span>
              </label>
              <input
                className='input input-bordered input-primary w-full bg-gray-100 dark:bg-gray-700' // Made read-only distinct
                type='number'
                value={calculatedGFlops.toFixed(2)}
                readOnly
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Select Training GPU</span>
                <span className='tooltip tooltip-right' data-tip="Select the GPU used for training. TFLOPs and Memory Bandwidth will be auto-filled.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <select
                className='select select-bordered w-full'
                value={selectedTrainingGpuName}
                onChange={(e) => {
                  const selectedName = e.target.value;
                  const selectedGpu = gpus.find(gpu => gpu.name === selectedName);
                  if (selectedGpu) {
                    setSelectedTrainingGpuName(selectedName);
                    setTrainingGpuTFlops(parseFloat(selectedGpu.flops.split(' ')[0]));
                    setTrainingMemoryBandwidth(parseFloat(selectedGpu.memory.split(' ')[0]));
                  }
                }}
              >
                {gpus.map((gpu) => (
                  <option key={gpu.name} value={gpu.name}>
                    {gpu.name} ({gpu.flops}, {gpu.memory})
                  </option>
                ))}
              </select>
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Training GPU TFlops (Selected)</span>
              </label>
              <input
                className='input input-bordered input-primary w-full bg-gray-100 dark:bg-gray-700'
                type='number'
                value={trainingGpuTFlops}
                readOnly
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Training GPU Memory Bandwidth (GB/s - Selected)</span>
              </label>
              <input
                className='input input-bordered input-primary w-full bg-gray-100 dark:bg-gray-700'
                type='number'
                value={trainingMemoryBandwidth}
                readOnly
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Model Parameters (Billions)</span>
                <span className='tooltip tooltip-right' data-tip="Should match 'Parameter Size' from Memory Estimator.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Number of Model Parameters (Billions)'
                type='number'
                onChange={(e) => {
                  setNumParams(parseFloat(e.target.value) || 0);
                }}
                value={numParams}
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>GPU Hourly Cost</span>
                <span className='tooltip tooltip-right' data-tip="The cost per hour for the selected training GPU, in your currency (e.g., USD).">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='GPU Hourly Cost'
                type='number'
                onChange={(e) => {
                  setGpuHourlyCost(parseFloat(e.target.value) || 0);
                }}
                value={gpuHourlyCost}
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Number of Epochs</span>
                <span className='tooltip tooltip-right' data-tip="The total number of times the model will iterate over the entire training dataset.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Number of Epochs'
                type='number'
                onChange={(e) => {
                  setNumEpochs(parseFloat(e.target.value) || 0);
                }}
                value={numEpochs}
              />
            </div>
            <div className='form-control w-full'>
              <label className='label'>
                <span className='label-text'>Datakset Size</span>
                <span className='tooltip tooltip-right' data-tip="Total number of training examples (sequences/items) in your dataset.">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-info shrink-0 w-4 h-4 ml-1 cursor-pointer"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </span>
              </label>
              <input
                className='input input-bordered input-primary w-full'
                placeholder='Dataset Size'
                type='number'
                onChange={(e) => {
                  setDatasetSize(parseFloat(e.target.value) || 0);
                }}
                value={datasetSize}
              />
            </div>
          </div>
          <p className='mt-4'>
            Estimated Cloud Training Cost: $
            <span className='font-bold text-lg'>
              {estimateTrainingCost(
                calculatedGFlops,          // GFLOPs from FLOPs estimator
                trainingGpuTFlops,         // TFLOPs from selected training GPU
                numParams * 10 ** 9,       // Actual number of parameters
                trainingMemoryBandwidth,   // GB/s from selected training GPU
                overheadFactor,
                gpuHourlyCost,
                numEpochs,
                datasetSize,
                bytes                        // model_precision_bits (e.g. 16 or 32)
              ).toFixed(2)}
            </span>{' '}
          </p>
        </div>
      </div>
    </main>
  );
}
