# LLM Explorer

LLM Explorer is an interactive Next.js application that helps machine-learning practitioners understand how a large language model is built, whether it will fit into a particular GPU setup, and how much it may cost to run. The explorer combines source-backed model architecture evidence, detailed VRAM breakdowns (weights, activations, KV cache and optimiser state), performance projections, hardware topology, and cloud pricing guidance.

## Key capabilities

- **Evidence-backed Hugging Face inspection** – resolve a model to an immutable Hub revision, read the API’s safetensors parameter totals, normalize architecture aliases from `config.json`, inspect optional generation/tokenizer metadata, and show the pinned model-card/source links. If the Hub total is unavailable, the browser can inspect safetensors headers with bounded HTTP range requests instead of downloading weights.
- **Transformers implementation map** – resolve the model type to the corresponding upstream Transformers directory, show a read-only source preview, and flag `auto_map`/remote-code indicators. Multimodal configs can expose separate image, video, audio, fusion, and language-stream stages instead of a generic projector box. The app never imports or executes downloaded Python.
- **Model structure visualisation** – switch between an overview and an inside-the-block flow, zoom from 65–175%, select each component, inspect tensor shapes/scaling notes, follow the sequential attention → residual → feed-forward flow, and load the linked Transformers implementation code directly inside the explorer.
- **Detailed VRAM analysis** – quantify memory consumption for model weights, activations, KV cache and optimiser state with a configurable execution mode (inference or training), precision, overhead factor and batch size.
- **Architecture-aware KV cache** – model head dimension and GQA/MQA key/value heads, hybrid local/global schedules, DeepSeek V4 compressed attention, and Nemotron state-space layers are accounted for before showing resident bytes and scaling with context length and concurrent sequences.
- **Current long-context defaults** – the default workspace starts with Gemma 4 12B and a 128K-token workload so long-context memory pressure is visible immediately.
- **Hardware fit recommendations** – compare the required VRAM against a curated list of GPUs, highlight whether a selected GPU has enough headroom and propose the closest alternatives.
- **Multi-vendor hardware inventory** – compare per-device versus aggregate memory, bandwidth, topology notes and precision-tagged compute fields across NVIDIA (including B200/B300, GB200/GB300 NVL72, HGX/DGX B300, DGX Spark, and DGX Station GB300), AMD (including Strix Halo and Instinct), and Intel accelerators.
- **Performance estimations** – estimate FLOPs per forward pass, decode tokens per second and milliseconds per token using the lower of compute and weight-memory bandwidth roofs, explicit compute/memory efficiency factors, and the best precision-specific catalog field available. Apple unified-memory Macs can use the bandwidth roof even when no comparable FP32 TFLOPS figure is published. Vendor peak AI figures remain labelled with their precision/sparsity assumptions.
- **Cloud cost calculator** – use provider-verified on-demand rates only when they match the exact selected GPU, see the pricing date and billing basis, or supply a current custom quote when no verified offering exists.

## Getting started

1. Install dependencies:

   ```bash
   npm ci
   ```

2. Start the development server:

   ```bash
   npm run dev
   ```

3. Navigate to `http://localhost:3000` and explore the default `google/gemma-4-12B` or search for a Hugging Face model such as `Qwen/Qwen3.8-27B`.

Public Hub repositories can be inspected directly from the static browser app. Gated/private repositories still require authentication and are reported as unavailable rather than accepting a token in the public client. The inspection flow reads metadata and source links only; it does not execute model code or download checkpoint weights.

## Refreshing source-backed catalogs

The generated catalogs are intentionally versioned snapshots so GitHub Pages remains a static export:

```bash
npm run refresh:catalogs        # refresh both model and hardware snapshots
npm run generate:model-catalog  # Hub API + SHA-pinned config/safetensors metadata
npm run generate:gpu-catalog    # validate and copy the source-backed hardware catalog
npm run verify:catalogs
```

Hardware records carry vendor URLs, the date checked, per-device capacity, aggregate system capacity where applicable, and topology caveats. Cloud rates are reference snapshots with provider pricing links; region, term, spot/reservation status, and availability can change, so verify before purchasing capacity.

## Testing

Run the unit test suite to verify estimator calculations:

```bash
npm test
```

Run the full local verification flow:

```bash
npm run lint:strict
npm run typecheck
npm test -- --runInBand
npm run validate:math
npm run build
npm run verify:export
```

The repo baseline is currently Node `24.14.1` with npm `11.11.0` or newer.

## Deployment

- The site is built as a **static export** (`output: 'export'`) and deployed to **GitHub Pages** from `.github/workflows/nextjs.yml`.
- Production builds use `NEXT_PUBLIC_SITE_URL` and `NEXT_PUBLIC_BASE_PATH` to generate correct canonical URLs, sitemap entries, and static asset paths for the repository Pages URL.

## Disclaimer

The estimator uses analytical approximations of transformer memory footprints, throughput and pricing. Results should be treated as indicative; always validate with real workloads before committing to production deployments.

Important distinctions shown in the UI:

- safetensors totals are the strongest available public parameter evidence; range-header fallback counts serialized tensor elements and may differ for tied/shared weights;
- architecture-derived parameter composition, activation memory, overhead, FLOPs, throughput and fit recommendations are analytical estimates;
- multi-GPU aggregate memory is not automatically one contiguous allocation; tensor/pipeline parallelism and interconnect support must be validated in the selected runtime;
- peak TOPS/TFLOPS values are not interchangeable across vendors unless precision and sparsity assumptions match.
- the interactive architecture explorer is a config-driven, implementation-linked map; it does not execute arbitrary Transformers code or claim to reproduce every custom kernel/control-flow branch.

## Contributing

Pull requests that improve model coverage, pricing data or UX are very welcome. Please open an issue to discuss substantial changes before contributing.
