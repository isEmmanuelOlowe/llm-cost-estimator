# LLM Cost Estimator

The LLM Cost Estimator is an interactive Next.js application that helps machine-learning practitioners quickly validate whether a large language model will fit into a particular GPU setup and how much it will cost to run. The calculator combines up-to-date GPU specifications, detailed VRAM breakdowns (weights, activations, KV cache and optimiser state), performance projections and cloud pricing guidance.

## Key capabilities

- **Automatic Hugging Face introspection** – fetch a model configuration from the Hugging Face Hub and auto-populate parameter counts, hidden sizes, attention heads and precision defaults.
- **Detailed VRAM analysis** – quantify memory consumption for model weights, activations, KV cache and optimiser state with a configurable execution mode (inference or training), precision, overhead factor and batch size.
- **Hardware fit recommendations** – compare the required VRAM against a curated list of GPUs, highlight whether a selected GPU has enough headroom and propose the closest alternatives.
- **Performance estimations** – estimate FLOPs per forward pass, tokens per second and milliseconds per token using the selected GPU’s compute throughput and an efficiency factor.
- **Cloud cost calculator** – explore on-demand pricing across popular AWS, GCP, Azure and independent GPU providers, override hourly rates and receive a total cost projection for the planned runtime.

## Getting started

1. Install dependencies:

   ```bash
   npm install
   ```

2. Start the development server:

   ```bash
   npm run dev
   ```

3. Navigate to `http://localhost:3000` and search for a Hugging Face model such as `meta-llama/Llama-2-7b-hf`.

## Testing

Run the unit test suite to verify estimator calculations:

```bash
npm test
```

## Disclaimer

The estimator uses analytical approximations of transformer memory footprints, throughput and pricing. Results should be treated as indicative; always validate with real workloads before committing to production deployments.

## Contributing

Pull requests that improve model coverage, pricing data or UX are very welcome. Please open an issue to discuss substantial changes before contributing.
