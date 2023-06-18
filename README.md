# LLM Cost Predictor
  A project developed to predict the computational costs associated with training and inferencing large language models (LLMs). The tool utilizes various key model parameters such as FLOPs, number of parameters, memory bandwidth, and others to provide an accurate estimate of GPU usage, inference time, and overall training cost.

## Features

- **GPU Memory Estimation**: Provide the number of parameters of an LLM and get an estimation of the total GPU memory required to run it.
- **FLOPs Calculation**: Enter your model's architecture description (number of layers, sequence length, hidden dimensions, number of heads, etc.), and ProphetAI will output the estimated total FLOPs of the model.
- **Inference Time Estimation**: ProphetAI can predict the inference time of your LLM on a specified GPU, factoring in memory bandwidth and software overhead.
- **Training Cost Estimation**: By inputting the necessary details (including GPU hourly cost, number of epochs, and dataset size), you can receive an estimate of the overall cost of training your LLM.

## Limitations and Disclaimer

The estimates provided are based on simplified models of computational requirements and costs. The actual requirements and costs can be higher due to a variety of factors not accounted for in these estimates, such as data loading time, network overhead, model serialization/deserialization time, potential additional costs like data transfer, storage, or CPU/RAM usage, and others.

Please use this tool as a rough guideline and always measure the actual requirements and costs by running your specific models on your target hardware and software stack.

## Contributions

Contributions are welcome!
