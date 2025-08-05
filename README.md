# TensorRT-LLM Model Management

A comprehensive toolkit for managing different quantizations, builds, and models for fast testing and benchmarking with TensorRT-LLM. This project provides a streamlined workflow to build, tag, and serve multiple model configurations for performance evaluation.

## Overview

This project simplifies the process of:
- Managing multiple model variants with tagged configurations
- Building TensorRT-LLM engines with different quantization settings
- Serving models for benchmarking and testing
- Comparing performance across different build configurations

### Multiple Model Engine Management

## Prerequisites

- Docker with GPU support
- NVIDIA TensorRT-LLM container
- HuggingFace access token (for downloading models)


## Getting Started by Building Container
Use the build-container wizard
```
./build-container
```

## Getting Started on Serving

### 1. Download Model from HuggingFace

Create a directory for your model in `model_weights/` and download the model files:

```bash

# Use git-lfs or huggingface-hub to download
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct model_weights/meta-llama_Llama-3.1-8B-Instruct

OR

huggingface-cli download nvidia/Llama-3_3-Nemotron-Super-49B-v1 --local-dir ./model_weights/ --local-dir-use-symlinks False

```

### 2. Create Model Serve Config for Pytorch
This path covers serving a model with pytorch as opposed to a native TensorRT-LLM engine. It's unclear if NVIDIA will be
supporting native TensorRT-LLM engines in the future because some folks at NVIDIA say they're moving towards pytorch.

Create a JSON configuration file in `./model_serve_args` that will be used to serve your model:

```bash
# Example: ./model_serve_args/meta-llama_Llama-3.1-8B-Instruct.default.json
```

**Configuration Format to serve with pytorch**
```json
{
  "notes": "From Alex Steiner NVIDIA saying serve it straight",
  "args": {
    "--backend": "pytorch",
    "--extra_llm_api_options": "pytorch-small-batch.yml",
    "--host": "0.0.0.0",
    "--port": "8000",
    "--max_batch_size": 128,
    "--tp_size": 1,
    "--max_num_tokens": 2048
  }
}
```

Notice that we're putting additional llm api options there. Let's create that file in ```./extra_llm_api_options```

```
# extra_llm_api_options/pytorch-small-batch.yml
print_iter_log: true
cuda_graph_config:
  batch_sizes: [1,2,4,8,16,32,64,128,256,512,1024,2048]
```

### 3. Start TensorRT Container

```bash
./trt-llab trtllm-serve
```
