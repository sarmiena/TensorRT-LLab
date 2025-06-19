# TensorRT-LLM Model Management

A comprehensive toolkit for managing different quantizations, builds, and models for fast testing and benchmarking with TensorRT-LLM. This project provides a streamlined workflow to build, tag, and serve multiple model configurations for performance evaluation.

## Overview

This project simplifies the process of:
- Managing multiple model variants with tagged configurations
- Building TensorRT-LLM engines with different quantization settings
- Serving models for benchmarking and testing
- Comparing performance across different build configurations

### Multiple Model Engine Management

The system supports multiple models engines simultaneously:
```bash
# Build different models
/scripts/build-engine.sh --model llama-7b --tag fp8-optimized
/scripts/build-engine.sh --model mistral-7b --tag int4-fast
/scripts/build-engine.sh --model codellama-13b --tag fp16-accuracy

# Each model maintains its own tagged builds
```

### Build Command Tracking

Each build saves its complete configuration to `build-command.json` in the engine directory, including:
- Timestamp
- Applied arguments
- Override parameters
- Full command history

This enables reproducible builds and performance comparisons.

## Project Structure

```
.
├── engines/                    # Built TensorRT engines (organized by model/tag)
├── model_weights/             # Downloaded HuggingFace models
├── scripts/
│   ├── models/               # Model configuration files
│   ├── build-engine.sh       # Main build script
│   ├── trtllm-serve.sh      # Model serving script
│   └── start-container.sh    # Docker container launcher
└── run-benchmark.sh          # Benchmarking script
```

## Prerequisites

- Docker with GPU support
- NVIDIA TensorRT-LLM container
- HuggingFace access token (for downloading models)

## Blackwell SM120 Support
At the time of this writing, PyPi hasn't approved TensorRT-LLM to have a bigger wheel size. As a result, Blackwell SM_120 (ie RTX PRO 6000 and 5090) don't have support on the main docker images. 

Instead we need to build from source starting at 0.21.0rc2

### Building for SM120
This will generate a container named tensorrt_llm/release
    
```                                                                 
sudo apt-get update && sudo apt-get -y install git git-lfs && \     
git lfs install && \                                                
git clone --depth 1 -b v0.21.0rc2 https://github.com/NVIDIA/TensorRT-LLM.git && \
cd TensorRT-LLM && \                                
git submodule update --init --recursive && \
git lfs pull


sudo make -C docker release_build CUDA_ARCHS="120-real"
```


## Getting Started

### 1. Download Model from HuggingFace

Create a directory for your model in `model_weights/` and download the model files:

```bash

# Use git-lfs or huggingface-hub to download
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct model_weights/meta-llama_Llama-3.1-8B-Instruct

OR

huggingface-cli download nvidia/Llama-3_3-Nemotron-Super-49B-v1 --local-dir ./model_weights/ --local-dir-use-symlinks False

```

### 2. Create Model Configuration

Create a JSON configuration file in `scripts/models/` with the same name as your model directory:

```bash
# Example: scripts/models/meta-llama_Llama-3.1-8B-Instruct.json
```

**Configuration Format:**
```json
{
  "quantize": {
    "model_dir": "/model",
    "dtype": "bfloat16",
    "qformat": "fp8",
    "kv_cache_dtype": "fp8",
    "output_dir": "/ckpt",
    "calib_size": "512"
  },
  "trtllm_build": {
    "checkpoint_dir": "/ckpt",
    "remove_input_padding": "enable",
    "kv_cache_type": "paged",
    "max_batch_size": "512",
    "max_num_tokens": "16355",
    "max_seq_len": "16355",
    "use_paged_context_fmha": "enable",
    "use_fp8_context_fmha": "enable",
    "gemm_plugin": "disable",
    "multiple_profiles": "enable"
  }
}
```

### 3. Start TensorRT Container

```bash
Usage:
  ./scripts/start-container.sh [<tensorrt-container-name>] --model <model-name> [--gpus <gpu-spec>]

Arguments:
  <tensorrt-container-name>   (Optional) Name of the TensorRT-LLM container image to use.
                              Defaults to "tensorrt-llm/release"
  --model <model-name>        (Required) Model name or path to mount inside the container.
  --gpus <gpu-spec>           (Optional) GPU specification passed to Docker (e.g., "all" or "device=0,1")

```

**Example:**
```bash
./scripts/start-container.sh --model meta-llama_Llama-3.1-8B-Instruct --gpus all
```

### 4. Quantize and Build the Engine

Inside the container, use the build script to create tagged engine builds. 

NOTE: Scripts within the container will use the model the container was given during the ./start-container.sh invocation (via --model)

```bash
/scripts/build-engine.sh --help

Usage:
  ./build-engine.sh --tag <tag> [--quantize-<arg> <value>] [--trtllm-build-<arg> <value>]
  ./build-engine.sh --list-tags
  ./build-engine.sh --show-build-metadata <tag>
  ./build-engine.sh --delete-tag <tag>
  ./build-engine.sh --help <tag>

/scripts/build-engine.sh --tag <tag-name> [overrides]
```

**Examples:**

Basic build:
```bash
ubuntu@host:TensorRT-LLab$ ./scripts/start-container.sh --model meta-llama_Llama-3.1-8B-Instruct --gpus all
root@container:# /scripts/build-engine.sh --tag default
```

You can also build with custom quantization and TensorRT options that will add/override the defaults in your model's json configuration.

Build with custom quantization:
```bash
root@container:# /scripts/build-engine.sh --model meta-llama_Llama-3.1-8B-Instruct --tag int4-awq \
  --quantize-qformat int4_awq \
  --quantize-kv_cache_dtype int8
```

Build with custom TensorRT settings:
```bash
root@container:# /scripts/build-engine.sh --model meta-llama_Llama-3.1-8B-Instruct --tag high-throughput \
  --trtllm-build-max_batch_size 1024 \
  --trtllm-build-max_num_tokens 32768
```

**Show build metadata:**
The ./build-engine.sh --show-build-metadata <tag> is useful to for inspecting exactly what the script produced during quantization and engine building.

```
root@container:# /scripts/build-engine.sh --show-build-metadata default

Build command for model 'meta-llama_Llama-3.1-8B-Instruct', tag 'default':
{
  "timestamp": "2025-06-19T18:02:33+00:00",
  "model": "meta-llama_Llama-3.1-8B-Instruct",
  "tag": "default",
  "commands": {
    "quantize": "python3 /app/tensorrt_llm/examples/quantization/quantize.py --qformat fp8 --kv_cache_dtype fp8 --dtype bfloat16 --calib_size 512 --output_dir /ckpt --model_dir /model",
    "trtllm_build": "trtllm-build --remove_input_padding enable  --checkpoint_dir /ckpt  --use_paged_context_fmha enable  --output_dir /engines/meta-llama_Llama-3.1-8B-Instruct/default --gemm_plugin disable  --max_batch_size 512  --multiple_profiles enable  --max_seq_len 16355  --kv_cache_type paged  --max_num_tokens 16355  --use_fp8_context_fmha enable "
  },
  "applied_arguments": {
    "quantize": {
      "--qformat fp8": "",
      "--kv_cache_dtype fp8": "",
      "--dtype bfloat16": "",
      "--calib_size 512": "",
      "--output_dir /ckpt": "",
      "--model_dir /model": ""
    },
    "trtllm_build": {
      "--remove_input_padding enable": "",
      "--checkpoint_dir /ckpt": "",
      "--use_paged_context_fmha enable": "",
      "--output_dir": "/engines/meta-llama_Llama-3.1-8B-Instruct/default",
      "--gemm_plugin disable": "",
      "--max_batch_size 512": "",
      "--multiple_profiles enable": "",
      "--max_seq_len 16355": "",
      "--kv_cache_type paged": "",
      "--max_num_tokens 16355": "",
      "--use_fp8_context_fmha enable": ""
    }
  },
  "overrides": {
    "quantize": {},
    "trtllm_build": {}
  }
}

```

**List available tags:**
```bash
root@container:# /scripts/build-engine.sh --list-tags

Available tags:
Model: meta-llama_Llama-3.1-8B-Instruct
  - default
  - tp_size2

```

**Delete a tagged build:**
```bash
root@container:# /scripts/build-engine.sh --list-tags

Available tags:
Model: meta-llama_Llama-3.1-8B-Instruct
  - default
  - tp_size2

root@container:# /scripts/build-engine.sh --delete-tag tp_size2

Do you want to delete 'tp_size2'? (y/N): y
Removing 'tp_size2'...

root@container:# /scripts ./build-engine.sh --list-tags

Available tags:
Model: meta-llama_Llama-3.1-8B-Instruct
  - default

```

## Running Models

### Serve Model

Inside the container:
```bash
root@container:# /scripts/trtllm-serve.sh --tag <tag-name>
```

**Examples:**
```bash
# Serve the fp8-default build
root@container:# /scripts/trtllm-serve.sh --tag fp8-default

# List available tags for the current model
root@container:# /scripts/trtllm-serve.sh --list-tags
```

The server will start on `localhost:8000` by default.

### Benchmark Model
Suggested to use inference-benchmarker from https://github.com/huggingface/inference-benchmarker from the host machine

```bash
ubuntu@host:TensorRT-LLab$ ./run-benchmark.example.sh
```
