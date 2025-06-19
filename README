# TensorRT-LLM Model Management

A comprehensive toolkit for managing different quantizations, builds, and models for fast testing and benchmarking with TensorRT-LLM. This project provides a streamlined workflow to build, tag, and serve multiple model configurations for performance evaluation.

## Overview

This project simplifies the process of:
- Building TensorRT-LLM engines with different quantization settings
- Managing multiple model variants with tagged configurations
- Serving models for benchmarking and testing
- Comparing performance across different build configurations

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

## Setup

### 1. Download Model from HuggingFace

Create a directory for your model in `model_weights/` and download the model files:

```bash
# Example: Download Llama 3.1 8B Instruct
mkdir -p model_weights/meta-llama_Llama-3.1-8B-Instruct

# Use git-lfs or huggingface-hub to download
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct model_weights/meta-llama_Llama-3.1-8B-Instruct
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

Use the provided example file as a template:
```bash
cp scripts/models/meta-llama_Llama-3.1-8B-Instruct.json.example scripts/models/your-model-name.json
```

## Building Models

### Start TensorRT Container

```bash
./scripts/start-container.sh <tensorrt-container-name> --model <model-name> [--gpus <gpu-spec>]
```

**Example:**
```bash
./scripts/start-container.sh nvcr.io/nvidia/tensorrt:24.02-py3 --model meta-llama_Llama-3.1-8B-Instruct --gpus all
```

### Build Engine

Inside the container, use the build script to create tagged engine builds:

```bash
/scripts/build-engine.sh --model <model-name> --tag <tag-name> [overrides]
```

**Examples:**

Basic build:
```bash
/scripts/build-engine.sh --model meta-llama_Llama-3.1-8B-Instruct --tag fp8-default
```

Build with custom quantization:
```bash
/scripts/build-engine.sh --model meta-llama_Llama-3.1-8B-Instruct --tag int4-awq \
  --quantize-qformat int4_awq \
  --quantize-kv_cache_dtype int8
```

Build with custom TensorRT settings:
```bash
/scripts/build-engine.sh --model meta-llama_Llama-3.1-8B-Instruct --tag high-throughput \
  --trtllm-build-max_batch_size 1024 \
  --trtllm-build-max_num_tokens 32768
```

### Managing Builds

**List available tags:**
```bash
/scripts/build-engine.sh --list-tags
```

**Show build configuration:**
```bash
/scripts/build-engine.sh --show-build-command <tag-name>
```

**Delete a tagged build:**
```bash
/scripts/build-engine.sh --delete-tag <tag-name>
```

## Running Models

### Serve Model

Inside the container:
```bash
/scripts/trtllm-serve.sh --tag <tag-name>
```

**Examples:**
```bash
# Serve the fp8-default build
/scripts/trtllm-serve.sh --tag fp8-default

# List available tags for the current model
/scripts/trtllm-serve.sh --list-tags
```

The server will start on `localhost:8000` by default.

### Benchmark Model

From the host machine, run the benchmarking script:

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Run benchmark
./run-benchmark.sh
```

The benchmark script is configured to:
- Test at 100 RPS
- Run for 60 seconds with 15s warmup
- Use 200-token prompts and responses
- Support up to 800 virtual users

## Advanced Usage

### Override Parameters

You can override any quantization or TensorRT build parameter at runtime:

**Quantization overrides:**
- `--quantize-<parameter> <value>`: Override quantize.py parameters
- Example: `--quantize-calib_size 1024`

**TensorRT build overrides:**
- `--trtllm-build-<parameter> <value>`: Override trtllm-build parameters  
- Example: `--trtllm-build-max_batch_size 256`

### Build Command Tracking

Each build saves its complete configuration to `build-command.json` in the engine directory, including:
- Timestamp
- Applied arguments
- Override parameters
- Full command history

This enables reproducible builds and performance comparisons.

### Multiple Model Management

The system supports multiple models simultaneously:
```bash
# Build different models
/scripts/build-engine.sh --model llama-7b --tag fp8-optimized
/scripts/build-engine.sh --model mistral-7b --tag int4-fast
/scripts/build-engine.sh --model codellama-13b --tag fp16-accuracy

# Each model maintains its own tagged builds
```

## Troubleshooting

**Container Issues:**
- Ensure GPU drivers and Docker GPU support are properly installed
- Verify the TensorRT container image is available and compatible

**Model Download Issues:**
- Check HuggingFace access permissions for gated models
- Ensure sufficient disk space in `model_weights/`

**Build Failures:**
- Review model configuration JSON for syntax errors
- Check GPU memory availability for large models
- Verify quantization format compatibility

**Serving Issues:**
- Confirm the tagged engine exists: `/scripts/trtllm-serve.sh --list-tags`
- Check GPU memory allocation
- Review engine build logs for errors

## Performance Tips

1. **Memory Optimization**: Adjust `max_batch_size` and `max_num_tokens` based on GPU memory
2. **Quantization**: Use FP8 for best performance/quality tradeoff, INT4 for maximum throughput
3. **Context Attention**: Enable `use_fp8_context_fmha` for longer sequences
4. **Batch Processing**: Increase `max_batch_size` for higher throughput scenarios

## Contributing

When adding new models:
1. Follow the naming convention: `organization_model-name`
2. Test with multiple quantization formats
3. Document any model-specific requirements
4. Update configuration examples

This toolkit provides a robust foundation for TensorRT-LLM experimentation and production deployment.