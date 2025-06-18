#!/bin/bash

MODEL_NAME=""
GPUS="all"  # Default value

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 --model <model-name> [--gpus <gpu-spec>]"
            exit 1
            ;;
    esac
done

# Validate --model arg
if [ -z "$MODEL_NAME" ]; then
    echo "Error: --model argument is required"
    echo "Usage: $0 --model <model-name>"
    exit 1
fi

# Check if directory exists
if [ ! -d "./model_weights/$MODEL_NAME" ]; then
    echo "Error: Directory './model_weights/$MODEL_NAME' not found in the current directory"
    exit 1
fi

# Run the container
docker run --gpus $GPUS \
    -e MODEL=$MODEL_NAME \
    -it --rm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./scripts/0.21.0rc1:/scripts \
    -v ./model_weights/$MODEL_NAME:/model \
    -v ./engines/0.21.0rc1:/engine \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    tensorrt_llm/release
