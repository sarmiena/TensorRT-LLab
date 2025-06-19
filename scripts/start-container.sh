#!/bin/bash

MODEL_NAME=""
GPUS="all"  # Default value
CONTAINER="tensorrt_llm/release"

# Define print_usage function first (before it's called)
print_usage() {
    echo "Usage:"
    echo "  $0 [<tensorrt-container-name>] --model <model-name> [--gpus <gpu-spec>]"
    echo
    echo "Arguments:"
    echo "  <tensorrt-container-name>   (Optional) Container image name."
    echo "                              Defaults to \"tensorrt-llm/release\""
    echo "  --model <model-name>        (Required) Path or name of the model to use."
    echo "  --gpus <gpu-spec>           (Optional) GPU spec for Docker (e.g., \"all\", \"device=0,1\")"
}

if [[ "$1" != --* ]]; then
    CONTAINER="$1"
    shift
fi


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
            print_usage
            exit 1
            ;;
    esac
done

# Validate --model arg
if [ -z "$MODEL_NAME" ]; then
    echo "Error: --model argument is required"
    print_usage
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
    -v ./scripts:/scripts \
    -v ./model_weights/$MODEL_NAME:/model \
    -v ./engines:/engines \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    $CONTAINER
