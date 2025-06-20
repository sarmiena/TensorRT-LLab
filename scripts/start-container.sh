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

list_available_models() {
    echo "Available models:"
    if [ -d "./model_weights" ]; then
        local found_models=false
        for model_dir in ./model_weights/*/; do
            if [ -d "$model_dir" ]; then
                model_name=$(basename "$model_dir")
                # Skip .gitkeep and other non-model files
                if [ "$model_name" != ".gitkeep" ] && [ -d "$model_dir" ]; then
                    echo "  - $model_name"
                    found_models=true
                fi
            fi
        done

        if [ "$found_models" = false ]; then
            echo "  No models found in ./model_weights/"
            echo
            echo "To add a model, download it to ./model_weights/<model-name>/"
            echo "Example:"
            echo "  git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct model_weights/meta-llama_Llama-3.1-8B-Instruct"
        fi
    else
        echo "  model_weights directory not found"
        echo "  Please create ./model_weights/ and add your models there"
    fi
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
            echo
            list_available_models
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
