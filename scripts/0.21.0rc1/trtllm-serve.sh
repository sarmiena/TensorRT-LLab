#!/bin/bash
TAG=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --list-tags)
            echo "Available tags for model $MODEL:"
            if [ -d "/engine/$MODEL" ]; then
                ls -1 /engine/$MODEL/ 2>/dev/null || echo "No tags found"
            else
                echo "No engine directory found for model $MODEL"
            fi
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 --tag <tag-name>"
            echo "       $0 --list-tags"
            exit 1
            ;;
    esac
done

# Validate --tag arg
if [ -z "$TAG" ]; then
    echo "Error: --tag argument is required"
    echo "Usage: $0 --tag <tag-name>"
    echo "       $0 --list-tags"
    exit 1
fi

# Check if MODEL environment variable is set
if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable is not set"
    exit 1
fi

# Check if engine directory exists
ENGINE_PATH="/engine/$MODEL/$TAG"
if [ ! -d "$ENGINE_PATH" ]; then
    echo "Error: Engine directory '$ENGINE_PATH' not found"
    echo "Available tags for model $MODEL:"
    if [ -d "/engine/$MODEL" ]; then
        ls -1 /engine/$MODEL/ 2>/dev/null || echo "No tags found"
    else
        echo "No engine directory found for model $MODEL"
    fi
    exit 1
fi

# Set environment variables and run trtllm-serve
export KV_CACHE_FREE_GPU_MEM_FRACTION=0.9 && \
export ENGINE_DIR="$ENGINE_PATH" && \
export TOKENIZER_DIR=/model
trtllm-serve ${ENGINE_DIR} --tokenizer ${TOKENIZER_DIR} --max_batch_size=512
