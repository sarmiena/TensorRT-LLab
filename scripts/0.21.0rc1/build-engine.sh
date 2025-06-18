#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Attempting to build $MODEL"
# Parse command line arguments
TAG=""
declare -A QUANTIZE_OVERRIDES
declare -A TRTLLM_OVERRIDES
LIST_TAGS=false
SHOW_BUILD_COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --list-tags)
            LIST_TAGS=true
            shift
            ;;
        --show-build-command)
            SHOW_BUILD_COMMAND="$2"
            shift 2
            ;;
        --quantize-*)
            # Extract the argument name (remove --quantize- prefix)
            ARG_NAME="${1#--quantize-}"
            QUANTIZE_OVERRIDES["$ARG_NAME"]="$2"
            shift 2
            ;;
        --trtllm-build-*)
            # Extract the argument name (remove --trtllm-build- prefix)
            ARG_NAME="${1#--trtllm-build-}"
            TRTLLM_OVERRIDES["$ARG_NAME"]="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_name> --tag <tag> [--quantize-<arg> <value>] [--trtllm-build-<arg> <value>]"
            echo "       $0 --list-tags [--model <model_name>]"
            echo "       $0 --show-build-command <tag> --model <model_name>"
            exit 1
            ;;
    esac
done

# Function to list all tags
list_tags() {
    local model_filter="$1"
    
    if [ ! -d "/engine" ]; then
        echo "No engine directory found at /engine"
        return
    fi
    
    echo "Available tags:"
    
    if [ -n "$model_filter" ]; then
        if [ -d "/engine/$model_filter" ]; then
            echo "Model: $model_filter"
            for tag_dir in "/engine/$model_filter"/*; do
                if [ -d "$tag_dir" ]; then
                    tag=$(basename "$tag_dir")
                    echo "  - $tag"
                fi
            done
        else
            echo "No builds found for model: $model_filter"
        fi
    else
        for model_dir in "/engine"/*; do
            if [ -d "$model_dir" ]; then
                model=$(basename "$model_dir")
                echo "Model: $model"
                for tag_dir in "$model_dir"/*; do
                    if [ -d "$tag_dir" ]; then
                        tag=$(basename "$tag_dir")
                        echo "  - $tag"
                    fi
                done
            fi
        done
    fi
}

# Function to show build command for a tag
show_build_command() {
    local model="$1"
    local tag="$2"
    
    if [ -z "$model" ] || [ -z "$tag" ]; then
        echo "Error: Both --model and --show-build-command (tag) are required"
        exit 1
    fi
    
    local build_command_file="/engine/$model/$tag/build-command.json"
    
    if [ ! -f "$build_command_file" ]; then
        echo "Error: Build command file not found: $build_command_file"
        exit 1
    fi
    
    echo "Build command for model '$model', tag '$tag':"
    cat "$build_command_file"
}

# Handle special modes
if [ "$LIST_TAGS" = true ]; then
    list_tags "$MODEL"
    exit 0
fi

if [ -n "$SHOW_BUILD_COMMAND" ]; then
    show_build_command "$MODEL" "$SHOW_BUILD_COMMAND"
    exit 0
fi

# Check if model and tag arguments were provided
if [ -z "$MODEL" ]; then
    echo "Error: --model argument is required"
    echo "Usage: $0 --model <model_name> --tag <tag> [--quantize-<arg> <value>] [--trtllm-build-<arg> <value>]"
    exit 1
fi

if [ -z "$TAG" ]; then
    echo "Error: --tag argument is required"
    echo "Usage: $0 --model <model_name> --tag <tag> [--quantize-<arg> <value>] [--trtllm-build-<arg> <value>]"
    exit 1
fi

# Function to build quantize.py command with overrides
build_quantize_cmd() {
    local base_args=("$@")
    local cmd="python3 /app/tensorrt_llm/examples/quantization/quantize.py"
    
    # Add base arguments
    for arg in "${base_args[@]}"; do
        cmd="$cmd $arg"
    done
    
    # Apply overrides and additions
    for key in "${!QUANTIZE_OVERRIDES[@]}"; do
        local value="${QUANTIZE_OVERRIDES[$key]}"
        # Check if this argument already exists in base_args and remove it
        local new_base_args=()
        local skip_next=false
        for i in "${!base_args[@]}"; do
            if [ "$skip_next" = true ]; then
                skip_next=false
                continue
            fi
            if [[ "${base_args[$i]}" == "--$key" ]]; then
                skip_next=true
                continue
            fi
            new_base_args+=("${base_args[$i]}")
        done
        
        # Add the override/new argument
        cmd="$cmd --$key $value"
    done
    
    echo "$cmd"
}

# Function to build trtllm-build command with overrides
build_trtllm_cmd() {
    local base_args=("$@")
    local cmd="trtllm-build"
    
    # Add base arguments
    for arg in "${base_args[@]}"; do
        cmd="$cmd $arg"
    done
    
    # Apply overrides and additions
    for key in "${!TRTLLM_OVERRIDES[@]}"; do
        local value="${TRTLLM_OVERRIDES[$key]}"
        # Check if this argument already exists in base_args and remove it
        local new_base_args=()
        local skip_next=false
        for i in "${!base_args[@]}"; do
            if [ "$skip_next" = true ]; then
                skip_next=false
                continue
            fi
            if [[ "${base_args[$i]}" == "--$key" ]]; then
                skip_next=true
                continue
            fi
            new_base_args+=("${base_args[$i]}")
        done
        
        # Add the override/new argument
        cmd="$cmd --$key $value"
    done
    
    echo "$cmd"
}

# Function to save build command details
save_build_command() {
    local output_dir="$1"
    local quantize_cmd="$2"
    local trtllm_cmd="$3"
    
    cat > "$output_dir/build-command.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "model": "$MODEL",
  "tag": "$TAG",
  "commands": {
    "quantize": "$quantize_cmd",
    "trtllm_build": "$trtllm_cmd"
  },
  "overrides": {
    "quantize": {
$(for key in "${!QUANTIZE_OVERRIDES[@]}"; do
    echo "      \"$key\": \"${QUANTIZE_OVERRIDES[$key]}\","
done | sed '$s/,$//')
    },
    "trtllm_build": {
$(for key in "${!TRTLLM_OVERRIDES[@]}"; do
    echo "      \"$key\": \"${TRTLLM_OVERRIDES[$key]}\","
done | sed '$s/,$//')
    }
  }
}
EOF
}

# Define output directory
OUTPUT_DIR="/engine/$MODEL/$TAG"

# Check if output directory exists and handle overwrite
if [ -d "$OUTPUT_DIR" ]; then
    echo "Warning: Directory $OUTPUT_DIR already exists."
    read -p "Do you want to delete and overwrite? (y/N): " confirm
    case $confirm in
        [Yy]* )
            echo "Removing existing directory..."
            rm -rf "$OUTPUT_DIR"
            ;;
        * )
            echo "Exiting without changes."
            exit 0
            ;;
    esac
fi

# Clean up previous builds (transient ckpt and specific engine files)
rm -rf /ckpt
rm -rf "$OUTPUT_DIR/config.json"
rm -rf "$OUTPUT_DIR"/*.engine

# Check if model configuration file exists
MODEL_CONFIG_FILE="$SCRIPT_DIR/models/${MODEL}.sh"
if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "Error: Model configuration file not found: $MODEL_CONFIG_FILE"
    echo "Available models:"
    if [ -d "$SCRIPT_DIR/models" ]; then
        for model_file in "$SCRIPT_DIR/models"/*.sh; do
            if [ -f "$model_file" ]; then
                basename "$model_file" .sh | sed 's/^/  - /'
            fi
        done
    else
        echo "  No models directory found"
    fi
    exit 1
fi

# Create output directory
mkdir -p /ckpt
mkdir -p "$OUTPUT_DIR"

# Source the model configuration
source "$MODEL_CONFIG_FILE"

# Execute the model build
echo "Building $MODEL with tag '$TAG'..."
echo "Output directory: $OUTPUT_DIR"

# Override the global OUTPUT_DIR variable for model configs
MODEL_OUTPUT_DIR="$OUTPUT_DIR"

build_model

echo "Build completed for model: $MODEL, tag: $TAG"

# Print summary of applied overrides if any
if [ ${#QUANTIZE_OVERRIDES[@]} -gt 0 ]; then
    echo "Applied quantize.py overrides:"
    for key in "${!QUANTIZE_OVERRIDES[@]}"; do
        echo "  --$key ${QUANTIZE_OVERRIDES[$key]}"
    done
fi

if [ ${#TRTLLM_OVERRIDES[@]} -gt 0 ]; then
    echo "Applied trtllm-build overrides:"
    for key in "${!TRTLLM_OVERRIDES[@]}"; do
        echo "  --$key ${TRTLLM_OVERRIDES[$key]}"
    done
fi
