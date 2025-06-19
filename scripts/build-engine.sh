#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
TAG=""
declare -A QUANTIZE_OVERRIDES
declare -A TRTLLM_OVERRIDES

declare -A COMPOSED_TRTLLM_ARGS_MAP
declare -A COMPOSED_QUANTIZE_ARGS_MAP


LIST_TAGS=false
SHOW_BUILD_COMMAND=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --delete-tag)
            DELETE_TAG="$2"
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
            ARG_NAME="${1#--quantize-}"
            QUANTIZE_OVERRIDES["--$ARG_NAME"]="$2"
            shift 2
            ;;
        --trtllm-build-*)
            ARG_NAME="${1#--trtllm-build-}"
            TRTLLM_OVERRIDES["--$ARG_NAME"]="$2"
            shift 2
            ;;
        *)
	    echo
            echo "Unknown argument: $1"
            echo
            echo "Usage:"
            echo "  $0 --tag <tag> [--quantize-<arg> <value>] [--trtllm-build-<arg> <value>]"
            echo "  $0 --list-tags"
            echo "  $0 --show-build-command <tag>"
            echo "  $0 --delete-tag <tag>"
            exit 1
            ;;
    esac
done

confirm_and_delete_tagged_engine() {
    local dir="$1"
    local tag=$(basename "$dir")

    if [[ ! -d "$dir" ]]; then
        echo "'$tag' does not exist. Skipping deletion."
        return 0
    fi

    echo
    echo -ne "\033[1;33mDo you want to $2 '$tag'? (y/N):\033[0m \c"
    read confirm
    case "$confirm" in
        [Yy]* )
            echo "Removing '$tag'..."
            rm -rf "$dir"
            ;;
        * )
            echo "Exiting without changes."
            exit 0
            ;;
    esac
}


# Function to list all tags
list_tags() {
    local model_filter="$1"
    
    if [ ! -d "/engines" ]; then
        echo "No engine directory found at /engines"
        return
    fi
    
    echo "Available tags:"
    
    if [ -n "$model_filter" ]; then
        if [ -d "/engines/$model_filter" ]; then
            echo "Model: $model_filter"
            for tag_dir in "/engines/$model_filter"/*; do
                if [ -d "$tag_dir" ]; then
                    tag=$(basename "$tag_dir")
                    echo "  - $tag"
                fi
            done
        else
            echo "No builds found for model: $model_filter"
        fi
    else
        for model_dir in "/engines"/*; do
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
    
    local build_command_file="/engines/$model/$tag/build-command.json"
    
    if [ ! -f "$build_command_file" ]; then
        echo "Error: Build command file not found: $build_command_file"
        exit 1
    fi
    
    echo
    echo -ne "\033[1;33mBuild command for model '$model', tag '$tag':\033[0m\c"
    echo
    cat "$build_command_file" | jq
}

build_model() {
    mapfile -t quantize_base_args < <(jq -r '.quantize | to_entries[] | "--\(.key) \(.value)"' "$MODEL_CONFIG_FILE")
    mapfile -t trtllm_base_args < <(jq -r '.trtllm_build | to_entries[] | "--\(.key) \(.value)"' "$MODEL_CONFIG_FILE")
    trtllm_base_args+=("--output_dir" "$MODEL_OUTPUT_DIR")

    build_quantize_cmd "${quantize_base_args[@]}"
    build_trtllm_cmd "${trtllm_base_args[@]}"

    echo "Running: $COMPOSED_QUANTIZE_COMMAND"
    eval "$COMPOSED_QUANTIZE_COMMAND"

    echo "Running: $COMPOSED_TRTLLM_COMMAND"
    eval "$COMPOSED_TRTLLM_COMMAND"

    save_build_command
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

if [[ -n "${DELETE_TAG:-}" ]]; then
    tag_dir="/engines/$MODEL/$DELETE_TAG"

    if [[ -d "$tag_dir" ]]; then
        confirm_and_delete_tagged_engine "$tag_dir" "delete"
	exit 1
    else
        echo -ne "\033[1;31mTag '$DELETE_TAG' does not exist at: $tag_dir\033[0m\c" >&2
	echo
        exit 1
    fi
fi

if [ -z "$TAG" ]; then
    echo "Error: --tag argument is required"
    echo "Usage: $0 --model <model_name> --tag <tag> [--quantize-<arg> <value>] [--trtllm-build-<arg> <value>]"
    exit 1
fi

if [ "$DELETE_TAG" ]; then
    echo
    echo -ne "\033[1;31mWarning: Tag '$(basename "$OUTPUT_DIR")' already exists.\033[0m \c"
    confirm_and_delete_tagged_engine "$OUTPUT_DIR" "delete and overwrite"
fi

# Function to build quantize.py command with overrides
build_quantize_cmd() {
    local base_args=("$@")
    local cmd="python3 /app/tensorrt_llm/examples/quantization/quantize.py"

    for ((i=0; i < ${#base_args[@]}; i++)); do
        key="${base_args[$i]}"
        next="${base_args[$((i + 1))]:-}"

        if [[ "$next" == --* ]] || [[ -z "$next" ]]; then
            val=""
        else
            val="$next"
            ((i++))
        fi

        COMPOSED_QUANTIZE_ARGS_MAP["$key"]="$val"
    done

    # Apply overrides
    for key in "${!QUANTIZE_OVERRIDES[@]}"; do
        COMPOSED_QUANTIZE_ARGS_MAP["$key"]="${QUANTIZE_OVERRIDES[$key]}"
    done

    # Rebuild command
    for key in "${!COMPOSED_QUANTIZE_ARGS_MAP[@]}"; do
        if [[ -n "${COMPOSED_QUANTIZE_ARGS_MAP[$key]}" ]]; then
            cmd+=" $key ${COMPOSED_QUANTIZE_ARGS_MAP[$key]}"
        else
            cmd+=" $key"
        fi
    done

    COMPOSED_QUANTIZE_COMMAND="$cmd"
}

# Function to build trtllm-build command with overrides
build_trtllm_cmd() {
    local base_args=("$@")
    local cmd="trtllm-build"

    for ((i=0; i < ${#base_args[@]}; i++)); do
        key="${base_args[$i]}"
        next="${base_args[$((i + 1))]:-}"

        if [[ "$next" == --* ]] || [[ -z "$next" ]]; then
            val=""
        else
            val="$next"
            ((i++))
        fi
    
        COMPOSED_TRTLLM_ARGS_MAP["$key"]="$val"
    done

    for key in "${!TRTLLM_OVERRIDES[@]}"; do
    	COMPOSED_TRTLLM_ARGS_MAP["$key"]="${TRTLLM_OVERRIDES[$key]}"
    done

    for key in "${!COMPOSED_TRTLLM_ARGS_MAP[@]}"; do
        cmd+=" $key ${COMPOSED_TRTLLM_ARGS_MAP[$key]}"
    done
    
    COMPOSED_TRTLLM_COMMAND="$cmd"
}

# Function to save build command details
save_build_command() {
    echo "Saving '$OUTPUT_DIR/build-command.json'"
    
    cat > "$OUTPUT_DIR/build-command.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "model": "$MODEL",
  "tag": "$TAG",
  "commands": {
    "quantize": "$COMPOSED_QUANTIZE_COMMAND",
    "trtllm_build": "$COMPOSED_TRTLLM_COMMAND"
  },
  "applied_arguments": {
    "quantize": {
$(for key in "${!COMPOSED_QUANTIZE_ARGS_MAP[@]}"; do
    echo "      \"$key\": \"${COMPOSED_QUANTIZE_ARGS_MAP[$key]}\","
done | sed '$s/,$//')
    },
    "trtllm_build": {
$(for key in "${!COMPOSED_TRTLLM_ARGS_MAP[@]}"; do
    echo "      \"$key\": \"${COMPOSED_TRTLLM_ARGS_MAP[$key]}\","
done | sed '$s/,$//')
    }
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
OUTPUT_DIR="/engines/$MODEL/$TAG"

# Check if output directory exists and handle overwrite
if [ -d "$OUTPUT_DIR" ]; then
    echo
    echo -ne "\033[1;31mWarning: Tag '$(basename "$OUTPUT_DIR")' already exists.\033[0m \c"
    confirm_and_delete_tagged_engine "$OUTPUT_DIR" "delete and overwrite"
fi

# Clean up previous builds (transient ckpt and specific engine files)
rm -rf /ckpt
rm -rf "$OUTPUT_DIR/config.json"
rm -rf "$OUTPUT_DIR"/*.engine

# Check if model configuration file exists
MODEL_CONFIG_FILE="$SCRIPT_DIR/models/${MODEL}.json"
if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "Error: Model configuration file not found: $MODEL_CONFIG_FILE"
    echo "Available models:"
    if [ -d "$SCRIPT_DIR/models" ]; then
        for model_file in "$SCRIPT_DIR/models"/*.json; do
            if [ -f "$model_file" ]; then
                basename "$model_file" .json | sed 's/^/  - /'
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
