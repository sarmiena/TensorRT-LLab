#!/bin/bash
TAG=""

show_help() {
    echo "Usage: $0 --tag <tag-name> [--config <config-name>]"
    echo "       $0 --list-tags"
    echo "       $0 --tag <tag-name> --list-configs"
    echo "       $0 --tag <tag-name> --show-config <config-name>"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
	--config)
            SERVE_CONFIG="$2"
            shift 2
            ;;
        --list-tags)
            echo "Available tags for model $MODEL:"
            if [ -d "/engines/$MODEL" ]; then
                ls -1 /engines/$MODEL/ 2>/dev/null || echo "No tags found"
            else
                echo "No engine directory found for model $MODEL"
            fi
            exit 0
            ;;
	 --list-configs)
            if [ -z "$TAG" ]; then
                echo "Error: --tag is required with --list-configs"
                exit 1
            fi
            ENGINE_PATH="/engines/$MODEL/$TAG"
            SERVE_ARGS_FILE="$ENGINE_PATH/serve-args.json"
            if [ -f "$SERVE_ARGS_FILE" ]; then
                echo "Available serve configurations for tag '$TAG':"
                jq -r 'keys[]' "$SERVE_ARGS_FILE" 2>/dev/null || echo "Error reading serve-args.json"
            else
                echo "No serve-args.json found for tag '$TAG'"
            fi
            exit 0
            ;;
        --show-config)
            if [ -z "$TAG" ]; then
                echo "Error: --tag is required with --show-config"
                exit 1
            fi
            SHOW_CONFIG_NAME="$2"
            shift 2
            ;;
        *)
	    echo
            echo "Unknown parameter: $1"
	    show_help
            ;;
    esac
done

# Validate --tag arg
if [ -z "$TAG" ]; then
    echo
    echo "Error: --tag argument is required"
    show_help
    exit 1
fi

# Check if MODEL environment variable is set
if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable is not set"
    exit 1
fi

# Check if engine directory exists
ENGINE_PATH="/engines/$MODEL/$TAG"
if [ ! -d "$ENGINE_PATH" ]; then
    echo "Error: Engine directory '$ENGINE_PATH' not found"
    echo "Available tags for model $MODEL:"
    if [ -d "/engines/$MODEL" ]; then
        ls -1 /engines/$MODEL/ 2>/dev/null || echo "No tags found"
    else
        echo "No engine directory found for model $MODEL"
    fi
    exit 1
fi

SERVE_ARGS_FILE="$ENGINE_PATH/serve-args.json"
if [ ! -f "$SERVE_ARGS_FILE" ]; then
    echo "Warning: serve-args.json not found at $SERVE_ARGS_FILE"
    echo "Using basic configuration..."
    BASIC_SERVE=true
fi

show_config() {
    local config_name="$1"
    if [ ! -f "$SERVE_ARGS_FILE" ]; then
        echo "Error: serve-args.json not found"
        exit 1
    fi

    if ! jq -e ".\"$config_name\"" "$SERVE_ARGS_FILE" > /dev/null 2>&1; then
        echo "Error: Configuration '$config_name' not found in serve-args.json"
        echo "Available configurations:"
        jq -r 'keys[]' "$SERVE_ARGS_FILE" 2>/dev/null
        exit 1
    fi

    echo "Configuration '$config_name' for tag '$TAG':"
    jq ".\"$config_name\"" "$SERVE_ARGS_FILE"
}

if [ -n "$SHOW_CONFIG_NAME" ]; then
    show_config "$SHOW_CONFIG_NAME"
    exit 0
fi

build_serve_command() {
    local serve_cmd="trtllm-serve ${ENGINE_DIR} --tokenizer ${TOKENIZER_DIR}"

    if [ "$BASIC_SERVE" = true ]; then
        # Fallback to basic serve command
        echo "$serve_cmd"
        return
    fi

    # Check if the configuration exists
    if ! jq -e ".\"$SERVE_CONFIG\"" "$SERVE_ARGS_FILE" > /dev/null 2>&1; then
        echo "Warning: Configuration '$SERVE_CONFIG' not found in serve-args.json"
        echo "Available configurations:"
        jq -r 'keys[]' "$SERVE_ARGS_FILE" 2>/dev/null
        echo "Using basic configuration..."
        echo "$serve_cmd"
        return
    fi

    # Extract args from the specified configuration
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            key=$(echo "$line" | cut -d'=' -f1)
            value=$(echo "$line" | cut -d'=' -f2-)

            # Skip example comments
            if [[ "$key" == "example_comment" ]]; then
                continue
            fi

            if [[ -n "$value" ]]; then
                serve_cmd+=" $key $value"
            else
                serve_cmd+=" $key"
            fi
            echo "Using serve config: $key $value"
        fi
    done < <(jq -r ".\"$SERVE_CONFIG\".args | to_entries[] | \"\(.key)=\(.value)\"" "$SERVE_ARGS_FILE" 2>/dev/null)

    SERVE_COMMAND="$serve_cmd"
}

echo "Starting TensorRT-LLM server for model '$MODEL', tag '$TAG', config '$SERVE_CONFIG'"
echo "Engine path: $ENGINE_PATH"

# Set environment variables
export KV_CACHE_FREE_GPU_MEM_FRACTION=0.9
export ENGINE_DIR="$ENGINE_PATH"
export TOKENIZER_DIR=/model

# Build and execute the serve command
build_serve_command
echo "Executing: $SERVE_COMMAND"
eval "$SERVE_COMMAND"
