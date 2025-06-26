#!/bin/bash

set -euo pipefail

# Global Configuration Variables
TENSORRT_LLM_VERSION=""
TENSORRT_MODEL_OPTIMIZER_VERSION=""
CUDA_ARCH="120-real"
TORCH_CUDA_ARCH_LIST="12.0+PTX"
BASE_IMAGE_TAG=""
FINAL_IMAGE_TAG="tensorrt-llab:latest"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repo paths
TENSORRT_LLM_PATH="$BUILD_DIR/TensorRT-LLM"
TENSORRT_MODEL_OPTIMIZER_PATH="$BUILD_DIR/TensorRT-Model-Optimizer"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if Docker is running
check_docker() {
    log "Checking Docker availability..."
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running or not accessible"
    fi
    success "Docker is running"
}

# Get user input for CUDA architecture settings
get_cuda_arch_settings() {
    echo
    echo "CUDA Architecture Configuration"
    echo "==============================="
    echo "Current defaults:"
    echo "  CUDA_ARCH: $CUDA_ARCH"
    echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    echo
    echo "Common architecture options:"
    echo "  - SM120 (RTX 5090, RTX PRO 6000): CUDA_ARCH='120-real', TORCH_CUDA_ARCH_LIST='12.0+PTX'"
    echo "  - SM90 (RTX 4090, H100): CUDA_ARCH='90-real', TORCH_CUDA_ARCH_LIST='9.0+PTX'"
    echo "  - SM89 (RTX 4080/4070): CUDA_ARCH='89-real', TORCH_CUDA_ARCH_LIST='8.9+PTX'"
    echo "  - Multiple: CUDA_ARCH='89-real;90-real;120-real', TORCH_CUDA_ARCH_LIST='8.9 9.0 12.0+PTX'"
    echo

    read -p "Enter CUDA_ARCH (press Enter for default: $CUDA_ARCH): " user_cuda_arch
    if [[ -n "$user_cuda_arch" ]]; then
        CUDA_ARCH="$user_cuda_arch"
    fi

    read -p "Enter TORCH_CUDA_ARCH_LIST (press Enter for default: $TORCH_CUDA_ARCH_LIST): " user_torch_arch
    if [[ -n "$user_torch_arch" ]]; then
        TORCH_CUDA_ARCH_LIST="$user_torch_arch"
    fi

    log "Using CUDA_ARCH: $CUDA_ARCH"
    log "Using TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
}

# Setup repository and get available versions
setup_repo() {
    local repo_path="$1"
    local repo_url="$2"
    local repo_name="$3"

    if [ ! -d "$repo_path" ]; then
        log "Cloning $repo_name..."
        git clone "$repo_url" "$repo_path"
    else
        log "$repo_name repository exists, fetching latest..."
        cd "$repo_path"
        git fetch --all --tags
        cd "$BUILD_DIR"
    fi

    success "$repo_name repository ready"
}

# Get version selection from user
select_version() {
    local repo_path="$1"
    local repo_name="$2"

    cd "$repo_path"

    # Get all tags in descending order
    local tags=($(git tag --sort=-version:refname 2>/dev/null || git tag | sort -V -r))

    echo
    echo "What version of $repo_name do you want to use?"
    echo "=============================================="
    echo "1. master (latest development)"

    local counter=2
    for tag in "${tags[@]}"; do
        echo "$counter. $tag"
        ((counter++))
    done

    echo
    read -p "Select version (1-$((${#tags[@]} + 1))): " selection

    if [[ "$selection" == "1" ]]; then
        echo "master"
    elif [[ "$selection" =~ ^[0-9]+$ ]] && [[ "$selection" -gt 1 ]] && [[ "$selection" -le $((${#tags[@]} + 1)) ]]; then
        local tag_index=$((selection - 2))
        echo "${tags[$tag_index]}"
    else
        error "Invalid selection"
    fi

    cd "$BUILD_DIR"
}

# Setup repositories and get versions
setup_repositories() {
    log "Setting up repositories..."

    setup_repo "$TENSORRT_LLM_PATH" "https://github.com/NVIDIA/TensorRT-LLM.git" "TensorRT-LLM"
    setup_repo "$TENSORRT_MODEL_OPTIMIZER_PATH" "https://github.com/NVIDIA/TensorRT-Model-Optimizer.git" "TensorRT-Model-Optimizer"

    TENSORRT_LLM_VERSION=$(select_version "$TENSORRT_LLM_PATH" "TensorRT-LLM")
    TENSORRT_MODEL_OPTIMIZER_VERSION=$(select_version "$TENSORRT_MODEL_OPTIMIZER_PATH" "TensorRT-Model-Optimizer")

    BASE_IMAGE_TAG="tensorrt-llm:${TENSORRT_LLM_VERSION}-cuda${CUDA_ARCH//[^0-9]/}"

    log "Selected TensorRT-LLM version: $TENSORRT_LLM_VERSION"
    log "Selected TensorRT-Model-Optimizer version: $TENSORRT_MODEL_OPTIMIZER_VERSION"
    log "Base image tag: $BASE_IMAGE_TAG"
}

# Update CUDA architecture settings in TensorRT-LLM
update_cuda_arch_settings() {
    log "Updating CUDA architecture settings in TensorRT-LLM..."
    log "Target CUDA_ARCH: ${CUDA_ARCH}"
    log "Target TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

    cd "$TENSORRT_LLM_PATH"

    # Update PyTorch installation script
    if [ -f "docker/common/install_pytorch.sh" ]; then
        log "Updating docker/common/install_pytorch.sh..."
        sed -i.bak "s/TORCH_CUDA_ARCH_LIST=\"[^\"]*\"/TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST}\"/" docker/common/install_pytorch.sh
        success "Updated PyTorch installation script"
    fi

    # Update wheel build script
    if [ -f "scripts/build_wheel.py" ]; then
        log "Updating scripts/build_wheel.py..."
        sed -i.bak "s/\"TORCH_CUDA_ARCH_LIST\": \"[^\"]*\"/\"TORCH_CUDA_ARCH_LIST\": \"${TORCH_CUDA_ARCH_LIST}\"/" scripts/build_wheel.py
        success "Updated wheel build script"
    fi

    # Update DeepSpeed Expert Parallelism installation script
    if [ -f "docker/common/install_deep_ep.sh" ]; then
        log "Updating docker/common/install_deep_ep.sh..."
        sed -i.bak "s/TORCH_CUDA_ARCH_LIST=\"[^\"]*\"/TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST}\"/" docker/common/install_deep_ep.sh
        success "Updated DeepSpeed EP installation script"
    fi

    # Update README documentation
    if [ -f "cpp/kernels/infra_v2/README.md" ]; then
        log "Updating cpp/kernels/infra_v2/README.md..."
        sed -i.bak "s/TORCH_CUDA_ARCH_LIST=[0-9.+PTX ]*/TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}/" cpp/kernels/infra_v2/README.md
        success "Updated README documentation"
    fi

    cd "$BUILD_DIR"
    success "CUDA architecture settings updated"
}

# Build base TensorRT-LLM container
build_tensorrt_llm() {
    log "Building TensorRT-LLM base container..."
    log "This may take 30-60 minutes..."

    cd "$TENSORRT_LLM_PATH"

    # Checkout the selected version
    log "Checking out TensorRT-LLM version: $TENSORRT_LLM_VERSION"
    if [[ "$TENSORRT_LLM_VERSION" == "master" ]]; then
        git checkout master
        git pull origin master
    else
        git checkout "tags/$TENSORRT_LLM_VERSION"
    fi

    # Initialize submodules
    git submodule update --init --recursive
    if command -v git-lfs >/dev/null 2>&1; then
        git lfs pull
    fi

    # Check if image already exists
    if docker image inspect "${BASE_IMAGE_TAG}" >/dev/null 2>&1; then
        warning "Base image ${BASE_IMAGE_TAG} already exists"
        read -p "Rebuild? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Skipping TensorRT-LLM build"
            cd "$BUILD_DIR"
            return 0
        fi
    fi

    # Build using proven method
    log "Building with CUDA_ARCHS=${CUDA_ARCH}..."
    make -C docker release_build CUDA_ARCHS="${CUDA_ARCH}"

    # Tag the built image
    BUILT_IMAGE=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep tensorrt_llm | head -1)
    if [ -n "$BUILT_IMAGE" ]; then
        docker tag "$BUILT_IMAGE" "${BASE_IMAGE_TAG}"
        success "Tagged TensorRT-LLM image as ${BASE_IMAGE_TAG}"
    else
        error "Could not find built TensorRT-LLM image"
    fi

    cd "$BUILD_DIR"
}

# Build TensorRT-LLab container with ModelOpt
build_tensorrt_llab() {
    log "Building TensorRT-LLab container with ModelOpt..."

    # Check if base image exists
    if ! docker image inspect "${BASE_IMAGE_TAG}" >/dev/null 2>&1; then
        error "Base image ${BASE_IMAGE_TAG} not found. Build TensorRT-LLM first."
    fi

    # Checkout TensorRT-Model-Optimizer version
    cd "$TENSORRT_MODEL_OPTIMIZER_PATH"
    log "Checking out TensorRT-Model-Optimizer version: $TENSORRT_MODEL_OPTIMIZER_VERSION"
    if [[ "$TENSORRT_MODEL_OPTIMIZER_VERSION" == "master" ]]; then
        git checkout master
        git pull origin master
    else
        git checkout "tags/$TENSORRT_MODEL_OPTIMIZER_VERSION"
    fi
    cd "$BUILD_DIR"

    # Create Dockerfile for TensorRT-LLab
    cat > Dockerfile.tensorrt-llab << EOF
# Use the built TensorRT-LLM as base
FROM ${BASE_IMAGE_TAG}

# Set up environment variables (from ModelOpt)
ARG PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"
ENV PIP_EXTRA_INDEX_URL=\$PIP_EXTRA_INDEX_URL \\
    PIP_NO_CACHE_DIR=off \\
    PIP_CONSTRAINT= \\
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Update PATH and LD_LIBRARY_PATH for TensorRT (from ModelOpt)
ENV LD_LIBRARY_PATH="/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:\${LD_LIBRARY_PATH}" \\
    PATH="/usr/local/tensorrt/targets/x86_64-linux-gnu/bin:\${PATH}"

# Export path for libcudnn.so (from ModelOpt)
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH}"

WORKDIR /workspace

# Install ModelOpt with all dependencies and precompile extensions (from ModelOpt)
RUN pip install -U "nvidia-modelopt[all,dev-test]"
RUN python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"

# Copy TensorRT-Model-Optimizer and install requirements
COPY TensorRT-Model-Optimizer/ ./TensorRT-Model-Optimizer/
RUN find TensorRT-Model-Optimizer/examples -name "requirements.txt" | grep -v "windows" | while read req_file; do \\
        echo "Installing from \$req_file"; \\
        pip install -r "\$req_file" || exit 1; \\
    done

# Install additional dependencies for TensorRT-LLab
RUN pip install \\
    huggingface-hub \\
    safetensors \\
    accelerate \\
    datasets

# Setup TensorRT-LLab
WORKDIR /app

# Copy TensorRT-LLab files
COPY scripts/ ./scripts/
COPY models/ ./models/ 2>/dev/null || true
COPY help/ ./help/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /model_weights /engines /ckpt

# Make scripts executable
RUN find scripts -name "*.sh" -exec chmod +x {} \\;
RUN chmod +x scripts/trt-llab 2>/dev/null || true

# Add scripts to PATH
ENV PATH="/app/scripts:\$PATH"

# Set permissions (from ModelOpt)
RUN chmod -R 777 /workspace /app

# Default working directory
WORKDIR /app

CMD ["/bin/bash"]
EOF

    # Build the final container
    docker build -f Dockerfile.tensorrt-llab -t "${FINAL_IMAGE_TAG}" .

    # Clean up temporary Dockerfile
    rm Dockerfile.tensorrt-llab

    success "TensorRT-LLab container built successfully"
}

# Test the built container
test_container() {
    log "Testing TensorRT-LLab container..."

    # Basic functionality test
    if docker run --rm --gpus all "${FINAL_IMAGE_TAG}" python -c "
import torch
import tensorrt_llm
import modelopt
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')
print(f'ModelOpt version: {modelopt.__version__}')
print('✅ All imports successful')
"; then
        success "Container test passed"
    else
        error "Container test failed"
    fi
}

# Show usage information
show_usage() {
    cat << EOF
TensorRT-LLab Build Script

Usage: $0 [OPTIONS]

Options:
    --skip-checks       Skip Docker and NVIDIA runtime checks
    --skip-tensorrt     Skip TensorRT-LLM build (use existing)
    --skip-test         Skip container testing
    --skip-arch-update  Skip CUDA architecture updates
    --clean             Remove existing images before building
    --help              Show this help message

Configuration will be prompted interactively:
    - CUDA Architecture settings (default: CUDA_ARCH=${CUDA_ARCH}, TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST})
    - TensorRT-LLM version selection
    - TensorRT-Model-Optimizer version selection

Examples:
    $0                          # Full interactive build
    $0 --skip-tensorrt          # Only build TensorRT-LLab layer
    $0 --skip-arch-update       # Build without updating CUDA arch settings
    $0 --clean                  # Clean build everything
EOF
}

# Parse command line arguments
SKIP_CHECKS=false
SKIP_TENSORRT=false
SKIP_TEST=false
SKIP_ARCH_UPDATE=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-checks)
            SKIP_CHECKS=true
            shift
            ;;
        --skip-tensorrt)
            SKIP_TENSORRT=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --skip-arch-update)
            SKIP_ARCH_UPDATE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Main execution
main() {
    log "Starting TensorRT-LLab build process..."

    # Clean up if requested
    if [ "$CLEAN" = true ]; then
        log "Cleaning existing images..."
        docker rmi "${FINAL_IMAGE_TAG}" 2>/dev/null || true
        # Note: We'll clean base image after getting version info
    fi

    # Run checks
    if [ "$SKIP_CHECKS" = false ]; then
        check_docker
    fi

    # Get user input for configuration
    get_cuda_arch_settings
    setup_repositories

    # Clean base image if requested (now that we have the tag)
    if [ "$CLEAN" = true ]; then
        docker rmi "${BASE_IMAGE_TAG}" 2>/dev/null || true
    fi

    # Setup and build TensorRT-LLM
    if [ "$SKIP_TENSORRT" = false ]; then
        if [ "$SKIP_ARCH_UPDATE" = false ]; then
            update_cuda_arch_settings
        fi
        build_tensorrt_llm
    fi

    # Build TensorRT-LLab
    build_tensorrt_llab

    # Test the container
    if [ "$SKIP_TEST" = false ]; then
        test_container
    fi

    success "Build completed successfully!"
    echo
    log "Final image: ${FINAL_IMAGE_TAG}"
    log "Base TensorRT-LLM image: ${BASE_IMAGE_TAG}"
    log "TensorRT-LLM version: ${TENSORRT_LLM_VERSION}"
    log "TensorRT-Model-Optimizer version: ${TENSORRT_MODEL_OPTIMIZER_VERSION}"
    log "CUDA Architecture: ${CUDA_ARCH}"
    log "PyTorch CUDA Arch List: ${TORCH_CUDA_ARCH_LIST}"
    echo
    log "You can now run TensorRT-LLab with:"
    echo "docker run --rm -it --gpus all -v \$(pwd)/model_weights:/model_weights ${FINAL_IMAGE_TAG}"
}

# Run main function
main "$@"
