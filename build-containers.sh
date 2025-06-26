#!/bin/bash

set -euo pipefail

# Configuration
TENSORRT_LLM_VERSION="v1.0.0rc0"
CUDA_ARCH="120-real"
TORCH_CUDA_ARCH_LIST="12.0+PTX"
BASE_IMAGE_TAG="tensorrt-llm:${TENSORRT_LLM_VERSION}-sm120"
FINAL_IMAGE_TAG="tensorrt-llab:latest"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Clone TensorRT-LLM if needed
setup_tensorrt_llm() {
    log "Setting up TensorRT-LLM source..."
    
    if [ ! -d "TensorRT-LLM" ]; then
        log "Cloning TensorRT-LLM ${TENSORRT_LLM_VERSION}..."
        git clone --depth 1 -b "${TENSORRT_LLM_VERSION}" https://github.com/NVIDIA/TensorRT-LLM.git
        cd TensorRT-LLM
        git lfs install
        git submodule update --init --recursive
        git lfs pull
        cd ..
    else
        log "TensorRT-LLM directory already exists"
        cd TensorRT-LLM
        # Ensure we're on the right version
        git fetch --depth 1 origin "${TENSORRT_LLM_VERSION}"
        git checkout "${TENSORRT_LLM_VERSION}"
        git lfs pull
        cd ..
    fi
    
    success "TensorRT-LLM source ready"
}

# Update CUDA architecture settings for target GPU
update_cuda_arch_settings() {
    log "Updating CUDA architecture settings..."
    log "Target CUDA_ARCH: ${CUDA_ARCH}"
    log "Target TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
    
    cd TensorRT-LLM
    
    # Update PyTorch installation script
    if [ -f "docker/common/install_pytorch.sh" ]; then
        log "Updating docker/common/install_pytorch.sh..."
        sed -i.bak "s/TORCH_CUDA_ARCH_LIST=\"[^\"]*\"/TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST}\"/" docker/common/install_pytorch.sh
        if [ $? -eq 0 ]; then
            success "Updated PyTorch installation script"
        else
            warning "Failed to update PyTorch installation script"
        fi
    fi
    
    # Update wheel build script
    if [ -f "scripts/build_wheel.py" ]; then
        log "Updating scripts/build_wheel.py..."
        sed -i.bak "s/\"TORCH_CUDA_ARCH_LIST\": \"[^\"]*\"/\"TORCH_CUDA_ARCH_LIST\": \"${TORCH_CUDA_ARCH_LIST}\"/" scripts/build_wheel.py
        if [ $? -eq 0 ]; then
            success "Updated wheel build script"
        else
            warning "Failed to update wheel build script"
        fi
    fi

    # Update DeepSpeed Expert Parallelism installation script
    if [ -f "docker/common/install_deep_ep.sh" ]; then
        log "Updating docker/common/install_deep_ep.sh..."
        sed -i.bak "s/TORCH_CUDA_ARCH_LIST=\"[^\"]*\"/TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST}\"/" docker/common/install_deep_ep.sh
        if [ $? -eq 0 ]; then
            success "Updated DeepSpeed EP installation script"
        else
            warning "Failed to update DeepSpeed EP installation script"
        fi
    fi
    
    # Update README documentation
    if [ -f "cpp/kernels/infra_v2/README.md" ]; then
        log "Updating cpp/kernels/infra_v2/README.md..."
        sed -i.bak "s/TORCH_CUDA_ARCH_LIST=[0-9.+PTX ]*/TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}/" cpp/kernels/infra_v2/README.md
        if [ $? -eq 0 ]; then
            success "Updated README documentation"
        else
            warning "Failed to update README documentation"
        fi
    fi
    
    # Check for any other TORCH_CUDA_ARCH_LIST references
    log "Checking for other TORCH_CUDA_ARCH_LIST references..."
    UPDATED_FILES=(
       "docker/common/install_pytorch.sh"
       "docker/common/install_deep_ep.sh"
       "scripts/build_wheel.py"
       "cpp/kernels/infra_v2/README.md"
    )

    # Build grep exclusion pattern
    EXCLUDE_PATTERN=""
    for file in "${UPDATED_FILES[@]}"; do
        EXCLUDE_PATTERN="${EXCLUDE_PATTERN} | grep -v \"${file}\""
    done

    OTHER_REFS=$(eval "grep -r 'TORCH_CUDA_ARCH_LIST' . --include='*.py' --include='*.sh' --include='*.md' --include='*.cmake' --exclude-dir=.git 2>/dev/null | grep -v '.bak:' ${EXCLUDE_PATTERN}")

    if [ "$OTHER_REFS" -gt 0 ]; then
       warning "Found $(echo "$OTHER_REFS" | wc -l) other TORCH_CUDA_ARCH_LIST references:"
       echo
       echo "$OTHER_REFS"
       echo
       warning "Some of these may need manual review. Continue anyway? [y/N]:"
       read -n 1 -r
       echo
       if [[ ! $REPLY =~ ^[Yy]$ ]]; then
           log "Build stopped by user. You can review the references above and re-run."
           exit 0
       fi
    fi



    OTHER_REFS=$(grep -r "TORCH_CUDA_ARCH_LIST" . --include="*.py" --include="*.sh" --include="*.md" --include="*.cmake" --exclude-dir=.git 2>/dev/null | grep -v ".bak:" | wc -l)
    if [ "$OTHER_REFS" -gt 0 ]; then
        warning "Found ${OTHER_REFS} other TORCH_CUDA_ARCH_LIST references. Review manually if needed:"
        grep -r "TORCH_CUDA_ARCH_LIST" . --include="*.py" --include="*.sh" --include="*.md" --include="*.cmake" --exclude-dir=.git 2>/dev/null | grep -v ".bak:" | head -5
    fi
    
    cd ..
    success "CUDA architecture settings updated"
}

# Build base TensorRT-LLM container
build_tensorrt_llm() {
    log "Building TensorRT-LLM base container..."
    log "This may take 30-60 minutes..."
    
    cd TensorRT-LLM
    
    # Check if image already exists
    if docker image inspect "${BASE_IMAGE_TAG}" >/dev/null 2>&1; then
        warning "Base image ${BASE_IMAGE_TAG} already exists"
        read -p "Rebuild? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Skipping TensorRT-LLM build"
            cd ..
            return 0
        fi
    fi
    
    # Build using your proven method
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
    
    cd ..
}

# Build TensorRT-LLab container with ModelOpt
build_tensorrt_llab() {
    log "Building TensorRT-LLab container with ModelOpt..."
    
    # Check if base image exists
    if ! docker image inspect "${BASE_IMAGE_TAG}" >/dev/null 2>&1; then
        error "Base image ${BASE_IMAGE_TAG} not found. Build TensorRT-LLM first."
    fi
    
    # Create Dockerfile for TensorRT-LLab based on ModelOpt's approach
    cat > Dockerfile.tensorrt-llab << EOF
# Use your proven TensorRT-LLM build as base
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
COPY models/ ./models/
COPY help/ ./help/

# Create necessary directories
RUN mkdir -p /model_weights /engines /ckpt

# Make scripts executable
RUN chmod +x scripts/*.sh scripts/trt-llab

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

Configuration:
    CUDA_ARCH: ${CUDA_ARCH}
    TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}

Examples:
    $0                          # Full build with arch updates
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
    log "Base image: ${BASE_IMAGE_TAG}"
    log "Final image: ${FINAL_IMAGE_TAG}"
    log "CUDA Architecture: ${CUDA_ARCH}"
    log "PyTorch CUDA Arch List: ${TORCH_CUDA_ARCH_LIST}"
    
    # Clean up if requested
    if [ "$CLEAN" = true ]; then
        log "Cleaning existing images..."
        docker rmi "${FINAL_IMAGE_TAG}" 2>/dev/null || true
        docker rmi "${BASE_IMAGE_TAG}" 2>/dev/null || true
    fi
    
    # Run checks
    if [ "$SKIP_CHECKS" = false ]; then
        check_docker
    fi
    
    # Setup and build TensorRT-LLM
    if [ "$SKIP_TENSORRT" = false ]; then
        setup_tensorrt_llm
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
    log "You can now run TensorRT-LLab with:"
    echo "docker run --rm -it --gpus all -v \$(pwd)/models:/model_weights ${FINAL_IMAGE_TAG}"
}

# Run main function
main "$@"
