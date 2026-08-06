#!/usr/bin/env bash


ROOT_DIR="$(pwd)/Repositories/hazGAN_for_CEs"

# Load root_dir/.env
source "${ROOT_DIR}/.env"

source "${CONDA_SOURCE}"
conda activate styleGAN

# Choose where CUDA is installed.
if [[ "${CUDA_MODE}" == "conda" ]]; then
    # CUDA is installed in the Conda environment.
    export CUDA_HOME="${CONDA_PREFIX}"

elif [[ "${CUDA_MODE}" == "external" ]]; then
    # CUDA is installed somewhere outside Conda.
    export CUDA_HOME="${CUDA_ROOT}"
fi

# Get c and c++ compilers (can be g++ or c++)
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
if [[ -f "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++" ]]; then
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
else
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
fi

# Get CUDA and Conda paths
export CUDAHOSTCXX="${CXX}"
export PATH="${CONDA_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
# Get header files
export CPATH="${CUDA_HOME}/include:${CONDA_PREFIX}/include:${CPATH:-}"

# Get library files
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST
export MAX_JOBS
export TORCH_EXTENSIONS_DIR

# Create the torch extensions dir
if [[ "${CLEAR_TORCH_EXTENSIONS:-0}" == "1" ]]; then
    rm -rf "${TORCH_EXTENSIONS_DIR}"
fi

mkdir -p "${TORCH_EXTENSIONS_DIR}"

