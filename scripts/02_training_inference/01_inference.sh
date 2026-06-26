#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=inference.out
#SBATCH --error=inference.err
#SBATCH --partition=GPU
#SBATCH --time=05:00:00
#SBATCH --dependency=afterok:116190

MODEL="00017-images-low_shot-kimg300-color-translation-cutout"
STEP=300
DATADIR=/data/ncas1/tb261/stylegan_events/training-runs/${MODEL}

# set -a
# source .env
# set +a

# # Initialise conda for bash
# if [[ -n "${CONDA_EXE:-}" ]]; then
#   eval "$("$CONDA_EXE" shell.bash hook)"
# else
#   echo "Could not initialise conda. Set CONDA_EXE or CONDA_BASE in .env." >&2
#   exit 1
# fi


# source /opt/conda/etc/profile.d/conda.sh
# conda activate /data/ncas2/tb261/penvs/styleGAN
# export CUDA_HOME="$CONDA_PREFIX"

# export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
# export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
# export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
# export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0"
# export MAX_JOBS=1

# export TORCH_EXTENSIONS_DIR="/data/ncas1/tb261/torch_extensions/debug_${SLURM_JOB_ID:-manual}"
# rm -rf "$TORCH_EXTENSIONS_DIR"
# mkdir -p "$TORCH_EXTENSIONS_DIR"

# # # micromamba activate styleGAN
# python styleGAN-DA/src/generate.py --outdir=${DATADIR}/results/trunc-1_0 --seeds=1-10000 --trunc=1.0 --network=${DATADIR}/network-snapshot-$(printf "%06d" $STEP).pkl




conda activate hazGAN

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/02_training_inference/01_inference.py --model=${MODEL} --step=${STEP} 
