#!/bin/bash
# # SBATCH --job-name=styleGAN2
# # SBATCH --output=train.out
# # SBATCH --error=train.err
# # SBATCH --partition=GPU
# # SBATCH --time=1-00:00:00

# DATADIR=/soge-home/projects/mistral/alison/data/stylegan_output
# source /lustre/soge1/users/spet5107/micromamba/etc/profile.d/micromamba.sh

# micromamba activate styleGAN
# DATADIR=/soge-home/projects/mistral/alison/data/stylegan
# python ../../styleGAN-DA/src/train.py --data=${DATADIR}/images.zip --outdir=${DATADIR}/training-runs --gpus=2 --DiffAugment=color,translation,cutout --kimg=300


IMGDIR=/data/ncas1/tb261/training/64x64_jja_0_1_3/images/gumbel/rgb
OUTDIR=/data/ncas1/tb261/stylegan_events/

mkdir -p ${OUTDIR}

python Repositories/hazGAN_for_CEs/styleGAN-DA/src/dataset_tool.py \
    --source=${IMGDIR} \
    --dest=${OUTDIR}/images.zip 


source /opt/conda/etc/profile.d/conda.sh
conda activate /data/ncas2/tb261/penvs/styleGAN
export CUDA_HOME="$CONDA_PREFIX"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0"
export MAX_JOBS=1

export TORCH_EXTENSIONS_DIR="/data/ncas1/tb261/torch_extensions/debug_${SLURM_JOB_ID:-manual}"
rm -rf "$TORCH_EXTENSIONS_DIR"
mkdir -p "$TORCH_EXTENSIONS_DIR"

CUDA_VISIBLE_DEVICES=0,1

DATADIR=/data/ncas1/tb261/stylegan_events

python Repositories/hazGAN_for_CEs/styleGAN-DA/src/train.py \
    --data=${DATADIR}/images.zip \
    --outdir=${DATADIR}/training-runs \
    --gpus=1 \
    --DiffAugment=color,translation,cutout \
    --kimg=300 \
    --metrics=none