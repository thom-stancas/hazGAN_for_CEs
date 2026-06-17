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


# IMGDIR=/data/ncas1/tb261/training/64x64_jja/images/gumbel/rgb
# OUTDIR=/data/ncas1/tb261/stylegan_events/

# mkdir -p ${OUTDIR}

# python Repositories/hazGAN_for_CEs/styleGAN-DA/src/dataset_tool.py \
#     --source=${IMGDIR} \
#     --dest=${OUTDIR}/images.zip 


conda activate styleGAN

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CXX=g++
export TORCH_EXTENSIONS_DIR=/data/ncas1/tb261/torch_extensions

rm -rf $TORCH_EXTENSIONS_DIR
mkdir -p $TORCH_EXTENSIONS_DIR

CUDA_VISIBLE_DEVICES=0,1

DATADIR=/data/ncas1/tb261/stylegan_events

python /home/users/t/tb261/Repositories/hazGAN_for_CEs/styleGAN-DA/src/train.py \
    --data=${DATADIR}/images.zip \
    --outdir=${DATADIR}/training-runs \
    --gpus=1 \
    --DiffAugment=color,translation,cutout \
    --kimg=300 \
    --metrics=none