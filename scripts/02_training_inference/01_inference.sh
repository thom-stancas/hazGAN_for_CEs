#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=inference.out
#SBATCH --error=inference.err
#SBATCH --partition=GPU
#SBATCH --time=05:00:00
#SBATCH --dependency=afterok:116190

MODEL="00013-images-low_shot-kimg300-color-translation-cutout"
STEP=300
DATADIR=/data/ncas1/tb261/stylegan_events/training-runs/${MODEL}

# conda activate styleGAN

# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH
# export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CPATH
# export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
# export CXX=g++
# export TORCH_EXTENSIONS_DIR=/data/ncas1/tb261/torch_extensions

# rm -rf $TORCH_EXTENSIONS_DIR
# mkdir -p $TORCH_EXTENSIONS_DIR

# CUDA_VISIBLE_DEVICES=0,1


# # # micromamba activate styleGAN
# python Repositories/hazGAN_for_CEs/styleGAN-DA/src/generate.py --outdir=${DATADIR}/results/trunc-1_0 --seeds=1-10000 --trunc=1.0 --network=${DATADIR}/network-snapshot-$(printf "%06d" $STEP).pkl

conda activate hazGAN

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python Repositories/hazGAN_for_CEs/scripts/02_training_inference/01_inference.py --model=${MODEL} --step=${STEP} 
