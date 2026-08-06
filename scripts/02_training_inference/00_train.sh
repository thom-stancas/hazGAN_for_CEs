#!/bin/bash
#SBATCH --job-name="train_hazgan"
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e err/%j.err

# Get the correct .env file for the project
ROOT_DIR="$(pwd)/Repositories/hazGAN_for_CEs"

# Load root_dir/.env
source "${ROOT_DIR}/.env"

# Get version from .env file variable, default to 0_1_3 if not set
VERSION=${VERSION:-0_1_3}

IMGDIR=${DATA_DIR}/training/64x64_jja_${VERSION}/images/gumbel/rgb
OUTDIR=${DATA_DIR}/stylegan_events/

mkdir -p ${OUTDIR}

source "${CONDA_SOURCE}"
conda activate styleGAN

python Repositories/hazGAN_for_CEs/styleGAN-DA/src/dataset_tool.py \
    --source=${IMGDIR} \
    --dest=${OUTDIR}/images.zip 


CLEAR_TORCH_EXTENSIONS=1
source ${ROOT_DIR}/scripts/02_training_inference/setup_cuda.sh


CUDA_VISIBLE_DEVICES=0,1

GEN_DATADIR=/data/ncas1/tb261/stylegan_events

python Repositories/hazGAN_for_CEs/styleGAN-DA/src/train.py \
    --data=${GEN_DATADIR}/images.zip \
    --outdir=${GEN_DATADIR}/training-runs \
    --gpus=1 \
    --DiffAugment=color,translation,cutout \
    --kimg=300 \
    --metrics=none