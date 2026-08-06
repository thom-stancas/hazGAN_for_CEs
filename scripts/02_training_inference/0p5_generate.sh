#!/bin/bash
#SBATCH --job-name="generate_hazgan"
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

# Get model num from the command line argument, throw an error if not provided
if [ -z "$1" ]; then
    echo "Error: Model number argument is required."
    exit 1
fi
MODEL_NUM=$1

MODEL="${MODEL_NUM}-images-low_shot-kimg300-color-translation-cutout"
STEP=300

# Get DATADIR from the .env file variable, exit if not set
if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR environment variable is not set. Please set it in the .env file."
    exit 1
fi

GEN_DATADIR=${DATA_DIR}/stylegan_events/training-runs/${MODEL}

source "${CONDA_SOURCE}"
conda activate styleGAN


CLEAR_TORCH_EXTENSIONS=1
source ${ROOT_DIR}/scripts/02_training_inference/setup_cuda.sh


CUDA_VISIBLE_DEVICES=0,1


python Repositories/hazGAN_for_CEs/styleGAN-DA/src/generate.py --outdir=${GEN_DATADIR}/results/trunc-1_0 --seeds=1-10000 --trunc=1.0 --network=${GEN_DATADIR}/network-snapshot-$(printf "%06d" $STEP).pkl