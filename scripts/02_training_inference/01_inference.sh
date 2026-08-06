#!/bin/bash
#SBATCH --job-name="inference_hazgan"
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e err/%j.err

# Get model num from the command line argument, throw an error if not provided
if [ -z "$1" ]; then
    echo "Error: Model number argument is required."
    exit 1
fi
MODEL_NUM=$1

MODEL="${MODEL_NUM}-images-low_shot-kimg300-color-translation-cutout"
STEP=300

# Get DATADIR from the .env file variable, exit if not set
if [ -z "$DATADIR" ]; then
    echo "Error: DATADIR environment variable is not set. Please set it in the .env file."
    exit 1
fi

DATADIR=${DATADIR}/stylegan_events/training-runs/${MODEL}

conda activate hazGAN

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python Repositories/hazGAN_for_CEs/scripts/02_training_inference/01_inference.py --model=${MODEL} --step=${STEP} 
