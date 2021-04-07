#!/bin/bash
#SBATCH --job-name=ophys-deepint
#SBATCH --mail-type=NONE
#SBATCH --ntasks=32
#SBATCH --mem=250gb
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=/allen/aibs/informatics/danielk/deepinterpolation/logs/fine_tune_%j.log
#SBATCH --partition braintv

image=docker://alleninstitutepika/deepinterpolation:develop
input_json=/allen/aibs/informatics/danielk/deepinterpolation/output/ophys_experiment_958741232/processed/fine_tune_input.json
TMPDIR=/scratch/fast/${SLURM_JOB_ID}

SINGULARITY_TMPDIR=${TMPDIR} singularity run \
    --bind /allen:/allen,${TMPDIR}:/tmp \
    --nv \
    ${image} python -m \
        deepinterpolation.cli.transfer_trainer \
        --input_json ${input_json}
