#!/bin/bash
#SBATCH --job-name=ophys-deepint-inference
#SBATCH --mail-type=NONE
#SBATCH --ntasks=32
#SBATCH --mem=250gb
#SBATCH --time=04:00:00
#SBATCH --output=/allen/aibs/informatics/danielk/deepinterpolation/logs/inference_%j.log
#SBATCH --partition braintv

image=docker://alleninstitutepika/deepinterpolation:develop
input_json=/allen/aibs/informatics/danielk/deepinterpolation/output/ophys_experiment_958741232/processed/inference_input.json

export TMPDIR=/scratch/fast/${SLURM_JOB_ID}
export SINGULARITY_CACHEDIR=/allen/ai/hpc/singularity/cache/danielk

SINGULARITY_TMPDIR=${TMPDIR} singularity run \
    --bind /allen:/allen,${TMPDIR}:/tmp \
    ${image} python -m \
        deepinterpolation.cli.inference \
        --input_json ${input_json}
