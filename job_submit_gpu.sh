#!/bin/bash

#SBATCH --job-name=yolo_train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=2-05:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=cosc028244
#SBATCH --nodelist=bp1-gpu035
#SBATCH --nodes=1


cd "${SLURM_SUBMIT_DIR}"


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

export COMET_API_KEY=4h2enjtP3OL4Q9bH9Sbf8sfuC


# Activate virtualenv
conda init
source ~/.bashrc
conda activate cow_project

# cd datasets
# ls -al
# cd ..

# Run Python script
python train.py

# Deactivate virtualenv
conda deactivate