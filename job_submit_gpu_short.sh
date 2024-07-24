#!/bin/bash

#SBATCH --job-name=yolo_train
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=cosc028244
#SBATCH --output=vit-bs8-40-seed42-roboAug.out


cd "${SLURM_SUBMIT_DIR}"


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

# export COMET_API_KEY=4h2enjtP3OL4Q9bH9Sbf8sfuC


# Activate virtualenv
conda init
source ~/.bashrc
conda activate cow_project

# cd datasets
# ls -al
# cd ..

# Run Python script
python train_vit.py

# Deactivate virtualenv
conda deactivate