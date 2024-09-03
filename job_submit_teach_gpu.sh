#!/bin/bash

#SBATCH --job-name=yolo_train
#SBATCH --partition=teach_gpu
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=cosc028244
#SBATCH --output=vit-bs8-seed42-test-offline-robo-randAugScript-best.out


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
python train.py
# python train_vit.py
# python train_vit_randAug.py
# python train_vit_parser.py --auto_augment "imagenet"


# Deactivate virtualenv
conda deactivate