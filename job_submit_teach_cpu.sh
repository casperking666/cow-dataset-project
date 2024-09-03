#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=10:00:00
#SBATCH --mem=24GB
#SBATCH --account=COMS031424
#SBATCH --output=yolo-hsv-search.log


cd "${SLURM_SUBMIT_DIR}"


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

# Activate virtualenv
conda init
source ~/.bashrc
conda activate cow_project

# Run Python script
python augmentation_search_general.py

# Deactivate virtualenv
conda deactivate