#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=20GB
#SBATCH --account=cosc028244
#SBATCH --output=output.log   # Specify the output file
#SBATCH --error=error.log     # Specify the error file

cd "${SLURM_SUBMIT_DIR}"

exec > output.log 2>&1

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
python dataset_split.py

# Deactivate virtualenv
conda deactivate