#!/bin/bash

#SBATCH --job-name=move_files
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --account=cosc028244

cd "${SLURM_SUBMIT_DIR}"

# Print some debug information
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This job runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"

# Source necessary profiles
# source ~/.bashrc

# Move files from source to destination
# rsync -av /group/nwc-group/tony/pmfeed_4_3_16_hand_labelled/ /user/home/yf20630/raw-cow-images/

# Define source and target directories
SOURCE_DIR="/path/to/source_folder"
TARGET_DIR="/path/to/target_folder/subfolder"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Use tar to move the files
tar cf - -C "$SOURCE_DIR" . | pv | tar xf - -C "$TARGET_DIR"


# Check if the move was successful
if [ $? -eq 0 ]; then
    echo "Files moved successfully."
else
    echo "Error occurred while moving files."
fi
