import os
import numpy as np

# Define the hyperparameter ranges
hsv_h_values = np.arange(0, 0.045, 0.005) 
hsv_s_values = np.arange(0.6, 1.1, 0.1)
hsv_v_values = np.arange(0, 0.6, 0.1)

# first round: s 0 - 0.6, v 0 - 0.6
# second round: s 0.6 - 1.1, v 0 - 0.6

# Create a directory to store the job scripts and logs
os.makedirs('jobs', exist_ok=True)
os.makedirs('logs_round2', exist_ok=True)

# SLURM job script template
slurm_template = """#!/bin/bash

#SBATCH --job-name=yolo_train_{job_id}
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=cosc028244
#SBATCH --output=logs/job_{job_id}.log
#SBATCH --error=logs/job_{job_id}.log

cd "${{SLURM_SUBMIT_DIR}}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${{SLURM_JOBID}}"
echo This jobs runs on the following machines:
echo "${{SLURM_JOB_NODELIST}}"

export COMET_API_KEY=4h2enjtP3OL4Q9bH9Sbf8sfuC

# Log hyperparameters
echo Hyperparameters for job {job_id}:
echo hsv_h: {hsv_h}
echo hsv_s: {hsv_s}
echo hsv_v: {hsv_v}

# Activate virtualenv
conda init
source ~/.bashrc
conda activate cow_project

# Run YOLO training and validation
yolo task=classify mode=train model=yolov8s-cls.yaml data=/user/work/yf20630/cow-dataset-project/datasets/subset_small epochs=150 save_period=1 imgsz=640 device=0 hsv_h={hsv_h} hsv_s={hsv_s} hsv_v={hsv_v} degrees=0.0 translate=0.0 scale=0.0 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.0 mosaic=0.0 mixup=0.0 copy_paste=0.0 project=augmentation2 name={job_id} &> logs_round2/job_{job_id}.log
yolo task=classify mode=val model=augmentation2/{job_id}/weights/best.pt data=/user/work/yf20630/cow-dataset-project/datasets/subset_small split=test &>> logs_round2/job_{job_id}.log

# Deactivate virtualenv
conda deactivate
"""

# Generate and submit job scripts
job_id = 0
for hsv_h in hsv_h_values:
    for hsv_s in hsv_s_values:
        for hsv_v in hsv_v_values:
            job_script = slurm_template.format(
                job_id=job_id,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v
            )
            
            script_path = f'jobs/job_{job_id}.sh'
            with open(script_path, 'w') as f:
                f.write(job_script)
            
            os.system(f'sbatch {script_path}')
            job_id += 1
