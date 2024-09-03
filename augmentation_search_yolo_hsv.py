import os
import numpy as np

# Define the hyperparameter ranges using np.linspace
hsv_h_values = np.linspace(0, 0.1, 21)  # 21 evenly spaced values from 0 to 0.1 (inclusive)
hsv_s_values = np.linspace(0, 1.0, 11)  # 11 evenly spaced values from 0 to 1.0 (inclusive)
hsv_v_values = np.linspace(0, 1.0, 11)  # 11 evenly spaced values from 0 to 1.0 (inclusive)

# Create directories to store the job scripts and logs
os.makedirs('jobs', exist_ok=True)
os.makedirs('yolo_hsv_logs', exist_ok=True)

# SLURM job script template
slurm_template = """#!/bin/bash

#SBATCH --job-name=yolo_train_{job_id}
#SBATCH --partition=teach_gpu
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=COMS031424
#SBATCH --output=yolo_hsv_logs/yolo-tiny-{param_value:.3f}{param_name}.log

cd "${{SLURM_SUBMIT_DIR}}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${{SLURM_JOBID}}"
echo This jobs runs on the following machines:
echo "${{SLURM_JOB_NODELIST}}"

# Log hyperparameters
echo Hyperparameters for job {job_id}:
echo {param_name}: {param_value:.3f}

# Activate virtualenv
conda init
source ~/.bashrc
conda activate cow_project

# Run YOLO training and validation
yolo task=classify mode=train model=yolov8s-cls.pt data=/user/work/yf20630/cow-dataset-project/datasets/subset_tiny epochs=400 imgsz=224 device=0 hsv_h={hsv_h:.3f} hsv_s={hsv_s:.3f} hsv_v={hsv_v:.3f} degrees=0.0 translate=0.0 scale=0.0 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.0 mosaic=0.0 mixup=0.0 copy_paste=0.0 erasing=0.0 project=yolo-hsv-aug name={job_id}
echo ---------------------------------------------------------------------------------
yolo task=classify mode=val model=yolo-hsv-aug/{job_id}/weights/best.pt data=/user/work/yf20630/cow-dataset-project/datasets/subset_tiny split=test

# Deactivate virtualenv
conda deactivate
"""

# Function to generate and submit job scripts
def submit_jobs(param_name, param_values):
    # Set default values
    default_hsv = {'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0}
    
    for value in param_values:
        job_id = f"{param_name}_{value:.3f}"
        
        # Set the current parameter to its test value, keep others at default
        hsv_values = default_hsv.copy()
        hsv_values[param_name] = value
        
        job_script = slurm_template.format(
            job_id=job_id,
            param_name=param_name,
            param_value=value,
            hsv_h=hsv_values['hsv_h'],
            hsv_s=hsv_values['hsv_s'],
            hsv_v=hsv_values['hsv_v']
        )
        
        script_path = f'jobs/job_{job_id}.sh'
        with open(script_path, 'w') as f:
            f.write(job_script)
        
        os.system(f'sbatch {script_path}')

# Submit jobs for each parameter separately
submit_jobs('hsv_h', hsv_h_values)
submit_jobs('hsv_s', hsv_s_values)
submit_jobs('hsv_v', hsv_v_values)