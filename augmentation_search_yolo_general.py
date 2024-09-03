import os
import numpy as np

# Define the hyperparameter ranges
# hsv_h_values = np.linspace(0, 0.1, 21)
# hsv_s_values = np.linspace(0, 1.0, 11)
# hsv_v_values = np.linspace(0, 1.0, 11)
degrees_values = [-10, -5, 5, 10, 15]
translate_values = [0.1, 0.2]
scale_values = np.linspace(0.1, 0.7, 7)
shear_values = [-15, -10, -5, 5, 10, 15]
erasing_values = np.linspace(0.1, 0.5, 5)
mixup_values = np.linspace(0.1, 0.5, 5)
fliplr_values = np.linspace(0.1, 0.5, 5)
auto_augment_values = ['randaugment', 'autoaugment', 'augmix']

# Create directories to store the job scripts and logs
os.makedirs('jobs', exist_ok=True)
os.makedirs('yolo_aug_logs', exist_ok=True)

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
#SBATCH --output=yolo_aug_logs/yolo-tiny-{param_name}_{param_value}.log

cd "${{SLURM_SUBMIT_DIR}}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${{SLURM_JOBID}}"
echo This jobs runs on the following machines:
echo "${{SLURM_JOB_NODELIST}}"

# Log hyperparameters
echo Hyperparameters for job {job_id}:
echo {param_name}: {param_value}

# Activate virtualenv
conda init
source ~/.bashrc
conda activate cow_project

# Run YOLO training and validation
yolo task=classify mode=train model=yolov8s-cls.pt data=/user/work/yf20630/cow-dataset-project/datasets/subset_tiny epochs=400 imgsz=224 device=0 {aug_params} project=yolo-aug-tuning name={job_id}
echo ---------------------------------------------------------------------------------
yolo task=classify mode=val model=yolo-aug-tuning/{job_id}/weights/best.pt data=/user/work/yf20630/cow-dataset-project/datasets/subset_tiny split=test

# Deactivate virtualenv
conda deactivate
"""

# Function to generate and submit job scripts
def submit_jobs(param_name, param_values):
    default_params = {
        'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
        'degrees': 0.0, 'translate': 0.0, 'scale': 0.0,
        'shear': 0.0, 'erasing': 0.0, 'mixup': 0.0,
        'fliplr': 0.0, 'flipud': 0.0, 'mosaic': 0.0,
        'copy_paste': 0.0, 'auto_augment': 'randaugment'
    }
    
    for value in param_values:
        if isinstance(value, float):
            job_id = f"{param_name}_{value:.3f}"
            param_value = f"{value:.3f}"
        elif isinstance(value, int):
            job_id = f"{param_name}_{value}"
            param_value = str(value)
        else:
            job_id = f"{param_name}_{value}"
            param_value = value
        
        # Set all parameters to default, then update the one we're testing
        aug_params = default_params.copy()
        aug_params[param_name] = value
        
        # Convert aug_params to a string for the YOLO command
        aug_params_str = ' '.join([f"{k}={v}" for k, v in aug_params.items()])
        
        job_script = slurm_template.format(
            job_id=job_id,
            param_name=param_name,
            param_value=param_value,
            aug_params=aug_params_str
        )
        
        script_path = f'jobs/job_{job_id}.sh'
        with open(script_path, 'w') as f:
            f.write(job_script)
        
        os.system(f'sbatch {script_path}')

# Submit jobs for each parameter separately
# submit_jobs('hsv_h', hsv_h_values)
# submit_jobs('hsv_s', hsv_s_values)
# submit_jobs('hsv_v', hsv_v_values)
submit_jobs('degrees', degrees_values)
submit_jobs('translate', translate_values)
submit_jobs('scale', scale_values)
submit_jobs('shear', shear_values)
submit_jobs('erasing', erasing_values)
submit_jobs('mixup', mixup_values)
submit_jobs('fliplr', fliplr_values)
submit_jobs('auto_augment', auto_augment_values)