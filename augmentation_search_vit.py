import os
import numpy as np

# Define the hyperparameter ranges
resize_values = [0, 112, 150, 200, 256]
random_resized_crop_values = [(0.8, 1.2), (0.7, 1.3), (0.6, 1.4)]
random_resize_values = [(180, 270), (160, 288), (140, 308)]
zoom_out_values = [(1.0, 1.3), (1.0, 1.5), (1.0, 1.7)]
rotation_degrees_values = [15, 30, 45]
affine_values = [(5, 0.05, 0.9, 1.1), (10, 0.1, 0.8, 1.2), (15, 0.15, 0.7, 1.3)]
perspective_values = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
color_jitter_values = [(0.1, 0.1, 0.1, 0.05), (0.2, 0.2, 0.2, 0.1), (0.3, 0.3, 0.3, 0.15)]
grayscale_p_values = [0.1, 0.2, 0.3]
gaussian_blur_values = [(3, 1.0), (5, 2.0), (7, 3.0)]
gaussian_noise_std_values = [0.03, 0.05, 0.07]
posterize_bits_values = [2, 4, 6]
solarize_threshold_values = [64, 128, 192]
sharpness_factor_values = [1.5, 2.0, 2.5]
autocontrast_p_values = [0.3, 0.5, 0.7]
equalize_p_values = [0.3, 0.5, 0.7]
auto_augment_values = ['imagenet', 'randaugment', 'trivialaugmentwide']

# Create directories to store the job scripts and logs
os.makedirs('vit_jobs', exist_ok=True)
os.makedirs('vit_aug_logs', exist_ok=True)

# SLURM job script template
slurm_template = """#!/bin/bash

#SBATCH --job-name=vit_train_{job_id}
#SBATCH --partition=teach_gpu
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=COMS031424
#SBATCH --output=vit_aug_logs/vit-{param_name}_{param_value}.log

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

# Run ViT training and validation
python train_vit_parser.py --batch_size 8 --num_epochs 20 --run_number {job_id} --dataset_path /user/work/yf20630/cow-dataset-project/datasets --seed 42 {aug_params}

# Deactivate virtualenv
conda deactivate
"""

# Function to generate and submit job scripts
def submit_jobs(param_name, param_values):
    for i, value in enumerate(param_values):
        job_id = f"{param_name}_{i}"
        
        if value is None:
            param_value = "None"
            aug_params = ""
        elif isinstance(value, tuple):
            param_value = '_'.join(map(str, value))
            aug_params = f"--{param_name} {' '.join(map(str, value))}"
        else:
            param_value = str(value)
            aug_params = f"--{param_name} {value}"
        
        job_script = slurm_template.format(
            job_id=job_id,
            param_name=param_name,
            param_value=param_value,
            aug_params=aug_params
        )
        
        script_path = f'vit_jobs/job_{job_id}.sh'
        with open(script_path, 'w') as f:
            f.write(job_script)
        
        os.system(f'sbatch {script_path}')

# Submit jobs for each parameter separately
submit_jobs('resize', resize_values)
submit_jobs('random_resized_crop', random_resized_crop_values)
submit_jobs('random_resize', random_resize_values)
submit_jobs('zoom_out', zoom_out_values)
submit_jobs('rotation_degrees', rotation_degrees_values)
submit_jobs('affine', affine_values)
submit_jobs('perspective', perspective_values)
submit_jobs('color_jitter', color_jitter_values)
submit_jobs('grayscale_p', grayscale_p_values)
submit_jobs('gaussian_blur', gaussian_blur_values)
submit_jobs('gaussian_noise_std', gaussian_noise_std_values)
submit_jobs('posterize_bits', posterize_bits_values)
submit_jobs('solarize_threshold', solarize_threshold_values)
submit_jobs('sharpness_factor', sharpness_factor_values)
submit_jobs('autocontrast_p', autocontrast_p_values)
submit_jobs('equalize_p', equalize_p_values)
submit_jobs('auto_augment', auto_augment_values)

print("All jobs submitted successfully!")