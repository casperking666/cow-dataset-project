#!/bin/bash

#SBATCH --job-name=yolo_train
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=cosc028244


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
yolo task=classify mode=train model=yolov8s-cls.yaml data=/user/work/yf20630/cow-dataset-project/datasets/subset_small epochs=400 imgsz=640 device=0 patience=20 save_period=1 hsv_h=0.0 hsv_s=0.7 hsv_v=0.0 degrees=0.0 translate=0.0 scale=0.0 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.0 mosaic=0.0 mixup=0.0 copy_paste=0.0 project=augmentation name=test
yolo task=classify mode=val model=augmentation/test/weights/best.pt data=/user/work/yf20630/cow-dataset-project/datasets/subset_small split=test


# Deactivate virtualenv
conda deactivate