#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --partition=econ-ssl
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --account=cosc028244

zip -r crops.zip /group/nwc-group/tony/crops
