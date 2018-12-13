#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1080ti:1
#SBATCH --job-name=vi_gru_a
#SBATCH --time=48:00:00
#SBATCH --mem=60GB


module purge
#which python
#module load anaconda3/5.3.0 cuda/9.0.176 cudnn/9.0v7.0.5
#which python
#source activate /home/sm7582/.conda/envs/denoising/bin/python

#source activate denoising
#which python
#export PYTHONPATH=$PYTHONPATH:/home/sm7582/.conda/envs/denoising/bin/python
# which python
srun python vi2en_gru_withattention.py >>vi2en_gru_withattention.log 2>&1
