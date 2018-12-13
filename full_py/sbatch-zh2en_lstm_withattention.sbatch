#!/bin/bash
#SBATCH --job-name=zh-lstm-a
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=49:59:59
#SBATCH --gres=gpu:p40:1


module purge
#which python
#module load anaconda3/5.3.0 cuda/9.0.176 cudnn/9.0v7.0.5
#which python
#source activate /home/sm7582/.conda/envs/denoising/bin/python

#source activate denoising
#which python
#export PYTHONPATH=$PYTHONPATH:/home/sm7582/.conda/envs/denoising/bin/python
which python
python zh2en_lstm_withattention.py
