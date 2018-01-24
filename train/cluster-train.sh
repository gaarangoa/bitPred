#!/bin/bash
#SBATCH -J bitPred
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -t 20:10:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:pascal:2

echo "Allocated GPU with ID $CUDA_VISIBLE_DEVICES"
# echo "Activate virtual environment: "
# export PYTHONNOUSERSITE=True
# source /work/newriver/gustavo1/deepLearning/bitPredEnv/bin/activate

source /home/gustavo1/deep_learning/bitPred/env/bin/activate
# source activate machine-learning

module load theano tensorflow

# change to the directory where the training scripts are
cd /home/gustavo1/deep_learning/bitPred/train/

# run master scrpt
python train.py

