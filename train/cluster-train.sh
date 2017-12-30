#!/bin/bash
#SBATCH -J bitPred
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -t 20:10:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:pascal:1

echo "Allocated GPU with ID $CUDA_VISIBLE_DEVICES"
# echo "Activate virtual environment: "
# export PYTHONNOUSERSITE=True
# source /work/newriver/gustavo1/deepLearning/bitPredEnv/bin/activate

module load anaconda2

source activate gustavo1
module load gcc cuda theano

cd /work/newriver/gustavo1/deepLearning/bitPred/train/
# module load cuda gcc/5.4.0 theano

# run master scrpt
python train.py

