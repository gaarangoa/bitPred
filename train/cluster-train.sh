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

# dont use the local installed modules
export PYTHONNOUSERSITE=True

# activate the virtual environment in conda
module load anaconda2 
source activate machine-learning

# load modules
module load gcc/5.4.0 cuda theano nccl anaconda2

# change to the directory where the training scripts are
cd /work/newriver/gustavo1/deepLearning/bitPred/train/

# run master scrpt
python train.py

