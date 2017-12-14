#!/bin/bash
#SBATCH -J train-full-genes
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -t 20:10:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:pascal:1

echo "Allocated GPU with ID $CUDA_VISIBLE_DEVICES"
echo "Activate virtual environment: "
export PYTHONNOUSERSITE=True
source ~/deepargTrainEnv/bin/activate

module load cuda gcc/5.4.0
python /work/newriver/gustavo1/deeparg/training/deeparg-ss/argdb/train_arc_genes.py