#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=flatness
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_jobs/%a.out

module purge
module load anaconda3/2020.07
module load cuda/10.2.89 

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh 
conda activate myenv

cd /home/db3484/gen_v_sharp/flatness/
# python train.py --ep=500 --dtype="mnist" --exp_num=$SLURM_ARRAY_TASK_ID
python invest_dl_prob.py --exp_num=$SLURM_ARRAY_TASK_ID --dtype="mnist"
