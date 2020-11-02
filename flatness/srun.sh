#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=flatness
#SBATCH --mail-type=END
##SBATCH --mail-user=db3484@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%a.out

module purge
module load anaconda3/2019.10
module load cuda/10.2.89 
module load cudnn/10.1v7.6.5.32

source /share/apps/anaconda3/2019.10/etc/profile.d/conda.sh 
conda activate myenv

cd /beegfs/db3484/gen_v_sharp/flatness/
python train.py --ep=500 --dtype="cifar10" --exp_num=$SLURM_ARRAY_TASK_ID
# python invest_dl_prob.py --exp_num=$SLURM_ARRAY_TASK_ID --dtype="mnist"
