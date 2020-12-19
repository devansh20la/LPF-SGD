#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=40:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=flatness
#SBATCH --mail-type=END
#SBATCH --mail-user=db3484@nyu.edu
#SBATCH --gres=gpu:p40_4:1
#SBATCH --output=slurm_logs/%a.out

module purge
module load anaconda3/2019.10
module load cuda/10.2.89 
module load cudnn/10.1v7.6.5.32

source /share/apps/anaconda3/2019.10/etc/profile.d/conda.sh 
conda activate myenv

cd /beegfs/db3484/gen_v_sharp/flatness/

# python train.py \
# 	--ep=500 \
# 	--dtype='cifar10' \
# 	--print_freq=500 \
# 	--exp_num=${SLURM_ARRAY_TASK_ID} && \
# python invest_dl_prob.py \
# 	--exp_num=${SLURM_ARRAY_TASK_ID} \
# 	--dtype='cifar10'

# python train_label_noise.py \
# 		--dtype='cifar10' \
# 		--ep=350 \
# 		--ms=0 \
# 		--mo=0.9 \
# 		--wd=5e-4 \
# 		--lr=0.1 \
# 		--bs=1024 \
# 		--ln=${SLURM_ARRAY_TASK_ID} \
# 		--print_freq=50
# python sharp_label_noise.py --dtype='cifar10' --ms=0 --bs=1024 --ln=$SLURM_ARRAY_TASK_ID

let SLURM_ARRAY_TASK_ID=$(($SLURM_ARRAY_TASK_ID*2))
python sharp_data_noise.py --dtype='cifar10_noisy' --ms=1 --bs=512 --dn=$SLURM_ARRAY_TASK_ID
# python train_data_noise.py \
# 			--dtype='cifar10_noisy' \
# 			--ep=350 \
# 			--ms=1 \
# 			--mo=0.9 \
# 			--wd=5e-4 \
# 			--lr=0.1 \
# 			--bs=128 \
# 			--dn=$SLURM_ARRAY_TASK_ID \
# 			--print_freq=50

# let SLURM_ARRAY_TASK_ID=$(($SLURM_ARRAY_TASK_ID*2))
# python train_double_descent.py \
# 		--dtype='cifar10' \
# 		--ep=4000 \
# 		--ms=0 \
# 		--mo=0.0 \
# 		--wd=0.0 \
# 		--lr=0.0001 \
# 		--bs=128 \
# 		--width=${SLURM_ARRAY_TASK_ID} \
# 		--print_freq=50

# python sharp_dd.py --dtype='cifar10' --ms=0 --bs=1024 --width=${SLURM_ARRAY_TASK_ID}