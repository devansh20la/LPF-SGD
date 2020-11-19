#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=flatness
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu
#SBATCH --output=slurm_jobs/%a.out

singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
	/bin/bash -c "cd /home/db3484/gen_v_sharp/flatness/; \
				  /ext3/anaconda3/bin/python train.py --ep=500 --print_freq=500 --dtype='cifar10' --exp_num=$SLURM_ARRAY_TASK_ID; \
				  /ext3/anaconda3/bin/python invest_dl_prob.py --exp_num=$SLURM_ARRAY_TASK_ID --dtype='cifar10' "
