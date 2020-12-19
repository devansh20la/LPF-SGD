#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=flatness
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_jobs/fsgd_train.out

# Regular
singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
	/bin/bash -c "cd /scratch/db3484/gen_v_sharp/optimization/; 
			  	  /ext3/anaconda3/bin/python fsgd_train.py \
					  	--ep=500 \
					  	--bs=128 \
					  	--dtype='cifar10' \
					  	--print_freq=100 \
					  	--mo=0.9 \
					  	--lr=0.1 \
					  	--ms=0 \
					  	--wd=5e-4;"
