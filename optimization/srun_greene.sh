#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=flatness
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_jobs/%j.out


# Entropy SGD
singularity exec --nv --overlay /scratch/$(whoami)/overlay-7.5GB-300K.ext3:ro \
	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
	/bin/bash -c "cd /scratch/$(whoami)/gen_v_sharp/optimization/; 
			  	  /ext3/anaconda3/bin/python ${1}_train.py \
					  	--ep=200 \
					  	--bs=128 \
					  	--dtype=${4} \
					  	--mtype=${3} \
					  	--print_freq=100 \
					  	--mo=0.9 \
					  	--lr=0.1 \
					  	--ms=${2} \
					  	--wd=5e-4 \
					  	--gamma_0=${5} \
					  	--gamma_1=${6}"
