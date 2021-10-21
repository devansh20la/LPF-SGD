#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=flatness
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_jobs/%j.out

####################### WRN ##############################################
# SGD - SAM - SmoothOut
singularity exec --nv --overlay /scratch/$(whoami)/overlay-25GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	/bin/bash -c "cd /scratch/$(whoami)/gen_v_sharp/adversarial/; 
		/ext3/anaconda3/bin/python main.py \
		--img_aug ${1} \
		--opt ${2} \
		--dtype ${3} \
		--seed ${4} \
		--mtype ${5} \
		--depth ${6} \
		--width_factor ${7} \
		--batch_size 1500"
