#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=machine_translation
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_jobs/%j.out

####################### Transformer ######################################
# ADAM / SAM
# singularity exec --nv --overlay /scratch/$(whoami)/overlay-25GB-500K.ext3:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/$(whoami)/gen_v_sharp/machine_translation/; 
# 		/ext3/anaconda3/bin/python _${1}train.py \
# 		--wd 0.0001 \
# 		--lr 0.0005 \
# 		--bs 256 \
# 		--seed ${2}"

# LPF ADAM
# singularity exec --nv --overlay /scratch/$(whoami)/overlay-25GB-500K.ext3:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/$(whoami)/gen_v_sharp/machine_translation/; 
# 		/ext3/anaconda3/bin/python _${1}train.py \
# 		--wd 0.0001 \
# 		--lr 0.0005 \
# 		--bs 256 \
# 		--seed ${2} \
# 		--std ${3} \
# 		--inc 15"

# SmoothOut
# singularity exec --nv --overlay /scratch/$(whoami)/overlay-25GB-500K.ext3:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/$(whoami)/gen_v_sharp/machine_translation/; 
# 		/ext3/anaconda3/bin/python _${1}train.py \
# 		--wd 0.0001 \
# 		--lr 0.0005 \
# 		--bs 256 \
# 		--seed ${2} \
# 		--smooth_out_a 0.009"

# ESGD
singularity exec --nv --overlay /scratch/$(whoami)/overlay-25GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
	/bin/bash -c "cd /scratch/$(whoami)/gen_v_sharp/machine_translation/; 
		/ext3/anaconda3/bin/python _${1}train.py \
		--wd 0.0001 \
		--lr 0.0005 \
		--bs 256 \
		--ts 25000 \
		--seed ${2} \
		--g0 ${3} \
		--g1 ${4} \
		--sgld_lr ${5}"