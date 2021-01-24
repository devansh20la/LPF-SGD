#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=flatness
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_jobs/%j.out

singularity exec --nv --overlay /scratch/$(whoami)/jax_overlay.ext3:ro \
	/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	/bin/bash -c "cd /scratch/$(whoami)/gen_v_sharp/sam/; 
			  	  /ext3/anaconda3/bin/python3 -m train \
			  	  --dataset ${1} \
			  	  --model_name WideResnet28x10 \
			  	  --output_dir checkpoints/ \
			  	  --image_level_augmentations ${2} \
			  	  --batch_level_augmentations ${3} \
			  	  --num_epochs 200 \
			  	  --weight_decay 0.0005 \
			  	  --batch_size 256 \
			  	  --learning_rate 0.1 \
			  	  --sam_rho -1 \
			  	  --ssgd_std ${4} \
			  	  --run_seed 0"