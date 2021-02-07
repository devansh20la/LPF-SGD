#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
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
		--output_dir checkpoints/ \
		--model_name WideResnet28x10 \
		--image_level_augmentations ${2} \
		--batch_level_augmentations ${3} \
		--num_epochs 200 \
		--weight_decay ${4} \
		--batch_size 32 \
		--learning_rate 0.1 \
		--sam_rho -1 \
		--ssgd_std 0.0001 \
		--std_inc ${5} \
		--M 8 \
		--run_seed 0"

python3 -m train \
		--dataset cifar100 \
		--output_dir checkpoints/ \
		--model_name WideResnet28x10 \
		--image_level_augmentations basic \
		--batch_level_augmentations none \
		--num_epochs 200 \
		--weight_decay 0.001 \
		--batch_size 32 \
		--learning_rate 0.1 \
		--sam_rho -1 \
		--ssgd_std 0.0001 \
		--std_inc 1 \
		--M 8 \
		--load_checkpoint "checkpoints/cifar100/exp_run/run_0/checkpoints/" \
		--run_seed 0