#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=flatness
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_jobs/%a.out

singularity exec --nv --overlay /scratch/$(whoami)/jax_overlay.ext3:ro \
	/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	/bin/bash -c "cd /scratch/$(whoami)/gen_v_sharp/sam/; 
			  	  /ext3/anaconda3/bin/python3 -m train \
			  	  --dataset cifar10 \
			  	  --model_name WideResnet28x10 \
			  	  --output_dir checkpoints/ \
			  	  --image_level_augmentations none \
			  	  --num_epochs 200 \
			  	  --sam_rho 0.00 \
			  	  --ssgd_std 1e-4 \
			  	  --run_seed $SLURM_ARRAY_TASK_ID \
			  	  --use_std_schedule"