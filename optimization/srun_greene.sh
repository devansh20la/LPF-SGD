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

dtype='cifar10'
mtype='resnet18'

# singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
# 	--overlay /scratch/work/public/imagenet/imagenet-train.sqf:ro \
# 	--overlay /scratch/work/public/imagenet/imagenet-val.sqf:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/db3484/gen_v_sharp/optimization/; 
# 			  	  /ext3/anaconda3/bin/python sam_train.py \
# 					  	--ep=100 \
# 					  	--bs=256 \
# 					  	--dtype=${dtype} \
# 					  	--mtype=${mtype} \
# 					  	--print_freq=100 \
# 					  	--mo=0.9 \
# 					  	--lr=0.01 \
# 					  	--ms=$SLURM_ARRAY_TASK_ID \
# 					  	--wd=5e-4;"
# singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/db3484/gen_v_sharp/optimization/; 
# 			  	  /ext3/anaconda3/bin/python test.py \
# 					  	--dtype=${dtype} \
# 					  	--mtype=${mtype} \
# 					  	--ms=$SLURM_ARRAY_TASK_ID \
# 					  	--optim='sgd';"

# FSGD
# singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
# 	--overlay /scratch/work/public/imagenet/imagenet-train.sqf:ro \
# 	--overlay /scratch/work/public/imagenet/imagenet-val.sqf:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/db3484/gen_v_sharp/optimization/; 
# 			  	  /ext3/anaconda3/bin/python fsgd_train.py \
# 					  	--ep=100 \
# 					  	--bs=256 \
# 					  	--std=0.0009 \
# 					  	--dtype=${dtype} \
# 					  	--mtype=${mtype} \
# 					  	--print_freq=100 \
# 					  	--mo=0.9 \
# 					  	--lr=0.1 \
# 					  	--ms=$SLURM_ARRAY_TASK_ID \
# 					  	--wd=1e-4"
# singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/db3484/gen_v_sharp/optimization/; 
# 			  	  /ext3/anaconda3/bin/python test.py \
# 					  	--dtype=${dtype} \
# 					  	--mtype=${mtype} \
# 					  	--optim='entropy_sgd' \
# 					  	--ms=$SLURM_ARRAY_TASK_ID" ; 


# Entropy SGD
singularity exec --nv --overlay /scratch/hz1922/overlay-7.5GB-300K.ext3:ro \
	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
	/bin/bash -c "cd /scratch/hz1922/gen_v_sharp/optimization/; 
			  	  /ext3/anaconda3/bin/python entropy_train.py \
					  	--ep=300 \
					  	--bs=128 \
					  	--dtype=${dtype} \
					  	--mtype=${mtype} \
					  	--print_freq=100 \
					  	--mo=0.9 \
					  	--lr=0.1 \
					  	--ms=$SLURM_ARRAY_TASK_ID \
					  	--wd=5e-4;"