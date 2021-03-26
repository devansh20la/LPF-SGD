#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=flatness
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_jobs/%a.out

# Regular
singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
	/bin/bash -c "cd /scratch/db3484/gen_v_sharp/flatness/; 
	/ext3/anaconda3/bin/python sharp_dl_prob.py \
		--exp_num=$SLURM_ARRAY_TASK_ID \
		--dtype='cifar10'"
			  	  # /ext3/anaconda3/bin/python train.py \
					  	# --ep=500 \
					  	# --print_freq=500 \
					  	# --dtype='cifar10' \
					  	# --exp_num=$SLURM_ARRAY_TASK_ID; \

# data noise
# let SLURM_ARRAY_TASK_ID=$(($SLURM_ARRAY_TASK_ID*2))
# singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/db3484/gen_v_sharp/flatness/; \
# 	/ext3/anaconda3/bin/python sharp_data_noise.py --dtype='cifar10_noisy' --ms=0 --bs=512 --dn=$SLURM_ARRAY_TASK_ID"

		# /ext3/anaconda3/bin/python train_data_noise.py \
		# 	--dtype='cifar10_noisy' \
		# 	--ep=350 \
		# 	--ms=1 \
		# 	--mo=0.9 \
		# 	--wd=5e-4 \
		# 	--lr=0.1 \
		# 	--bs=128 \
		# 	--dn=$SLURM_ARRAY_TASK_ID \
		# 	--print_freq=50"
# /ext3/anaconda3/bin/python sharp_data_noise.py --dtype='cifar10_noisy' --ms=0 --bs=512 --dn=$SLURM_ARRAY_TASK_ID

# label noise
# singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/db3484/gen_v_sharp/flatness/; \
# 	/ext3/anaconda3/bin/python sharp_label_noise.py --dtype='cifar10' --ms=0 --bs=512 --ln=$SLURM_ARRAY_TASK_ID"

					# /ext3/anaconda3/bin/python train_label_noise.py \
					# 	--dtype='cifar10' \
					# 	--ep=350 \
					# 	--ms=1 \
					# 	--mo=0.9 \
					# 	--wd=5e-4 \
					# 	--lr=0.1 \
					# 	--bs=128 \
					# 	--ln=$SLURM_ARRAY_TASK_ID \
					# 	--print_freq=50"


# Double descent
# let SLURM_ARRAY_TASK_ID=$(($SLURM_ARRAY_TASK_ID*2))
# singularity exec --nv --overlay /scratch/db3484/overlay-7.5GB-300K.ext3:ro \
# 	/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
# 	/bin/bash -c "cd /scratch/db3484/gen_v_sharp/flatness/; \
# 				  /ext3/anaconda3/bin/python train_double_descent.py \
# 				  	--dtype='cifar10' \
# 				  	--ep=4000 \
# 				  	--ms=0 \
# 				  	--mo=0.0 \
# 				  	--wd=0.0 \
# 				  	--lr=0.0001 \
# 				  	--bs=128 \
# 				  	--width=$SLURM_ARRAY_TASK_ID \
# 					--print_freq=50"


# /ext3/anaconda3/bin/python sharp_dd.py --dtype='cifar10' --ms=0 --bs=1024 --width=$SLURM_ARRAY_TASK_ID
