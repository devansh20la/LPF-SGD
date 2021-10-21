#!/bin/bash

########################## Transformer ###################################
# for opt in "adam"; do
# 	for seed in 0; do
# 		sbatch srun_greene.sh ${opt} ${seed} ${rho}
# 	done
# done

# LPF_ADAM
# for opt in "lpfadam"; do
# 	for seed in 0 1 2; do
# 		for std in 0.0005; do
# 			sbatch srun_greene.sh ${opt} ${seed} ${std}
# 		done
# 	done
# done

# SmoothOut
# for opt in "smoothout"; do
# 	for seed in 0 1 2; do
# 		sbatch srun_greene.sh ${opt} ${seed} 0.009
# 	done
# done

# ESGD
# for opt in "esgd"; do
# 	for seed in 0 1 2; do
# 		for g0 in 0.5; do
# 			for g1 in 0.0001; do
# 				for lr in 0.1; do
# 					sbatch srun_greene.sh ${opt} ${seed} ${g0} ${g1} ${lr}
# 				done
# 			done
# 		done
# 	done
# done


# for opt in "adam" "sam"; do
# 	for seed in 0 1 2; do
# 		python _test.py --cp_dir="checkpoints/${opt}/run_ms_${seed}/"
# 	done
# done



