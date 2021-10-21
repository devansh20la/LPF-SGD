#!/bin/bash


########################## SHAKE SHAKE / PyramidNet ###################################

# for aug in 'basic_none' 'basic_cutout' 'autoaugment_cutout'; do
# 	for opt in 'sgd' 'sam';  do
# 		for dtype in 'cifar10' 'cifar100'; do
# 			for seed in 0 1 2 3 4; do
# 				# sbatch srun_greene.sh ${aug} ${opt} ${dtype} ${seed} "wrn" 16 8
# 				sbatch srun_greene.sh ${aug} ${opt} ${dtype} ${seed} "wrn" 28 10
# 			done
# 		done
# 	done
# done

for aug in 'basic_none' 'basic_cutout' 'autoaugment_cutout'; do
	for opt in 'sgd' 'sam'; do
		for dtype in 'cifar10' 'cifar100'; do
			for seed in 0 1 2 3 4; do
				sbatch srun_greene.sh ${aug} ${opt} ${dtype} ${seed} "pyramidnet" 110 270
			done
		done
	done
done
