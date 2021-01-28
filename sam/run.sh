#!/bin/bash

# query_device() {
# 	for d in 0 1 2 3
# 	do
# 		dmem="$(nvidia-smi -q -i ${d} -d Memory | grep -A4 GPU | grep Used | grep -Eo '[0-9]{1,5}')"
# 		if ((${dmem} < 50)); then
# 		    device=${d}
# 		fi
# 	done
# }

# get_device() {
# 	unset device
# 	while [[ -z "${device}" ]]
# 	do
# 		query_device
# 		if [[ -z "${device}" ]]; then
# 			echo "All devices are busy sleeping for 10s"
# 			sleep 5
# 		fi
# 	done
# }

for dtype in 'cifar10'; do
	for img_aug in 'basic'; do
		for img_batch_aug in 'none'; do
			for stdinc in 1 5 10; do
				sbatch srun_greene.sh ${dtype} ${img_aug} ${img_batch_aug} ${stdinc}
			done
		done
	done
done

# for dtype in 'cifar10' 'cifar100'; do
# 	for img_aug in 'basic' 'autoaugment'; do
# 		if [[ $img_aug == "basic" ]]
# 		then
# 			for img_batch_aug in 'none' 'cutout'; do
# 				for std in 0.0001; do
# 					sbatch srun_greene.sh ${dtype} ${img_aug} ${img_batch_aug} ${std}
# 				done
# 			done
# 		else
# 			for img_batch_aug in 'cutout'; do
# 				for std in 0.0001; do
# 					sbatch srun_greene.sh ${dtype} ${img_aug} ${img_batch_aug} ${std}
# 				done
# 			done
# 		fi
# 	done
# done