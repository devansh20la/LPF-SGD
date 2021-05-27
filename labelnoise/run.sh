#!/bin/bash

query_device() {
	for d in 0 1 2 3
	do
		dmem="$(nvidia-smi -q -i ${d} -d Memory | grep -A4 GPU | grep Used | grep -Eo '[0-9]{1,5}')"
		if ((${dmem} < 50)); then
		    device=${d}
		fi
	done
}

get_device() {
	unset device
	while [[ -z "${device}" ]]
	do
		query_device
		if [[ -z "${device}" ]]; then
			echo "All devices are busy sleeping for 10s"
			sleep 5
		fi
	done
}

# for ms in 0 1; do
# 	for opt in 'ssgd'; do
# 		for noise in 0 20 60 80; do
# 			get_device
# 			CUDA_VISIBLE_DEVICES=$((device)) nohup python ${opt}_train.py \
# 					--ep=300 \
# 					--bs=128 \
# 					--dtype='cifar10' \
# 					--mtype='resnet32' \
# 					--print_freq=100 \
# 					--mo=0.9 \
# 					--lr=0.1 \
# 					--ms=${ms} \
# 					--wd=5e-4 \
# 					--noise=${noise} > jobs/${opt}_${noise}_${ms}.out 2>&1 &
# 			sleep 10
# 		done
# 	done
# done

for ms in 0 1; do
	for opt in 'ssgd'; do
		for noise in 0 20 40 60 80; do
			sbatch srun_greene.sh ${opt} ${ms} ${noise}
		done
	done
done

# for ms in 0 1; do
# 	for opt in 'ssgd'; do
# 		for noise in 20 40 60 80; do
# 			sbatch srun_greene.sh ${opt} ${ms} ${noise}
# 		done
# 	done
# done