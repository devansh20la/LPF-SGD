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

# for ms in 0 1 2 3 4; do
# 	get_device
# 	CUDA_VISIBLE_DEVICES=$((device)) nohup python -u test.py \
# 	-m LeNet -l2 5e-4 -L 5 --gamma 1e-4 --scoping 1e-3 --noise 1e-4 --lr 0.01 -s ${ms} & 
# 	sleep 2
# done


for ms in 0 1 2 3 4; do
	get_device
	CUDA_VISIBLE_DEVICES=$((device)) nohup python -u train.py \
	--l2 0.0005 -L 5 --gamma 1e-4 --scoping 1e-3 --noise 1e-4 --lr 0.01 -s ${ms} > jobs/${ms}.out 2>&1  & 
	sleep 30
done
