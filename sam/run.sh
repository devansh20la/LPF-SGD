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

for ms in 0 1 2 3 4; do
	get_device
	CUDA_VISIBLE_DEVICES=$((device)) nohup python3 -m train \
		--dataset cifar10 \
		--model_name WideResnet28x10 \
		--output_dir checkpoints/ \
		--image_level_augmentations basic \
		--num_epochs 200 \
		--ssgd_std 0.001 \
		--run_seed ${ms} & 
	sleep 30
done

