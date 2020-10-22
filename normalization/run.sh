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
			echo "All devices are busy sleeping for 60s"
			sleep 60
		fi
	done
}

for ms in {1..10}
do
	get_device
	CUDA_VISIBLE_DEVICES=$((device)) nohup python train.py \
		--ms=${ms} \
		--optim="sgd" \
		--lr=0.01  \
		--wd=0.0  \
		--m=0.9  \
		--bs=64  \
		--dtype="cifar10" &
	sleep 30
done

