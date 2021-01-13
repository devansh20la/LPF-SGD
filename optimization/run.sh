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

# for mtype in 'resnet18'; do
# 	for optim in 'sgd' 'sam_sgd' 'fsgd'; do
# 		for dtype in 'tinyimagenet'; do
# 			for ms in {0..4}; do
# 				get_device
# 				CUDA_VISIBLE_DEVICES=$((device)) nohup python test.py --dtype=${dtype} --mtype=${mtype} --ms=${ms} --optim=${optim} &
# 				sleep 2
# 			done
# 		done
# 	done
# done
# for mtype in 'resnet18' 'resnet50' 'resnet101'; do
# 	for dtype in 'cifar10' 'cifar100'; do
# 		python train.py --ep=10 --bs=128 --dtype=${dtype} --mtype=${mtype} --print_freq=100 --mo=0.9 --lr=0.1 --ms=0 --wd=5e-4
# 	done
# done
# for mtype in 'resnet18' 'resnet50' 'resnet101'; do
# 	for dtype in 'cifar10' 'cifar100'; do
# 		python sam_train.py --ep=10 --bs=128 --dtype=${dtype} --mtype=${mtype} --print_freq=100 --mo=0.9 --lr=0.1 --ms=0 --wd=5e-4
# 	done
# done
# for mtype in 'resnet18' 'resnet50' 'resnet101'; do
# 	for dtype in 'cifar10' 'cifar100'; do
# 		python fsgd_train.py --std=0.002 --ep=10 --bs=128 --dtype=${dtype} --mtype=${mtype} --print_freq=100 --mo=0.9 --lr=0.1 --ms=0 --wd=5e-4
# 	done
# done


# for ms in 0 1 2 3 4; do
# 	get_device
# 	CUDA_VISIBLE_DEVICES=$((device)) nohup python train.py \
# 	  	--ep=100 \
# 	  	--bs=128 \
# 	  	--dtype='cifar10' \
# 	  	--mtype='resnet18' \
# 	  	--print_freq=100 \
# 	  	--mo=0.9 \
# 	  	--lr=0.1 \
# 	  	--ms=${ms} \
# 	  	--wd=1e-4 &
# 	sleep 30
# done


for ms in 0 1 2 3 4; do
	get_device
	CUDA_VISIBLE_DEVICES=$((device)) nohup python entropy_train.py \
	  	--ep=200 \
	  	--bs=128 \
	  	--dtype='cifar10' \
	  	--mtype='resnet50' \
	  	--print_freq=100 \
	  	--mo=0.9 \
	  	--lr=0.1 \
	  	--ms=${ms} \
	  	--wd=5e-4 &
	sleep 30
done
