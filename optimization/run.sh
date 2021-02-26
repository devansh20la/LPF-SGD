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
# 	for optim in 'fsgd'; do
# 		for dtype in 'imagenet'; do
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

############## compute sharpness #######################
# for ms in 0; do
# 	for mtype in 'lenet'; do
# 		for dtype in 'mnist'; do
# 			get_device
# 			CUDA_VISIBLE_DEVICES=$((device)) nohup python sharp_dl.py \
# 				--dtype=${dtype} \
# 				--mtype=${mtype} \
# 				--cp_dir=checkpoints/mnist/lenet/sgd/run_ms_${ms} > jobs/sgd.out 2>&1 &
# 			sleep 10
# 			get_device
# 			CUDA_VISIBLE_DEVICES=$((device)) nohup python sharp_dl.py \
# 				--dtype=${dtype} \
# 				--mtype=${mtype} \
# 				--cp_dir=checkpoints/mnist/lenet/sam_sgd/run_ms_${ms} > jobs/sam.out 2>&1 &
# 			sleep 10
# 			get_device
# 			CUDA_VISIBLE_DEVICES=$((device)) nohup python sharp_dl.py \
# 				--dtype=${dtype} \
# 				--mtype=${mtype} \
# 				--cp_dir=checkpoints/mnist/lenet/fsgd/run_ms_${ms} > jobs/fsgd.out 2>&1 &
# 			sleep 10
# 		done
# 	done
# done

# for ms in 0 1 2 3 4; do
# 	for dtype in 'cifar10' 'cifar100'; do
# 		for mtype in 'resnet50' 'resnet101'; do
# 			for opt in 'sgd' 'sam_sgd' 'fsgd'; do
# 				sbatch srun_greene.sh ${mtype} ${dtype} ${ms} ${opt}
# 			done
# 		done
# 	done
# done
