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
	for dtype in 'cifar10' 'cifar100'; do
		for opt in 'sgd' 'sam' 'lpf'; do
			get_device
			CUDA_VISIBLE_DEVICES=$((device)) nohup python ${opt}_train.py \
			  	--ep=150 \
			  	--bs=128 \
			  	--dtype=${dtype} \
			  	--mtype=${mtype} \
			  	--print_freq=100 \
			  	--mo=0.9 \
			  	--lr=0.1 \
			  	--ms=${ms} \
			  	--wd=5e-4 \
			  	--gamma_0=${g0} \
			  	--gamma_1=${g1} > jobs/entropy_${g0}_${g1}_${ms}.out 2>&1 &
		done
	done
done


# for ms in 0; do
# 	for dtype in 'mnist'; do
# 		for mtype in 'lenet'; do
# 			for g0 in 0.001; do
# 				for g1 in 0.0001; do
# 					get_device
# 					CUDA_VISIBLE_DEVICES=$((device)) nohup python entropy_train.py \
# 					  	--ep=150 \
# 					  	--bs=128 \
# 					  	--dtype=${dtype} \
# 					  	--mtype=${mtype} \
# 					  	--print_freq=100 \
# 					  	--mo=0.9 \
# 					  	--lr=0.1 \
# 					  	--ms=${ms} \
# 					  	--wd=5e-4 \
# 					  	--gamma_0=${g0} \
# 					  	--gamma_1=${g1} > jobs/entropy_${g0}_${g1}_${ms}.out 2>&1 &
# 					sleep 10
# 					# sbatch srun_greene.sh ${opt} ${ms} ${mtype} ${dtype} ${g0} ${g1}
# 				done
# 			done
# 		done
# 	done
# done