import yaml
import os
import re
import argparse
import time
import torch
import utils
import numpy as np
import youtokentome as yttm
from tqdm import tqdm
import argparse
from models import *
import _data as data
from utils import *
import glob
import shutil
import logging
from torch.utils.tensorboard import SummaryWriter
import pickle

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def create_path(path):
	if os.path.isdir(path) is False:
		os.makedirs(path)

	return 


if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	bpe_model = yttm.BPE(model="data/bpe.32000.model")
	model = Seq2SeqTransformer(
		num_encoder_layers=3,
		num_decoder_layers=3,
		emb_size=512,
		nhead=8,
		vocab_size=bpe_model.vocab_size(),
		dim_feedforward=512,
		dropout=0.1
	)
	model = model.to(device)

	try:
		with open('checkpoints/scores_dump.pickle', 'rb') as handle:
			scores = pickle.load(handle)
	except:
		scores = {}

	for seed in [0, 1, 2]:
		for opt in ["esgd_0.5_0.0001_0.1"]:
			if f"{opt}_{seed}_val" in scores:
				print(f"Exists: {opt}_{seed}_val")
				continue

			if f"{opt}_{seed}_test" in scores:
				print(f"Exists: {opt}_{seed}_test")
				continue

			state = torch.load(f"checkpoints/{opt}/run_ms_{seed}/best_model.pth.tar")

			model.load_state_dict(state)
			model.eval()

			criterion = torch.nn.CrossEntropyLoss(ignore_index=1)

			# Create dataset
			dset_loaders = {
				'val': data.load("data/",
								split='dev',
								batch_size=1,
								shuffle=False,
								bpe_model=bpe_model),
				'test': data.load("data/",
								split='test',
								batch_size=1,
								shuffle=False,
								bpe_model=bpe_model)
			} 

			scores[f"{opt}_{seed}_val"] = compute_bleu(model, dset_loaders['val'], bpe_model, device)
			scores[f"{opt}_{seed}_test"] = compute_bleu(model, dset_loaders['test'], bpe_model, device)
			print(scores)
	
			with open('checkpoints/scores_dump.pickle', 'wb') as handle:
				pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

		