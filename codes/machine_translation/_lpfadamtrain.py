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


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def create_path(args, path):
	if os.path.isdir(path) is False:
		os.makedirs(path)
	else:
		if len(glob.glob(f"{path}/*.log", recursive=True)) > 0:
			file = glob.glob(f"{path}/*.log", recursive=True)[0]
			with open(file, 'r') as f:
				text = f.read()
			if f"Step: {args.ts - 1}" in text:
				print("File exists")
				quit()
			else:
				shutil.rmtree(path)
				os.makedirs(path)
				print("Removing old files")
		else:
			shutil.rmtree(path)
			os.makedirs(path)
			print("Removing old files")
				
	return 


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# optim config
	parser.add_argument("--ts", default=100000, type=int, help="Total number of epochs.")
	parser.add_argument("--lr", default=0.0005, type=float, help="Base learning rate at the start of the training.")
	parser.add_argument("--wd", default=0.0001, type=float, help="L2 weight decay.")
	parser.add_argument("--bs", default=256, type=int, help="Batch size used in the training and validation loop.")
	parser.add_argument("--dp", default=0.1, type=float)
	
	parser.add_argument("--std", default=0.0005, type=float)
	parser.add_argument("--inc", default=15, type=int)
	parser.add_argument("--M", default=8, type=int)


	# seed
	parser.add_argument("--seed", default=123, type=int, help="seed")
	parser.add_argument("--print_freq", type=int, default=1000, help="print frequency")

	args = parser.parse_args()

	initialize(args.seed)

	# initialize directory
	args.cp_dir = f"checkpoints/lpfadam_{args.std}_{args.inc}_{args.M}/run_ms_{args.seed}/"
	create_path(args, args.cp_dir)
	for file in glob.glob("**/*.py", recursive=True):
		if "checkpoints" in file or "data" in file or "results" in file:
			continue
		os.makedirs(os.path.dirname(f"{args.cp_dir}/codes/{file}"), exist_ok=True)
		shutil.copy(file, f"{args.cp_dir}/codes/{file}")

	# initialize logging
	train_log = os.path.join(args.cp_dir, time.strftime("%Y%m%d-%H%M%S") + '.log')
	logging.basicConfig(
		format="%(name)s: %(message)s",
		level="INFO",
		handlers=[
			logging.FileHandler(train_log),
			logging.StreamHandler()
		]
	)

	writer = SummaryWriter(log_dir=args.cp_dir)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	bpe_model = yttm.BPE(model="data/bpe.32000.model")
	logging.info(f"Vocab length: {bpe_model.vocab_size()}")

	# Create dataset
	dset_loaders = {
		'train': data.load("data/",
						   split='train',
						   batch_size=args.bs,
						   bpe_model=bpe_model,
						   workers=8),
		'val': data.load("data/",
						 split='dev',
						 batch_size=args.bs,
						 shuffle=False,
						 bpe_model=bpe_model),
		'test': data.load("data/",
						 split='dev',
						 batch_size=1,
						 shuffle=False,
						 bpe_model=bpe_model)
	} 
	args.ts = math.ceil(args.ts / len(dset_loaders['train'])) * len(dset_loaders['train'])
	logging.info(args)

	model = Seq2SeqTransformer(
		num_encoder_layers=3,
		num_decoder_layers=3,
		emb_size=512,
		nhead=8,
		vocab_size=bpe_model.vocab_size(),
		dim_feedforward=512,
		dropout=args.dp
	)
	model = model.to(device)

	criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=args.lr,
		betas=(0.9, 0.999), 
		eps=1e-08,
		weight_decay=args.wd)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
														   mode='min',
														   factor=0.5,
														   patience=1,
														   threshold=0.01,
														   min_lr=1e-6)

	
	std_scheduler = CosineInc(args.std, args.ts, args.inc - 1)
	std = std_scheduler(0)

	best_loss = float('inf')
	step = 0
	epoch = 0
	current_step = 0.0
	while (step < args.ts):
		trainloss = AverageMeter()
		model.train()
		
		for batch_idx, batch in enumerate(dset_loaders['train']):
			inputs, targets = (b.to(device) for b in batch)

			inputs1 = torch.split(inputs, int(args.bs / args.M), dim=1)
			targets1 = torch.split(targets, int(args.bs / args.M), dim=1)

			for inputs, targets in zip(inputs1, targets1):
				targets_input = targets[:-1, :]
				targets_out = targets[1:, :]

				# new technique
				with torch.no_grad():
					noise = []
					for mp in model.parameters():
						if len(mp.shape) > 1:
							sh = mp.shape
							sh_mul = np.prod(sh[1:])
							temp = mp.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(mp.shape)
							temp = torch.normal(0, std*temp).to(mp.data.device)
						else:
							temp = torch.empty_like(mp, device=mp.data.device)
							temp.normal_(0, std*(mp.view(-1).norm().item() + 1e-16))
						noise.append(temp)
						mp.data.add_(noise[-1])

				# single sample convolution approximation
				with torch.set_grad_enabled(True):
					inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask = create_mask(inputs, targets_input, device)
					outputs = model(inputs, targets_input, inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
					
					batch_loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets_out.reshape(-1)) / args.M
					batch_loss.backward()

				# going back to without theta
				with torch.no_grad():
					for mp, n in zip(model.parameters(), noise):
						mp.data.sub_(n)

				with torch.no_grad():
					trainloss.update(batch_loss.item(), inputs.shape[1])
				
			step+=1
			optimizer.step()
			optimizer.zero_grad()                
			std = std_scheduler(step)

		writer.add_scalar('Train/train_loss', trainloss.avg, step)
		writer.add_scalar('Params/lr', optimizer.param_groups[0]['lr'], step)
		
		inputs = "Republican leaders justified their policy by the need to combat electoral fraud."
		inputs = translate(model, inputs, bpe_model, device)
		logging.info(f"Translated Sentence \n {inputs}")

		model.eval()
		valloss = AverageMeter()

		with torch.no_grad():
			for batch_idx, batch in enumerate(dset_loaders['val']):
				inputs, targets = (b.to(device) for b in batch)
				targets_input = targets[:-1, :]
				targets_out = targets[1:, :]

				inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask = create_mask(inputs, targets_input, device)
				outputs = model(inputs, targets_input, inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
				
				batch_loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets_out.reshape(-1))
			
				valloss.update(batch_loss.item(), inputs.shape[1])
					
				if batch_idx % args.print_freq == 0:
					logging.info('Batch: {0}/{1}, Val_Loss = {2}'.format(batch_idx, len(dset_loaders['val']), valloss.avg))

		writer.add_scalar('Val/val_loss', valloss.avg, step)
		writer.add_scalar('Param/lr', optimizer.param_groups[0]['lr'], step)
		logging.info('Step: {0}/{1}, Train_Loss = {2}, Val_Loss = {3}'.format(step, args.ts, trainloss.avg, valloss.avg))

		if valloss.avg < best_loss:
			torch.save(model.state_dict(), f"{args.cp_dir}/best_model.pth.tar")
			best_loss = valloss.avg
		
		scheduler.step(valloss.avg)
		epoch+=1
		if epoch % 2 == 0:
			with torch.no_grad():
				score = compute_bleu(model, dset_loaders['test'], bpe_model, device)
			writer.add_scalar('Test/bleu_score', score,  step)
			logging.info('Bleu_score = {0}'.format(score))

		