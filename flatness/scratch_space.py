import pickle
import os
import glob

for fol in glob.glob("*"):
	for file in glob.glob(f"{fol}/run_ms_0/*.pkl"):
		with open(file, 'rb') as f:
			mtr = pickle.load(f)
		try:
			if mtr["fro_norm"] < 1:
				print(fol, mtr)
		except:
			print(fol, mtr)