import numpy as np
import h5py
import os
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from Utils import Data, Network, train_loop, val_loop, var_bins
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_var(datasets, dataset_names, test_train, var_name, plot_dir, bins):

	cs = ["#DE0C62",
	      "#9213E0",
	      "#094234"]  

	if not os.path.exists(plot_dir):
		os.mkdir(plot_dir)

	if not os.path.exists(plot_dir+test_train+"/"):
		os.mkdir(plot_dir+test_train+"/")

	for count, dataset in enumerate(datasets):

		plt.hist(dataset, bins = bins, histtype = "step", density = True, color = cs[count], label = dataset_names[count], linewidth = 3)
		plt.hist(dataset, bins = bins, density = True, color = cs[count], alpha = 0.1)

	plt.draw()

	plt.legend()
	plt.xlabel(var_name)
	plt.ylabel("Normalmized Number of Events")

	plt.savefig(plot_dir+"/"+test_train+"/"+var_name+".png")
	plt.savefig(plot_dir+"/"+test_train+"/"+var_name+".pdf")
	plt.clf()

#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################


# import config
file = open("Config.yaml")
config = yaml.load(file, Loader=yaml.FullLoader)

variables = config["variables"]

dataset_locations = ["/gpfs/home/epp/phubsg/MVA_Discriminant_V2/samples/bbbb_sig_1p0_run2/",
                     "/gpfs/home/epp/phubsg/MVA_Discriminant_V2/samples/bb_bkg_run2/",
					 "/gpfs/home/epp/phubsg/MVA_Discriminant_V2/samples/bbbb_bkg_run2/"]

dataset_names = ["bbbb_sig",
                 "2b2j_data",
				 "bbbb_bkg"]

datasets_train = []
datasets_test = []


for dataset_location in dataset_locations:

	train_file = h5py.File(dataset_location+"train.h5")
	test_file = h5py.File(dataset_location+"test.h5")

	train_tensors, test_tensors = [],[]

	for var in variables:
		train_tensors.append(torch.from_numpy(train_file["Data"][var]))
		test_tensors.append(torch.from_numpy(test_file["Data"][var]))

	train_vecs = torch.stack((train_tensors),-1)

	test_vecs = torch.stack((test_tensors),-1)

	datasets_train.append(train_vecs.numpy())
	datasets_test.append(test_vecs.numpy())

	# Closes h5 files
	test_file.close()
	train_file.close()

for count, var in enumerate(variables):

	bins = var_bins[var]

	print(var)
	plot_var([datasets_train[0][:,count],datasets_train[1][:,count],datasets_train[2][:,count]], dataset_names, "train", var, "plots/inputs", bins)
	plot_var([datasets_test[0][:,count],datasets_test[1][:,count],datasets_test[2][:,count]], dataset_names, "test", var, "plots/inputs", bins)

