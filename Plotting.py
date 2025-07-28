import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch import nn
from torch.utils.data import DataLoader,Dataset
from Utils import test_loop, Network, Data
import yaml
import pandas as pd
import skink as skplt

def discriminant(arr: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return np.log((arr+eps)/(1-arr+eps))

def calc_rej(discs, labels, sig_eff):

	cutvalue = np.percentile(discs[labels == 1], 100* (1.0-sig_eff))


	sum_pass = 0
	sum_bkg = np.sum(labels==0)

	for event, disc in enumerate(discs):
		if disc > cutvalue and labels[event] == 0:
			sum_pass += 1

	rejection = sum_bkg/sum_pass

	return rejection

#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################

print("Running")

# import config
file = open("Config.yaml")
config = yaml.load(file, Loader=yaml.FullLoader)

variables = config["variables"]
structure = config["training"]["structure"]["hidden_layers"]
learning = config["training"]["learning"]
signal = config["training"]["signal"]
background = config["training"]["background"]

net_name = config["training"]["add_name"]+"_"+config["training"]["signal_name"]+"_"+config["training"]["background_name"]

save_to_label_dict = {"bbbb_sig": "4b signal mc",
					  "bb_data": "2b2j data",
					  "bbbb_bkg": "4b background prediction"}

sig_name = save_to_label_dict[config["training"]["signal_name"]]
bkg_name = save_to_label_dict[config["training"]["background_name"]]

for i in range(len(structure)):
	net_name += "_"+str(structure[i])

net_name += "_"+str(learning["learning_rate"])+"_"+str(learning["epochs"])

plot_dir = "plots/network/"+net_name+"/"

if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

# Selects the best trained epoch
ckpt_list = os.listdir("trained_networks/"+net_name)
ckpt_list.sort()
ckpt_list.pop(-1)

ckpt_nums = []
for ckpt in ckpt_list:
	for char in range(len(ckpt)):
		if ckpt[char].isdigit() and ckpt[char+1].isdigit():
			ckpt_nums.append(int(ckpt[char:char+2]))
			break
		elif ckpt[char].isdigit():
			ckpt_nums.append(int(ckpt[char]))
			break

losses = []
for num, ckpt in enumerate(ckpt_list):
	losses.append(ckpt_list[num][15+len(str(ckpt_nums[num])):-4])

print(ckpt_list[np.argmin(losses)])

# Downloads trained model
model = torch.load("trained_networks/"+net_name+"/"+ckpt_list[np.argmin(losses)], weights_only = False)

# Initialize loss function
loss_function = nn.BCELoss(reduction="none")
###################################################################

test_rw_arr = []
# Imports training and testing samples

# Grab Signal test and train files
test_file = h5py.File(signal+"test.h5")

# Converts Files to dataset
test_tensors = []

for var in variables:
	test_tensors.append(torch.from_numpy(test_file["Data"][var]))

test_vecs = torch.stack((test_tensors),-1)
test_weights = torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1)
test_labels = torch.from_numpy(np.ones_like(test_file["Data"]["label"])).unsqueeze(1)

# Closes h5 files
test_file.close()

# Do it all again with background
# Grab Signal test and train files
test_file = h5py.File(background+"test.h5")

# Converts Files to dataset
test_tensors = []

for var in variables:
	test_tensors.append(torch.from_numpy(test_file["Data"][var]))

test_vecs = torch.cat((test_vecs,torch.stack((test_tensors),-1)))
test_weights = torch.cat((test_weights,torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1)))
test_labels = torch.cat((test_labels,torch.from_numpy(np.zeros_like(test_file["Data"]["label"])).unsqueeze(1)))

# Closes h5 files
test_file.close()
test_data = Data(test_vecs, test_weights, test_labels)

# Conversts datsets to datlaoaders for training
test_dataloader = DataLoader(test_data, batch_size=learning["batch_size"], shuffle=False)

# Runs over validation/test dataset to generate arrays of the network inputs and outputs
outputs, labels = test_loop(test_dataloader,model,loss_function)

# Discriminant calculations
discs = np.apply_along_axis(discriminant,0,outputs)
ratios = np.exp(discs)

# Rejection Plotting

eff_space = np.linspace(0.5,1,50)

rejs = []


for eff in eff_space:
	rejs.append(calc_rej(discs,labels.flatten(),eff))

print("Plotting")

## Output Plotting

bins  = skplt.get_bins(0,1,40)

HistogramPlot = skplt.HistogramPlot(bins = bins, xlabel = "Output Score", ylabel = "Number of Events")

HistogramPlot.add(data = outputs[labels.flatten() == 1], label = sig_name)
HistogramPlot.add(data = outputs[labels.flatten() == 0], label = bkg_name)

HistogramPlot.Plot(plot_dir+"outputs")

## Ratio Plotting

norm = True
bins = skplt.get_bins(0,1000,40)

HistogramPlot = skplt.HistogramPlot(bins = bins, xlabel = "Ratio", ylabel = "Number of Events", logy = True)

HistogramPlot.Add(data = ratios[labels.flatten() == 1], label = sig_name)
HistogramPlot.Add(data = ratios[labels.flatten() == 0], label = bkg_name)	

HistogramPlot.Plot(plot_dir+"ratio")

## Discriminant Plotting

bins  = skplt.get_bins(-10,10,40)

HistogramPlot = skplt.HistogramPlot(bins = bins, xlabel = "Discriminant Value", ylabel = "Number of Events", ratio = True, logy = False)

HistogramPlot.Add(data = discs[labels.flatten() == 1], label = sig_name, refernece = True)
HistogramPlot.Add(data = discs[labels.flatten() == 0], label = bkg_name)

HistogramPlot.Plot(plot_dir+"discs")


## disc ratio plotting

ratio_ratio = np.divide(np.histogram(discs[labels.flatten() == 1], bins = bins)[0],np.histogram(discs[labels.flatten() == 0], bins = bins)[0])

LinePlot = skplt.LinePlot(xs = bins[1], xlabel = "Discriminant Value", ylabel = "Ratio of Discriminant Values", logy = True, ratio = True)

LinePlot.Add(ys = ratio_ratio, label = "Ratio of ("+sig_name+")/("+bkg_name+")", refernece = True)
LinePlot.Add(ys = np.exp(bins[1]), label = "Ideal Ratio", drawstyle = "steps", drawstyle = "--", colour = "grey")

LinePlot.Plot(plot_dir+"discs_ratio")

## ROC curve

LinePlot = skplt.LinePlot(xs = eff_space, xlabel = sig_name+" Efficiency", ylabel = bkg_name+" Rejection", logy = True)

LinePlot.Add(ys = rejs)

LinePlot.Plot(plot_dir+"ROC")

print("Done")
