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

def plot_hist(data_arr, bins, norm, colour, label, linewidth = 3, histtype = "step", alpha = 0.1, ax = None, weights = None):

	if not ax:
		plt.hist(data_arr, bins = bins, density = norm,	color = colour,	label = label, histtype = histtype, linewidth = linewidth, weights = weights)
		plt.hist(data_arr, bins = bins,	density = norm,	color = colour,	alpha = alpha, weights = weights)
	else:
		ax.hist(data_arr, bins = bins, density = norm,	color = colour,	label = label, histtype = histtype, linewidth = linewidth, weights = weights)
		ax.hist(data_arr, bins = bins,	density = norm,	color = colour,	alpha = alpha, weights = weights)

def transport_distance_p_1(data_arr_sig,sig_weights,data_arr_bkg,bkg_weights,bins):

	sig_hist = np.histogram(data_arr_sig, bins = bins, weights = sig_weights)[0]
	bkg_hist = np.histogram(data_arr_bkg, bins = bins, weights = bkg_weights)[0]

	sig_hist = sig_hist/np.sum(sig_hist)
	bkg_hist = bkg_hist/np.sum(bkg_hist)

	sig_cdf = np.cumsum(sig_hist)
	bkg_cdf = np.cumsum(bkg_hist)

	cdf_diff = np.subtract(sig_cdf,bkg_cdf)

	distance = np.sum(np.abs(cdf_diff))

	# sig_sorted = np.sort(data_arr_sig)
	# bkg_sorted = np.sort(data_arr_bkg)

	# distance = np.sum(np.abs(np.subtract(sig_sorted,bkg_sorted)))/sig_sorted.size


	return distance

#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################

# import config
file = open("Config_V3.yaml")
config = yaml.load(file, Loader=yaml.FullLoader)

variables = config["variables"]
structure = config["training"]["structure"]["hidden_layers"]
learning = config["training"]["learning"]
signal = config["training"]["signal"]
background = config["training"]["background"]

net_name = config["training"]["signal_name"]+"_"+config["training"]["background_name"]

for i in range(len(structure)):
	net_name += "_"+str(structure[i])

net_name += "_"+str(learning["learning_rate"])+"_"+str(learning["epochs"])

plot_dir = "plots/network/"+net_name+"/"

if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

# Selects the best trained epoch
ckpt_list = os.listdir("trained_networks/"+net_name)

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
loss_function = nn.BCELoss()
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

## Colour Scheme we want:
##   - E78AA1 (Orange)
##   - DE0C62 (Red)
##   - 9213E0 (Purple)
##   - 11B7E0 (Blue)
##   - 00B689 (Green)

c1 = "#DE0C62"
c2 = "#9213E0"

## Loss curve

ckpt_nums_sorted, losses_sorted = zip(*sorted(zip(ckpt_nums,losses)))


plt.plot(ckpt_nums_sorted,[round(float(loss),4) for loss in losses_sorted], linewidth = 3, color = c1)
plt.draw()

plt.xlabel("Epoch")
plt.ylabel("Avg. Validation Loss")

plt.savefig(plot_dir+"loss.png")
plt.savefig(plot_dir+"loss.pdf")
plt.clf()

## Output Plotting

norm = True
bins  = np.linspace(0,1,40)

plot_hist(outputs[labels.flatten() == 1], bins = bins, norm = norm, colour = c1, label = "Signal")
plot_hist(outputs[labels.flatten() == 0], bins = bins, norm = norm, colour = c2, label = "Background")

plt.draw()

plt.legend()
plt.xlabel("Output Score")
plt.ylabel("Normalised Number of Events" if norm else "Number of Events")

plt.savefig(plot_dir+"outputs.png")
plt.savefig(plot_dir+"outputs.pdf")
plt.clf()

## Ratio Plotting

norm = True
bins = np.linspace(0,1000,40)

plot_hist(ratios[labels.flatten() == 1], bins = bins, norm = norm, colour = c1, label = "Signal")
plot_hist(ratios[labels.flatten() == 0], bins = bins, norm = norm, colour = c2, label = "Background")

plt.draw()

plt.legend()
plt.xlabel("Ratio")
plt.ylabel("Normalised Number of Events" if norm else "Number of Events")

plt.savefig(plot_dir+"ratio.png")
plt.savefig(plot_dir+"ratio.pdf")
plt.clf()

## Ratio Ratio Plotting

norm = True
# bins = np.linspace(0,1000,4)
bins = np.logspace(0,10,40,base = np.e)

ratio_ratio = np.divide(np.histogram(ratios[labels.flatten() == 1], bins = bins)[0],np.histogram(ratios[labels.flatten() == 0], bins = bins)[0])

bins = (bins[1:]+bins[:-1])/2

plt.plot(bins, ratio_ratio, color = c1,	label = "Ratio of Signal/Background", drawstyle = "steps", linewidth = 3)

plt.draw()

plt.legend()
plt.xlabel("Ratio")
plt.ylabel("Normalised Number of Events" if norm else "Number of Events")

plt.savefig("temp.png")
plt.savefig("temp.pdf")
plt.clf()

## Discriminant Plotting

norm = False
bins  = np.linspace(-10,10,40)

plot_hist(discs[labels.flatten() == 1], bins = bins, norm = norm, colour = c1, label = "Signal")
plot_hist(discs[labels.flatten() == 0], bins = bins, norm = norm, colour = c2, label = "Background")

ax = plt.gca()
ax.set_yscale("log")

plt.draw()

plt.legend()
plt.xlabel("Discriminant Value")
plt.ylabel("Normalised Number of Events" if norm else "Number of Events")

plt.savefig(plot_dir+"discs.png")
plt.savefig(plot_dir+"discs.pdf")
plt.clf()

## disc ratio plotting

norm = True
bins  = np.linspace(-10,10,40)

ratio_ratio = np.divide(np.histogram(discs[labels.flatten() == 1], bins = bins)[0],np.histogram(discs[labels.flatten() == 0], bins = bins)[0])

bins = (bins[1:]+bins[:-1])/2

plt.plot(bins, ratio_ratio, color = c1,	label = "Ratio of Signal/Background", drawstyle = "steps", linewidth = 3)

ax = plt.gca()
ax.set_yscale("log")

plt.draw()

plt.legend()
plt.xlabel("Discriminant Value")
plt.ylabel("Normalised Number of Events" if norm else "Number of Events")

plt.savefig(plot_dir+"discs_ratio.png")
plt.savefig(plot_dir+"discs_ratio.pdf")
plt.clf()

## ROC curve

plt.plot(eff_space,rejs, linewidth = 3, color = c1)
plt.draw()
plt.yscale("log")

plt.xlabel("Signal Efficiency")
plt.ylabel("Background Rejection")

plt.savefig(plot_dir+"ROC.png")
plt.savefig(plot_dir+"ROC.pdf")
plt.clf()

print("Done")
