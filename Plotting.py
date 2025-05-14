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

def transport_distance_p_1(data_arr_sig,data_arr_bkg, bins):

	hist_sig = np.histogram(data_arr_sig, bins = bins, density = True)[0]
	hist_bkg = np.histogram(data_arr_bkg, bins = bins, density = True)[0]

	print(np.sum(hist_sig)*(bins[1]-bins[0]))
	print(np.sum(hist_bkg))


#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################

var_RW = "eta_h1"

file = open("Config.yaml")
config = yaml.load(file, Loader=yaml.FullLoader)
hyper_params = config["hyper_params"]["training"]
variables = config["inputs"]["variables"]
klambda = config["inputs"]["files"]["klambda"]

plot_dir = "plots/network/"+config["net_name"]+"/"

if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)


# Selects the best trained epoch
ckpt_list = os.listdir("trained_networks/"+config["net_name"])

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
model = torch.load("trained_networks/"+config["net_name"]+"/"+ckpt_list[np.argmin(losses)], weights_only = False)

# Initialize loss function
loss_function = nn.BCELoss()
###################################################################

years_mc_data = [("mc20a",["15","16"]),
				 ("mc20d",["17"]),
				 ("mc20e",["18"])]

test_vecs, test_weights, test_lables = [],[],[]

test_rw_arr = []

for year_count, year_tuple in enumerate(years_mc_data):

	file_loc = "samples/"+klambda+"/"

	data_years_name = ""
	for i in year_tuple[1]:
		data_years_name = data_years_name + i
		if i != year_tuple[1][-1]:
			data_years_name = data_years_name + ","
	year_name = year_tuple[0]+"("+data_years_name+")"

	test_file = h5py.File("samples/"+klambda+"/"+year_name+"_test.h5","r")

	# Converts h5 files to dataset
	test_tensors = []

	for var in variables:
		test_tensors.append(torch.from_numpy(test_file["Data"][var]))

	if year_count == 0:
		test_vecs = torch.stack((test_tensors),-1)
		test_weights = torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1)
		test_labels = torch.from_numpy(test_file["Data"]["label"]).unsqueeze(1)

		test_rw_arr = test_file["Data"][var_RW]

	else:
		test_vecs = torch.cat((test_vecs,torch.stack((test_tensors),-1)))
		test_weights = torch.cat((test_weights,torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1)))
		test_labels = torch.cat((test_labels,torch.from_numpy(test_file["Data"]["label"]).unsqueeze(1)))

		test_rw_arr = np.append(test_rw_arr,test_file["Data"][var_RW])


	# Closes h5 files
	test_file.close()

test_data = Data(test_vecs, test_weights, test_labels)

# #####################################
# # Imports training and testing samples
# test_file = h5py.File("samples/test.h5","r")

# test_tensors = []

# for var in variables:
# 	test_tensors.append(torch.from_numpy(test_file["Data"][var]))

# test_data = Data(torch.stack((test_tensors),-1),
# 				 torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1),
# 				 torch.from_numpy(test_file["Data"]["label"]).unsqueeze(1)
# 				 )

# #####################################################

# test_reweight_dataframe = pd.DataFrame({var_RW : test_file["Data"][var_RW]})

test_reweight_dataframe = pd.DataFrame({var_RW : test_rw_arr})

# # Closes h5 file
# test_file.close()

# Conversts datsets to datlaoaders for training
test_dataloader = DataLoader(test_data, batch_size=hyper_params["batch_size"], shuffle=False)

# Runs over validation/test dataset to generate arrays of the network inputs and outputs
outputs, labels = test_loop(test_dataloader,model,loss_function)

# Discriminant calculations
discs = np.apply_along_axis(discriminant,0,outputs)

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


## Discriminant Plotting

norm = True
bins  = np.linspace(-10,10,40)

plot_hist(discs[labels.flatten() == 1], bins = bins, norm = norm, colour = c1, label = "Signal")
plot_hist(discs[labels.flatten() == 0], bins = bins, norm = norm, colour = c2, label = "Background")

plt.draw()

plt.legend()
plt.xlabel("Discriminant Value")
plt.ylabel("Normalised Number of Events" if norm else "Number of Events")

plt.savefig(plot_dir+"discs.png")
plt.savefig(plot_dir+"discs.pdf")
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


## Reweight


bkg_to_sig_weights = []

for count, label in enumerate(labels.flatten()):
	if label == 0 and (1-outputs[count] > 0):
		bkg_to_sig_weights.append(((outputs[count])/(1-outputs[count]))[0])
	elif label == 1:
		bkg_to_sig_weights.append(1) 

bkg_to_sig_weights = np.array(bkg_to_sig_weights)

bkg_to_sig_weights = pd.DataFrame({"weights": bkg_to_sig_weights})

bins = np.linspace(-2.5,2.5,40)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (10,5))

plot_hist(test_reweight_dataframe[var_RW][labels.flatten() == 1],
          bins = bins, norm = True, colour = c1, label = "Signal", ax = ax1)
plot_hist(test_reweight_dataframe[var_RW][labels.flatten() == 0],
          bins = bins, norm = True, colour = c2, label = "Background", ax = ax1)


plot_hist(test_reweight_dataframe[var_RW][labels.flatten() == 1],
          bins = bins, norm = True, colour = c1, label = "Signal", ax = ax2)
plot_hist(test_reweight_dataframe[var_RW][labels.flatten() == 0], weights = bkg_to_sig_weights[labels.flatten() == 0],
          bins = bins, norm = True, colour = c2, label = "Background_RW", ax = ax2)

print(transport_distance_p_1((test_reweight_dataframe[var_RW][labels.flatten() == 1]),
      test_reweight_dataframe[var_RW][labels.flatten() == 0],
      bins))

plt.draw()
ax1.set_xlabel(var_RW)
ax2.set_xlabel(var_RW)
ax1.set_ylabel("Normalized Number of Events")
ax1.legend()
ax2.legend()
plt.savefig("test.png")
plt.savefig("test.pdf")
plt.clf()

print("Done")
