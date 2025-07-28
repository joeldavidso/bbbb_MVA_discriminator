import numpy as np
import h5py
import os
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from Utils import Data, Network, train_loop, val_loop, plot_var, var_bins, plot_losses
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl

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
structure = config["training"]["structure"]["hidden_layers"]
learning = config["training"]["learning"]
signal = config["training"]["signal"]
background = config["training"]["background"]

net_name = config["training"]["add_name"]+"_"+config["training"]["signal_name"]+"_"+config["training"]["background_name"]

for i in range(len(structure)):
	net_name += "_"+str(structure[i])

net_name += "_"+str(learning["learning_rate"])+"_"+str(learning["epochs"])

# Opens network directory for saving
if not os.path.exists("trained_networks/"+net_name):
	os.mkdir("trained_networks/"+net_name)

print("Net Name: "+ net_name)

# Imports training and testing samples

# Grab Signal test and train files
train_file = h5py.File(signal+"train.h5")
test_file = h5py.File(signal+"test.h5")

# Converts Files to dataset

train_tensors, test_tensors = [],[]

for var in variables:
	train_tensors.append(torch.from_numpy(train_file["Data"][var]))
	test_tensors.append(torch.from_numpy(test_file["Data"][var]))

train_vecs = torch.stack((train_tensors),-1)
train_weights = torch.from_numpy(train_file["Data"]["weight"]).unsqueeze(1)
train_labels = torch.from_numpy(np.ones_like(train_file["Data"]["label"])).unsqueeze(1)

test_vecs = torch.stack((test_tensors),-1)
test_weights = torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1)
test_labels = torch.from_numpy(np.ones_like(test_file["Data"]["label"])).unsqueeze(1)

# Closes h5 files
test_file.close()
train_file.close()


# Do it all again with background
# Grab Signal test and train files
train_file = h5py.File(background+"train.h5")
test_file = h5py.File(background+"test.h5")

# Converts Files to dataset

train_tensors, test_tensors = [],[]

for var in variables:
	train_tensors.append(torch.from_numpy(train_file["Data"][var]))
	test_tensors.append(torch.from_numpy(test_file["Data"][var]))

train_vecs = torch.cat((train_vecs,torch.stack((train_tensors),-1)))
train_weights = torch.cat((train_weights,torch.from_numpy(train_file["Data"]["weight"]).unsqueeze(1)))
train_labels = torch.cat((train_labels,torch.from_numpy(np.zeros_like(train_file["Data"]["label"])).unsqueeze(1)))

test_vecs = torch.cat((test_vecs,torch.stack((test_tensors),-1)))
test_weights = torch.cat((test_weights,torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1)))
test_labels = torch.cat((test_labels,torch.from_numpy(np.zeros_like(test_file["Data"]["label"])).unsqueeze(1)))

# Closes h5 files
test_file.close()
train_file.close()

train_data = Data(train_vecs, train_weights, train_labels)
test_data = Data(test_vecs, test_weights, test_labels)

print("No. Train: "+str(len(train_data)))

# Plotting inputs

print(train_data.vecs[:,2])

for count, var in enumerate(variables):

	bins = var_bins[var]

	plot_var(train_data.vecs[:,count], train_data.labels.flatten(), "train", var, "trained_networks/"+net_name, bins, config["training"]["signal_name"], config["training"]["background_name"])
	plot_var(test_data.vecs[:,count], test_data.labels.flatten(), "test", var, "trained_networks/"+net_name, bins, config["training"]["signal_name"], config["training"]["background_name"])


# Converts datsets to datlaoaders for training
train_dataloader = DataLoader(train_data, batch_size=learning["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=learning["batch_size"], shuffle=True)

# Outputs the structure of the data
train_vecs, train_weights, train_labels = next(iter(train_dataloader))
print(f"Vector batch shape: {train_vecs.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Get training device
device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Training device set to "+device)

# Create instance of Network and move to device
model = Network(input_dim = len(variables),
				output_dim = 1,
				hidden_layers = structure
				).to(device)
print("Model created with structure:")
print(model)

# Initialize loss function with class weights
loss_function = nn.BCELoss(reduction="none")

# Initialize the optimizer with Stockastic Gradient Descent function
if learning["optimizer"] == "SGD":
	optimizer = torch.optim.SGD(model.parameters(), lr=float(learning["learning_rate"]), momentum = 0.8)
elif learning["optimizer"] == "Adam":
	optimizer = torch.optim.Adam(model.parameters(), lr=float(learning["learning_rate"]))
else:
	raise("Please Select a Valid Optimizer")


# Initialize the learning rate schduler
if learning["scheduler"] == "Exponential":
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
elif learning["scheduler"] == "Cosine":
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = learning["epochs"], eta_min = 0)

train_losses = []
val_losses = []

# Loop over epochs to train and validate
for e in range(learning["epochs"]):

	print("__________________________________")
	print("Epoch "+str(e+1)+" :")
	print("----------------------------------")
	print("Training:")
	train_losses.append(train_loop(train_dataloader, model, loss_function, optimizer,scheduler,e+1, int(learning["batch_size"])))
	val_losses.append(val_loop(test_dataloader, model, loss_function, e+1))

	plot_losses(train_losses,val_losses,e+1,"trained_networks/"+net_name+"/plots/")

	# Saves network eopochs
	torch.save(model, "trained_networks/"+net_name+"/CKPT_"+str(e)+"_VAL_LOSS_"+str(val_losses[e])+".pth")


print("Training Done!")
