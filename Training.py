import numpy as np
import h5py
import os
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from Utils import Data, Network, train_loop, val_loop, plot_var, var_bins, plot_losses, plot_lrs, normlayer
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

#############################################################
#############################################################
######                                                 ######
######                  Arg Parsing                    ######
######                                                 ######
#############################################################
#############################################################

# import config
file = open("Config.yaml")
config = yaml.load(file, Loader=yaml.FullLoader)

## values from config that do not want arguments parsed
variables = config["variables"]

## training file location dictionary
file_dict = {"Sig": config["training"]["Signal_file"],
             "Data": config["training"]["Data_file"],
             "Bkg": config["training"]["Background_file"]}

############################################################################################
## use config values unless args parsed

parser = argparse.ArgumentParser()

# parser.add_argument("square", help = "Hello")

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))


arg_default = config["training"]["structure"]["hidden_layers"]
parser.add_argument("--structure",
                    type = list_of_ints,
                    help = "list of hidden layer sizes, default: "+str(arg_default),
					default = arg_default)

arg_default = config["training"]["learning"]["learning_rate"]
parser.add_argument("-lr", "--learning_rate",
                    type = float,
   				    help = "the initial learning rate of the training, default: "+str(arg_default),
					default = arg_default)

arg_default = "Sig"
parser.add_argument("--signal",
                    type = str,
   				    help = "Sets signal file ('Sig', & 'Data' only compatable), default: "+str(arg_default),
					default = arg_default)

arg_default = "Data"
parser.add_argument("--background",
                    type = str,
   				    help = "Sets background file ('Data', & 'Bkg' only compatable), default: "+str(arg_default),
					default = arg_default)

arg_default = ""
parser.add_argument("--addname",
                    type = str,
   				    help = "additional name at front of file for easy loaction e.g. the date",
					default = arg_default)

arg_default = config["training"]["learning"]["epochs"]
parser.add_argument("--epochs",
                    type = int,
   				    help = "n_epochs in training, default: "+str(arg_default),
					default = arg_default)

arg_default = config["training"]["learning"]["batch_size"]
parser.add_argument("--batch_size",
                    type = int,
   				    help = "batch size, default: "+str(arg_default),
					default = arg_default)

# arg_default =
# parser.add_argument("",
#                     type = ,
#    				    help = "",
# 					default = )

args = parser.parse_args()

# raise("HI")

############################################################################################

net_name = args.addname+"_"+args.signal+"_"+args.background

for i in range(len(args.structure)):
	net_name += "_"+str(args.structure[i])

net_name += "_"+str(args.learning_rate)+"_"+str(args.epochs)

#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################

# Opens network directory for saving
if not os.path.exists("trained_networks/"+net_name):
	os.mkdir("trained_networks/"+net_name)
if not os.path.exists("trained_networks/"+net_name+"/ckpts"):
	os.mkdir("trained_networks/"+net_name+"/ckpts")

print("Net Name: "+ net_name)

# Imports training and valing samples

# Grab Signal val and train files
train_file = h5py.File(file_dict[args.signal]+"train.h5")
val_file = h5py.File(file_dict[args.signal]+"val.h5")

# Converts Files to dataset

train_tensors, val_tensors = [],[]

for var in variables:
	train_tensors.append(torch.from_numpy(train_file["Data"][var]))
	val_tensors.append(torch.from_numpy(val_file["Data"][var]))

train_vecs = torch.stack((train_tensors),-1)
train_weights = torch.from_numpy(train_file["Data"]["weight"]).unsqueeze(1)
train_labels = torch.from_numpy(np.ones_like(train_file["Data"]["label"])).unsqueeze(1)

val_vecs = torch.stack((val_tensors),-1)
val_weights = torch.from_numpy(val_file["Data"]["weight"]).unsqueeze(1)
val_labels = torch.from_numpy(np.ones_like(val_file["Data"]["label"])).unsqueeze(1)

# Closes h5 files
val_file.close()
train_file.close()


# Do it all again with background
# Grab Signal val and train files
train_file = h5py.File(file_dict[args.background]+"train.h5")
val_file = h5py.File(file_dict[args.background]+"val.h5")

# Converts Files to dataset

train_tensors, val_tensors = [],[]

for var in variables:
	train_tensors.append(torch.from_numpy(train_file["Data"][var]))
	val_tensors.append(torch.from_numpy(val_file["Data"][var]))

train_vecs = torch.cat((train_vecs,torch.stack((train_tensors),-1)))
train_weights = torch.cat((train_weights,torch.from_numpy(train_file["Data"]["weight"]).unsqueeze(1)))
train_labels = torch.cat((train_labels,torch.from_numpy(np.zeros_like(train_file["Data"]["label"])).unsqueeze(1)))

val_vecs = torch.cat((val_vecs,torch.stack((val_tensors),-1)))
val_weights = torch.cat((val_weights,torch.from_numpy(val_file["Data"]["weight"]).unsqueeze(1)))
val_labels = torch.cat((val_labels,torch.from_numpy(np.zeros_like(val_file["Data"]["label"])).unsqueeze(1)))

# Closes h5 files
val_file.close()
train_file.close()

train_data = Data(train_vecs, train_weights, train_labels)
val_data = Data(val_vecs, val_weights, val_labels)

print("No. Train: "+str(len(train_data)))

# Plotting inputs

print(train_data.vecs[:,2])

for count, var in enumerate(variables):

	bins = var_bins[var]

	plot_var(np.array(train_data.vecs[:,count]), np.array(train_data.labels.flatten()), "train", var, "trained_networks/"+net_name, bins, args.signal, args.background)
	plot_var(np.array(val_data.vecs[:,count]), np.array(val_data.labels.flatten()), "val", var, "trained_networks/"+net_name, bins, args.signal, args.background)


# Converts datsets to datlaoaders for training
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

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
				hidden_layers = args.structure,
				init_layer = normlayer(train_data.vecs)
				).to(device)
print("Model created with structure:")
print(model)

# Initialize loss function with class weights
loss_function = nn.BCELoss(reduction="none")

# Initialize the optimizer with Stockastic Gradient Descent function
optim_name = config["training"]["learning"]["optimizer"]
if optim_name == "SGD":
	optimizer = torch.optim.SGD(model.parameters(), lr=float(args.learning_rate), momentum = 0.8)
elif optim_name == "Adam":
	optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
else:
	raise("Please Select a Valid Optimizer")


# Initialize the learning rate schduler
sched_name = config["training"]["learning"]["scheduler"]
if sched_name == "Exponential":
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
elif sched_name == "Cosine":
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = args.epochs, eta_min = 0)

train_losses = []
val_losses = []
lrs = []

# Loop over epochs to train and validate
for e in range(args.epochs):

	lrs.append(scheduler.get_last_lr()[-1])
	print("__________________________________")
	print("Epoch "+str(e+1)+" :")
	print("----------------------------------")
	print("Training:")
	train_losses.append(train_loop(train_dataloader, model, loss_function, optimizer,scheduler,e+1, int(args.batch_size)))
	val_losses.append(val_loop(val_dataloader, model, loss_function, e+1))

	plot_losses(train_losses,val_losses,e+1,"trained_networks/"+net_name+"/plots/")
	plot_lrs(lrs,e+1,"trained_networks/"+net_name+"/plots/")

	# Saves network eopochs
	torch.save(model, "trained_networks/"+net_name+"/ckpts/CKPT_"+str(e)+"_VAL_LOSS_"+str(val_losses[e])+".pth")


print("Training Done!")
