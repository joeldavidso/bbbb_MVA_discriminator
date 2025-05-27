import numpy as np
import h5py
import os
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from Utils import Data, Network, train_loop, val_loop
import yaml

#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################

kappa_lambdas = ["m1p0",
				 "0p0",
				 "1p0",
				 "2p5",
				 "5p0",
				 "10p0"]

# import config

file = open("Config.yaml")
config = yaml.load(file, Loader=yaml.FullLoader)

hyper_params = config["hyper_params"]["training"]
net_name = config["net_name"]
variables = config["inputs"]["variables"]

# Grabs the signal and background file locations
signal_dir = config["inputs"]["files"]["signal_dir"]
signal_files = config["inputs"]["files"]["signal_files"]

background_dir = config["inputs"]["files"]["background_dir"]
background_files = config["inputs"]["files"]["background_files"]

# Grabs the kappa lambda used
contained_lambdas = [klambda in signal_file for signal_file in signal_files for klambda in kappa_lambdas]

if np.sum(contained_lambdas) != len(signal_files):
	raise Exception("Unclear kappa lambdas in signal files!")

kappa_lambda = kappa_lambdas[np.nonzero([klambda in signal_files[0] for klambda in kappa_lambdas])[0][0]]


# Imports training and testing samples
# And loops of all years to creat one dataset

years_mc_data = [("mc20a",["15","16"]),
				 ("mc20d",["17"]),
				 ("mc20e",["18"])]

train_vecs, train_lables = [],[]
test_vecs, test_lables = [],[]

for year_count, year_tuple in enumerate(years_mc_data):

	file_loc = "samples/"+kappa_lambda+"/"

	data_years_name = ""
	for i in year_tuple[1]:
		data_years_name = data_years_name + i
		if i != year_tuple[1][-1]:
			data_years_name = data_years_name + ","
	year_name = year_tuple[0]+"("+data_years_name+")"

	train_file = h5py.File("samples/"+kappa_lambda+"/"+year_name+"_train.h5","r")
	test_file = h5py.File("samples/"+kappa_lambda+"/"+year_name+"_test.h5","r")

# Converts h5 files to datasets

	train_tensors = []

	for var in variables:
		train_tensors.append(torch.from_numpy(train_file["Data"][var]))

	if year_count == 0:
		train_vecs = torch.stack((train_tensors),-1)
		train_weights = torch.from_numpy(train_file["Data"]["weight"]).unsqueeze(1)
		train_labels = torch.from_numpy(train_file["Data"]["label"]).unsqueeze(1)

	else:
		train_vecs = torch.cat((train_vecs,torch.stack((train_tensors),-1)))
		train_weights = torch.cat((train_weights,torch.from_numpy(train_file["Data"]["weight"]).unsqueeze(1)))
		train_labels = torch.cat((train_labels,torch.from_numpy(train_file["Data"]["label"]).unsqueeze(1)))

	test_tensors = []

	for var in variables:
		test_tensors.append(torch.from_numpy(test_file["Data"][var]))

	if year_count == 0:
		test_vecs = torch.stack((test_tensors),-1)
		test_weights = torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1)
		test_labels = torch.from_numpy(test_file["Data"]["label"]).unsqueeze(1)

	else:
		test_vecs = torch.cat((test_vecs,torch.stack((test_tensors),-1)))
		test_weights = torch.cat((test_weights,torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1)))
		test_labels = torch.cat((test_labels,torch.from_numpy(test_file["Data"]["label"]).unsqueeze(1)))


	# Closes h5 files
	test_file.close()
	train_file.close()

	train_data = Data(train_vecs, train_weights, train_labels)
	test_data = Data(test_vecs, test_weights, test_labels)

# Converts datsets to datlaoaders for training
train_dataloader = DataLoader(train_data, batch_size=hyper_params["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=hyper_params["batch_size"], shuffle=True)

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
				hidden_layers = config["hyper_params"]["network"]["hidden_layer_structure"]
				).to(device)
print("Model created with structure:")
print(model)

# Initialize loss function with class weights
loss_function = nn.BCELoss(reduction="none")

# Initialize the optimizer with Stockastic Gradient Descent function
optimizer = torch.optim.SGD(model.parameters(), lr=float(hyper_params["learning_rate"]))

# Initialize the learning rate schduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

# Opens network directory for saving
if not os.path.exists("trained_networks/"+config["net_name"]):
	os.mkdir("trained_networks/"+config["net_name"])

# Loop over epochs to train and validate
for e in range(hyper_params["epochs"]):

	print("__________________________________")
	print("Epoch "+str(e+1)+" :")
	print("----------------------------------")
	print("Training:")
	train_loop(train_dataloader, model, loss_function, optimizer,scheduler,e+1, int(hyper_params["batch_size"]))
	val_loss = val_loop(test_dataloader, model, loss_function, e+1)

	# Saves network eopochs
	torch.save(model, "trained_networks/"+config["net_name"]+"/CKPT_"+str(e)+"_VAL_LOSS_"+str(val_loss)+".pth")


print("Training Done!")
