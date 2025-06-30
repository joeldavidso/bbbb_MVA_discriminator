import numpy as np
import random
import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import matplotlib.pyplot as plt

#############################################################
#############################################################
######                                                 ######
######                 Definitions                     ######
######                                                 ######
#############################################################
#############################################################

var_bins = {"dEta_hh": np.linspace(0,2.5,31),
            "eta_h1": np.linspace(-2.5,2.5,31),
            "eta_h2": np.linspace(-2.5,2.5,31),
            "m_h1": np.linspace(100,150,31),
            "m_h2": np.linspace(100,150,31),
            "m_hh": np.linspace(0,1000,31),
            "pt_h1": np.linspace(0,600,31),
            "pt_h2": np.linspace(0,600,31),
            "X_hh": np.linspace(0,1.6,31),
            "X_wt_tag": np.linspace(0,12,31)}

# Creates dataset class
class Data(Dataset):

	def __init__(self, input_vecs, input_weights, input_labels):
		self.labels = input_labels
		self.weights = input_weights
		self.vecs = input_vecs
		
	def __len__(self):
		return len(self.labels)

	def __getitem__(self,index):
		vec = self.vecs[index]
		weight = self.weights[index]
		label = self.labels[index]
		return vec,weight,label

# Define the Network
class Network(nn.Module):

	def __init__(self, input_dim, output_dim, hidden_layers):

		super(Network, self).__init__()

		model_layers = []
		in_dim = input_dim
		for i, out_dim in enumerate(hidden_layers):
			model_layers.append(nn.Linear(in_dim, out_dim))
			model_layers.append(nn.ReLU())
			in_dim = out_dim

		model_layers.append(nn.Linear(in_dim, output_dim))
		model_layers.append(nn.Sigmoid())


		self.operation = nn.Sequential(*model_layers)

	def forward(self, input):
		out = self.operation(input)
		return out


# The Training Loop
def train_loop(dataloader, model, loss_fn, optimizer, scheduler, epoch, batch_size):

	# Set model to training mode and get size of data
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.train()
	loss,tot_loss = 0,0

	# Loop over batches
	for batch, (vec,weight,label) in enumerate(dataloader):


		# Apply model to batch and calculate loss
		pred = model(vec)

		loss_intermediate = loss_fn(pred,label)
		# loss = torch.mean(weight*loss_intermediate)

		sum_loss = torch.sum(weight*loss_intermediate)
		sum_weight = torch.sum(weight)

		loss = torch.divide(sum_loss,sum_weight)

		tot_loss+=loss.item()

		# Backpropagation
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		# Print current progress after each 1% of batches per epoch
		if int(100*(batch-1)/num_batches) != int(100*(batch)/num_batches):
			loss, current = loss.item(), batch*batch_size+len(label)
			print(f"\rloss: "+str(loss)+"   ["+str(current)+"/"+str(size)+"]   ["+str(int(100*current/size))+"%]", end = "", flush = True)
	scheduler.step()


# The validation loop
def val_loop(dataloader, model, loss_fn, epoch):

	# Set model to evaluation mode (same reasoning as train mode)
	model.eval()
	size=len(dataloader.dataset)
	num_batches=len(dataloader)
	val_loss, val_weight, correct = 0,0,0

	# torch.no_grad() allows for evaluation with no gradient calculation (more efficient)
	with torch.no_grad():
		# Loop over all data in test/val dataloader
		for vec,weight,label in dataloader:
			# Calculates accuracy and avg loss for outputting
			pred = model(vec)

			sum_loss = torch.sum(weight*loss_fn(pred,label))
			sum_weight = torch.sum(weight)

			# loss = torch.divide(sum_loss,sum_weight)

			val_loss+=sum_loss.item()
			val_weight+=sum_weight.item()
			correct+=(torch.round(pred) == label).type(torch.float).sum().item()

	# Normalizes loss and accuracy then prints
	val_loss = val_loss/val_weight
	correct /= size
	print("")
	print("Validation:")
	print("Accuracy: "+str(100*correct)+", Avg Loss: "+str(val_loss))

	return val_loss

# Testing/Plotting loop
def test_loop(dataloader, model, loss_fn):

	print("Running Test Loop")

	# Set model to evaluation mode (same reasoning as train mode)
	model.eval()
	test_loss, correct = 0,0

	# Creates array for outputting
	out = []
	labels = []

	# torch.no_grad() allows for evaluation with no gradient calculation (more efficient)
	with torch.no_grad():
			# Loop over all data in test/val dataloader
			for vecs,weight,label in dataloader:
					# Calculates accuracy and avg loss for outputting
					pred=model(vecs)
					test_loss+=loss_fn(pred,label).item()
					correct+=(torch.round(pred) == label).type(torch.float).sum().item()
					# Turn pytorch tensors to numpy arrays for easier use later in plotting
					label_num = label.numpy()
					pred_num = pred.numpy()

					# Loop over entries in the batch
					for count, vec in enumerate(vecs):
						out.append(pred_num[count])
						labels.append(label_num[count])
	
	return np.array(out), np.array(labels)

def plot_var(dataset, labels, test_train, var_name, plot_dir, bins, sig_name, bkg_name):

	c1 = "#DE0C62"
	c2 = "#9213E0"

	if not os.path.exists(plot_dir+"/plots/"):
		os.mkdir(plot_dir+"/plots/")

	if not os.path.exists(plot_dir+"/plots/"+test_train+"/"):
		os.mkdir(plot_dir+"/plots/"+test_train+"/")

	plt.hist(dataset[labels == 1], bins = bins, histtype = "step", density = False, color = c1, label = sig_name, linewidth = 3)
	plt.hist(dataset[labels == 0], bins = bins, histtype = "step", density = False, color = c2, label = bkg_name, linewidth = 3)

	plt.hist(dataset[labels == 1], bins = bins, density = False, color = c1, alpha = 0.1)
	plt.hist(dataset[labels == 0], bins = bins, density = False, color = c2, alpha = 0.1)

	plt.draw()

	plt.legend()
	plt.xlabel(var_name)
	plt.ylabel("Number of Events")

	plt.savefig(plot_dir+"/plots/"+test_train+"/"+var_name+".png")
	plt.savefig(plot_dir+"/plots/"+test_train+"/"+var_name+".pdf")
	plt.clf()
