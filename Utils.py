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
import skink as skplt

#############################################################
#############################################################
######                                                 ######
######                 Definitions                     ######
######                                                 ######
#############################################################
#############################################################

var_bins = {"dEta_hh": skplt.get_bins(0,1.5,30),
            "eta_h1": skplt.get_bins(-2.5,2.5,30),
            "eta_h2": skplt.get_bins(-2.5,2.5,30),
            "m_h1": skplt.get_bins(100,150,30),
            "m_h2": skplt.get_bins(100,150,30),
            "m_hh": skplt.get_bins(0,1000,30),
            "pt_h1": skplt.get_bins(0,600,30),
            "pt_h2": skplt.get_bins(0,600,30),
            "X_hh": skplt.get_bins(0,1.6,30),
            "X_wt_tag": skplt.get_bins(0,10,30)}

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


# creates a linear layer that normalizes the inputs
def normlayer(xs, n_input):

  bias = torch.mean(xs, 0)
  cov = torch.cov(xs.t())

  l = nn.Linear(n_input , n_input)

  eigvals , eigvec = torch.linalg.eigh(cov)

  trans = \
    torch.matmul \
    ( eigvec
    , torch.matmul \
      ( torch.diag(1.0 / torch.sqrt(eigvals))
      , eigvec.t()
      )
    )

  assert int(np.isnan(bias).sum()) == 0
  assert int(np.isnan(trans).sum()) == 0

  l.weight = nn.Parameter(trans)
  l.bias = nn.Parameter(torch.matmul(trans, -bias))

  l.bias.requires_grad = False
  l.weight.requires_grad = False

  return l


# Define the Network
class Network(nn.Module):

	def __init__(self, input_dim, output_dim, hidden_layers, init_layer = False):

		super(Network, self).__init__()

		model_layers = [init_layer] if init_layer is not None else []
		in_dim = input_dim
		for i, out_dim in enumerate(hidden_layers):
			model_layers.append(nn.Linear(in_dim, out_dim))
			model_layers.append(nn.ReLU())
			in_dim = out_dim

		model_layers.append(nn.Linear(in_dim, output_dim))
		model_layers.append(nn.Sigmoid())

		self.operation = nn.Sequential(*model_layers)

	# Defines the forward pass
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

	return tot_loss/(batch+1)

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

	if not os.path.exists(plot_dir+"/plots/"):
		os.mkdir(plot_dir+"/plots/")

	if not os.path.exists(plot_dir+"/plots/"+test_train+"/"):
		os.mkdir(plot_dir+"/plots/"+test_train+"/")

	if sig_name == "bb_bkg":
		sig_name = "2b2j_data"
	elif bkg_name == "bb_bkg":
		bkg_name = "2b2j_data"

	HistogramPlot = skplt.HistogramPlot(bins = bins, xlabel = var_name, ylabel = "Number of Events", ratio = True, plot_unc = True)

	HistogramPlot.Add(data = dataset[labels == 1], label = sig_name, fill = "A0", reference = True)
	HistogramPlot.Add(data = dataset[labels == 0], label = bkg_name, fill = "A0")

	HistogramPlot.Plot(plot_dir+"/plots/"+test_train+"/"+var_name)

def plot_losses(train_losses, val_losses, epochs, plot_dir):

	LinePlot = skplt.LinePlot(xs = [e for e in range(epochs)], xlabel = "Epoch", ylabel = "Loss", ratio = False, plot_unc = False)

	LinePlot.Add(ys = [float(loss) for loss in train_losses], label = "Avg. Train Loss", marker_size = 0, linewidth = 1)
	LinePlot.Add(ys = [float(loss) for loss in val_losses], label = "Avg. Val Loss", marker_size = 0, linewidth = 1)

	LinePlot.Plot(plot_dir+"loss")

def plot_lrs(lrs, epochs, plot_dir):


	LinePlot = skplt.LinePlot(xs = [e for e in range(epochs)], xlabel = "Epoch", ylabel = "Learning Rate", ratio = False, plot_unc = False)

	LinePlot.Add(ys = lrs, label = "Learning Rate", marker_size = 0, linewidth = 1)

	LinePlot.Plot(plot_dir+"lr")
