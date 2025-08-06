import numpy as np
import math
import pandas as pd
import pyarrow.dataset as pyd
import torch
import h5py
import yaml
import os



# Returns the dataset tha is used to store events for h5 saving later on
# the dataset is created from a data dictionary of all input variables
def create_dataset_datatype(variables):

	# print("Variables for Training: ")
	# print(variables)

	dict = {}

	for i in range(len(variables)):
		dict[variables[i]] = np.float32

	dict["label"] = np.float32
	dict["weight"] = np.float32

	# Create datatype for the h5file
	datatype = \
	np.dtype \
	( list \
		( dict.items()
		)
	)

	return datatype


# Reutrns a list of all vriables including variables required for cuts
# This is used when loading the batches from the parquet files
def add_cuts(variables, selections):

	# Cut definitions
	cut_vars = {}

	for cut in selections:
		cut_vars[cut[0]] = 0

	cut_vars = list(cut_vars.keys())

	# add cut vars to variable list
	full_variables = list(np.concatenate((variables,cut_vars)))

	return full_variables


def calc_xhh(mh1s,mh2s):

	left = (mh1s-124)/(0.1*mh1s)
	right = (mh2s-117)/(0.1*mh2s)

	return np.sqrt(left**2+right**2)


# Grabs events of specific class and adds them to the dataset
def prep_class(class_dataset, batch_size, target, variables, selections, smear):

	# Create data array
	datatype = create_dataset_datatype(variables)

	# Create full variable list for loading from parquet files
	
	if selections[0] != "None":
		full_variables = add_cuts(variables, selections)
	else:
		full_variables = variables

	full_variables = list(dict.fromkeys(full_variables))

	out_dataset = []

	# Run over Class datasets to get wanted variables from events
	for batch_number, batch in enumerate(class_dataset.to_batches(columns = full_variables, batch_size = batch_size)):

		# Event Selection via cuts
		pass_arr = np.full(np.array(batch[variables[0]]).shape,True)
		for cut in selections:
			if cut == "None":
				break
			cut_string = "np.array(batch['"+cut[0]+"'])"+cut[1]
			pass_arr = np.logical_and(pass_arr, eval(cut_string))

		temp_dataset = np.recarray(np.array(batch[variables[0]])[pass_arr].shape, dtype = datatype)

		# Remove nan events
		nan_bool_arr =[]
		for i in range(len(variables)):
			# Fills dataset with events
			temp_dataset[variables[i]] = batch[variables[i]].to_numpy(zero_copy_only = False)[pass_arr]

			if i == 0:
				nan_bool_arr = np.isnan(temp_dataset[variables[i]])
			else:
				nan_bool_arr = np.logical_or(nan_bool_arr, np.isnan(temp_dataset[variables[i]]))


		# If Smearing required then do it here and recalculate X_hh from smeard mh1 mh2 dists
		if smear:
			temp_dataset["m_h1"] = temp_dataset["m_h1"] + np.random.normal(0,1,temp_dataset["m_h1"].shape)
			temp_dataset["m_h2"] = temp_dataset["m_h2"] + np.random.normal(0,1,temp_dataset["m_h2"].shape)

			temp_dataset["X_hh"] = calc_xhh(temp_dataset["m_h1"],temp_dataset["m_h2"])

			temp_dataset = temp_dataset[temp_dataset["X_hh"] < 1.6]


		if np.sum(nan_bool_arr) > 0:
			print(np.sum(nan_bool_arr))
			raise("NOOOOOOOOOOOO")

		temp_dataset["label"] = np.zeros_like(temp_dataset[variables[0]])
		temp_dataset["weight"] = np.ones_like(temp_dataset[variables[0]])

		# Add temp dataset to event dataset
		if batch_number == 0:
			out_dataset = temp_dataset
		else:
			out_dataset = np.append(out_dataset, temp_dataset)

		print(f"\r["+str(round(100*len(out_dataset)/target,3))+"%]", end = "", flush = True)

		if len(out_dataset) > target:
			break

	print("                                                  ", end = "\r")

	return out_dataset

def resample(dataset, target):

	N_events = len(dataset)
	
	print("Resampling from "+str(N_events)+" events to "+str(target)+" events")
	print("^^ Events are resampled "+str(round( - 1 + target/N_events,2))+" times !!!")

	dataset = np.random.choice(dataset, target) if N_events < target else dataset[:(target-N_events)]

	return dataset

#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################



with open("Config.yaml") as file:

	config = yaml.load(file, Loader=yaml.FullLoader)

	# Grabs info from config
	classes_dict = config["samples"]["files"]
	variables = config["variables"]
	target = int(config["samples"]["Target_Events"])
	batch_size = 1000

	# Create data dict and datatype for use later
	datatype = create_dataset_datatype(variables)

	# Runs over each class and creates training and testing samples
	for count, sample_class in enumerate(classes_dict):

		sample_class = classes_dict[sample_class]

		print("-------------------------------")
		print("Class: "+sample_class["save_name"])

		# Grab files and turn to dataset
		dataset = pyd.dataset([sample_class["directory"] + file for file in sample_class["files"]])

		# get number of events in dataset
		n_events = dataset.count_rows()
		print("NEVENTS PRE-CUTS: "+str(n_events))

		dataset = prep_class(dataset, batch_size, min(n_events,target), variables, sample_class["selections"], sample_class["smear"])
		np.random.shuffle(dataset)

		N_events = len(dataset)

		print("NEVENTS POST-CUTS: "+str(N_events))

		ten_percent_cutoff = int(0.1*N_events)

		# Create File location for test and train files
		if not os.path.exists("samples/"+sample_class["save_name"]):
			os.mkdir("samples/"+sample_class["save_name"])
 
		test_file = h5py.File("samples/"+sample_class["save_name"]+"/test.h5","w")
		test_file.create_dataset("Data", data = dataset[:ten_percent_cutoff])
		test_file.close()

		# Resample training dataset to have equal number of events
		dataset = resample(dataset[ten_percent_cutoff:], int(0.9*target)) if N_events != target else dataset[ten_percent_cutoff:]

		# h5 file createion
		train_file = h5py.File("samples/"+sample_class["save_name"]+"/train.h5","w")
		train_file.create_dataset("Data", data = dataset)
		train_file.close()

	print("-------------------------------")
	print("Samples Prepared For Training!!")
