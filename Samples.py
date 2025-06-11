import numpy as np
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


# Grabs events of specific class and adds them to the dataset
def prep_class(class_datasets, batch_size, n_batches, variables, selections):

	# Create data array
	datatype = create_dataset_datatype(variables)

	# Create full variable list for loading from parquet files
	
	if selections[0] != "None":
		full_variables = add_cuts(variables, selections)
	else:
		full_variables = variables

	classes_dataset = []

	# Run over Class datasets to get wanted variables from events
	for dataset_number, class_dataset in enumerate(class_datasets):

		file_dataset = []

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

			# temp_dataset = temp_dataset[np.invert(nan_bool_arr)]

			if np.sum(nan_bool_arr) > 0:
				print(np.sum(nan_bool_arr))
				raise("NOOOOOOOOOOOO")

			temp_dataset["label"] = np.zeros_like(temp_dataset[variables[0]])
			temp_dataset["weight"] = np.ones_like(temp_dataset[variables[0]])

			# Add temp dataset to event dataset
			if batch_number == 0:
				file_dataset = temp_dataset
			else:
				file_dataset = np.append(file_dataset, temp_dataset)

			print(f"\r["+str(100*batch_number/n_batches)+"%]", end = "", flush = True)

			if batch_number == n_batches:
				break

		print("                                                  ")
		print(temp_dataset.shape)
		
		# Add file dataset to class dataset
		if dataset_number == 0:
			classes_dataset = file_dataset
		else:
			classes_dataset = np.append(classes_dataset, file_dataset)


	return classes_dataset



#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################



with open("Config_V2.yaml") as file:

	config = yaml.load(file, Loader=yaml.FullLoader)

	# Grabs info from config
	classes_dict = config["samples"]["files"]
	variables = config["variables"]
	n_batches = config["samples"]["N_batches"]
	batch_size = config["samples"]["batch_size"]

	# Create data dict and datatype for use later
	datatype = create_dataset_datatype(variables)

	# print(classes_dict)

	# Runs over each class and creates training and testing samples
	for count, sample_class in enumerate(classes_dict):

		sample_class = classes_dict[sample_class]

		print("-------------------------------")
		print("Class: "+sample_class["save_name"])

		# Grab datasets
		datasets = [pyd.dataset(sample_class["directory"]+file) for file in sample_class["files"]]

		# Get Number of Events
		N_events = sum(datasets[i].count_rows() for i in range(len(datasets)))
		print("NEVENTS: "+str(N_events))

		N_target = n_batches*batch_size

		if N_target > N_events:
			print("Resampling ratio of: "+str(np.round(N_target/N_events,2)))

		dataset = prep_class(datasets, batch_size, n_batches, variables, sample_class["selections"])

		# Resample datasets to have equal number of events
		N_events = len(dataset)

		if N_events < N_target*len(datasets):
			dataset = np.random.choice(dataset, N_target*len(datasets))

		np.random.shuffle(dataset)

		ten_percent_cutoff = int(0.1*(len(dataset)))

		# Create File location for test and train files
		if not os.path.exists("samples/"+sample_class["save_name"]):
			os.mkdir("samples/"+sample_class["save_name"])

		# h5 file createion
		train_file = h5py.File("samples/"+sample_class["save_name"]+"/train.h5","w")
		train_file.create_dataset("Data", data = dataset[ten_percent_cutoff:])
		train_file.close()

		test_file = h5py.File("samples/"+sample_class["save_name"]+"/test.h5","w")
		test_file.create_dataset("Data", data = dataset[:ten_percent_cutoff])
		test_file.close()

	print("Samples Prepared For Training!!")
