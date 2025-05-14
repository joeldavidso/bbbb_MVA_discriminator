import numpy as np
import pandas as pd
import pyarrow.dataset as pyd
import torch
import h5py
import yaml
import os

tot_batch = 1000


# Returns the dataset tha is used to store events for h5 saving later on
# the dataset is created from a data dictionary of all input variables
def create_dataset(variables):

	print("Variables for Training: ")
	print(variables)

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

	# Create data dictionary to save as h5 files
	dataset = np.recarray(np.array([]), dtype = datatype)

	return dataset



# Reutrns a list of all vriables including variables required for cuts
# This is used when loading the batches from the parquet files
def add_cuts(variables, selections):

	# Cut definitions
	cut_vars = {}

	for cut in selections["signal"]:
		cut_vars[cut[0]] = 0
	for cut in selections["background"]:
		cut_vars[cut[0]] = 0

	# add cut vars to variable list
	full_variables = list(np.concatenate((variables,cut_vars)))

	return full_variables



# Grabs events of specific class and adds them to the dataset
def prep_class(class_files, N_class, N_target, dataset, variables):

	return True


#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################



# Function for creating samples
def prep_samples(signal_file, background_files, variables, selections, klambda, year_name):

	# Create dictionary for datatype

	print("Variables: ")
	print(variables)

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

	# Cut definitions
	cut_vars = {}

	for cut in selections["signal"]:
		cut_vars[cut[0]] = 0
	for cut in selections["background"]:
		cut_vars[cut[0]] = 0


	# Grab Parquet Files
	dataset_signal = pyd.dataset(signal_file)
	dataset_backgrounds = [pyd.dataset(background_files[i]) for i in range(len(background_files))]

	n_sig = dataset_signal.count_rows()
	n_bkg = sum(dataset_backgrounds[i].count_rows() for i in range(len(background_files)))

	print("NSIG: "+ str(n_sig))
	print("NBKG: "+str(n_bkg))

	# Create data dictionary to save as h5 files
	data_dict = np.recarray(np.array([]), dtype = datatype)

	cut_vars = list(cut_vars.keys())

	full_variables = list(np.concatenate((variables,cut_vars)))

	# Run over signal file to get wanted variable data
	for batch_n, batch in enumerate(dataset_signal.to_batches(columns = full_variables, batch_size = 1_000)):

		# Event selection
		pass_arr = np.full(np.array(batch[variables[0]]).shape,True)
		for cut in selections["signal"]:
			cut_string = "np.array(batch['"+cut[0]+"'])"+cut[1]
			pass_arr = np.logical_and(pass_arr, eval(cut_string))

		temp_data_dict = np.recarray(np.array(batch[variables[0]])[pass_arr].shape, dtype = datatype)

		# Save succesful events
		for i in range(len(variables)):
			temp_data_dict[variables[i]] = np.nan_to_num(batch[variables[i]].to_numpy(zero_copy_only = False)[pass_arr])

		temp_data_dict["label"] = np.ones_like(np.nan_to_num(batch[variables[i]]))[pass_arr]
		temp_data_dict["weight"] = np.ones_like(np.nan_to_num(batch[variables[i]]))[pass_arr]

		if batch_n != 0:
			data_dict = np.append(data_dict, temp_data_dict)
		else:
			data_dict = temp_data_dict
		
		print(f"\rSignal Events: ["+str(100*1_000*batch_n/n_sig)+"%]", end = "", flush = True)
		print(f"\r                                                                              ", end = "", flush = True)


	# Run over background files to get wanted variable data

	for bkg_n, background_dset in enumerate(dataset_backgrounds):
		for batch_n, batch in enumerate(background_dset.to_batches(columns = full_variables, batch_size = 1_000)):

			# Event selection
			pass_arr = np.full(np.array(batch[variables[0]]).shape,True)
			for cut in selections["background"]:
				cut_string = "np.array(batch['"+cut[0]+"'])"+cut[1]
				pass_arr = np.logical_and(pass_arr, eval(cut_string))

			temp_data_dict = np.recarray(np.array(batch[variables[0]])[pass_arr].shape, dtype = datatype)

			# Save Succesfull events
			for i in range(len(variables)):
				temp_data_dict[variables[i]] = np.nan_to_num(batch[variables[i]].to_numpy(zero_copy_only = False)[pass_arr])

			temp_data_dict["label"] = np.zeros_like(np.nan_to_num(batch[variables[i]]))[pass_arr]
			temp_data_dict["weight"] = np.zeros_like(np.nan_to_num(batch[variables[i]]))[pass_arr]

			data_dict = np.append(data_dict, temp_data_dict)

			print(f"\r["+str(100*batch_n/tot_batch)+"%]", end = "", flush = True)

			if batch_n == tot_batch:
				break

	# Shuffle and split for training and test dataset 90:10 split
	np.random.shuffle(data_dict)

	ten_percent_cutoff = int(0.1*(len(data_dict)))

	# weight calculations

	n_sig_train = np.sum(data_dict["label"][ten_percent_cutoff:])
	n_bkg_train = len(data_dict["label"][ten_percent_cutoff:]) - n_sig_train

	sig_weight = n_bkg_train/n_sig_train

	print("Weight:  "+str(sig_weight))

	data_dict["weight"] = np.select([data_dict["label"] == 1, data_dict["label"] == 0], [sig_weight, 1])

	# Create File location for klambda and then year within 
	if not os.path.exists("samples/"+klambda):
		os.mkdir("samples/"+klambda)

	# h5 file createion
	train_file = h5py.File("samples/"+klambda+"/"+year_name+"_train.h5","w")
	train_file.create_dataset("Data", data = data_dict[ten_percent_cutoff:])
	train_file.close()

	test_file = h5py.File("samples/"+klambda+"/"+year_name+"_test.h5","w")
	test_file.create_dataset("Data", data = data_dict[:ten_percent_cutoff])
	test_file.close()

	print("Samples Prepared For Training!!")


#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################

years_mc_data = [("mc20a",["15","16"]),
				 ("mc20d",["17"]),
				 ("mc20e",["18"])]


with open("Config.yaml") as file:
	config = yaml.load(file, Loader=yaml.FullLoader)

	n_events = []
	year_names = []

	for year_count, year_tuple in enumerate(years_mc_data):

		# Signal File Multiple lines for clarity
		signal_file = config["inputs"]["files"]["signal_pre_klambda"]+config["inputs"]["files"]["klambda"]
		signal_file = signal_file+config["inputs"]["files"]["signal_mid"]
		signal_file = signal_file+year_tuple[0]+config["inputs"]["files"]["signal_post_year"]

		background_files = []
		for year in year_tuple[1]:
			background_files.append(config["inputs"]["files"]["background_pre_year"]+year+config["inputs"]["files"]["background_post_year"])
	
		data_years_name = ""
		for i in year_tuple[1]:
			data_years_name = data_years_name + i
			if i != year_tuple[1][-1]:
				data_years_name = data_years_name + ","
		year_name = year_tuple[0]+"("+data_years_name+")"
		
		year_names.append(year_name)

		prep_samples(config["inputs"]["files"]["signal_dir"]+signal_file, 
					[config["inputs"]["files"]["background_dir"]+background_files[i] for i in range(len(background_files))],
					config["inputs"]["variables"],
					config["inputs"]["selections"],
					config["inputs"]["files"]["klambda"],
					year_name)
