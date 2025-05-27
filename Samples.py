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

# def resample(dataset, N_target):

# 	N_dataset = len(dataset)


# 	R_floored = int(np.floor(N_target/N_dataset))
# 	N_add_one = N_target - R_floored * N_dataset


# 	duplicate_dataset = np.tile(dataset,R_floored)
# 	duplicate_dataset = np.append(duplicate_dataset, dataset[0:N_add_one])


# 	return duplicate_dataset

# Grabs events of specific class and adds them to the dataset
def prep_class(class_datasets, class_name, batch_size, n_batches, variables, selections):

	# Create data array
	datatype = create_dataset_datatype(variables)

	# Create full variable list for loading from parquet files
	full_variables = add_cuts(variables, selections)

	classes_dataset = []

	# Run over Class datasets to get wanted variables from events
	for dataset_number, class_dataset in enumerate(class_datasets):

		file_dataset = []

		for batch_number, batch in enumerate(class_dataset.to_batches(columns = full_variables, batch_size = batch_size)):


			# Event Selection via cuts
			pass_arr = np.full(np.array(batch[variables[0]]).shape,True)
			for cut in selections:
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

			if class_name == "signal":
				temp_dataset["label"] = np.ones_like(temp_dataset[variables[0]])
			else:
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

years_mc_data = [("mc20a",["15","16"]),
				 ("mc20d",["17"]),
				 ("mc20e",["18"])]

kappa_lambdas = ["m1p0",
				 "0p0",
				 "1p0",
				 "2p5",
				 "5p0",
				 "10p0"]

with open("Config.yaml") as file:
	config = yaml.load(file, Loader=yaml.FullLoader)

	# Grabs info from config
	variables = config["inputs"]["variables"]
	n_batches = config["inputs"]["files"]["N_batches"]
	batch_size = config["inputs"]["files"]["batch_size"]

	selections = config["inputs"]["selections"]

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

	# Runs over years and mc campaigns and processes samples
	for year_count, year_tuple in enumerate(years_mc_data):

		print("----------------------")
		data_years_name = ""
		for i in year_tuple[1]:
			data_years_name = data_years_name + i
			if i != year_tuple[1][-1]:
				data_years_name = data_years_name + ","
		year_name = year_tuple[0]+"("+data_years_name+")"

		print(year_name)

		# Get index for signal and background files used in this iteration

		sig_file_use = [year_tuple[0] in signal_files[i] for i in range(len(signal_files))]
		bkg_file_use = [year_tuple[1][j] in background_files[i] for j in range(len(year_tuple[1])) for i in range(len(background_files))]

		if len(year_tuple[1]) > 1:
			bkg_file_use = [bkg_file_use[i] or bkg_file_use[i+4] for i in range(len(background_files))]
		

		# Create data dict and datatype for use later
		datatype = create_dataset_datatype(variables)

		# Grab Parquet Files
		signal_datasets = [pyd.dataset(signal_dir+file) for file in np.array(signal_files)[sig_file_use]]
		background_datasets = [pyd.dataset(background_dir+file) for file in np.array(background_files)[bkg_file_use]]

		# Get Numbers of total signal and background events

		N_sig = sum(signal_datasets[i].count_rows() for i in range(np.sum(sig_file_use)))
		N_bkg = sum(background_datasets[i].count_rows() for i in range(np.sum(bkg_file_use)))

		print("NSIG: "+ str(N_sig))
		print("NBKG: "+str(N_bkg))

		N_target = n_batches*batch_size

		if N_target > max(N_sig,N_bkg):
			raise Exception("Target number of events ("+str(N_target)+") greater than both signal ("+str(N_sig)+") and background ("+str(N_bkg)+") numbers. Choose a lower target.")

		print("Signal:")
		signal_dataset = prep_class(signal_datasets, "signal", batch_size, n_batches, variables, selections["signal"])

		print("Background")
		background_dataset = prep_class(background_datasets, "background", batch_size, n_batches, variables, selections["background"])

		# Resample datasets to have equal number of eventss
		N_Sig = len(signal_dataset)
		N_Bkg = len(background_dataset)
		
		print("N Signal: " +str(N_Sig))
		print("N Background: "+str(N_Bkg))

		if N_Sig < N_Bkg:
			signal_dataset = np.random.choice(signal_dataset, N_Bkg)
		elif N_Bkg < N_Sig:
			background_dataset = np.random.choice(background_dataset, N_Sig)

		np.random.shuffle(signal_dataset)
		np.random.shuffle(background_dataset)

		ten_percent_cutoff_signal = int(0.1*(len(signal_dataset)))
		ten_percent_cutoff_background = int(0.1*(len(background_dataset)))

		# Create File location for klambda and then year within 
		if not os.path.exists("samples/"+kappa_lambda):
			os.mkdir("samples/"+kappa_lambda)

		# h5 file createion
		train_file = h5py.File("samples/"+kappa_lambda+"/"+year_name+"_train.h5","w")
		train_file.create_dataset("Data", data = np.append(signal_dataset[ten_percent_cutoff_signal:], background_dataset[ten_percent_cutoff_background:]))
		train_file.close()

		test_file = h5py.File("samples/"+kappa_lambda+"/"+year_name+"_test.h5","w")
		test_file.create_dataset("Data", data = np.append(signal_dataset[:ten_percent_cutoff_signal], background_dataset[:ten_percent_cutoff_background]))
		test_file.close()

	print("Samples Prepared For Training!!")
