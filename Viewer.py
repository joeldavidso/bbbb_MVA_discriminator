import pyarrow.parquet as pa
import pyarrow.dataset as pd
import pyarrow as py
import numpy as np
import matplotlib.pyplot as plt


def plot_hist(data_arr, bins, norm, colour, label, linewidth = 3):

    # With input of np.hsitogram we take this hist as weights and fill histogram 
    # plot with single elements of this weight at the bincenters

    bin_centres = []

    for i in range(len(bins)-1):
        bin_centres.append((bins[i+1]+bins[i])/2)

    plt.hist(bin_centres,
             bins = bins,
             weights = data_arr,
             density = norm,
             color = colour,
             label = label,
             histtype = "step",
             linewidth = linewidth)
    plt.hist(bin_centres,
             bins = bins,
             weights = data_arr,
             density = norm,
             color = colour,
             alpha = 0.1)

def batch_cuts(batch):

    cutflow_nums = [batch.num_rows]

    pass_arr = np.full(cutflow_nums[0], True)

    cuts = [
            # "njets']) == 4"
            # ,"ntag']) == 2"
            # ,"bucket']) > 0"
            "pass_resolved'])"
            ]

    for count, cut in enumerate(cuts):

        pass_arr = pass_arr & np.array(eval("np.array(batch['"+cut))

        cutflow_nums.append(np.sum(pass_arr))


    return pass_arr, cutflow_nums

cut_args = ["njets", "ntag", "bucket","pass_resolved"]


var_list = [[["njets"], [0,15,15], True, True],
            [["m_hh"], [0,4000,40], True, True],
            [["ntag"], [0,7,7], True, True],
            [["m_h1", "m_h2"], [0,2500,40], True, True],
            [["pt_h1", "pt_h2"], [0,2000,40], True, True],
            [["eta_h1", "eta_h2"], [-2.5,2.5,40], False, True],
            [["X_hh"], [0,40,40], False, True],
            [["luminosity_resolved"],[0,30,30],False,True],
            [["mc_sf"],[-5e-5,5e-5,50],True,True],
            [["HC_j4_m"],[0,1000,40],True,True],
            [["bucket"],[-10,10,40],True,True]]

parquet_locations = [
                     "/storage/epp2/phubsg/bbbb_samples/bbbb_signal_mc/mc20/"
                     ,"/storage/epp2/phubsg/bbbb_samples/bbbb_signal_mc/mc20/",
                     "/storage/epp2/phubsg/bbbb_samples/bb_background_data/"
                     ,"/storage/epp2/phubsg/bbbb_samples/bbbb_background_prediction/data161718-gn277_orth1_4b_sr_124_cr_x1.6_detaCut/resolved_blind_h1h2/"
                     ]

parquet_labels = [
                  "combined_skim_ggFhh_chhh1p0_mc20a__Nominal.parquet",
                  "combined_skim_ggFhh_chhh1p0_mc20d__Nominal.parquet",
                  "combined_skim_data16__Nominal.parquet",
                  "df_pred__pipe_16_10xGP_mean.parquet"
                  ]

parquet_names = [
                 "bbbb chhh1p0 mc20a",
                 "bbbb chhh1p0 mc20d",
                 "bb data16",
                 "bbbb predicted background 16"]

parquet_files = [parquet_locations[i]+parquet_labels[i] for i in range(len(parquet_locations))]

dataset_target = 1

datasets = [pd.dataset(file) for file in parquet_files]

## Colour Scheme we want:
##   - E78AA1 (Orange)
##   - DE0C62 (Red)
##   - 9213E0 (Purple)
##   - 11B7E0 (Blue)
##   - 00B689 (Green)

colours = ["#E78AA1",
           "#9213E0",
           "#00B689"]


for vars in var_list:

    var_bins = np.linspace(vars[1][0],vars[1][1],vars[1][2]+1)

    for var in vars[0]:

        print("__________________________")
        print(var+":")
        print("----------")

        dataset_hists = []


        for count, dataset in enumerate(datasets):

            var_hist = np.histogram([],bins=var_bins)[0]

            print("* "+parquet_names[count]+":")

            n_nan = 0
            n_passed = 0

            for batch in dataset.to_batches(columns = list(dict.fromkeys([var]+cut_args))):

                cut_arr, cutflow_nums = batch_cuts(batch)

                cut_batch = batch[0].to_numpy(zero_copy_only = False)[cut_arr]

                n_nan += np.sum(np.isnan(cut_batch))

                n_passed += np.sum(cut_arr)

                var_hist += np.histogram(cut_batch, bins = var_bins)[0]

            dataset_hists.append(var_hist)

            plot_hist(var_hist, var_bins, vars[3], colour = colours[count], label = parquet_labels[count])

            print("    N_events: "+str(dataset.count_rows()))
            print("    N_passed: "+str(n_passed))
            print("    N_nan: "+str(n_nan))

        if vars[2]:
            plt.yscale("log")

        plt.draw()
        plt.legend(loc = "upper right")
        plt.xlabel(var)
        plt.ylabel("Normalised Number of Events" if vars[3] else "Number of Events")

        plt.savefig("plots/parquet/"+var+".pdf")
        plt.savefig("plots/parquet/"+var+".png")
        plt.clf()