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


def KL_Divergence(hist_arrs, target):

    hist_target = hist_arrs[target]/np.sum(hist_arrs[target])
    hist_not_target = hist_arrs[1-target]/np.sum(hist_arrs[1-target])


    KL_div = 0    
    # 1-target i the non target distribution as target is either 0 or 1
    for bin in range(len(hist_not_target)):
        if hist_target[bin] == 0:
            KL_div += 0
        elif hist_not_target[bin] == 0:
            KL_div += 0
        else:
            KL_div += hist_target[bin]*np.log(hist_target[bin]/hist_not_target[bin])
    return KL_div

def Mean_Distance(hist_arrs, target):
    
    tot_dist = 0
    # 1-target i the non target distribution as target is either 0 or 1
    for bin in range(len(hist_arrs[1-target])):
        tot_dist += np.abs(hist_arrs[target][bin]-hist_arrs[1-target][bin])
    return tot_dist/len(hist_arrs[1-target])

var_list = [[["njets"], [0,15,15], True, True],
            [["m_hh"], [0,4000,40], True, True],
            [["ntag"], [0,7,7], True, True],
            [["m_h1", "m_h2"], [0,2500,40], True, True],
            [["pt_h1", "pt_h2"], [0,2000,40], True, True],
            [["eta_h1", "eta_h2"], [-2.5,2.5,40], False, True],
            [["X_hh"], [0,40,40], False, True],
            [["luminosity_resolved"],[0,30,30],False,True],
            [["mc_sf"],[-5e-5,5e-5,50],True,True]]

parquet_locations = ["/storage/epp2/phubsg/bbbb_samples/bbbb_signal_mc/mc20/"
                     ,"/storage/epp2/phubsg/bbbb_samples/bbbb_signal_mc/mc20/"
                    #  ,"/storage/epp2/phubsg/bbbb_samples/bb_background_data/"
                     ]

parquet_labels = ["combined_skim_ggFhh_chhh1p0_mc20a__Nominal.parquet"
                  ,"combined_skim_ggFhh_chhh1p0_mc20d__Nominal.parquet"
                #   ,"combined_skim_data16__Nominal.parquet"
                  ]

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
           "#9213E0"]


for vars in var_list:

    var_bins = np.linspace(vars[1][0],vars[1][1],vars[1][2]+1)

    for var in vars[0]:

        print(var)

        dataset_hists = []

        for count, dataset in enumerate(datasets):

            var_hist = np.histogram([],bins=var_bins)[0]

            for batch in dataset.to_batches(columns = [var,"pass_resolved"]):

                # if var == "mc_sf":
                #     print(batch[0].to_numpy(zero_copy_only = False)[batch["pass_resolved"]])

                var_hist += np.histogram(batch[0].to_numpy(zero_copy_only = False)[batch["pass_resolved"]], bins = var_bins)[0]

            dataset_hists.append(var_hist)

            plot_hist(var_hist, var_bins, vars[3], colour = colours[count], label = parquet_labels[count])

        if vars[2]:
            plt.yscale("log")

        plt.draw()
        plt.legend(loc = "upper right")
        plt.xlabel(var)
        plt.ylabel("Normalised Number of Events" if vars[3] else "Number of Events")

        KL_div = KL_Divergence(dataset_hists, dataset_target)
        plt.text(0, 0.5,"KL Divergence = "+str(round(KL_div,4)),fontsize = 10)
        # Mean_dist = Mean_Distance(dataset_hists, dataset_target)
        # plt.text(0, 0.5,"Mean Distance = "+str(int(Mean_dist)),fontsize = 10)

        plt.savefig("plots/parquet/"+var+".pdf")
        plt.savefig("plots/parquet/"+var+".png")
        plt.clf()