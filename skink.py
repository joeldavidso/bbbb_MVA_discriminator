from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Dictionary of colours

C_Pallets = {"Pastel": {"Orange" : "#E78AA1",
                        "Blue" : "#11B7E0",
                        "Red" : "#DE0C62",
                        "Green" : "#00B689",
                        "Purple" : "#9213E0",},

             "Bold": {"Orange": "#ff6600",
                      "Blue": "#0033cc",
                      "Red": "#b80053",
                      "Green": "#009933",
                      "Purple": "#670178"}
            }


                                   ######################################################
##############################################       Misc Plot Functions      ###########################################
                                   ######################################################


# Simple function for creating bin edges and bin centres for use in plotting    
def get_bins(xmin, xmax, nbins):
    
    bin_edges = np.linspace(xmin,xmax,nbins+1)
    bin_centres = (bin_edges[:-1]+bin_edges[1:])/2
    
    return [bin_edges,bin_centres]


                                   ######################################################
##############################################         Plot Base Class        ###########################################
                                   ######################################################

class PlotBase:

    def __init__(self, xlabel, ylabel, add_ax = False, density = False, sizex = 6, sizey = 6, logy = False):
        
        # Create figure and axes 
        fig, ax = plt.subplots(figsize = (sizex, sizey + (1.2 if add_ax else 0)))

        self.figure = fig
        self.primary_ax = ax

        self.density = density

        self.logy = logy

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density from " + ylabel if density else ylabel)

        ax.grid(linestyle = "--")
        ax.set_axisbelow(True)

        ax.xaxis.minorticks_on()
        ax.yaxis.minorticks_on()

        # Creates and configure additional axis
        if add_ax:

            # Move x axis ticks to below additional axis plot
            ax.tick_params(labelbottom = False)
            
            # Append additional axis under primary plot
            divider = make_axes_locatable(ax)
            ax_add = divider.append_axes("bottom", 1.2, pad = 0, sharex = ax)

            self.add_ax = ax_add
            self.primary_ax.set_xlabel("")

            # Axes label and tick stuff and things
            ax_add.tick_params(labelleft = False, labelright = True)        
            ax_add.set_xlabel(xlabel)
            ax_add.xaxis.minorticks_on()

            ax_add.yaxis.set_label_position("right")
            ax_add.yaxis.set_ticks_position("right")
            ax_add.set_ylabel("Ratio")
            ax_add.yaxis.minorticks_on()
            
            ax_add.grid(linestyle = "--")
            ax_add.set_axisbelow(True)

        else:
            self.add_ax = None


                                   ######################################################
############################################           Hist Plot Class          ###########################################
                                   ######################################################


class HistogramPlot(PlotBase):
    

    def __init__(self, bins, xlabel = "X", ylabel = "Y", ratio = False, residual = False, density = False,
                 plot_unc = True, cpallet = "Pastel", sizex = 5, sizey = 5, logy = False):
        
        # Initialize relevant callable values
        self.histograms = []
        self.bin_edges = bins[0]
        self.bin_centres = bins[1]
        self.colours = C_Pallets[cpallet]
        self.plot_unc = plot_unc
        self.residual = residual
        self.ratio = ratio

        if ratio and residual:
            raise("Can only plot one of ratio or residual, not both !!!")

        super().__init__(xlabel, ylabel, add_ax = (ratio or residual), density = density, sizex = 5, sizey = 5, logy = logy)

    def Add(self, data, label = "Add Label !!!!!!!!!!", colour = None, uncs = None, drawstyle = "bar",
                 fill = "70", shrink = None, linewidth = 1.5, linecolour = "black", linestyle = "-",
                 orientation = "vertical", reference = False, addthis = True, weights = None):

        # Set weights to one if None given
        weights = np.ones_like(data) if weights is None else weights

        # Bin Data into numpy histogram and get uncertainties and density factors
        n_entries = np.sum(weights[np.logical_and(data > self.bin_edges[0],data < self.bin_edges[-1])])

        uncs = np.sqrt(np.histogram(data, weights = weights**2, bins = self.bin_edges)[0]) if uncs is None else uncs
        data = np.histogram(data, weights = weights, bins = self.bin_edges)[0]

        density_factors = 1/(n_entries*(self.bin_edges[1:] - self.bin_edges[:-1])) if self.density else np.array([1 for b in self.bin_centres])

        # If no uncertatinty given then use sqrt(bin value)
        # / by 1/sqrt(n_entries * bin_width) if density is True
        # uncs = density_factors * uncs 


        self.histograms.append({"data":  data,
                                "label": label,
                                "colour": colour,
                                "uncs": uncs,
                                "density_factors": density_factors,
                                "fill": "ff" if fill == "full" else fill,
                                "shrink": shrink,
                                "linewidth": linewidth,
                                "linecolour": linecolour,
                                "linestyle": linestyle,
                                "orientation": orientation,
                                "reference": reference,
                                "add": addthis,
                                "drawstyle": drawstyle})
    
    def Plot_Unc(self, axis, hist):

        # Do errors for each bin rather than whole plot so that gaps can occur when shrink != None
        for bin_N in range(len(self.bin_centres)):

            x1 = self.bin_edges[bin_N]
            x2 = self.bin_edges[bin_N+1]

            if hist["shrink"] is not None:
                x1 = (1+hist["shrink"])*self.bin_centres[bin_N] - hist["shrink"]*x1
                x2 = (1+hist["shrink"])*self.bin_centres[bin_N] - hist["shrink"]*x2

            # split into upper and lower error for alpha matching whith overlap with transparent hist

            # Adding alphas is non-linear
            # For alpha_a on top of alpha_b alpha_both = alpha_a +(1-alpha_a)*alpha_b
            # In this case, alpha_a = fill_alpha/255 and alha_b = 0.3 
            alpha_unc = 0.3
        

            # Lower            
            axis.fill_between(x = [x1,x2],
                              y1 = np.array([hist["data"][bin_N] - hist["uncs"][bin_N], hist["data"][bin_N] - hist["uncs"][bin_N]])*hist["density_factors"][bin_N],
                              y2 = np.array([hist["data"][bin_N], hist["data"][bin_N]])*hist["density_factors"][bin_N],
                              color = hist["colour"], zorder = 1, step = "pre", edgecolor = None, alpha = alpha_unc)

            # Upper 
            axis.fill_between(x = [x1,x2],
                              y1 = np.array([hist["data"][bin_N], hist["data"][bin_N]])*hist["density_factors"][bin_N],
                              y2 = np.array([hist["data"][bin_N] + hist["uncs"][bin_N], hist["data"][bin_N] + hist["uncs"][bin_N]])*hist["density_factors"][bin_N],
                              color = hist["colour"], zorder = 1, step = "pre", edgecolor = None, alpha = min(alpha_unc + (1-alpha_unc)*(int(hist["fill"],16)/255), 1))


    def Plot_Ratio(self, hist):
        
        ref_bool = [hists["reference"] for hists in self.histograms]

        if np.sum(ref_bool) != 1:
            raise("Wrong Number of Refernece Hist Specified !!!")

        ref_hist = self.histograms[np.nonzero(ref_bool)[0][0]]

        self.add_ax.set_ylabel("Ratio wrt "+ref_hist["label"])

        temp_hist = {"data":  (hist["data"]*hist["density_factors"])/(ref_hist["data"]*ref_hist["density_factors"]),
                     "uncs": ((hist["data"]*hist["density_factors"])/(ref_hist["data"]*ref_hist["density_factors"])) 
                     *np.sqrt((ref_hist["uncs"]/ref_hist["data"])**2 + (hist["uncs"]/hist["data"])**2), # density factors cancel in square roots!
                     "density_factors": [1 for b in self.bin_centres],
                     "shrink": None,
                     "fill": "00",
                     "colour": hist["colour"],
                     "label": hist["label"]}

        self.Plot_Unc(self.add_ax, temp_hist) if self.plot_unc else None

        self.add_ax.plot(self.bin_centres, temp_hist["data"], color = hist["colour"], label = temp_hist["label"], drawstyle = "steps-mid")


    def Plot_Residual(self, hist):
        
        ref_bool = [hists["reference"] for hists in self.histograms]

        if np.sum(ref_bool) != 1:
            raise("Wrong Number of Refernece Hist Specified !!!")

        ref_hist = self.histograms[np.nonzero(ref_bool)[0][0]]

        self.add_ax.set_ylabel("Residual wrt "+ref_hist["label"])

        temp_hist = {"data":  ((hist["data"]*hist["density_factors"]) - (ref_hist["data"]*ref_hist["density_factors"]))
                              / np.sqrt((ref_hist["density_factors"]*ref_hist["uncs"])**2 + (hist["density_factors"]*hist["uncs"])**2)
                              ,
                     "uncs": None,
                     "density_factors": [1 for b in self.bin_centres],
                     "shrink": None,
                     "fill": "00",
                     "colour": hist["colour"],
                     "label": hist["label"]}

        self.add_ax.plot(self.bin_centres, temp_hist["data"], color = hist["colour"], label = temp_hist["label"], drawstyle = "steps-mid")


    def Plot(self, plot_path, legend_loc = ["upper right", "upper right"], frame = False):

        # Plot each histogram on primary axis
        for count, hist in enumerate(self.histograms):

            hist["colour"] = hist["colour"] if hist["colour"] is not None else self.colours[list(self.colours.keys())[count]]
        
            # Uncertainty Plotting
            self.Plot_Unc(self.primary_ax, hist) if self.plot_unc else None

            self.primary_ax.hist(self.bin_centres, self.bin_edges, weights = hist["data"], ec = hist["linecolour"],
                                 fc = hist["colour"] + hist["fill"], linestyle = hist["linestyle"],
                                 linewidth = hist["linewidth"], rwidth = hist["shrink"], orientation = hist["orientation"], label = hist["label"],
                                 density = self.density, histtype = hist["drawstyle"])


            # Ratio Plotting
            self.Plot_Ratio(hist) if self.ratio and hist["add"] else None

            # Ratio Plotting
            self.Plot_Residual(hist) if self.residual and hist["add"] else None

        self.primary_ax.margins(x = 0, y = 0.1)

        self.add_ax.margins(x = 0, y = 0.05) if self.ratio else None
        self.add_ax.set_ylim(bottom = max(0,self.add_ax.get_ylim()[0])) if self.ratio else None

        self.primary_ax.set_yscale("log") if self.logy else None

        self.primary_ax.legend(loc = legend_loc[0], frameon = frame)

        if self.add_ax is not None:

            self.add_ax.legend(loc = legend_loc[1], frameon = frame)

        plt.draw()

        plt.savefig(plot_path+".pdf", bbox_inches = "tight")
        plt.savefig(plot_path+".png", bbox_inches = "tight")

        plt.close(self.figure)

                                   ######################################################
############################################           Line Plot Class          ###########################################
                                   ######################################################


class LinePlot(PlotBase):
    
    def __init__(self, xs, xlabel = "X", ylabel = "Y", ratio = False, residual = False, plot_unc = False, logy = False, cpallet = "Pastel", sizex = 5, sizey = 5):
        
        # Initialize relevant callable values
        self.lines = []
        self.xs = xs
        self.colours = C_Pallets[cpallet]
        self.plot_unc = plot_unc
        self.residual = residual
        self.ratio = ratio

        if ratio and residual:
            raise("Can Only Plot Ratio or Residual Individually Not Together!!!!")

        super().__init__(xlabel, ylabel, add_ax = (ratio or residual), density = False, sizex = 5, sizey = 5, logy = logy)

    
    def Add(self, ys, label = "Add Label !!!!!!!!!!", linecolour = None, uncs = None, residualthis = True,
            linestyle = "-", linewidth = 2, marker = ".", marker_size = 10, reference = False, ratiothis = True):

        # Force data to be in binned form
        if len(ys) != len(self.xs):
            raise("xs & ys Not Same Shape !!!!")

        # If no uncertatinty given and plot_unc = True require uncs
        if self.plot_unc and uncs is None:
            raise("Plot Uncertainty True but No Uncertainty Given!!!!")         

        self.lines.append({"ys":  np.array(ys),
                           "label": label,
                           "uncs": np.array(uncs),
                           "linewidth": linewidth,
                           "linecolour": linecolour,
                           "linestyle": linestyle,
                           "marker": marker,
                           "marker_size": marker_size,
                           "reference": reference,
                           "ratio": ratiothis,
                           "residual": residualthis})


    def Plot_Unc(self, axis, line):

        axis.fill_between(x = self.xs,
                          y1 = line["ys"] - line["uncs"],
                          y2 = line["ys"] + line["uncs"],
                          color = line["linecolour"], zorder = 1, edgecolor = None, alpha = 0.3)
        

    def Plot_Ratio(self, line):
        
        ref_bool = [line["reference"] for line in self.lines]

        if np.sum(ref_bool) != 1:
            raise("Wrong Number of Refernece Lines Specified !!!")

        ref_line = self.lines[np.nonzero(ref_bool)[0][0]]

        self.add_ax.set_ylabel("Ratio wrt "+ref_line["label"])

        temp_line = {"ys":  line["ys"]/ref_line["ys"],
                     "uncs": (line["ys"]/ref_line["ys"]) * np.sqrt((ref_line["uncs"]/ref_line["ys"])**2 + (line["uncs"]/line["ys"])**2) if self.plot_unc else None,
                     "label": line["label"],
                     "linecolour": line["linecolour"]}

        self.Plot_Unc(self.add_ax, temp_line) if self.plot_unc else None

        self.add_ax.plot(self.xs, temp_line["ys"], color = line["linecolour"], label = temp_line["label"])


    def Plot_Residual(self,line):

        ref_bool = [line["reference"] for line in self.lines]

        if np.sum(ref_bool) != 1:
            raise("Wrong Number of Refernece Lines Specified !!!")

        ref_line = self.lines[np.nonzero(ref_bool)[0][0]]

        self.add_ax.set_ylabel("Residual wrt "+ref_line["label"])

        temp_line = {"ys":  line["ys"] - ref_line["ys"],
                     "uncs": np.sqrt(ref_line["uncs"]**2 + line["uncs"]**2) if self.plot_unc else None,
                     "label": line["label"],
                     "linecolour": line["linecolour"]}

        self.Plot_Unc(self.add_ax, temp_line) if self.plot_unc else None

        self.add_ax.plot(self.xs, temp_line["ys"], color = line["linecolour"], label = temp_line["label"])


    def Plot(self, plot_path, legend_loc = ["upper right", "upper right"], frame = False, ymax = None, ymin = None):

        # Plot each line on primary axis
        for count, line in enumerate(self.lines):

            line["linecolour"] = line["linecolour"] if line["linecolour"] is not None else self.colours[list(self.colours.keys())[count]]
    
            # Uncertainty Plotting
            self.Plot_Unc(self.primary_ax, line) if self.plot_unc else None

            self.primary_ax.plot(self.xs, line["ys"], color = line["linecolour"], linestyle = line["linestyle"], marker = line["marker"], linewidth = line["linewidth"],
                                 ms = line["marker_size"], label = line["label"])

            # Ratio Plotting
            self.Plot_Ratio(line) if self.ratio and line["ratio"] else None

            # Residual Plotting
            self.Plot_Residual(line) if self.residual and line["residual"] else None

        self.primary_ax.margins(x = 0, y = 0.1)

        self.primary_ax.set_ylim(bottom =  ymin) if ymin is not None else None
        self.primary_ax.set_ylim(top =  ymax) if ymax is not None else None

        self.add_ax.margins(x = 0, y = 0.05) if self.ratio else None
        self.add_ax.set_ylim(bottom = max(0,self.add_ax.get_ylim()[0])) if self.ratio else None

        self.primary_ax.set_yscale("log") if self.logy else None

        self.primary_ax.legend(loc = legend_loc[0], frameon = frame)

        if self.add_ax is not None:
            self.add_ax.legend(loc = legend_loc[1], frameon = frame)

        plt.draw()

        plt.savefig(plot_path+".pdf", bbox_inches = "tight")
        plt.savefig(plot_path+".png", bbox_inches = "tight")

        plt.close(self.figure)


                                   ######################################################
############################################          Hist2D Plot Class         ###########################################
                                   ######################################################

class Hist2D(PlotBase):

    def __init__(self, xbins, ybins, xlabel = "X", ylabel = "Y", cmap = "cividis", sizex = 5, sizey = 5, margins = True, plot_unc = True):
        
        # Initialize relevant callable values
        self.xbin_edges = xbins[0]
        self.xbin_centres = xbins[1]
        self.ybin_edges = ybins[0]
        self.ybin_centres = ybins[1]
        self.cmap = cmap
        self.margins = margins
        self.plot_unc = plot_unc

        super().__init__(xlabel, ylabel, sizex = 5, sizey = 5)


    def Set(self, xdata, ydata, colour = "#0000FF", fill = "90", linecolour = "black", linewidth = 1.5):

        self.hist = {"xdata":  xdata,
                     "ydata": ydata,
                     ## The rest is for margins
                     "colour": colour,
                     "fill": fill,
                     "linecolour": linecolour,
                     "linewidth": linewidth
                     }
        
    def PlotMargins(self):
        
        divider = make_axes_locatable(self.primary_ax)

        ax_histx = divider.append_axes("top", 0.6, pad=0, sharex=self.primary_ax)
        ax_histy = divider.append_axes("right", 0.6, pad= -0.03 if self.plot_unc else 0, sharey=self.primary_ax)

        ax_histx.axis("off")
        ax_histy.axis("off")

        self.PlotUnc(ax_histx, self.hist["xdata"], self.xbin_edges, self.xbin_centres)
        self.PlotUnc(ax_histy, self.hist["ydata"], self.ybin_edges, self.ybin_centres)

        ax_histx.hist(self.hist["xdata"], self.xbin_edges, fc = self.hist["colour"] + "90", ec = "black", linewidth = 1.5)
        ax_histy.hist(self.hist["ydata"], self.ybin_edges, fc = self.hist["colour"] + "90", ec = "black", linewidth = 1.5, orientation = "horizontal")

    def PlotUnc(self, axis, data, bin_edges, bin_centres):

        data_hist = np.histogram(data, bin_edges)[0]

        hist = {"data": data_hist,
                "uncs": np.sqrt(data_hist)}

        # Do errors for each bin rather than whole plot so that gaps can occur when shrink != None
        for bin_N in range(len(bin_centres)):

            x1 = bin_edges[bin_N]
            x2 = bin_edges[bin_N+1]

            # split into upper and lower error for alpha matching whith overlap with transparent hist

            # Adding alphas is non-linear
            # For alpha_a on top of alpha_b alpha_both = alpha_a +(1-alpha_a)*alpha_b
            # In this case, alpha_a = fill_alpha/255 and alha_b = 0.3 
            alpha_unc = 0.3
            
            # Lower            
            axis.fill_between(x = [x1,x2],
                              y1 = [hist["data"][bin_N] - hist["uncs"][bin_N], hist["data"][bin_N] - hist["uncs"][bin_N]],
                              y2 = [hist["data"][bin_N], hist["data"][bin_N]],
                              color = self.hist["colour"], zorder = 1, step = "pre", edgecolor = None, alpha = alpha_unc)

            # Upper 
            axis.fill_between(x = [x1,x2],
                              y1 = [hist["data"][bin_N], hist["data"][bin_N]],
                              y2 = [hist["data"][bin_N] + hist["uncs"][bin_N], hist["data"][bin_N] + hist["uncs"][bin_N]],
                              color = self.hist["colour"], zorder = 1, step = "pre", edgecolor = None, alpha = min(alpha_unc + (1-alpha_unc)*(int(self.hist["fill"],16)/255), 1))

    def Plot(self, plot_path):

    
        self.primary_ax.hist2d(self.hist["xdata"],self.hist["ydata"], bins = [self.xbin_edges, self.ybin_edges], cmap = "cividis")

        self.PlotMargins() if self.margins else None

        plt.draw()

        plt.savefig(plot_path+".pdf", bbox_inches = "tight")
        plt.savefig(plot_path+".png", bbox_inches = "tight")

        plt.close(self.figure)

