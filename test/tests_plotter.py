"""
this class contains useful functions that can used for plotting the comparison between different runs
make sure to run this file at correct level if relative path is used
"""

from os.path import join
import json
import matplotlib.pyplot as plotter
import numpy as np

# the path to the test results
results_path = [
    "output_100_master",
    "output_calllback",
    "output_callback_update",
    "output_master"
]

# name of the results to be used in legend

names = [
    "master 1 model",
    "callback",
    "updated callback",
    "master 4 model"
]

# plotting styles, one to one corresponding to result_path

styles = [
    {"color": "blue", "marker": "o", "alpha": 0.3},
    {"color": "orange", "marker": "o", "alpha": 0.6},
    {"color": "green", "marker": "o", "alpha": 0.7},
    {"color": "brown", "marker": "o", "alpha": 0.3}
]

# observables, each should appear in all metrics
observables = [
        "th_pt",
        "th_y",
        "th_phi",
        "th_e",
        "tl_pt",
        "tl_y",
        "tl_phi",
        "tl_e"
    ]

# these are functions for extracting the information from metric files
# if something doesn't work, check if the metric format has changed

def read_metric(observable, result_path):
    """
    read and return the metric for the observable in result_path

    arguments
    ---------
    observable: str
        the observable name
    result_path: str
        path to the result folder

    returns
    -------
    metric: dict
        dictionary containing the content of the metric file
    """

    path = join(result_path, "Metrics", observable+".json")
    with open(path, "r") as file:
        metric = json.load(file)
    return metric

def nominal(metric):
    """
    read and return the nominal part of the metric

    arguments
    ---------
    metric:
        metric read from metric file

    returns
    -------
    nominal: dict
        dictionary containing the content of the metric file that corresponds to nominal
    """
    return metric["nominal"]

def resample(metric):
    """
    read and return the resample part of the metric

    arguments
    ---------
    metric:
        metric read from metric file

    returns
    -------
    nominal: dict
        dictionary containing the content of the metric file that corresponds to nominal
    """
    return metric["resample"]

def iterations(metric):
    """
    extract the number of iterations from existing read metric

    arguments
    ---------
    metric: dict
        read metric from original metric file
    
    returns
    -------
    iterations: numpy 1d array
        iteration number from 0 to n iterations - 1
    """
    return np.array(nominal(metric)["Chi2"]["iterations"])

def chi2(metric):
    """
    extract the chi2/ndf values from existing dictionary that represents either the "nominal" or "resample" part of the metric

    arguments
    ---------
    metric: dict
        a dictionary representing either the "nominal" or "resample" part of the metric

    returns
    -------
    chi2: numpy array
        a 1d array consisting of the chi2/ndf values for each iteration if metric is "nominal"
        a 2d array of shape (nruns, n iterations) of chi2/ndf values if metric is "resample"
    """
    return np.array(metric["Chi2"]["chi2/ndf"])

def delta(metric):
    """
    extract the delta values from existing dictionary that represents either the "nominal" or "resample" part of the metric

    arguments
    ---------
    metric: dict
        a dictionary representing either the "nominal" or "resample" part of the metric

    returns
    -------
    chi2: numpy array
        a 1d array consisting of the delta values for each iteration if metric is "nominal"
        a 2d array of shape (nruns, n iterations) of delta values if metric is "resample"
    """
    return np.array(metric["Delta"]["delta"])

def binerror(metric):
    """
    extract the bin error values from existing dictionary that represents either the "nominal" or "resample" part of the metric

    arguments
    ---------
    metric: dict
        a dictionary representing either the "nominal" or "resample" part of the metric

    returns
    -------
    chi2: numpy array
        a 2d array consisting of the bin error values for each iteration if metric is "nominal", in the shape (n iterations, n bins)
        a 3d array of bin error values if metric is "resample", in the shape (nruns, n iterations, n bins)
    """
    return np.array(metric["BinErrors"]["percentage"])

def binedges(metric):
    """
    extract the bin edges from existing dictionary that represents either the "nominal" or "resample" part of the metric

    arguments
    ---------
    metric: dict
        a dictionary representing either the "nominal" or "resample" part of the metric

    returns
    -------
    chi2: numpy array
        a numpy array indicating the bin edges in the shape of (n bins + 1,)
    """
    return np.array(metric["BinErrors"]["bin edges"])

# these are functions for plotting, they can be called directly to produce plots

def compare_delta_variance(save_location):
    """
    compares the variance of the deltas, can be interpreted as a measure of algorithm stability

    arguments
    ---------
    save_location: str
        the path to where the generated plot is to be saved
    """
    plotter.clf()
    for idx, metric in enumerate(metrics):
        vard = np.var(delta(resample(metric)), axis=0)
        plotter.plot(iterations(metric), vard, **styles[idx])
    plotter.title("variance in delta against iterations")
    plotter.xlabel("number of iterations")
    plotter.ylabel("variance in delta")
    plotter.grid(True)
    plotter.legend(names)
    plotter.savefig(save_location)
    plotter.clf()

def compare_delta(save_location, average=True):
    """
    compare the delta between tests. When there are the same number of runs in each test, a sum is the equivalent to an average

    arguments
    ---------
    save_location: str
        path to where generated plot will be saved
    average: boolean
        whether the plot will be averaged based on the number of runs
    """
    plotter.clf()
    for idx, metric in enumerate(metrics):
        sumd = np.sum(delta(resample(metric)), axis=0)
        n = np.shape(delta(resample(metric)))[0] # first dimension is nruns
        compval = sumd / n if average else sumd
        plotter.plot(iterations(metric), compval, **styles[idx])
    typeplot = "average" if average else "sum"
    title = typeplot + " delta against iterations"
    plotter.title(title)
    plotter.xlabel("number of iterations")
    plotter.ylabel(typeplot + " of delta")
    plotter.grid(True)
    plotter.legend(names)
    plotter.savefig(save_location)
    plotter.clf()

def compare_bin_errors(save_location):
    """
    compare the bin errors between tests.

    arguments
    ---------
    save_location: str
        path to where generated plot will be saved
    """        
    plotter.clf()
    x = np.arange(len(binedges(nominal(metrics[0]))) - 1) # all unfold results should have the save bin config
    width = 0.2

    # change the figsize argument if the generated plot can't be clearly seen
    fig, ax = plotter.subplots(len(iterations(metrics[0])), 1, figsize=(10,10))

    ax = ax.flatten()

    for plot_idx, plot in enumerate(ax):
        for idx, metric in enumerate(metrics):
            plot.bar(x +  width * idx, np.array(binerror(nominal(metric))[plot_idx]), width, label=names[idx])
        plot.set_ylabel("bin error")
        plot.set_title("bin error at iteration " + str(plot_idx))
        plot.set_xlabel("bin starting number")
        plot.set_xticks(x, binedges(nominal(metrics[0]))[:len(x)])
        if plot_idx == len(ax)-1: plot.legend(loc="lower left")

    fig.tight_layout()
    plotter.savefig(save_location)

# put plotting commands here
for observable in observables:
    # initialize metric
    metrics = []
    for result_path in results_path:
        metrics += [read_metric(observable, result_path)[observable]]
    
    # plot commands here
    compare_delta(join("plots", observable + "_delta.png"))
    compare_delta_variance(join("plots", observable + "_delta_variance.png"))
    compare_bin_errors(join("plots", observable + "_bin_error.png"))