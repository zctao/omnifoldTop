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
    "output_only_standardization",
    "output_tmp"
]

# plotting styles, one to one corresponding to result_path

styles = [

]

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

def nominal(observable, result_path):
    """
    read and return the nominal part of the metric for the observable in result_path

    arguments
    ---------
    observable: str
        the observable name
    result_path: str
        path to the result folder

    returns
    -------
    nominal: dict
        dictionary containing the content of the metric file that corresponds to nominal
    """
    metric = read_metric(observable, result_path)
    return metric["nominal"]

def resample(observable, result_path):
    """
    read and return the resample part of the metric for the observable in result_path

    arguments
    ---------
    observable: str
        the observable name
    result_path: str
        path to the result folder

    returns
    -------
    resample: dict
        dictionary containing the content of the metric file that corresponds to nominal
    """
    metric = read_metric(observable, result_path)
    return metric["resample"]

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

def binerror(metric):
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