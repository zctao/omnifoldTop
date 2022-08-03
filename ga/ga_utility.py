"""
containing various utility functions that are used in the run_genetic_optimization module
"""

import numpy as np
from os.path import join
import json

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
    return metric[observable]

def extract_nominal_pval(observable, path):
    """
    read and return the nominal pvalue from the metric file

    arguments
    ---------
    observable: str
        the observable name
    result_path: str
        path to the result folder

    returns
    -------
    pval: numpy array
        the nominal pvalues corresponding to each iteration
    """
    metric = read_metric(observable, path)
    return np.array(metric["nominal"]["Chi2"]["pvalue"])

def extract_rerun_delta_std(observable, path):
    """
    read and return the standard deviation of the rerun deltas from the metric file

    arguments
    ---------
    observable: str
        the observable name
    result_path: str
        path to the result folder

    returns
    -------
    std: numpy array
        the standard deviation of the rerun deltas corresponding to each iteration
    """
    metric = read_metric(observable, path)
    return np.sqrt(np.var(np.array(metric["resample"]["Delta"]["delta"]), axis=0))

def select_best_iteration(observables, path):
    