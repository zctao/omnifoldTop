"""
this class contains useful functions that can used for plotting the comparison between different runs
make sure to run this file at correct level if relative path is used
"""

from os.path import join
import json
import matplotlib.pyplot as plotter

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
    read and return the metric for each observable in result_path

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
