"""
a genetic algorithm optimizer
"""
import numpy as np

# observables for the run, should match observable config
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

# extra factors that will be taken into the fit function and their weights
metric_item_and_weights = {
    "time" : 1
}

# max number of deep layers
max_layers = 10
# max nodes per layer
max_nodes_per_layer = 200

def generate_data_type_list():
    """
    returns a list of datatypes for the genes
    """
    return [float] + [int for i in range max_layers]
def generate_gene_space_list():
    """
    returns a list of gene spaces (upper and lower limits for each entry)
    """
    return [{'low': 0.00001, 'high': 0.01}] + [{'low': 0, 'high': max_nodes_per_layer} for i in range max_layers]

