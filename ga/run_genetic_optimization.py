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
