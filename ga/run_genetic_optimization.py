"""
a genetic algorithm optimizer
"""
import numpy as np
import time
from os.path import isfile, join

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

def learning_rate(solution):
    """
    returns
    -------
    learning rate of the solution
    """
    return solution[0]

def write_config(solution):
    """
    write the solution to run config

    arguments
    ---------
    solution: a solution generated from pygad
    """
    run = {
        "data": ["/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_hw_3_pseudotop_parton_ejets.root"],
        "signal": [
            "/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_1_pseudotop_parton_ejets.root",
            "/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_4_pseudotop_parton_ejets.root",
            "/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_5_pseudotop_parton_ejets.root",
            "/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_6_pseudotop_parton_ejets.root"
        ],
        "observable_config" : "configs/observables/vars_ttbardiffXs_pseudotop.json",
        "iterations" : 4,
        "nruns": 8,
        "truth_known" : True,
        "normalize" : True,
        "batch_size" : 20000,
        "plot_verbosity" : 0,
        "outputdir" : "output_ga"
    }
    
    lrs = {
        "initial_learning_rate": 0.001,
        "scheduler_names": ["warmc"],
        "scheduler_args": {
            "warm_up_epochs": 5
        },
        "reduce_on_plateau": 0
    }

def fitness_func(solution, solution_idx):
    """
    evaluates the fitness value of each solution, which has a higher value the better it performs
    the most important factor (currently) is the p-values corresponding to unfolding quality
    the fitness_function also takes into its account the unfolding time

    arguments
    ---------
    solution: np.ndarray
        the solution in the form [learning_rate, nodes in 1st layer, ... , nodes in last layer]
        note that in evaluating fitness, layers with zero nodes will be skipped, meaning that a sample
        solution like [0.01, 0, 0, 100, 0, 100, 0, 100, 0, 0] will be converted to a network of 3 100 nodes
        layers with 0.01 learning rate.
    solution_idx: int
        index of solution related to internal tracking of pygad, not used

    returns
    -------
    promised fitness value of the solutions
    """
    write_config(solution)


