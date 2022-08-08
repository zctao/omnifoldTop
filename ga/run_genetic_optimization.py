"""
a genetic algorithm optimizer
"""
from lib2to3 import refactor
import numpy as np
from time import time
from os.path import isfile, join, exists
from os import makedirs
import json
import subprocess
import ga_utility
import pygad
import matplotlib.pyplot as plt

# if False, the script will attempt to read a previous ga save file and plot the results from it
run_mode = True

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
metric_weight = {
    "time" : 2,
    "pval": 5,
    "std" : 2
}

# max number of deep layers
max_layers = 10
# max nodes per layer
max_nodes_per_layer = 200

# name of output folder
output_folder = "output_ga"

# path to save generations
generation_save = join("ga", "generations")

# path to log, None for no log
log_path = join("ga", "run.log")

# load ref run
with open(join("ga", "ref.json"), "r") as file:
    ref = json.load(file)

def log(msg):
    """
    write log message to log file

    arguments
    ---------
    msg: str
        message to be recorded
    """
    with open(log_path, "a") as log:
        log.write(msg+"\n")

def generate_data_type_list():
    """
    returns a list of datatypes for the genes
    """
    return [float] + [int for i in range(max_layers)]
def generate_gene_space_list():
    """
    returns a list of gene spaces (upper and lower limits for each entry)
    """
    return [{'low': 0.00001, 'high': 0.01}] + [{'low': 0, 'high': max_nodes_per_layer} for i in range(max_layers)]

def learning_rate(solution):
    """
    returns
    -------
    learning rate of the solution
    """
    return solution[0]

def node_list(solution):
    """
    returns
    -------
    node list from a solution
    """
    return [node for node in solution[1:] if node != 0]

def write_config(solution):
    """
    write the solution to run config

    arguments
    ---------
    solution: a solution generated from pygad
    """

    run_path = join("configs", "run", "ga.json")
    lrs_path = join("configs", "lrs", "lrsga.json")

    # composed according to how they are parsed in modelUtils
    model_name = "dense"
    for node in node_list(solution):
        model_name += "_"+str(node)
    # run config here
    run = {
        "data": ["/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_hw_3_pseudotop_parton_ejets.root"],
        "signal": [
            "/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_1_pseudotop_parton_ejets.root",
            "/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_4_pseudotop_parton_ejets.root",
            "/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_5_pseudotop_parton_ejets.root",
            "/fast_scratch/xyyu/model_learning_iteration_test_data/ttbar_6_pseudotop_parton_ejets.root"
        ],
        "observable_config" : "configs/observables/vars_ttbardiffXs_pseudotop.json",
        "iterations" : 2,
        "nruns": 2,
        "truth_known" : True,
        "normalize" : True,
        "batch_size" : 20000,
        "plot_verbosity" : 0,
        "outputdir" : output_folder,
        "lrscheduler_config": lrs_path,
        "model_name": model_name
    }
    
    # lrs config here, identicle to default setting except for learning rate
    lrs = {
        "initial_learning_rate": learning_rate(solution),
        "scheduler_names": ["warmc"],
        "scheduler_args": {
            "warm_up_epochs": 5
        },
        "reduce_on_plateau": 0
    }

    # create files if not there
    option = "w" if isfile(run_path) else "x"
    with open(run_path, option) as file:
        json.dump(run, file, indent="")
    option = "w" if isfile(lrs_path) else "x"
    with open(lrs_path, option) as file:
        json.dump(lrs, file, indent="")

def shift_zero(solution):
    """
    shifts all non-zero layers to the left by ignoring zero layers
    """
    for each in solution:
        nonzero = each[each != 0]
        zero = each[each == 0]
        each[:len(nonzero)] = nonzero
        each[len(nonzero):] = zero

def on_mutation(ga_instance, offspring_mutation):
    """
    shifts all non-zero layers to the left by ignoring zero layers
    """
    shift_zero(offspring_mutation)

def on_crossover(ga_instance, offspring_crossover):
    """
    shifts all non-zero layers to the left by ignoring zero layers
    """
    shift_zero(offspring_crossover)

def on_generation(ga_instance):
    """
    callback for every generation, currently for anouncing beginning of generations in log and saving current generation

    arguments
    ---------
    ga_instance: pygad.GA
        ga instance
    """
    log("++++++++++++++++++++++++")
    log(str(ga_instance.generations_completed) + " generations completed")
    
    if not exists(generation_save):
        makedirs(generation_save)
    ga.save(join(generation_save, "gen_"+str(ga_instance.generations_completed)))
    log("++++++++++++++++++++++++")

def calculate_std_score(stds):
    """
    calculates the model stability score by comparing the standard deviation of the last iteration to reference run value

    arguments
    ---------
    stds: np array of float
        a np array of standard deviations in the same order as the list of observables
    """
    ref_std = []
    for observable in observables:
        ref_std += [ref[observable+"_delta_std"]]
    ref_std = np.array(ref_std)
    
    return (ref_std - stds) / ref_std

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
    log("Evaluate solution: "+str(solution))
    # runtime
    write_config(solution)
    run_path = join("configs", "run", "ga.json")
    start = time()
    subprocess.run(["./run_unfold.py", run_path])
    duration = time() - start

    pvals, stds = [], []
    for observable in observables:
        # extract pval from last run
        pvals += [(ga_utility.extract_nominal_pval(observable, output_folder))[-1]]
        stds += [(ga_utility.extract_rerun_delta_std(observable, output_folder))[-1]]
    pvals = np.array(pvals)
    stds = np.array(stds)

    # temporary placeholder fitness
    fitness = 0 # start from 0
    fitness += metric_weight["pval"] * np.sum(pvals)
    fitness += metric_weight["time"] * (ref["time"] - duration) / ref["time"]
    fitness += metric_weight["std"] * np.sum(calculate_std_score(stds))
    log("Fitness Score: "+str(fitness))
    return fitness
    
initial_population = np.array(
    [
        [0.001, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0],
        [0.001, 50, 50, 50, 50, 50, 50, 50, 50, 50, 0],
        [0.001, 1000, 10, 10000, 0, 0, 0, 0, 0, 0, 0],
        [0.001, 50, 200, 50, 0, 0, 0, 0, 0, 0, 0],
        [0.001, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0],
        [0.005, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0],
        [0.005, 50, 50, 50, 50, 50, 50, 50, 50, 50, 0],
        [0.005, 1000, 10, 10000, 0, 0, 0, 0, 0, 0, 0],
        [0.005, 50, 200, 50, 0, 0, 0, 0, 0, 0, 0],
        [0.005, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0],
        [0.01, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0],
        [0.01, 50, 50, 50, 50, 50, 50, 50, 50, 50, 0],
        [0.01, 1000, 10, 10000, 0, 0, 0, 0, 0, 0, 0],
        [0.01, 50, 200, 50, 0, 0, 0, 0, 0, 0, 0],
        [0.01, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0]
    ]
)

if run_mode:
    ga = pygad.GA(fitness_func=fitness_func, initial_population=initial_population,
                num_genes=11,
                gene_type=generate_data_type_list(), gene_space=generate_gene_space_list(),
                parent_selection_type="tournament",
                crossover_type="two_points", crossover_probability=0.1,
                mutation_type="adaptive", mutation_probability=[0.25, 0.1],
                on_mutation=on_mutation, on_crossover=on_crossover,
                callback_generation=on_generation,
                save_best_solutions=True, save_solutions=True,
                num_generations=3, num_parents_mating = int(len(initial_population) / 3)
                )
    log("---------------------------------")
    log("Run Begins")
    ga.run()
    ga.save(join("ga", "ga_save"))

ga = pygad.load(join("ga", "ga_save"))
ga.plot_fitness(save_dir=join("ga", "fitness"))
print("best solutions: ", ga.best_solutions)
log("---------------------------------")
log("Run Completed")