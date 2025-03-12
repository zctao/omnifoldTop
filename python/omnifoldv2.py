import os
import numpy as np
import matplotlib.pyplot as plt

import plotter
import modelUtils
from modelUtils import reportGPUMemUsage
from nnreweighter import train_and_reweight

import gc

import logging
logger = logging.getLogger('omnifold')

B_TO_MB = 2**-20 # constant for converting size in bytes to MBs

# write a logger message regarding the size of an object in number of bytes
def log_size_bytes(
    name, # str, name of the object
    size # type of numpy.ndarray.nbytes, size of object in number of bytes
    ):
    logger.debug(f"Size of the {name}: {size * B_TO_MB:.3f} MB")

def set_model_paths(
    iteration, # int
    step, # int or str
    save_dir='', # str
    load_dir='', # str
    model_name_prefix='model',
    start_from_previous=False
    ):

    model_name = model_name_prefix+"_iter{}_step{}"

    if save_dir:
        fpath_model_save = os.path.join(save_dir, model_name.format(iteration, step))
    else:
        fpath_model_save = None

    if load_dir:
        fpath_model_load = os.path.join(load_dir, model_name.format(iteration, step))
    elif start_from_previous and save_dir and iteration > 0:
        fpath_model_load = os.path.join(load_dir, model_name.format(iteration-1, step))
    else:
        fpath_model_load = None

    return fpath_model_save, fpath_model_load

def omnifold(
    # Data
    X_data, # feature array of observed data
    X_sim, # feature array of signal simulation at reco level
    X_gen, # feature array of signal simulation at truth level
    X_bkg, # feature array of background simulation at reco level
    w_data, # event weights of observed data
    w_sim, # reco weights of signal simulation events
    w_gen, # MC weights of signal simulation events
    w_bkg, # reco weights of background simulation events
    # Event selection flags
    passcut_sim, # flags to indicate if signal events pass reco level selections
    passcut_gen, # flags to indicate if signal events pass truth level selections
    # Parameters
    niterations, # number of iterations
    model_type='dense_100x3', # name of the model type 
    save_models_to='', # directory to save models to if provided
    load_models_from='', # directory to load trained models if provided
    continue_training=False, # If True, continue to train even if model loading from load_models_from fails
    start_from_previous_iter=False, # If True, initialize model with the previous iteration
    fast_correction=False, # If True, assign weights of 1 to events
    plot=False, # If True, plot training history and make other status plots
    feature_names_sim=None,
    feature_names_gen=None,
    # Model training parameters
    batch_size=256,
    epochs=100,
    verbose=1,
    output_dataset=None, # hdf5 dataset for storing the event weights
    output_dataset_reco=None, # hdf5 dataset for storing the reco level event weights
    run_index=0
    ):
    """
    OmniFold
    arXiv:1911.09107, arXiv:2105.04448

    """
    # Logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    ################
    # Prepare data arrays for training
    assert(len(X_data)==len(w_data))
    assert(len(X_sim)==len(w_sim))
    assert(len(X_gen)==len(w_gen))
    assert(len(X_sim)==len(passcut_sim))
    assert(len(X_gen)==len(passcut_gen))
    if X_bkg is not None:
        assert(len(X_bkg)==len(w_bkg))

    ################
    # Prepare models
    if save_models_to and not os.path.isdir(save_models_to):
        logger.info("Make directory for saving models")
        os.makedirs(save_models_to)

    if load_models_from and not os.path.isdir(load_models_from):
        raise RuntimeError(f"Cannot load models from {load_models_from}: directory does not exist!")

    ################
    # Plots
    if plot:
        # Initialize input ratio plots
        if feature_names_sim is None:
            feature_names_sim = [''] * X_sim.shape[1]
        fig_step1, axes_step1 = plotter.init_training_input_ratio_plot(
            niterations, feature_names_sim)

        if feature_names_gen is None:
            feature_names_gen = [''] * X_gen.shape[1]
        fig_step2, axes_step2 = plotter.init_training_input_ratio_plot(
            niterations, feature_names_gen)

    ################
    # Common run arguments
    runstep_args = {
        "model_type" : model_type,
        "skip_training" : True if load_models_from else False,
        "resume_training" : continue_training,
        "batch_size" : batch_size,
        "epochs" : epochs,
        "verbose" : verbose,
        "nsplit_cv" : 1,
        "calibrate" : False,
        "clip_weights" : True,
        "plot" : plot
    }

    ################
    # Start iterations
    # Weights
    weights_push = np.ones(shape=(modelUtils.n_models_in_parallel, len(X_sim)), dtype=np.float64)
    weights_pull = np.ones(shape=(modelUtils.n_models_in_parallel, len(X_gen)), dtype=np.float64)

    if output_dataset is None:
        weights_unfold = np.empty(shape=(modelUtils.n_models_in_parallel, niterations, np.count_nonzero(passcut_gen)), dtype=np.float64)
        # shape: (modelUtils.n_models_in_parallel, n_iterations, n_events[passcut_gen])
    #else:
        # write directly to output_dataset[run_index*modelUtils.n_models_in_parallel:(run_index+1)*modelUtils.n_models_in_parallel, i, :]

    if logger.isEnabledFor(logging.DEBUG):
        reportGPUMemUsage(logger)

    for i in range(niterations):
        logger.info(f"Iteration {i}")

        #####
        # step 1: reweight to sim to data
        logger.info("Step 1")
        model_save_1, model_load_1 = set_model_paths(
            i, "1", save_dir = save_models_to, load_dir = load_models_from,
            start_from_previous = start_from_previous_iter)

        weights_pull[:,passcut_sim] = weights_push[:,passcut_sim] * train_and_reweight(
            # Inputs
            X_target = X_data,
            X_source = X_sim[passcut_sim],
            X_bkg = X_bkg,
            w_target = w_data,
            w_source = weights_push[:,passcut_sim] * w_sim[passcut_sim],
            w_bkg = w_bkg,
            # model paths
            model_filepath_load = model_load_1,
            model_filepath_save = model_save_1,
            # other arguments
            ax_input_ratio = axes_step1[i] if plot else None,
            **runstep_args
        )

        gc.collect()

        if logger.isEnabledFor(logging.DEBUG):
            reportGPUMemUsage(logger)

        #####
        # step 1b: deal with events that do not pass reco cuts
        if np.any(~passcut_sim):
            logger.info("Step 1b")
            if fast_correction:
                logger.info("Assign an average weight of one")
                weights_pull[:, ~passcut_sim] = 1.
            else:
                # Estimate the average weights: <w|x_true>
                model_save_1b, model_load_1b = set_model_paths(
                    i, "1b", save_dir = save_models_to, load_dir = load_models_from,
                    start_from_previous = start_from_previous_iter)

                weights_pull[:, ~passcut_sim] = train_and_reweight(
                    # Inputs
                    X_target = X_gen[passcut_sim & passcut_gen],
                    X_source = X_gen[passcut_sim & passcut_gen],
                    w_target = weights_pull[:, passcut_sim & passcut_gen] * w_gen[passcut_sim & passcut_gen],
                    w_source = w_gen[passcut_sim & passcut_gen],
                    X_pred = X_gen[~passcut_sim],
                    # model paths
                    model_filepath_load = model_load_1b,
                    model_filepath_save = model_save_1b,
                    # other arguments
                    **runstep_args
                )

            gc.collect()

        weights_pull /= np.mean(weights_pull, axis=1)[:,None]

        if output_dataset_reco is not None:
            output_dataset_reco[run_index*modelUtils.n_models_in_parallel:(run_index+1)*modelUtils.n_models_in_parallel, i, :] = weights_pull[:, passcut_sim]

        if logger.isEnabledFor(logging.DEBUG):
            reportGPUMemUsage(logger)
        gc.collect()

        #####
        # step 2
        logger.info("Step 2")
        model_save_2, model_load_2 = set_model_paths(
            i, "2", save_dir = save_models_to, load_dir = load_models_from,
            start_from_previous = start_from_previous_iter)

        rw_step2 = 1. # always reweight against the prior
        #rw_step2 = 1. if i==0 else weights_push[:,passcut_gen] # previous iteration

        weights_push[:, passcut_gen] = rw_step2 * train_and_reweight(
            # Inputs
            X_target = X_gen[passcut_gen],
            X_source = X_gen[passcut_gen],
            w_target = weights_pull[:,passcut_gen] * w_gen[passcut_gen],
            w_source = w_gen[passcut_gen] * rw_step2,
            # model paths
            model_filepath_load = model_load_2,
            model_filepath_save = model_save_2,
            # other arguments
            ax_input_ratio = axes_step2[i] if plot else None,
            **runstep_args
        )

        gc.collect()

        if logger.isEnabledFor(logging.DEBUG):
            reportGPUMemUsage(logger)

        #####
        # step 2b: events that do not pass truth cuts
        if np.any(~passcut_gen):
            logger.info("Step 2b")
            if fast_correction:
                logger.info("Assign an average weight of one")
                weights_push[:, ~passcut_gen] = 1.
            else:
                # Estimate the average weights: <w|x_reco>
                model_save_2b, model_load_2b = set_model_paths(
                    i, "2b", save_dir = save_models_to, load_dir = load_models_from,
                    start_from_previous = start_from_previous_iter)

                weights_push[:, ~passcut_gen] = train_and_reweight(
                    # Inputs
                    X_target = X_sim[passcut_sim & passcut_gen],
                    X_source = X_sim[passcut_sim & passcut_gen],
                    w_target = weights_push[:, passcut_sim & passcut_gen] * w_sim[passcut_sim & passcut_gen],
                    w_source = w_sim[passcut_sim & passcut_gen],
                    X_pred = X_sim[~passcut_gen],
                    # model paths
                    model_filepath_load = model_load_2b,
                    model_filepath_save = model_save_2b,
                    # other arguments
                    **runstep_args
                )

            gc.collect()

        weights_push /= np.mean(weights_push, axis=1)[:,None]

        # save truth level weights of this iteration
        if output_dataset is None:
            weights_unfold[:,i,:] = weights_push[:, passcut_gen]
        else:
            output_dataset[run_index*modelUtils.n_models_in_parallel:(run_index+1)*modelUtils.n_models_in_parallel, i, :] = weights_push[:, passcut_gen]

        if logger.isEnabledFor(logging.DEBUG):
            reportGPUMemUsage(logger)

        gc.collect()

    # end of iteration loop

    if plot:
        figname_input1_ratio = os.path.join(save_models_to, "inputs_ratio_step1.png")
        logger.info(f"Plot ratio of step 1 training inputs: {figname_input1_ratio}")

        # legend
        handles1, labels1 = axes_step1.flat[-1].get_legend_handles_labels()
        fig_step1.legend(handles1, labels1, loc="upper right")

        # save
        fig_step1.savefig(figname_input1_ratio)
        plt.close(fig_step1)

        figname_input2_ratio = os.path.join(save_models_to, "inputs_ratio_step2.png")
        logger.info(f"Plot ratio of step 2 training inputs: {figname_input2_ratio}")

        # legend
        handles2, labels2 = axes_step2.flat[-1].get_legend_handles_labels()
        fig_step2.legend(handles2, labels2, loc="upper right")

        # save
        fig_step2.savefig(figname_input2_ratio)
        plt.close(fig_step2)

    if output_dataset is None:
        return weights_unfold