import os
import numpy as np

import plotter
from modelUtils import get_model, get_callbacks, train_model, n_models_in_parallel

from util import reportGPUMemUsage

import tensorflow as tf

import logging
logger = logging.getLogger('omnifold')
logger.setLevel(logging.DEBUG)

B_TO_MB = 2**-20 # constant for converting size in bytes to MBs

def set_up_model(
    model_type, # str, type of the network
    input_shape, # tuple, shape of the input layer
    iteration = 0, # int, iteration index
    name_prefix = 'model', # str, prefix of the model name
    save_models_to = '', # str, directory to save the trained model to
    load_models_from = '', # str, directory to load trained model weights
    start_from_previous_iter = False, # bool, if True, initialize model from previous iteration
    ):

    # get network
    model = get_model(input_shape, nclass=2, model_name=model_type)

    # name of the model checkpoint
    mname = name_prefix + "_iter{}".format(iteration)

    # callbacks
    filepath_save = None
    if save_models_to:
        filepath_save = os.path.join(save_models_to, mname)

    callbacks = get_callbacks(filepath_save)
    
    # load trained model if needed
    if load_models_from:
        filepath_load = os.path.join(load_models_from, mname)
        model.load_weights(filepath_load).expect_partial()
        logger.info(f"Load model from {filepath_load}")
    else:
        if start_from_previous_iter and save_models_to and iteration > 0:
            # initialize model weights from the previous iteration
            mname_prev = name_prefix+"_iter{}".format(iteration-1)
            filepath_load = os.path.join(save_models_to, mname_prev)
            model.load_weights(filepath_load)
            logger.debug(f"Initialize model from {filepath_load}")

    return model, callbacks

def reweight(model, events, batch_size, figname=None):
    events_list = [events for i in range(n_models_in_parallel)]
    preds_list = [np.squeeze(pred) for pred in model.predict(events_list, batch_size=batch_size)] if n_models_in_parallel > 1 else model.predict(events_list, batch_size=batch_size)[:,1]
    r = np.array([np.nan_to_num( preds / (1. - preds) ) for preds in preds_list]) if n_models_in_parallel > 1 else np.nan_to_num( preds_list / (1. - preds_list) )

    if figname: # plot the distribution
        for i in range(n_models_in_parallel)
            logger.info(f"Plot likelihood ratio distribution {figname}{i}")
            plotter.plot_LR_distr(figname+str(i), [r[i]])

    return r

# write a logger message regarding the size of an object in number of bytes
def log_size_bytes(
    name, # str, name of the object
    size # type of numpy.ndarray.nbytes, size of object in number of bytes
    ):
    logger.debug(f"Size of the {name}: {size * B_TO_MB:.3f} MB")
    
def omnifold(
    # Data
    X_data, # feature array of observed data
    X_sim, # feature array of signal simulation at reco level
    X_gen, # feature array of signal simulation at truth level
    w_data, # event weights of observed data
    w_sim, # reco weights of signal simulation events
    w_gen, # MC weights of signal simulation events
    # Event selection flags
    passcut_data, # flags to indicate if data events pass reco level selections
    passcut_sim, # flags to indicate if signal events pass reco level selections
    passcut_gen, # flags to indicate if signal events pass truth level selections
    # Parameters
    niterations, # number of iterations
    model_type='dense_100x3', # name of the model type 
    save_models_to='', # directory to save models to if provided
    load_models_from='', # directory to load trained models if provided
    start_from_previous_iter=False, # If True, initialize model with the previous iteration
    plot=False, # If True, plot training history and make other status plots
    # Model training parameters
    batch_size=256,
    epochs=100,
    verbose=1
    ):
    """
    OmniFold
    arXiv:1911.09107, arXiv:2105.04448

    """
    ################
    # Prepare data arrays for training
    assert(len(X_data)==len(w_data))
    assert(len(X_sim)==len(w_sim))
    assert(len(X_gen)==len(w_gen))
    assert(len(X_sim)==len(passcut_sim))
    assert(len(X_gen)==len(passcut_gen))

    # Expand the weights arrays
    w_data = np.array([w_data for i in range(n_models_in_parallel)])
    w_sim = np.array([w_sim for i in range(n_models_in_parallel)])
    w_gen = np.array([w_gen for i in range(n_models_in_parallel)])

    # Step 1
    # Use events that pass reco level selections
    # features
    X_step1 = np.concatenate([ X_data[passcut_data], X_sim[passcut_sim] ])
    # labels: data=1, sim=0
    Y_step1 = np.concatenate([ np.ones(len(X_data[passcut_data])), np.zeros(len(X_sim[passcut_sim])) ])

    log_size_bytes("feature array for step 1", X_step1.nbytes)
    log_size_bytes("label array for step 1", Y_step1.nbytes)

    # Step 1b
    if np.any(~passcut_sim):
        X_step1b = np.concatenate([ X_gen[passcut_sim & passcut_gen], X_gen[passcut_sim & passcut_gen] ])
        Y_step1b = np.concatenate([ np.ones(len(X_gen[passcut_sim & passcut_gen])), np.zeros(len(X_gen[passcut_sim & passcut_gen])) ])

        log_size_bytes("feature array for step 1b", X_step1b.nbytes)
        log_size_bytes("label array for step 1b", Y_step1b.nbytes)

    # Step 2
    # features
    X_step2 = np.concatenate([ X_gen[passcut_gen], X_gen[passcut_gen] ])
    # labels
    Y_step2 = np.concatenate([ np.ones(len(X_gen[passcut_gen])), np.zeros(len(X_gen[passcut_gen])) ])

    log_size_bytes("feature array for step 2", X_step2.nbytes)
    log_size_bytes("label array for step 2", Y_step2.nbytes)

    # Step 2b
    if np.any(~passcut_gen):
        X_step2b = np.concatenate([ X_sim[passcut_sim & passcut_gen], X_sim[passcut_sim & passcut_gen] ])
        Y_step2b = np.concatenate([ np.ones(len(X_sim[passcut_sim & passcut_gen])), np.zeros(len(X_sim[passcut_sim & passcut_gen])) ])

        log_size_bytes("feature array for step 2b", X_step2b.nbytes)
        log_size_bytes("label array for step 2b", Y_step2b.nbytes)

    ################
    # Prepare models
    if save_models_to and not os.path.isdir(save_models_to):
        logger.info("Make directory for saving models")
        os.makedirs(save_models_to)

    if load_models_from and not os.path.isdir(load_models_from):
        raise RuntimeError(f"Cannot load models from {load_models_from}: directory does not exist!")

    ################
    # Start iterations
    # Weights
    weights_push = [np.ones(len(X_sim)) for i in range(n_models_in_parallel)]
    weights_pull = [np.ones(len(X_gen)) for i in range(n_models_in_parallel)]

    weights_unfold = np.empty(shape=(n_models_in_parallel, niterations, len(X_gen[passcut_gen])))
    # shape: (n_models_in_parallel, n_iterations, n_events[passcut_gen])

    reportGPUMemUsage(logger)

    for i in range(niterations):
        logger.info(f"Iteration {i}")
        #####
        # step 1: reweight to sim to data
        logger.info("Step 1")
        # set up the model
        model_step1, cb_step1 = set_up_model(
            model_type, X_step1.shape[1:], i, "model_step1",
            save_models_to, load_models_from, start_from_previous_iter)

        if load_models_from:
            logger.info("Use trained model for reweighting")
        else: # train model
            w_step1 = [np.concatenate([
                w_data[passcut_data], (weights_push*w_sim)[passcut_sim]
                ]) for i in range(n_models_in_parallel)]

            logger.info("Start training")
            fname_preds = save_models_to + f"/preds_step1_{i}" if save_models_to and plot else ''
            train_model(model_step1, X_step1, Y_step1, w_step1,
                        callbacks = cb_step1,
                        #figname = fname_preds,
                        batch_size=batch_size, epochs=epochs, verbose=verbose
                        )
            logger.info("Training done")

        # reweight
        logger.info("Reweight")
        fname_rdistr = save_models_to + f"/rdistr_step1_{i}" if save_models_to and plot else ''
        weights_pull = weights_push * reweight(model_step1, X_sim, batch_size, fname_rdistr) if n_models_in_parallel>1 else weights_push * np.array([reweight(model_step1, X_sim, batch_size, fname_rdistr)])

        #####
        # step 1b: deal with events that do not pass reco cuts
        if np.any(~passcut_sim):
            logger.info("Step 1b")
            # weights_pull[~passcut_sim] = 1.
            # Or alternatively, estimate the average weights: <w|x_true>
            model_step1b, cb_step1b = set_up_model(
                model_type, X_step1b.shape[1:], i, "model_step1b",
                save_models_to, load_models_from, start_from_previous_iter)

            if load_models_from:
                logger.info("Use trained model for reweighting")
            else: # train model
                w_step1b = [np.concatenate([
                    (weights_pull*w_gen)[passcut_sim & passcut_gen],
                    w_gen[passcut_sim & passcut_gen]
                    ]) for i in range(n_models_in_parallel)]

                logger.info("Start training")
                train_model(model_step1b, X_step1b, Y_step1b, w_step1b,
                            callbacks = cb_step1b, batch_size=batch_size,
                            epochs=epochs, verbose=verbose)
                logger.info("Training done")

            # reweight
            logger.info("Reweight")
            fname_rdistr = save_models_to + f"/rdistr_step1b_{i}" if save_models_to and plot else ''
            step_reweight = reweight(model_step1b, X_gen[~passcut_sim], batch_size, fname_rdistr)
            step_reweight = np.array([step_reweight]) if n_models_in_parallel == 1 else step_reweight
            for j in range(n_models_in_parallel):
                weights_pull[j][~passcut_sim] = step_reweight[j]

        # TODO: check this
        weights_pull /= np.mean(weights_pull)

        reportGPUMemUsage(logger)

        #####
        # step 2
        logger.info("Step 2")
        model_step2, cb_step2 = set_up_model(
            model_type, X_step2.shape[1:], i, "model_step2",
            save_models_to, load_models_from, start_from_previous_iter)

        rw_step2 = 1. # always reweight against the prior
#        rw_step2 = 1. if i==0 else weights_unfold[i-1] # previous iteration

        if load_models_from:
            logger.info("Use trained model for reweighting")
        else: # train model
            w_step2 = [np.concatenate([
                (weights_pull*w_gen)[passcut_gen], w_gen[passcut_gen]*rw_step2
                ]) for i in range(n_models_in_parallel)]

            logger.info("Start training")
            fname_preds = save_models_to + f"/preds_step2_{i}" if save_models_to and plot else ''
            train_model(model_step2, X_step2, Y_step2, w_step2,
                        callbacks = cb_step2,
                        #figname = fname_preds,
                        batch_size=batch_size, epochs=epochs, verbose=verbose)
            logger.info("Training done")

        # reweight
        logger.info("Reweight")
        fname_rdistr = save_models_to + f"/rdistr_step2_{i}" if save_models_to and plot else ''
        step_reweight = rw_step2 * reweight(model_step2, X_gen[passcut_gen], batch_size, fname_rdistr)
        step_reweight = np.array([step_reweight]) if n_models_in_parallel == 1 else step_reweight
        for j in range(n_models_in_parallel):
            weights_push[j][passcut_gen] = step_reweight[j]

        #####
        # step 2b: deal with events that do not pass truth cuts
        if np.any(~passcut_gen):
            logger.info("Step 2b")
            # weights_push[~passcut_gen] = 1.
            # Or alternatively, estimate the average weights: <w|x_reco>
            model_step2b, cb_step2b = set_up_model(
                model_type, X_step2b.shape[1:], i, "model_step2b",
                save_models_to, load_models_from, start_from_previous_iter)

            if load_models_from:
                logger.info("Use trained model for reweighting")
            else: # train model
                w_step2b = [np.concatenate([
                    (weights_push*w_sim)[passcut_sim & passcut_gen],
                    w_sim[passcut_sim & passcut_gen]
                    ]) for i in range(n_models_in_parallel)]

                logger.info("Start training")
                train_model(model_step2b, X_step2b, Y_step2b, w_step2b,
                            callbacks = cb_step2b, batch_size=batch_size,
                            epochs=epochs, verbose=verbose)
                logger.info("Training done")

            # reweight
            logger.info("Reweight")
            fname_rdistr = save_models_to + f"/rdistr_step2b_{i}" if save_models_to and plot else ''
            step_reweight = reweight(model_step2b, X_sim[~passcut_gen], batch_size, fname_rdistr)
            step_reweight = np.array([step_reweight]) if n_models_in_parallel == 1 else step_reweight
            for j in range(n_models_in_parallel):
                weights_push[j][~passcut_gen] = step_reweight[j]

        # TODO: check this
        weights_push /= np.mean(weights_push)

        for j in range(n_models_in_parallel):
        # save truth level weights of this iteration
            weights_unfold[j][i,:] = weights_push[j][passcut_gen]

        reportGPUMemUsage(logger)

    # end of iteration loop

    return weights_unfold
