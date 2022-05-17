import os
import numpy as np

import plotter
from modelUtils import get_model, get_callbacks, train_model

from util import reportGPUMemUsage

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
    mname = name_prefix + "_{}".format(iteration)

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
            mname_prev = name_prefix+"_{}".format(iteration-1)
            filepath_load = os.path.join(save_models_to, mname_prev)
            model.load_weights(filepath_load)
            logger.debug(f"Initialize model from {filepath_load}")

    return model, callbacks

def reweight(model, events, figname=None):
    preds = model.predict(events, batch_size=int(0.01 * len(events)))[:,1]
    r = np.nan_to_num( preds / (1. - preds) )

    if figname: # plot the distribution
        logger.info(f"Plot likelihood ratio distribution {figname}")
        plotter.plot_LR_distr(figname, [r])

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
    **fitargs
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
    weights_push = np.ones(len(X_sim))
    weights_pull = np.ones(len(X_gen))

    weights_unfold = np.empty(shape=(niterations, len(X_gen[passcut_gen])))
    # shape: (n_iterations, n_events[passcut_gen])

    reportGPUMemUsage(logger)

    for i in range(niterations):
        logger.info(f"Iteration {i}")
        # #####
        # # step 1: reweight to sim to data

        logger.info("Step 1")
        w_step1 = w_step(w_data, weights_push * w_sim, passcut_data, passcut_sim) if not load_models_from else None
        print(weights_pull[0:20])
        train_step(X_step1, Y_step1, w_step1, weights_pull, weights_push, None, X_sim, i, "step1", 
            model_type, save_models_to, load_models_from, start_from_previous_iter, plot, **fitargs)
        print(weights_pull[0:20])

        #####
        # step 1b: deal with events that do not pass reco cuts

        if np.any(~passcut_sim):
            logger.info("Step 1b")
            print(weights_pull[0:20])
            w_step1b = w_step(weights_pull * w_gen, w_gen, passcut_sim & passcut_gen, passcut_sim & passcut_gen) if not load_models_from else None
            train_step(X_step1b, Y_step1b, w_step1b, weights_pull, None, ~passcut_sim, X_gen[~passcut_sim], i, "step1b", 
                model_type, save_models_to, load_models_from, start_from_previous_iter, plot, **fitargs)
            print(weights_pull[0:20])

        # TODO: check this
        weights_pull /= np.mean(weights_pull)

        reportGPUMemUsage(logger)

        #####
        # step 2

        logger.info("Step 2")
        rw_step2 = 1. # always reweight against the prior
# #        rw_step2 = 1. if i==0 else weights_unfold[i-1] # previous iteration
        w_step2 = w_step(weights_pull * w_gen, w_gen, passcut_gen, passcut_gen, reweight_second = rw_step2) if not load_models_from else None # rw_step2 currently set to 1.
        print(weights_push[0:20])
        train_step(X_step2, Y_step2, w_step2, weights_push, None, passcut_gen, X_gen[passcut_gen], i, "step2", 
            model_type, save_models_to, load_models_from, start_from_previous_iter, plot, **fitargs)
        weights_push[passcut_gen] *= rw_step2
        print(weights_push[0:20])

        #####
        # step 2b: deal with events that do not pass truth cuts

        if np.any(~passcut_gen):
            logger.info("Step 2b")
            print(weights_pull[0:20])
            w_step2b = w_step(weights_push * w_sim, w_sim, passcut_sim & passcut_gen, passcut_sim & passcut_gen) if not load_models_from else None
            train_step(X_step2b, Y_step2b, w_step2b, weights_push, None, ~passcut_gen, X_sim[~passcut_gen], i, "step2b", 
                model_type, save_models_to, load_models_from, start_from_previous_iter, plot, **fitargs)
            print(weights_pull[0:20])

        # TODO: check this
        weights_push /= np.mean(weights_push)

        # save truth level weights of this iteration
        weights_unfold[i,:] = weights_push[passcut_gen]

        reportGPUMemUsage(logger)

    # end of iteration loop

    return weights_unfold

# evaluate a step in OmniFold method, training a model if needed and train the model
# weights_push is modified for reweighting
# Return: None
def train_step(
    # Data
    X_step, # feature array of observed data, for this step
    Y_step, # label array, for this step
    w_step, # w_array, for this step
    weights_update, # arraylike, weights_pull or weights_push to be updated
    weights_multiplier, # arraylike, weights_pull or weights_push to be multiplied to be reweight factor, weights_update becomes reweight if None is supplied
    weights_update_range, # range of weights_update to be updated, the entire weights_update_range is updated if NONE is supplied
    weights_update_events, # arraylike, events used for reweighting weights_update
    # Parameters
    iteration, # the number of iterations up to this call
    name, # str, name of the model
    model_type, # name of the model type 
    save_models_to, # directory to save models to
    load_models_from, # directory to load trained models
    start_from_previous_iter, # If True, initialize model with the previous iteration
    plot, # If True, plot training history and make other status plots
    **fitargs
    ):
    model, cb = set_up_model(model_type, X_step.shape[1:], iteration, "model_{0}".format(name), save_models_to, load_models_from, start_from_previous_iter)
    logger.debug("mode_{0}".format(name))

    if load_models_from:
        logger.info("Use trained model for reweighting")
    else: # train model
        logger.info("Start training")
        train_model(model, X_step, Y_step, w_step,callbacks = cb, **fitargs)
        logger.info("Training done")

    # reweight
    logger.info("Reweight")
    fname_rdistr = (save_models_to + f"/rdistr_{0}_{1}".format(name, iteration)) if save_models_to and plot else ''
    if weights_update_range is not None:
        if weights_multiplier is not None:
            weights_update[weights_update_range] = weights_multiplier[weights_update_range] *  reweight(model, weights_update_events, fname_rdistr) # TODO
        else:
            weights_update[weights_update_range] = reweight(model, weights_update_events, fname_rdistr)
    else:
        if weights_multiplier is not None:
            weights_update = weights_multiplier *  reweight(model, weights_update_events, fname_rdistr)
        else:
            weights_update = reweight(model, weights_update_events, fname_rdistr)


# generate the weights for a step in OmniFold
# Return: arraylike weight from combination of w_first[range_first] and w_second[range_second]
def w_step(
    w_first, # first part of the final combined w
    w_second, # second part of the final combined w
    range_first, # array specifying the range of data for first part
    range_second, # array specifying the range of data for second part
    reweight_first = 1., # if reweighting against the prior, always do if none
    reweight_second =  1., # if reweighting against the prior, always do if none
    ):
    return np.concatenate([
        reweight_first * w_first[range_first], reweight_second * w_second[range_second]
        ])
