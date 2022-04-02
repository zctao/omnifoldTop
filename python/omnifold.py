import os
import numpy as np

import plotter
from modelUtils import get_model, get_callbacks, train_model

import logging
logger = logging.getLogger('omnifold')
logger.setLevel(logging.DEBUG)

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
    preds = model.predict(events, batch_size=int(0.01*len(events)))[:,1]
    r = np.nan_to_num( preds / (1. - preds) )

    if figname: # plot the distribution
        logger.info(f"Plot likelihood ratio distribution {figname}")
        plotter.plot_LR_distr(figname, [r])

    return r

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

    logger.debug(f"Size of the feature array for step 1: {X_step1.nbytes*2**-20:.3f} MB")
    logger.debug(f"Size of the label array for step 1: {Y_step1.nbytes*2**-20:.3f} MB")

    # Step 1b
    if np.any(~passcut_sim):
        X_step1b = np.concatenate([ X_gen[passcut_sim & passcut_gen], X_gen[passcut_sim & passcut_gen] ])
        Y_step1b = np.concatenate([ np.ones(len(X_gen[passcut_sim & passcut_gen])), np.zeros(len(X_gen[passcut_sim & passcut_gen])) ])

        logger.debug(f"Size of the feature array for step 1b: {X_step1b.nbytes*2**-20:.3f} MB")
        logger.debug(f"Size of the label array for step 1b: {Y_step1b.nbytes*2**-20:.3f} MB")

    # Step 2
    # features
    X_step2 = np.concatenate([ X_gen[passcut_gen], X_gen[passcut_gen] ])
    # labels
    Y_step2 = np.concatenate([ np.ones(len(X_gen[passcut_gen])), np.zeros(len(X_gen[passcut_gen])) ])

    logger.debug(f"Size of the feature array for step 2: {X_step2.nbytes*2**-20:.3f} MB")
    logger.debug(f"Size of the label array for step 2: {Y_step2.nbytes*2**-20:.3f} MB")

    # Step 2b
    if np.any(~passcut_gen):
        X_step2b = np.concatenate([ X_sim[passcut_sim & passcut_gen], X_sim[passcut_sim & passcut_gen] ])
        Y_step2b = np.concatenate([ np.ones(len(X_sim[passcut_sim & passcut_gen])), np.zeros(len(X_sim[passcut_sim & passcut_gen])) ])

        logger.debug(f"Size of the feature array for step 2b: {X_step2b.nbytes*2**-20:.3f} MB")
        logger.debug(f"Size of the label array for step 2b: {Y_step2b.nbytes*2**-20:.3f} MB")

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
            w_step1 = np.concatenate([
                w_data[passcut_data], (weights_push*w_sim)[passcut_sim]
                ])

            logger.info("Start training")
            fname_preds = save_models_to + f"/preds_step1_{i}" if save_models_to and plot else ''
            train_model(model_step1, X_step1, Y_step1, w_step1,
                        callbacks = cb_step1,
                        #figname = fname_preds,
                        **fitargs)
            logger.info("Training done")

        # reweight
        logger.info("Reweight")
        fname_rdistr = save_models_to + f"/rdistr_step1_{i}" if save_models_to and plot else ''
        weights_pull = weights_push * reweight(model_step1, X_sim, fname_rdistr)

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
                w_step1b = np.concatenate([
                    (weights_pull*w_gen)[passcut_sim & passcut_gen],
                    w_gen[passcut_sim & passcut_gen]
                    ])

                logger.info("Start training")
                train_model(model_step1b, X_step1b, Y_step1b, w_step1b,
                            callbacks = cb_step1b, **fitargs)
                logger.info("Training done")

            # reweight
            logger.info("Reweight")
            fname_rdistr = save_models_to + f"/rdistr_step1b_{i}" if save_models_to and plot else ''
            weights_pull[~passcut_sim] = reweight(model_step1b, X_gen[~passcut_sim], fname_rdistr)

        # TODO: check this
        weights_pull /= np.mean(weights_pull)

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
            w_step2 = np.concatenate([
                (weights_pull*w_gen)[passcut_gen], w_gen[passcut_gen]*rw_step2
                ])

            logger.info("Start training")
            fname_preds = save_models_to + f"/preds_step2_{i}" if save_models_to and plot else ''
            train_model(model_step2, X_step2, Y_step2, w_step2,
                        callbacks = cb_step2,
                        #figname = fname_preds,
                        **fitargs)
            logger.info("Training done")

        # reweight
        logger.info("Reweight")
        fname_rdistr = save_models_to + f"/rdistr_step2_{i}" if save_models_to and plot else ''
        weights_push[passcut_gen] = rw_step2 * reweight(model_step2, X_gen[passcut_gen], fname_rdistr)

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
                w_step2b = np.concatenate([
                    (weights_push*w_sim)[passcut_sim & passcut_gen],
                    w_sim[passcut_sim & passcut_gen]
                    ])

                logger.info("Start training")
                train_model(model_step2b, X_step2b, Y_step2b, w_step2b,
                            callbacks = cb_step2b, **fitargs)
                logger.info("Training done")

            # reweight
            logger.info("Reweight")
            fname_rdistr = save_models_to + f"/rdistr_step2b_{i}" if save_models_to and plot else ''
            weights_push[~passcut_gen] = reweight(model_step2b, X_sim[~passcut_gen], fname_rdistr)

        # TODO: check this
        weights_push /= np.mean(weights_push)

        # save truth level weights of this iteration
        weights_unfold[i,:] = weights_push[passcut_gen]
    # end of iteration loop

    return weights_unfold

