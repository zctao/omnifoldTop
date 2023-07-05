import numpy as np
from sklearn.model_selection import KFold
import hist

from modelUtils import n_models_in_parallel, get_model, get_callbacks, train_model
import histogramming as myhu
import plotter

import logging

logger = logging.getLogger("nnreweighter")

def check_weights(w, nevents=None):
    if w is None:
        if nevents is None:
            raise RuntimeError(f"Number of events is not specified")
        else:
            w = [np.ones(nevents)] * n_models_in_parallel
    elif not isinstance(w, list):
        w = [w] * n_models_in_parallel
    elif len(w) != n_models_in_parallel:
        raise RuntimeError(f"The number of weight arrays provided in the list ({len(w)}) is inconsistent with the number of parallel models ({n_models_in_parallel})")

    return w

def set_up_model(
    model_type, # str, type of the network
    input_shape,
    model_filepath_load = None, # str, filepath to load trained model weights
    resume_training = False, # bool
    ):

    model = get_model(input_shape=input_shape, nclass=2, model_type=model_type)

    # load model
    if model_filepath_load:
        try:
            model.load_weights(model_filepath_load)
            logger.info(f"Load model from {model_filepath_load}")
        except:
            if resume_training:
                logger.debug(f"Cannot load model from {model_filepath_load}. Continue to train models from here.")
            else:
                logger.critical(f"Cannot load model from {model_filepath_load}")
                raise RuntimeError("Model loading failure")

    return model

def predict(model, events, batch_size):
    events_list = [events for _ in range(n_models_in_parallel)]
    preds = model.predict(events_list, batch_size=batch_size)
    preds = np.squeeze(preds)

    if n_models_in_parallel == 1:
        preds = np.reshape(preds, (n_models_in_parallel,)+np.shape(preds)) # happens after squeezing, so that we keep the first dimension

    return preds

def reweight(preds):
    return np.nan_to_num( preds / (1. - preds))

def train_and_reweight(
    # Inputs
    X_target, 
    X_source,
    w_target = None,
    w_source = None,
    # model
    model_type = 'dense_100x3',
    skip_training = False,
    model_filepath_load = None,
    model_filepath_save = None,
    # Training
    resume_training = False,
    nsplit_cv = 2,
    batch_size = 20000,
    epochs = 100,
    # Reweight
    calibrate=False,
    # Logging
    verbose=False,
    # plot
    plot=False
    ):

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # event weights
    w_target = check_weights(w_target, nevents = len(X_target))
    w_source = check_weights(w_source, nevents = len(X_source))

    # labels
    Y_target = np.ones(len(X_target))
    Y_source = np.zeros(len(X_source))

    # model outputs
    preds_source = np.empty(shape=(n_models_in_parallel, len(X_source)))

    # calibrator
    if calibrate:
        nbins = 10 + int(len(X_source) ** (1. / 3.))
        logger.debug(f"calibrator nbins = {nbins}")
        histogram_1 = [ hist.Hist(hist.axis.Regular(nbins, 0., 1.), storage=hist.storage.Weight()) for _ in range(n_models_in_parallel) ]
        histogram_0 = [ hist.Hist(hist.axis.Regular(nbins, 0., 1.), storage=hist.storage.Weight()) for _ in range(n_models_in_parallel) ]

    # K-fold cross validation
    logger.debug(f"nsplit_cv = {nsplit_cv}")

    kf = KFold(nsplit_cv)

    for icv, ((index_train_source, index_test_source), (index_train_target, index_test_target)) in enumerate(zip(kf.split(X_source), kf.split(X_target))):
        logger.info(f"KFold: {icv}")

        model_filepath_load_icv = f"{model_filepath_load}_cv{icv}" if model_filepath_load else None
        model_filepath_save_icv = f"{model_filepath_save}_cv{icv}" if model_filepath_save else None

        # set up network
        classifier = set_up_model(
            model_type, 
            input_shape = X_source.shape[1:],
            model_filepath_load = model_filepath_load_icv,
            resume_training = resume_training
            )

        if skip_training:
            logger.info("Use trained model for reweighting")

        else:
            logger.info("Start training")

            X_train = np.concatenate([X_source[index_train_source], X_target[index_train_target]])
            Y_train = np.concatenate([Y_source[index_train_source], Y_target[index_train_target]])
            w_train = np.concatenate([np.asarray(w_source)[:,index_train_source], np.asarray(w_target)[:,index_train_target]], axis=1)

            train_model(
                classifier,
                X_train, Y_train, w_train,
                batch_size = batch_size, epochs = epochs,
                callbacks = get_callbacks(model_filepath_save_icv),
                )
            
            logger.info("Training done")

            if model_filepath_save_icv:
                logger.info("Save model to " + model_filepath_save_icv)
                classifier.save_weights(model_filepath_save_icv)

        # reweight
        logger.info("Reweight")

        preds_source[:,index_test_source] = predict(
            classifier, X_source[index_test_source], batch_size=batch_size
            )

        if calibrate:
            preds_test_target = predict(
                classifier, X_target[index_test_target], batch_size=batch_size
                )

            for i in range(n_models_in_parallel):
                histogram_1[i].fill(
                    preds_test_target[i], weight=w_target[i][index_test_target]
                    )

                histogram_0[i].fill(
                    preds_source[i,index_test_source], weight=w_source[i][index_test_source]
                    )

    # end of cross validation loop

    if calibrate:
        rw = np.empty_like(preds_source)
        for i in range(n_models_in_parallel):
            histogram_r = myhu.divide(histogram_1[i], histogram_0[i])
            rw[i] = myhu.read_histogram_at_locations(preds_source[i], histogram_r)

            if plot and model_filepath_save:
                plotter.plot_histograms_and_ratios(
                    figname = f"{model_filepath_save}_preds_r{i}",
                    hists_numerator = [histogram_1[i]],
                    hist_denominator = histogram_0[i],
                    draw_options_numerator = [{'label':'Target'}],
                    draw_option_denominator = {'label':'Source'},
                    xlabel = 'NN Output',
                    ylabel_ratio = 'Target / Source'
                )

        return rw
    else:
        # direct reweighting
        return reweight(preds_source)