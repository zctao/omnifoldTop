import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import hist

import modelUtils
from modelUtils import get_model, get_callbacks, train_model
import histogramming as myhu
import plotter

import logging

logger = logging.getLogger("nnreweighter")

def check_weights(w, nevents=None):
    if w is None:
        if nevents is None:
            raise RuntimeError(f"Number of events is not specified")
        else:
            w = np.ones(shape=(modelUtils.n_models_in_parallel, nevents))
    elif np.asarray(w).ndim == 1:
        w = [w] * modelUtils.n_models_in_parallel
    elif len(w) != modelUtils.n_models_in_parallel:
        raise RuntimeError(f"The number of weight arrays provided in the list ({len(w)}) is inconsistent with the number of parallel models ({modelUtils.n_models_in_parallel})")

    return np.asarray(w)

def predict(model, events, batch_size):
    events_list = [events for _ in range(modelUtils.n_models_in_parallel)]
    preds = model.predict(events_list, batch_size=batch_size)
    preds = np.squeeze(preds)

    if modelUtils.n_models_in_parallel == 1:
        preds = np.reshape(preds, (modelUtils.n_models_in_parallel,)+np.shape(preds)) # happens after squeezing, so that we keep the first dimension

    return preds

def reweight(preds):
    return np.nan_to_num( preds / (1. - preds))

def _train_impl(
    # Inputs
    X_1,
    w_1,
    X_0,
    w_0,
    X_b = None,
    w_b = None,
    # model
    model_type = 'dense_100x3',
    skip_training = False,
    model_filepath_load = None,
    model_filepath_save = None,
    # Training
    resume_training = False,
    batch_size = 20000,
    epochs = 100
    ):

    # set up network
    classifier = get_model(model_type=model_type, input_shape=X_0.shape[1:], nclass=2)

    if model_filepath_load:
        try:
            classifier.load_weights(model_filepath_load)
            logger.info(f"Load model from {model_filepath_load}")
        except:
            if resume_training:
                logger.debug(f"Cannot load model from {model_filepath_load}. Continue to train models from here.")
                skip_training = False
            else:
                logger.critical(f"Cannot load model from {model_filepath_load}")
                raise RuntimeError("Model loading failure")

    if skip_training:
        logger.info("Use trained model for reweighting")

    else:
        logger.info("Start training")

        if X_b is None:
            X_train = np.concatenate([X_0, X_1])
            Y_train = np.concatenate([np.zeros(len(X_0)), np.ones(len(X_1))])
            w_train = np.concatenate([w_0, w_1], axis=1)
        else:
            X_train = np.concatenate([X_0, X_1, X_b])
            Y_train = np.concatenate([np.zeros(len(X_0)), np.ones(len(X_1)+len(X_b))])
            w_train = np.concatenate([w_0, w_1, -1*w_b], axis=1)

        train_model(
            classifier,
            X_train, Y_train, w_train,
            batch_size = batch_size, epochs = epochs,
            callbacks = get_callbacks(model_filepath_save),
            )

        logger.info("Training done")

        if model_filepath_save:
            logger.info("Save model to " + model_filepath_save)
            classifier.save_weights(model_filepath_save)

    return classifier

def _reweight_impl(
    classifier,
    X_0,
    w_0 = None,
    X_1 = None,
    w_1 = None,
    X_b = None,
    w_b = None,
    X_pred = None,
    batch_size = 20000,
    calibrate = False,
    hists_calib_0 = [],
    hists_calib_1 = [],
    ):
    logger.info("Reweight")

    if X_pred is None:
        preds = predict(classifier, X_0, batch_size)
    else:
        preds = predict(classifier, X_pred, batch_size)

    if calibrate:
        if not hists_calib_0 or not hists_calib_1:
            raise RuntimeError("Reweight with calibration but no histogram is initialized")

        if X_1 is None or w_0 is None or w_1 is None:
            raise RuntimeError("Reweight with calibration but either target array or weights are provided")

        preds_1 = predict(classifier, X_1, batch_size)
        preds_b = predict(classifier, X_b, batch_size) if X_b is not None else None
        preds_0 = preds if X_pred is None else predict(classifier, X_0, batch_size)

        for i in range(modelUtils.n_models_in_parallel):
            hists_calib_1[i].fill(preds_1[i], weight=w_1[i])

            if preds_b is not None:
                hists_calib_1[i].fill(preds_b[i], weight=-1*w_b[i])

            hists_calib_0[i].fill(preds_0[i], weight=w_0[i])

    return preds

def train_and_reweight(
    # Inputs
    X_target, 
    X_source,
    w_target = None,
    w_source = None,
    X_bkg = None,
    w_bkg = None,
    # Array used for prediction only. If None, use X_source
    X_pred = None,
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
    plot=False,
    ax_input_ratio=None
    ):

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # event weights
    w_target = check_weights(w_target, nevents = len(X_target))
    w_source = check_weights(w_source, nevents = len(X_source))

    # background to be subtracted from target
    if X_bkg is not None:
        # shuffle background
        X_bkg, w_bkg = shuffle(X_bkg, w_bkg)
        w_bkg = check_weights(w_bkg, nevents = len(X_bkg))

    # plot input variable ratios
    if plot and ax_input_ratio is not None:
        plotter.draw_inputs_ratio(ax_input_ratio, X_source, w_source, X_target, w_target, X_bkg, w_bkg)

    # model outputs
    npreds = len(X_source) if X_pred is None else len(X_pred)
    preds_out = np.empty(shape=(modelUtils.n_models_in_parallel, npreds))

    # calibrator
    if calibrate:
        nbins = 10 + int(len(X_source) ** (1. / 3.))
        logger.debug(f"calibrator nbins = {nbins}")
        histogram_1 = [ hist.Hist(hist.axis.Regular(nbins, 0., 1.), storage=hist.storage.Weight()) for _ in range(modelUtils.n_models_in_parallel) ]
        histogram_0 = [ hist.Hist(hist.axis.Regular(nbins, 0., 1.), storage=hist.storage.Weight()) for _ in range(modelUtils.n_models_in_parallel) ]
    else:
        histogram_1, histogram_0 = [], []

    if nsplit_cv < 2 or X_pred is not None:
        # assume X_source and X_pred are separate arrays, therefore no need for cross validation

        classifier = _train_impl(
            X_target, w_target,
            X_source, w_source,
            X_bkg, w_bkg,
            model_type = model_type,
            skip_training = skip_training,
            model_filepath_load = model_filepath_load,
            model_filepath_save = model_filepath_save,
            resume_training = resume_training,
            batch_size = batch_size,
            epochs = epochs
        )

        preds_out[:] = _reweight_impl(
            classifier,
            X_source, w_source,
            X_target, w_target,
            X_bkg, w_bkg,
            X_pred = X_pred,
            batch_size = batch_size,
            calibrate = calibrate,
            hists_calib_0 = histogram_0,
            hists_calib_1 = histogram_1
        )

    else:
        # K-fold cross validation
        logger.debug(f"nsplit_cv = {nsplit_cv}")
        kf = KFold(nsplit_cv)

        i_source_gen = kf.split(X_source)
        i_target_gen = kf.split(X_target)
        i_bkg_gen = kf.split(X_bkg) if X_bkg is not None else None

        if i_bkg_gen is None:
            index_gen_zip = zip(i_source_gen, i_target_gen)
        else:
            index_gen_zip = zip(i_source_gen, i_target_gen, i_bkg_gen)

        for icv, indices in enumerate(index_gen_zip):
            logger.info(f"KFold: {icv}")

            # unpack indices
            indices_train_source, indices_test_source = indices[0]
            indices_train_target, indices_test_target = indices[1]

            if len(indices) > 2:
                indices_train_bkg, indices_test_bkg = indices[2]
            else:
                indices_train_bkg, indices_test_bkg = None, None

            # model file paths
            model_filepath_load_icv = f"{model_filepath_load}_cv{icv}" if model_filepath_load else None
            model_filepath_save_icv = f"{model_filepath_save}_cv{icv}" if model_filepath_save else None

            classifier = _train_impl(
                X_target[indices_train_target],
                w_target[:, indices_train_target],
                X_source[indices_train_source],
                w_source[:, indices_train_source],
                X_bkg[indices_train_bkg] if indices_train_bkg is not None else None,
                w_bkg[:, indices_train_bkg] if indices_train_bkg is not None else None,
                model_type = model_type,
                skip_training = skip_training,
                model_filepath_load = model_filepath_load_icv,
                model_filepath_save = model_filepath_save_icv,
                resume_training = resume_training,
                batch_size = batch_size,
                epochs = epochs
            )

            preds_out[:, indices_test_source] = _reweight_impl(
                classifier,
                X_source[indices_test_source],
                w_source[:,indices_test_source],
                X_target[indices_test_target],
                w_target[:,indices_test_target],
                X_bkg[indices_test_bkg] if indices_test_bkg is not None else None,
                w_bkg[:, indices_test_bkg] if indices_test_bkg is not None else None,
                X_pred = X_pred,
                batch_size = batch_size,
                calibrate = calibrate,
                hists_calib_0 = histogram_0,
                hists_calib_1 = histogram_1
            )

        # end of cross validation loop

    if calibrate:
        rw = np.empty_like(preds_out)
        for i in range(modelUtils.n_models_in_parallel):
            histogram_r = myhu.divide(histogram_1[i], histogram_0[i])
            rw[i] = myhu.read_histogram_at_locations(preds_out[i], histogram_r)

            if plot and model_filepath_save:
                plotter.plot_histograms_and_ratios(
                    figname = f"{model_filepath_save}_preds_{i}",
                    hists_numerator = [histogram_1[i]],
                    hist_denominator = histogram_0[i],
                    draw_options_numerator = [{'label':'Target'}],
                    draw_option_denominator = {'label':'Source'},
                    xlabel = 'NN Output',
                    ylabel_ratio = 'Target / Source'
                )
    else:
        # direct reweighting
        rw = reweight(preds_out)

    if plot and model_filepath_save:
        for i in range(modelUtils.n_models_in_parallel):
            figname_r = f"{model_filepath_save}_lr_{i}"
            logger.info(f"Plot likelihood ratio distribution {figname_r}")
            plotter.plot_LR_distr(figname_r, [rw[i]])

    return rw