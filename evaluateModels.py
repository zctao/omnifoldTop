#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from model import get_model, get_callbacks

from datahandler import DataHandler, DataToy
from util import configGPUs, expandFilePath, read_dict_from_json
from util import get_bins, write_chi2, write_ks, write_triangular_discriminators
import plotting
import logging

msgfmt = '%(asctime)s %(levelname)-7s %(name)-15s %(message)s'
datefmt = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(format=msgfmt, datefmt=datefmt)
logger = logging.getLogger("EvaluateModels")
logger.setLevel(logging.DEBUG)

def get_training_inputs(variables, dataHandle, simHandle, rw_type=None, vars_dict=None):
    ###
    X_d, Y_d = dataHandle.get_dataset(variables, 1)
    X_s, Y_s = simHandle.get_dataset(variables, 0)

    X = np.concatenate([X_d, X_s])

    Xmean = np.mean(np.abs(X), axis=0)
    Xoom = 10**(np.log10(Xmean).astype(int)) # Order of Magnitude
    X /= Xoom

    ###
    Y = tf.keras.utils.to_categorical(np.concatenate([Y_d, Y_s]))

    ###
    # event weights
    w_d = dataHandle.get_weights(rw_type=rw_type, vars_dict=vars_dict)
    w_s = simHandle.get_weights()

    # normalize data weights to mean of one
    w_d /= np.mean(w_d)
    # scale simulation total weights to data
    w_s *= w_d.sum()/w_s.sum()

    w = np.concatenate([w_d, w_s])

    return X, Y, w

def set_up_model(model_name, input_shape, outputdir):
    model_dir = os.path.join(outputdir, 'Models', model_name)
    if not os.path.isdir(model_dir):
        logger.info("Create directory {}".format(model_dir))
        os.makedirs(model_dir)

    model = get_model(input_shape, nclass=2) # fixme

    # callbacks
    callbacks = get_callbacks(model_dir)

    return model, callbacks

def reweight(model, events):
    preds = model.predict(events, batch_size=int(0.1*len(events)))[:,1]
    r = np.nan_to_num( preds / (1. - preds) )
    return r

def evaluateModels(**parsed_args):
    #################
    # Variables
    #################
    observable_dict = read_dict_from_json(parsed_args['observable_config'])

    logger.info("Features used in training: {}".format(', '.join(parsed_args['observables'])))
    # detector level
    vars_det = [ observable_dict[key]['branch_det'] for key in parsed_args['observables'] ]
    # truth level
    vars_mc = [ observable_dict[key]['branch_mc'] for key in parsed_args['observables'] ]

    # event weights
    wname = parsed_args['weight']

    #################
    # Load data
    #################
    logger.info("Loading data")

    fnames_d = parsed_args['data']
    logger.info("(Pseudo) data files: {}".format(' '.join(fnames_d)))
    dataHandle = DataHandler(fnames_d, wname, variable_names=vars_det+vars_mc)

    fnames_s = parsed_args['signal']
    logger.info("Simulation files: {}".format(' '.join(fnames_s)))
    simHandle = DataHandler(fnames_s, wname, variable_names=vars_det+vars_mc)

    ####
    #dataHandle = DataToy(1000000, 1, 1.5)
    #simHandle = DataToy(1000000, 0, 1)
    #vars_mc = ['x_truth']
    ####

    #################
    # Training data
    #################
    # Training arrays
    # Truth level
    X, Y, w = get_training_inputs(vars_mc, dataHandle, simHandle, rw_type=parsed_args['reweight_data'], vars_dict=observable_dict)

    # Split into training, validation, and test sets: 75%, 15%, 10%
    X_train, X_test, Y_train, Y_test, w_train, w_test = train_test_split(X, Y, w, test_size=0.25)
    X_val, X_test, Y_val, Y_test, w_val, w_test = train_test_split(X_test, Y_test, w_test, test_size=0.4)

    #################
    # Model
    #################
    model, callbacks = set_up_model('Model', input_shape=X.shape[1:], outputdir=parsed_args['outputdir'])

    # Train
    if parsed_args['load_model'] is None:
        history = model.fit(X_train, Y_train, sample_weight=w_train,
                            validation_data=(X_val, Y_val, w_val),
                            callbacks=callbacks,
                            batch_size=parsed_args['batch_size'],
                            epochs=200, verbose=1)

        logger.info("Plot training history")
        fname_loss = os.path.join(parsed_args['outputdir'], 'Loss')
        plotting.plot_train_loss(fname_loss, history.history['loss'], history.history['val_loss'])
    else:
        logger.info("Try loading model weights from {}".format(parsed_args['load_model']))
        try:
            model.load_weights(parsed_args['load_model']).expect_partial()
        except:
            logger.error("Failed to load model weights from {}".format(parsed_args['load_model']))
            return

    #################
    # Evaluate performance
    logger.info("Plot prediction distributions")
    pred_train = model.predict(X_train, batch_size=int(0.1*len(X_train)))[:,1]
    pred_val = model.predict(X_val, batch_size=int(0.1*len(X_val)))[:,1]
    fname_preds = os.path.join(parsed_args['outputdir'], 'preds')
    plotting.plot_training_vs_validation(fname_preds, pred_train, Y_train, w_train, pred_val, Y_val, w_val)

    # ROC curve
    Y_pred = model.predict(X_test, batch_size=int(0.1*len(X_test)))[:,1]
    Y_true = np.argmax(Y_test, axis=1)

    logger.info("Plot ROC curve")
    fname_roc = os.path.join(parsed_args['outputdir'], 'ROC')
    plotting.plot_roc_curves(fname_roc, [Y_pred], Y_true, w_test)

    # Calibration plot
    logger.info("Plot reliability curve")
    fname_cal = os.path.join(parsed_args['outputdir'], 'Calibration')
    plotting.plot_calibrations(fname_cal, [Y_pred], Y_true)

    # Reweight simulation prior to truth in pseudo data
    w_d = dataHandle.get_weights(rw_type=parsed_args['reweight_data'], vars_dict=observable_dict)

    w_s = simHandle.get_weights()
    # rescale simulation weights to pseudo data
    ndata = w_d.sum()
    nsim = w_s.sum()
    w_s *= ndata / nsim

    # New weights for simulation
    X_prior = X[np.argmax(Y, axis=1)==0]
    lr = reweight(model, X_prior)

    fname_hlr = os.path.join(parsed_args['outputdir'], 'rhist')
    plotting.plot_LR_distr(fname_hlr, [lr])

    w_s_rw = w_s * lr

    # Compare reweighted simulation prior to pseudo truth
    for varname in parsed_args['observables']:
        bins = get_bins(varname, parsed_args['binning_config'])
        vname_mc = observable_dict[varname]['branch_mc']

        # simulation prior
        hist_prior, hist_prior_err = simHandle.get_histogram(vname_mc, w_s, bins)

        # reweighted simulation distribution
        hist_rw, hist_rw_err = simHandle.get_histogram(vname_mc, w_s_rw, bins)

        # pseudo truth
        hist_truth, hist_truth_err = dataHandle.get_histogram(vname_mc, w_d, bins)

        # plot histograms and their ratio
        figname = os.path.join(parsed_args['outputdir'], 'Reweight_{}'.format(varname))
        logger.info("Plot reweighted distribution: {}".format(figname))

        # Compute chi2s
        text_chi2 = write_chi2(hist_truth, hist_truth_err, [hist_rw, hist_prior], [hist_rw_err, hist_truth_err], labels=['Reweighted', 'Prior'])
        logger.info("  "+"    ".join(text_chi2))

        # Compute triangular discriminator
        text_tria = write_triangular_discriminators(hist_truth, [hist_rw, hist_prior], labels=['Reweighted', 'Prior'])
        logger.info("  "+"    ".join(text_tria))

        # Compute KS test statistic
        arr_truth = dataHandle.get_variable_arr(vname_mc)
        arr_sim = simHandle.get_variable_arr(vname_mc)
        text_ks = write_ks(arr_truth, w_d, [arr_sim, arr_sim], [w_s_rw, w_s], labels=['Reweighted', 'Prior'])

        logger.info("  "+"    ".join(text_ks))

        plotting.plot_results(bins, (hist_prior, hist_prior_err), (hist_rw, hist_rw_err), histogram_truth=(hist_truth, hist_truth_err), figname=figname, texts=text_ks, **observable_dict[varname])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--observables', dest='observables',
                        nargs='+',
                        default=['th_pt','th_y','th_phi','th_e','tl_pt','tl_y','tl_phi','tl_e'],
                        help="List of features to train models")
    parser.add_argument('-d', '--data', required=True, nargs='+',
                        type=str,
                        help="Pseudo data npz file names")
    parser.add_argument('-s', '--signal', required=True, nargs='+',
                        type=str,
                        help="Signal MC npz file names")
    parser.add_argument('-o', '--outputdir', default='./output_models',
                        help="Output directory")
    parser.add_argument('-r', '--reweight-data', dest='reweight_data',
                        choices=['linear_th_pt', 'gaussian_bump', 'gaussian_tail'], default=None,
                        help="Reweight strategy of the input spectrum for stress tests")
    parser.add_argument('-g', '--gpu',
                        type=int, choices=[0, 1], default=None,
                        help="Manually select one of the GPUs to run")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=512,
                        help="Batch size for training")
    parser.add_argument('--weight', default='w',
                        help="name of event weight")
    parser.add_argument('--observable-config', dest='observable_config',
                        default='configs/observables/vars_klfitter.json',
                        help="JSON configurations for observables")
    parser.add_argument('--binning-config', dest='binning_config',
                        default='configs/binning/bins_10equal.json', type=str,
                        help="Binning config file for variables")
    parser.add_argument('--load-model', dest='load_model',
                        default=None, type=str,
                        help="Path to trained model weights to be loaded")

    args = parser.parse_args()
    
    if not os.path.isdir(args.outputdir):
        logger.info("Create output directory {}".format(args.outputdir))
        os.makedirs(args.outputdir)

    # check if configuraiton files exist and expand the file path
    fullpath_obsconfig = expandFilePath(args.observable_config)
    if fullpath_obsconfig is not None:
        args.observable_config = fullpath_obsconfig
    else:
        logger.error("Cannot find file: {}".format(args.observable_config))
        sys.exit("Config Failure")

    configGPUs(args.gpu)
    evaluateModels(**vars(args))
