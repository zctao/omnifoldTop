#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from model import get_model, get_callbacks

from datahandler import DataHandler, DataToy
import util
from util import configGPUs, configRootLogger, expandFilePath, read_dict_from_json
from util import get_bins, write_chi2, write_ks, write_triangular_discriminators, ks_2samp_weighted
import plotting
import logging

def get_training_inputs(variables, dataHandle, simHandle, rw_type=None, vars_dict=None):
    ###
    X_d = dataHandle[variables]
    Y_d = util.labels_for_dataset(X_d, 1)
    X_s = simHandle[variables]
    Y_s = util.labels_for_dataset(X_s, 0)

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

def set_up_model(model_name, input_shape, model_dir):
    if not os.path.isdir(model_dir):
        logger.info("Create directory {}".format(model_dir))
        os.makedirs(model_dir)

    model = get_model(input_shape, model_name, nclass=2)

    # callbacks
    callbacks = get_callbacks(os.path.join(model_dir, model_name))

    return model, callbacks

def reweight(model, events):
    preds = model.predict(events, batch_size=int(0.1*len(events)))[:,1]
    r = np.nan_to_num( preds / (1. - preds) )
    return r

def train_model(Data_train, Data_val, Data_test, model_name, model_dir,
                batch_size, load_model=None):

    X_train, Y_train, w_train = Data_train
    X_val, Y_val, w_val = Data_val
    X_test, Y_test, w_test = Data_test

    # zip event weights with labels
    Yw_train = np.column_stack((Y_train, w_train))
    Yw_val = np.column_stack((Y_val, w_val))
    Yw_test = np.column_stack((Y_test, w_test))

    model, callbacks = set_up_model(model_name, model_dir=model_dir,
                                    input_shape=X_train.shape[1:])

    if load_model is None:
        # start training
        history = model.fit(X_train, Yw_train,
                            validation_data=(X_val, Yw_val),
                            callbacks=callbacks,
                            batch_size=batch_size,
                            epochs=200, verbose=1)

        logger.info("Plot training history")
        fname_loss = os.path.join(model_dir, 'Loss')
        plotting.plot_train_loss(fname_loss, history.history['loss'], history.history['val_loss'])
    else:
        logger.info("Try loading model weights from {}".format(load_model))
        try:
            model.load_weights(parsed_args['load_model']).expect_partial()
        except:
            logger.error("Failed to load model weights from {}".format(parsed_args['load_model']))
            exit()

    # report performance
    pred_train = model.predict(X_train, batch_size=int(0.1*len(X_train)))[:,1]
    pred_val = model.predict(X_val, batch_size=int(0.1*len(X_val)))[:,1]

    logger.info("Plot prediction distributions")
    fname_preds = os.path.join(model_dir, 'preds')
    plotting.plot_training_vs_validation(fname_preds, pred_train, Y_train, w_train, pred_val, Y_val, w_val)

    #
    Y_pred = model.predict(X_test, batch_size=int(0.1*len(X_test)))[:,1]
    Y_true = np.argmax(Y_test, axis=1)

    # ROC curve
    logger.info("Plot ROC curve")
    fname_roc = os.path.join(model_dir, 'ROC')
    plotting.plot_roc_curves(fname_roc, [Y_pred], Y_true, w_test)

    # Calibration plot
    logger.info("Plot reliability curve")
    fname_cal = os.path.join(model_dir, 'Calibration')
    plotting.plot_calibrations(fname_cal, [Y_pred], Y_true)

    return model

def evaluateModels(**parsed_args):

    logger = logging.getLogger('EvalModel')

    # log arguments
    for argkey, argvalue in sorted(parsed_args.items()):
        if argvalue is None:
            continue
        logger.info('Argument {}: {}'.format(argkey, argvalue))

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
    logger.info("Total number of pseudo data events: {}".format(len(dataHandle)))

    fnames_s = parsed_args['signal']
    logger.info("Simulation files: {}".format(' '.join(fnames_s)))
    simHandle = DataHandler(fnames_s, wname, variable_names=vars_det+vars_mc)
    logger.info("Total number of simulation events: {}".format(len(simHandle)))

    ####
    #dataHandle = DataToy(1000000, 1, 1.5)
    #simHandle = DataToy(1000000, 0, 1)
    #vars_mc = ['x_truth']
    ####

    #################
    # Event weights
    # pseudo data weights
    w_d = dataHandle.get_weights(rw_type=parsed_args['reweight_data'], vars_dict=observable_dict)

    # prior simulation weights
    w_s = simHandle.get_weights()

    # normalize simulation weights to pseudo data
    ndata = w_d.sum()
    nsim = w_s.sum()
    w_s *= ndata / nsim

    #################
    # Input datasets
    #################
    # Training arrays
    # Truth level

    # FIXME hard code input variables for pfn for now
    if parsed_args['model_name'] == 'pfn':
        vars_mc = [['th_pt_MC', 'th_y_MC', 'th_phi_MC', 'th_e_MC'],
                   ['tl_pt_MC', 'tl_y_MC', 'tl_phi_MC', 'tl_e_MC']]

    X, Y, w = get_training_inputs(vars_mc, dataHandle, simHandle, rw_type=parsed_args['reweight_data'], vars_dict=observable_dict)

    # Split into training, validation, and test sets: 75%, 15%, 10%
    X_train, X_test, Y_train, Y_test, w_train, w_test = train_test_split(X, Y, w, test_size=0.25)
    X_val, X_test, Y_val, Y_test, w_val, w_test = train_test_split(X_test, Y_test, w_test, test_size=0.4)

    #################
    # Train model and reweight simulation
    weights_rw = []
    for i in range(parsed_args['nrun']):
        logger.info("RUN {}".format(i))

        model_dir = os.path.join(parsed_args['outputdir'], 'Models_{}'.format(i))

        model = train_model((X_train, Y_train, w_train),
                            (X_val,   Y_val,   w_val),
                            (X_test,  Y_test,  w_test),
                            model_name = parsed_args['model_name'],
                            model_dir = model_dir,
                            batch_size = parsed_args['batch_size'],
                            load_model = parsed_args['load_model'])

        # Reweight simulation to the truth in pseudo data
        # reweighting factors
        X_prior = X[np.argmax(Y, axis=1)==0]
        lr = reweight(model, X_prior)

        logger.info("Plot distribution of reweighitng factors")
        fname_hlr = os.path.join(model_dir, 'rhist')
        plotting.plot_LR_distr(fname_hlr, [lr])

        # New weights for simulation
        weights_rw.append(w_s * lr)

    #################
    # Compare reweighted simulation prior to pseudo truth

    w_s_rw = weights_rw[0]

    for varname in parsed_args['observables']:
        logger.info(varname)
        bins = get_bins(varname, parsed_args['binning_config'])
        vname_mc = observable_dict[varname]['branch_mc']

        # pseudo truth
        hist_truth, hist_truth_err = dataHandle.get_histogram(vname_mc, w_d, bins)

        # simulation prior
        hist_prior, hist_prior_err = simHandle.get_histogram(vname_mc, w_s, bins)

        # reweighted simulation distributions
        hists_rw, hists_rw_err = simHandle.get_histogram(vname_mc, weights_rw, bins)

        # plot the first reweighted distribution
        assert(len(hists_rw) > 0)
        hist_rw = hists_rw[0]
        hist_rw_err = hists_rw_err[0]
        #hist_rw = np.mean(np.asarray(hists_rw), axis=0)
        #hist_rw_err = np.std(np.asarray(hists_rw), axis=0, ddof=1)

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
        arr_truth = dataHandle[vname_mc]
        arr_sim = simHandle[vname_mc]
        text_ks = write_ks(arr_truth, w_d, [arr_sim, arr_sim], [w_s_rw, w_s], labels=['Reweighted', 'Prior'])

        logger.info("  "+"    ".join(text_ks))

        plotting.plot_results(bins, (hist_prior, hist_prior_err), (hist_rw, hist_rw_err), histogram_truth=(hist_truth, hist_truth_err), figname=figname, texts=text_ks, **observable_dict[varname])

        ####
        # plot all trials
        if len(hists_rw) > 1:
            figname_all = os.path.join(parsed_args['outputdir'], 'Reweight_{}_allruns'.format(varname))
            plotting.plot_hists_resamples(figname_all, bins, hists_rw, hist_prior, hist_truth, **observable_dict[varname])

        # plot the distribution of KS test statistic
        ks_list = []
        for rw_s in weights_rw:
            ks = ks_2samp_weighted(arr_truth, arr_sim, w_d, rw_s)[0]
            ks_list.append(ks)
        hist_ks, bins_ks = np.histogram(ks_list)
        fname_ks = os.path.join(parsed_args['outputdir'], 'KSDistr_{}'.format(varname))
        plotting.plot_histograms1d(fname_ks, bins_ks, [hist_ks], xlabel="KS")

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
    parser.add_argument('-m', '--model-name', dest='model_name', type=str,
                        default = 'dense_3hl',
                        help="Model name")
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
    parser.add_argument('-n', '--nrun', type=int, default=1,
                        help="Number of times to repeat the reweighting")
    parser.add_argument('--weight', default='totalWeight_nominal',
                        help="name of event weight")
    parser.add_argument('-v', '--verbose',
                        action='count', default=0,
                        help="Verbosity level")
    parser.add_argument('--observable-config', dest='observable_config',
                        default='configs/observables/vars_ttbardiffXs.json',
                        help="JSON configurations for observables")
    parser.add_argument('--binning-config', dest='binning_config',
                        default='configs/binning/bins_10equal.json', type=str,
                        help="Binning config file for variables")
    parser.add_argument('--load-model', dest='load_model',
                        default=None, type=str,
                        help="Path to trained model weights to be loaded")

    args = parser.parse_args()

    logfile = os.path.join(args.outputdir, 'log.txt')
    configRootLogger(filename=logfile)
    logger = logging.getLogger('EvalModel')
    logger.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)

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
