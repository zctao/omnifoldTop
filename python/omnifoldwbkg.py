import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import plotting
from datahandler import DataHandler
from model import get_model, get_callbacks
from util import add_histograms, write_chi2
import logging
logger = logging.getLogger('OmniFoldwBkg')
logger.setLevel(logging.DEBUG)

class OmniFoldwBkg(object):
    def __init__(self, variables_det, variables_truth, iterations=4, outdir='.', binned_rw=False):
        # list of detector and truth level variable names used in training
        self.vars_reco = variables_det 
        self.vars_truth = variables_truth
        # number of iterations
        self.iterations = iterations
        # reweighting method
        self.binned_rw = binned_rw
        # category labels
        self.label_obs = 1
        self.label_sig = 0
        self.label_bkg = 0
        # data handlers
        self.datahandle_obs = None
        self.datahandle_sig = None
        self.datahandle_bkg = None
        # arrays for training
        self.X_step1 = None
        self.Y_step1 = None
        self.X_step2 = None
        self.Y_step2 = None
        # arrays for reweigting
        self.X_sim = None
        self.X_gen = None
        # nominal event weights from samples
        self.weights_obs = None
        self.weights_sim = None
        self.weights_bkg = None
        # unfoled weights
        self.unfolded_weights = None
        self.unfolded_weights_resample = None
        # output directory
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        self.outdir = outdir.rstrip('/')+'/'

    def prepare_inputs(self, obsHandle, simHandle, bkgHandle=None,
                        plot_corr=False, standardize=True, reweight_type=None,
                        vars_dict={}):
        # observed data
        self.datahandle_obs = obsHandle
        logger.info("Total number of observed events: {}".format(self.datahandle_obs.get_nevents()))

        # simulation
        self.datahandle_sig = simHandle
        logger.info("Total number of simulated events: {}".format(self.datahandle_sig.get_nevents()))

        # background simulation if needed
        self.datahandle_bkg = bkgHandle
        if self.datahandle_bkg is not None:
            logger.info("Total number of simulated background events: {}".format(self.datahandle_bkg.get_nevents()))

        # plot input variable correlations
        if plot_corr:
            logger.info("Plot input variable correlations")
            corr_obs_reco = obsHandle.get_correlations(self.vars_reco)
            plotting.plot_correlations(corr_obs_reco, os.path.join(self.outdir, 'correlations_det_obs'))
            corr_sim_reco = simHandle.get_correlations(self.vars_reco)
            plotting.plot_correlations(corr_sim_reco, os.path.join(self.outdir, 'correlations_det_sig'))
            corr_sim_gen = simHandle.get_correlations(self.vars_truth)
            plotting.plot_correlations(corr_sim_gen, os.path.join(self.outdir, 'correlations_gen_sig'))

        logger.info("Prepare arrays")
        # arrays for step 1
        # self.X_step1, self.Y_step1, self.X_sim
        self._set_arrays_step1(self.datahandle_obs, self.datahandle_sig, self.datahandle_bkg, standardize)

        # arrays for step 2
        # self.X_step2, self.Y_step2, self.X_gen
        self._set_arrays_step2(self.datahandle_sig, standardize)

        # event weights for training
        self._set_event_weights(rw_type=reweight_type, vars_dict=vars_dict,
                                rescale=True)

    def run(self, error_type='sumw2', nresamples=0, load_previous_iteration=True,
            batch_size=256, epochs=100):
        assert(self.datahandle_obs is not None)
        assert(self.datahandle_sig is not None)

        fitargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}

        self.unfolded_weights = self._unfold(load_previous_iter=load_previous_iteration, fname_event_weights='weights.npz', **fitargs)

        # bootstrap uncertainty
        if error_type in ['bootstrap_full', 'bootstrap_stat', 'bootstrap_model']:
            self._unfold_resample(nresamples, error_type, load_previous_iteration, fname_event_weights='weights_resample{}.npz'.format(nresamples), **fitargs)

    def load(self, unfolded_weight_files):
        # load unfolded event weights from the saved file
        logger.info("Skip training")

        wfilelist = list(unfolded_weight_files)
        assert(len(wfilelist) > 0)
        logger.info("Load unfolded weights: {}".format(wfilelist[0]))
        self.unfolded_weights = self._read_weights_from_file(wfilelist[0])
        logger.debug("unfolded_weights.shape: {}".format(self.unfolded_weights.shape))

        if len(wfilelist) > 1:
            logger.info("Load unfolded weights from resampling: {}".format(wfilelist[1]))
            self.unfolded_weights_resample = self._read_weights_from_file(wfilelist[1], array_name='weights_resample')
            # TODO: load weights from multiple files
            logger.debug("unfolded_weights_resample.shape: {}".format(self.unfolded_weights_resample.shape))

    def get_unfolded_distribution(self, variable, bins, all_iterations=False,
                                  bootstrap_uncertainty=True, normalize=True):
        ws = self.unfolded_weights if all_iterations else self.unfolded_weights[-1]
        hist_uf, hist_uf_err = self.datahandle_sig.get_histogram(variable, ws, bins)

        bin_corr = None # bin correlations
        if bootstrap_uncertainty:
            if self.unfolded_weights_resample is not None:
                hist_uf_err, bin_corr = self._get_unfolded_uncertainty(variable, bins, all_iterations)
            else:
                logger.warn("  Unable to compute bootstrap uncertainty. Use sum of weights squared in each bin instead.")

        if normalize:
            # renormalize the unfolded histograms and its error to the nominal signal simulation weights
            hist_uf *= self.weights_sim.sum() / self.unfolded_weights[-1].sum()
            hist_uf_err *= self.weights_sim.sum() / self.unfolded_weights[-1].sum()
            # all iterations are rescaled based on the weights of the last one
            # renormalize histograms of each iteration according to their individual norm instead?

        return hist_uf, hist_uf_err, bin_corr

    def plot_distributions_reco(self, varname, varConfig, bins):
        # observed
        hist_obs, hist_obs_err = self.datahandle_obs.get_histogram(varConfig['branch_det'], self.weights_obs, bins)

        # signal simulation
        hist_sim, hist_sim_err = self.datahandle_sig.get_histogram(varConfig['branch_det'], self.weights_sim, bins)

        # background simulation
        if self.datahandle_bkg is None:
            hist_simbkg, hist_simbkg_err = None, None
        else:
            hist_simbkg, hist_simbkg_err = self.datahandle_bkg.get_histogram(varConfig['branch_det'], self.weights_bkg, bins)

        # plot
        figname = os.path.join(self.outdir, 'Reco_{}'.format(varname))
        logger.info("  Plot detector-level distribution: {}".format(figname))
        plotting.plot_reco_variable(bins, (hist_obs, hist_obs_err),
                                    (hist_sim, hist_sim_err),
                                    (hist_simbkg, hist_simbkg_err),
                                    figname=figname, log_scale=False,
                                    **varConfig)

    def plot_distributions_unfold(self, varname, varConfig, bins, ibu=None, iteration_history=False):
        # unfolded distribution
        hist_uf, hist_uf_err, hist_uf_corr = self.get_unfolded_distribution(varConfig['branch_mc'], bins, all_iterations=False)

        if ibu:
            hist_ibu, hist_ibu_err, hist_ibu_corr = ibu.get_unfolded_distribution()
        else:
            hist_ibu, hist_ibu_err, hist_ibu_corr = None, None, None

        # signal prior distribution
        hist_gen, hist_gen_err = self.datahandle_sig.get_histogram(varConfig['branch_mc'], self.weights_sim, bins)

        # MC truth if known
        if self.datahandle_obs.truth_known:
            hist_truth, hist_truth_err = self.datahandle_obs.get_histogram(varConfig['branch_mc'], self.weights_obs, bins)

            # subtract background if needed
            if self.datahandle_bkg is not None:
                hist_genbkg, hist_genbkg_err = self.datahandle_bkg.get_histogram(varConfig['branch_mc'], self.weights_bkg, bins)
                hist_truth, hist_truth_err = add_histograms(hist_truth, hist_genbkg, hist_truth_err, hist_genbkg_err, c1=1., c2=-1.)
        else:
            hist_truth, hist_truth_err = None, None

        # compute chi2s
        text_td = []
        if self.datahandle_obs.truth_known:
            text_td = write_chi2(hist_truth, hist_truth_err, [hist_uf, hist_ibu, hist_gen], [hist_uf_err, hist_ibu_err, hist_gen_err], labels=['OmniFold', 'IBU', 'Prior'])
            logger.info("  "+"    ".join(text_td))

        # plot
        figname = os.path.join(self.outdir, 'Unfold_{}'.format(varname))
        logger.info("  Plot unfolded distribution: {}".format(figname))
        plotting.plot_results(bins, (hist_gen, hist_gen_err),
                              (hist_uf, hist_uf_err),
                              (hist_ibu, hist_ibu_err),
                              (hist_truth, hist_truth_err),
                              figname=figname, texts=text_td, **varConfig)

        # bin correlations
        if hist_uf_corr is not None:
            figname_of_corr = os.path.join(self.outdir, 'BinCorrelations_{}_OmniFold'.format(varname))
            logger.info("  Plot bin correlations: {}".format(figname_of_corr))
            plotting.plot_correlations(hist_uf_corr, figname_of_corr)
        if hist_ibu_corr is not None:
            figname_ibu_corr = os.path.join(self.outdir, 'BinCorrelations_{}_IBU'.format(varname))
            logger.info("  Plot bin correlations: {}".format(figname_ibu_corr))
            plotting.plot_correlations(hist_ibu_corr, figname_ibu_corr)

        # plot iteration history
        if iteration_history:
            iteration_dir = os.path.join(self.outdir, 'Iterations')
            if not os.path.isdir(iteration_dir):
                logger.info("Create directory {}".format(iteration_dir))
                os.makedirs(iteration_dir)

            figname_prefix = os.path.join(iteration_dir, varname)

            hists_uf, hists_uf_err = self.get_unfolded_distribution(varConfig['branch_mc'], bins, all_iterations=True)[:2]
            plotting.plot_iteration_distributions(figname_prefix+"_OmniFold_iterations", bins, hists_uf, hists_uf_err, **varConfig)

            if ibu:
                hists_ibu, hists_ibu_err = ibu.get_unfolded_distribution(all_iterations=True)[:2]
                plotting.plot_iteration_distributions(figname_prefix+"_IBU_iterations", bins, hists_ibu, hists_ibu_err, **varConfig)
            else:
                hists_ibu, hists_ibu_err = [], []

            plotting.plot_iteration_diffChi2s(figname_prefix+"_diffChi2s", [hists_ibu, hists_uf], [hists_ibu_err, hists_uf_err], labels=["IBU", "OmniFold"])
            if self.datahandle_obs.truth_known:
                plotting.plot_iteration_chi2s(figname_prefix+"_chi2s_wrt_Truth", hist_truth, hist_truth_err, [hists_ibu, hists_uf], [hists_ibu_err, hists_uf_err], labels=["IBU", "OmniFold"])

    def _unfold(self, resample_data=False, model_name='Models',
                reweight_only=False, load_previous_iter=True,
                val_size=0.2, fname_event_weights='weights.npz', **fitargs):
        ################
        # model directory
        model_dir = os.path.join(self.outdir, model_name) if model_name else None
        if model_dir and not os.path.isdir(model_dir):
            logger.info("Create directory {}".format(model_dir))
            os.makedirs(model_dir)

        ################
        # event weights for training
        wobs, wsim, wbkg = self._get_event_weights(normalize=True, resample=resample_data)

        ################
        # start iterations
        ws_t = np.empty(shape=(self.iterations+1, len(wsim)))
        ## shape: (n_iterations+1, n_events)
        ws_t[0,:] = wsim

        for i in range(self.iterations):
            logger.info("Iteration {}".format(i))
            ####
            # step 1: reweight sim to look like data
            logger.info("Step 1")
            # set up the model for iteration i
            model_step1, cb_step1 = self._set_up_model_step1(self.X_step1.shape[1:], i, model_dir, reweight_only, load_previous_iter)

            # push the latest truth-level weights to the detector level
            wm_push_i = ws_t[i] # for i=0, this is wsim

            if not reweight_only:
                # prepare weight array for training
                if wbkg is None:
                    w_step1 = np.concatenate([wobs, wm_push_i])
                else:
                    w_step1 = np.concatenate([wobs, wm_push_i, wbkg])
                assert(len(w_step1)==len(self.X_step1))

                # split data into training and test sets
                X_step1_train, X_step1_test, Y_step1_train, Y_step1_test, w_step1_train, w_step1_test = train_test_split(self.X_step1, self.Y_step1, w_step1, test_size=val_size)

                logger.info("Start training")
                fname_preds1 = model_dir+'/preds_step1_{}'.format(i) if model_dir else None
                self._train_model(model_step1, X_step1_train, Y_step1_train, w_step1_train, callbacks=cb_step1, val_data=(X_step1_test, Y_step1_test, w_step1_test), figname_preds=fname_preds1, **fitargs)

            # reweight
            logger.info("Reweighting")
            fname_rhist1 = model_dir+'/rhist_step1_{}'.format(i) if model_dir and not reweight_only else None
            wm_i = wm_push_i * self._reweight_step1(model_step1, self.X_sim, fname_rhist1)
            # normalize the weight to the initial one
            if False: # TODO check performance
                wm_i *= (wsim.sum()/wm_i.sum())
            logger.debug("Iteration {} step 1: wm.sum() = {}".format(i, wm_i.sum()))

            ####
            # step 2: reweight the simulation prior to the learned weights
            logger.info("Step 2")
            # set up the model for iteration i
            model_step2, cb_step2 = self._set_up_model_step2(self.X_step2.shape[1:], i, model_dir, reweight_only, load_previous_iter)

            # pull the learned weights from detector level to the truth level
            wt_pull_i = wm_i

            if not reweight_only:
                # prepare weight array for training
                w_step2 = np.concatenate([wt_pull_i, ws_t[i]])

                # split data into training and test sets
                X_step2_train, X_step2_test, Y_step2_train, Y_step2_test, w_step2_train, w_step2_test = train_test_split(self.X_step2, self.Y_step2, w_step2, test_size=val_size)

                # train model
                logger.info("Start training")
                fname_preds2 = model_dir+'/preds_step2_{}'.format(i) if model_dir else None
                self._train_model(model_step2, X_step2_train, Y_step2_train, w_step2_train, callbacks=cb_step2, val_data=(X_step2_test, Y_step2_test, w_step2_test), figname_preds=fname_preds2, **fitargs)

            # reweight
            logger.info("Reweighting")
            fname_rhist2 = model_dir+'/rhist_step2_{}'.format(i) if model_dir and not reweight_only else None
            wt_i = wt_pull_i * self._reweight_step2(model_step2, self.X_gen, fname_rhist2)
            # normalize the weight to the initial one
            if False: # TODO check performance
                wt_i *= (wsim.sum()/wt_i.sum())
            logger.debug("Iteration {} step 2: wt.sum() = {}".format(i, wt_i.sum()))
            ws_t[i+1,:] = wt_i
        # end of iterations
        #assert(not np.isnan(ws_t).any())

        # rescale unfolded weights from training to the nominal sim weights
        logger.info("Rescale unfolded weights according to the nominal signal simulation weights and the weights used in the training")
        ws_t *= self.weights_sim.sum() / wsim.sum()
        logger.debug("Sum of unfolded weights = {}".format(ws_t[-1].sum()))

        # normalize unfolded weights to the nominal signal simulation weights
        #logger.info("Normalize to nominal signal simulation weights")
        #ws_t *= (self.weights_sim.sum() / ws_t.sum(axis=1)[:,np.newaxis])
        #logger.debug("Sum of unfolded weights after normalization = {}".format(ws_t[-1].sum()))

        # save the weights
        if fname_event_weights:
            weights_file = os.path.join(self.outdir, fname_event_weights)
            np.savez(weights_file, weights = ws_t)

        # Plot training log
        if model_dir and not reweight_only:
            logger.info("Plot model training history")
            for csvfile in glob.glob(os.path.join(model_dir, '*.csv')):
                logger.info("  Plot training log {}".format(csvfile))
                plotting.plot_train_log(csvfile)

        return ws_t

    def _unfold_resample(self, nresamples, error_type='bootstrap_full',
                         load_previous_iter=True, fname_event_weights=None,
                         **fitargs):
        if not nresamples > 1:
            return

        self.unfolded_weights_resample = np.empty(shape=(nresamples, self.iterations+1, self.datahandle_sig.get_nevents()))
        # shape: (nresamples, n_iterations+1, n_events)

        model_name = 'Models' if error_type=='bootstrap_stat' else 'Models_rs{}'
        reweight_only = True if error_type=='bootstrap_stat' else False
        resample_data = False if error_type=='bootstrap_model' else True

        for iresample in range(nresamples):
            logger.info("Resample {}".format(iresample))
            ws = self._unfold(resample_data, model_name.format(iresample), reweight_only, load_previous_iter, fname_event_weights=None, **fitargs)
            self.unfolded_weights_resample[iresample,:,:] = ws

        if fname_event_weights:
            weights_file = os.path.join(self.outdir, fname_event_weights)
            np.savez(weights_file, weights_resample = self.unfolded_weights_resample)

    def _get_unfolded_uncertainty(self, variable, bins, all_iterations=False):
        #assert(self.unfolded_weights_resample is not None)
        hists_resample = []
        for iresample in range(len(self.unfolded_weights_resample)):
            if all_iterations:
                ws = self.unfolded_weights_resample[iresample]
            else:
                ws = self.unfolded_weights_resample[iresample][-1]

            hist = self.datahandle_sig.get_histogram(variable, ws, bins)[0]
            hists_resample.append(hist)

        hists_err, hists_corr = None, None
        if hists_resample:
            hists_err = np.std(np.asarray(hists_resample), axis=0, ddof=1)
            # shape = (n_iteration, n_bins) if all_iterations
            # otherwise, shape = (n_bins,)

            # bin correlations
            if all_iterations:
                hists_corr = []
                # hists_resample shape: (n_resamples, n_iterations, n_bins)
                for i in range(self.iterations):
                    df_i = pd.DataFrame(np.asarray(hists_resample)[:,i,:])
                    hists_corr.append(df_i.corr()) # ddof = 1
                    # sanity check
                    #assert((np.asarray(df_i.std()) == hists_err[i]).all())
            else:
                # hists_resample shape: (n_resamples, n_bins)
                df_i = pd.DataFrame(np.asarray(hists_resample))
                hists_corr = df_i.corr()
                # sanity check
                #assert((np.asarray(df_i.std()) == hists_err).all())

        return  hists_err, hists_corr

    def _read_weights_from_file(self, weights_file, array_name='weights'):
        # load unfolded weights from saved file
        wfile = np.load(weights_file)
        weights = wfile[array_name]
        wfile.close()
        return weights

    def _set_arrays_step1(self, obsHandle, simHandle, bkgHandle=None, standardize=True):
        # step 1: observed data vs simulation at detector level
        X_obs, Y_obs = obsHandle.get_dataset(self.vars_reco, self.label_obs, standardize=False)
        X_sim, Y_sim = simHandle.get_dataset(self.vars_reco, self.label_sig, standardize=False)

        if bkgHandle is None:
            self.X_step1 = np.concatenate([X_obs, X_sim])
            self.Y_step1 = np.concatenate([Y_obs, Y_sim])
        else:
            X_simbkg, Y_simbkg = bkgHandle.get_dataset(self.vars_reco, self.label_bkg, standardize=False)
            self.X_step1 = np.concatenate([X_obs, X_sim, X_simbkg])
            self.Y_step1 = np.concatenate([Y_obs, Y_sim, Y_simbkg])

        self.X_sim = X_sim

        # make Y categorical
        self.Y_step1 = tf.keras.utils.to_categorical(self.Y_step1)

        if standardize:
            Xmean = np.mean(self.X_step1, axis=0)
            Xstd = np.std(self.X_step1, axis=0)
            self.X_step1 -= Xmean
            self.X_step1 /= Xstd
            self.X_sim -= Xmean
            self.X_sim /= Xstd

        logger.info("Size of the feature array for step 1: {:.3f} MB".format(self.X_step1.nbytes*2**-20))
        logger.info("Size of the label array for step 1: {:.3f} MB".format(self.Y_step1.nbytes*2**-20))

    def _set_arrays_step2(self, simHandle, standardize=True):
        # step 2: update simulation weights at truth level
        self.X_gen = simHandle.get_dataset(self.vars_truth, self.label_sig, standardize=False)[0]
        nsim = len(self.X_gen)

        self.X_step2 = np.concatenate([self.X_gen, self.X_gen])
        self.Y_step2 = tf.keras.utils.to_categorical(np.concatenate([np.ones(nsim), np.zeros(nsim)]))

        if standardize:
            Xmean = np.mean(self.X_step2, axis=0)
            Xstd = np.std(self.X_step2, axis=0)
            self.X_step2 -= Xmean
            self.X_step2 /= Xstd
            self.X_gen -= Xmean
            self.X_gen /= Xstd

        logger.info("Size of the feature array for step 2: {:.3f} MB".format(self.X_step2.nbytes*2**-20))
        logger.info("Size of the label array for step 2: {:.3f} MB".format(self.Y_step2.nbytes*2**-20))

    def _set_event_weights(self, rw_type=None, vars_dict={}, rescale=True):
        self.weights_obs = self.datahandle_obs.get_weights(rw_type=rw_type,
                                                           vars_dict=vars_dict)
        self.weights_sim = self.datahandle_sig.get_weights()
        self.weights_bkg = None if self.datahandle_bkg is None else self.datahandle_bkg.get_weights()

        # rescale signal and background simulation weights to data
        if rescale:
            ndata = self.weights_obs.sum()
            nsig = self.weights_sim.sum()
            nbkg = self.weights_bkg.sum() if self.datahandle_bkg else 0.
            nsim = nsig + nbkg

            self.weights_sim *= ndata / nsim
            if self.datahandle_bkg:
                self.weights_bkg *= ndata / nsim

        logger.debug("weights_obs.sum() = {}".format(self.weights_obs.sum()))
        logger.debug("weights_sim.sum() = {}".format(self.weights_sim.sum()))
        if self.datahandle_bkg:
            logger.debug("weights_bkg.sum() = {}".format(self.weights_bkg.sum()))

    def _get_event_weights(self, normalize=False, resample=False):
        wobs = self.weights_obs
        wsim = self.weights_sim
        wbkg = self.weights_bkg if self.weights_bkg else None

        if normalize: # normalize to len(weights)
            wobs = wobs / np.mean(wobs)
            wsim = wsim / np.mean(wsim)
            if wbkg:
                wbkg = wbkg / np.mean(wbkg)

        if resample:
            wobs *= np.random.poisson(1, size=len(wobs))

        return wobs, wsim, wbkg

    def _set_up_model(self, input_shape, filepath_save=None, filepath_load=None,
                      reweight_only=False):
        # get model
        model = get_model(input_shape)

        # callbacks
        callbacks = get_callbacks(filepath_save)

        # load weights from the previous model if available
        if filepath_load:
            logger.info("Load model weights from {}".format(filepath_load))
            if reweight_only:
                model.load_weights(filepath_load).expect_partial()
            else:
                model.load_weights(filepath_load)

        return model, callbacks

    def _set_up_model_step1(self, input_shape, iteration, model_dir,
                            reweight_only=False, load_previous_iter=True):
        # model filepath
        model_fp = os.path.join(model_dir, 'model_step1_{}') if model_dir else None

        if reweight_only:
            # load the previously trained model from model_dir
            # apply it directly in reweighting without training
            assert(model_fp)
            return self._set_up_model(input_shape, filepath_save=None, filepath_load=model_fp.format(iteration), reweight_only=True)
        else:
            # set up model for training
            if load_previous_iter and iteration > 0:
                # initialize model based on the previous iteration
                assert(model_fp)
                return self._set_up_model(input_shape, filepath_save=model_fp.format(iteration), filepath_load=model_fp.format(iteration-1))
            else:
                return self._set_up_model(input_shape, filepath_save=model_fp.format(iteration), filepath_load=None)

    def _set_up_model_step2(self, input_shape, iteration, model_dir,
                            reweight_only=False, load_previous_iter=True):
        # model filepath
        model_fp = os.path.join(model_dir, 'model_step2_{}') if model_dir else None

        if reweight_only:
            # load the previously trained model from model_dir
            # apply it directly in reweighting without training
            assert(model_fp)
            return self._set_up_model(input_shape, filepath_save=None, filepath_load=model_fp.format(iteration), reweight_only=True)
        else:
            # set up model for training
            if load_previous_iter and iteration > 0:
                # initialize model based on the previous iteration
                assert(model_fp)
                return self._set_up_model(input_shape, filepath_save=model_fp.format(iteration), filepath_load=model_fp.format(iteration-1))
            else:
                return self._set_up_model(input_shape, filepath_save=model_fp.format(iteration), filepath_load=None)

    def _train_model(self, model, X, Y, w, callbacks=[], val_data=None, figname_preds='', **fitargs):
        if callbacks:
            fitargs.setdefault('callbacks', []).extend(callbacks)

        val_dict = {'validation_data': val_data} if val_data is not None else {}

        model.fit(X, Y, sample_weight=w, **fitargs, **val_dict)

        if figname_preds:
            preds_train = model.predict(X, batch_size=int(0.1*len(X)))[:,1]
            X_val, Y_val, w_val = val_data
            preds_val = model.predict(X_val, batch_size=int(0.1*len(X_val)))[:,1]
            logger.info("Plot model output distribution: {}".format(figname_preds))
            plotting.plot_training_vs_validation(figname_preds, preds_train, Y, w, preds_val, Y_val, w_val)

    def _reweight(self, model, events, plotname=None):
        preds = model.predict(events, batch_size=int(0.1*len(events)))[:,1]
        r = preds / (1. - preds + 10**-50)

        if plotname: # plot the ratio distribution
            logger.info("Plot likelihood ratio distribution "+plotname)
            plotting.plot_LR_distr(plotname, [r])

        return r

    #def _reweight_binned(self):

    def _reweight_step1(self, model, events, plotname=None):
        return self._reweight(model, events, plotname)

    def _reweight_step2(self, model, events, plotname=None):
        return self._reweight(model, events, plotname)
