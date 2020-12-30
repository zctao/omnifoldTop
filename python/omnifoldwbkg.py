import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import plotting
from datahandler import DataHandler
from model import get_model, get_callbacks
from util import getLogger, add_histograms, write_chi2
logger = getLogger('OmniFoldwBkg')

class OmniFoldwBkg(object):
    def __init__(self, variables_det, variables_truth, iterations, outdir='.', binned_rw=False):
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

    def run(self, load_previous_iteration=True, unfolded_weights_file=None,
            unfolded_weights_file_resample=None, nresamples=0):
        assert(self.datahandle_obs is not None)
        assert(self.datahandle_sig is not None)

        if unfolded_weights_file:
            # load unfolded weights directly from the saved file without training
            logger.info("Skip training")
            logger.info("Read unfolded weights from file {}".format(unfolded_weights_file))
            self.unfolded_weights = self._read_weights_from_file(unfolded_weights_file)
            if unfolded_weights_file_resample:
                self.unfolded_weights_resample = self._read_weights_from_file(unfolded_weights_file_resample, array_name='weights_resample')
        else:
            self.unfolded_weights = self._unfold(load_previous_iteration, model_name='Models', save_weights_fname='weights.npz')

            # bootstrap uncertainty
            if nresamples:
                self._unfold_resample(nresamples, load_previous_iteration, save_weights_fname='weights_resample{}.npz'.format(nresamples))

    def get_unfolded_distribution(self, variable, bins, all_iterations=False,
                                    bootstrap_uncertainty=True):
        ws = self.unfolded_weights if all_iterations else self.unfolded_weights[-1]
        hist_uf, hist_uf_err = self.datahandle_sig.get_histogram(variable, ws, bins)

        if bootstrap_uncertainty:
            if self.unfolded_weights_resample is not None:
                hist_uf_err = self._get_unfolded_uncertainty(variable, bins, all_iterations)
            else:
                logger.warn("  Unable to compute bootstrap uncertainty. Use sum of weights squared in each bin instead.")

        return hist_uf, hist_uf_err

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
        hist_uf, hist_uf_err = self.get_unfolded_distribution(varConfig['branch_mc'], bins, all_iterations=False)

        if ibu:
            hist_ibu, hist_ibu_err = ibu.get_unfolded_distribution()
        else:
            hist_ibu, hist_ibu_err = None, None

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

        if iteration_history:
            iteration_dir = os.path.join(self.outdir, 'Iterations')
            if not os.path.isdir(iteration_dir):
                logger.info("Create directory {}".format(iteration_dir))
                os.makedirs(iteration_dir)

            figname_prefix = os.path.join(iteration_dir, varname)

            hists_uf, hists_uf_err = self.get_unfolded_distribution(varConfig['branch_mc'], bins, all_iterations=True)
            plotting.plot_iteration_distributions(figname_prefix+"_OmniFold_iterations", bins, hists_uf, hists_uf_err, **varConfig)

            if ibu:
                hists_ibu, hists_ibu_err = ibu.get_unfolded_distribution(True)
                plotting.plot_iteration_distributions(figname_prefix+"_IBU_iterations", bins, hists_ibu, hists_ibu_err, **varConfig)
            else:
                hists_ibu, hists_ibu_err = [], []

            plotting.plot_iteration_diffChi2s(figname_prefix+"_diffChi2s", [hists_ibu, hists_uf], [hists_ibu_err, hists_uf_err], labels=["IBU", "OmniFold"])
            if self.datahandle_obs.truth_known:
                plotting.plot_iteration_chi2s(figname_prefix+"_chi2s_wrt_Truth", hist_truth, hist_truth_err, [hists_ibu, hists_uf], [hists_ibu_err, hists_uf_err], labels=["IBU", "OmniFold"])

    def _unfold(self, load_previous_iter=False, bootstrap=False, val_size=0.2,
                model_name=None, save_weights_fname='weights.npz'):
        ################
        # model filepaths
        model_dir = os.path.join(self.outdir, model_name) if model_name else None
        if model_dir:
            if not os.path.isdir(model_dir):
                logger.info("Create directory {}".format(model_dir))
                os.makedirs(model_dir)

            model_step1_fp = os.path.join(model_dir, 'model_step1_{}')
            model_step2_fp = os.path.join(model_dir, 'model_step2_{}')
        else:
            model_step1_fp = None
            model_step2_fp = None
            if load_previous_iter:
                logger.warn("Cannot load model from the previous iteration since the model checkpoints are not saved.")
                load_previous_iter=False

        ################
        # event weights for training
        wobs, wsim, wbkg = self._get_event_weights(normalize=True, resample=bootstrap)

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
            step1_fp = model_step1_fp.format(i) if model_step1_fp else None
            step1_fp_prev = model_step1_fp.format(i-1) if model_step1_fp and i > 0 and load_previous_iter else None
            model_step1, cb_step1 = self._set_up_model_step1(self.X_step1.shape[1:], step1_fp, step1_fp_prev)

            # push the latest truth-level weights to the detector level
            wm_push_i = ws_t[i] # for i=0, this is wsim

            # prepare weight array for training
            if wbkg is None:
                w_step1 = np.concatenate([wobs, wm_push_i])
            else:
                w_step1 = np.concatenate([wobs, wm_push_i, wbkg])
            assert(len(w_step1)==len(self.X_step1))

            # split data into training and test sets
            X_step1_train, X_step1_test, Y_step1_train, Y_step1_test, w_step1_train, w_step1_test = train_test_split(self.X_step1, self.Y_step1, w_step1, test_size=val_size)

            logger.info("Start training")
            self._train_model(model_step1, X_step1_train, Y_step1_train, w_step1_train, callbacks=cb_step1, val_data=(X_step1_test, Y_step1_test, w_step1_test), filepath=step1_fp)

            # reweight
            logger.info("Reweighting")
            fname_rhist1 = step1_fp+'_rhist' if step1_fp else None
            wm_i = wm_push_i * self._reweight_step1(model_step1, self.X_sim,
                                                    fname_rhist1)
            # normalize the weight to the initial one
            if True: # TODO check performance
                wm_i *= (wsim.sum()/wm_i.sum())
            logger.debug("Iteration {} step 1: wm.sum() = {}".format(i, wm_i.sum()))

            ####
            # step 2: reweight the simulation prior to the learned weights
            logger.info("Step 2")
            # set up the model for iteration i
            step2_fp = model_step2_fp.format(i) if model_step2_fp else None
            step2_fp_prev = model_step2_fp.format(i-1) if model_step2_fp and i > 0 and load_previous_iter else None
            model_step2, cb_step2 = self._set_up_model_step2(self.X_step2.shape[1:], step2_fp, step2_fp_prev)

            # pull the learned weights from detector level to the truth level
            wt_pull_i = wm_i

            # prepare weight array for training
            w_step2 = np.concatenate([wt_pull_i, ws_t[i]])

            # split data into training and test sets
            X_step2_train, X_step2_test, Y_step2_train, Y_step2_test, w_step2_train, w_step2_test = train_test_split(self.X_step2, self.Y_step2, w_step2, test_size=val_size)

            # train model
            logger.info("Start training")
            self._train_model(model_step2, X_step2_train, Y_step2_train, w_step2_train, callbacks=cb_step2, val_data=(X_step2_test, Y_step2_test, w_step2_test), filepath=step2_fp)

            # reweight
            logger.info("Reweighting")
            fname_rhist2 = step2_fp+'_rhist' if step2_fp else None
            wt_i = wt_pull_i * self._reweight_step2(model_step2, self.X_gen,
                                                    fname_rhist2)
            # normalize the weight to the initial one
            if True: # TODO check performance
                wt_i *= (wsim.sum()/wt_i.sum())
            logger.debug("Iteration {} step 2: wt.sum() = {}".format(i, wt_i.sum()))
            ws_t[i+1,:] = wt_i
        # end of iterations
        #assert(not np.isnan(ws_t).any())
        logger.debug("Sum of unfolded weights = {}".format(ws_t[-1].sum()))

        # save the weights
        if save_weights_fname:
            weights_file = os.path.join(self.outdir, save_weights_fname)
            np.savez(weights_file, weights = ws_t)

        # normalize unfolded weights to the nominal signal simulation weights
        logger.info("Normalize to nominal signal simulation weights")
        ws_t *= (self.weights_sim.sum() / ws_t.sum(axis=1)[:,np.newaxis])
        logger.debug("Sum of unfolded weights after normalization = {}".format(ws_t[-1].sum()))

        # Plot training log
        if model_dir:
            logger.info("Plot model training history")
            for csvfile in glob.glob(os.path.join(model_dir, '*.csv')):
                logger.info("  Plot training log {}".format(csvfile))
                plotting.plot_train_log(csvfile)

        return ws_t

    def _unfold_resample(self, nresamples, load_previous_iteration=False,
                         save_weights_fname=None):
        self.unfolded_weights_resample = np.empty(shape=(nresamples, self.iterations+1, self.datahandle_sig.get_nevents()))
        # shape: (nresamples, n_iterations+1, n_events)
        for iresample in range(nresamples):
            # only save the model checkpoints if load_previous_iteration
            if load_previous_iteration:
                mname = 'Models_rs{}'.format(iresample)
            else:
                mname = None

            ws = self._unfold(load_previous_iteration, bootstrap=True, model_name=mname, save_weights_fname=None)
            self.unfolded_weights_resample[iresample,:,:] = ws

        if save_weights_fname:
            weights_file = os.path.join(self.outdir, save_weights_fname)
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

        if hists_resample:
            hists_err = np.std(np.asarray(hists_resample), axis=0)
        else:
            hists_err = None

        return  hists_err
        # shape = (n_iteration, n_bins) if all_iterations
        # otherwise, shape = (n_bins,)

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
        self.weights_obs = self.datahandle_obs.get_weights(normalize=True, rw_type=rw_type, vars_dict=vars_dict)
        self.weights_sim = self.datahandle_sig.get_weights(normalize=False)
        self.weights_bkg = None if self.datahandle_bkg is None else self.datahandle_bkg.get_weights(normalize=False)

        # rescale signal and background simulation weights to data
        if rescale:
            ndata = self.weights_obs.sum()
            nsig = self.weights_sim.sum()
            nbkg = self.weights_bkg.sum() if self.datahandle_bkg else 0.

            # sum of simulation original weights
            sumw_sig = self.datahandle_sig.sumw
            sumw_bkg = self.datahandle_bkg.sumw if self.datahandle_bkg else 0.
            sumw_sim = sumw_sig + sumw_bkg

            self.weights_sim *= (sumw_sig / sumw_sim * ndata / nsig)
            if self.datahandle_bkg:
                self.weights_bkg *= (sumw_bkg / sumw_sim * ndatat / nbkg)

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

    def _set_up_model(self, input_shape, model_filepath=None, previous_model_filepath=None):
        # get model
        model = get_model(input_shape)

        # callbacks
        callbacks = get_callbacks(model_filepath)

        # load weights from the previous model if available
        if previous_model_filepath:
            model.load_weights(previous_model_filepath)

        return model, callbacks

    def _set_up_model_step1(self, input_shape, model_filepath=None, previous_model_filepath=None):
        return self._set_up_model(input_shape, model_filepath, previous_model_filepath)

    def _set_up_model_step2(self, input_shape, model_filepath=None, previous_model_filepath=None):
        return self._set_up_model(input_shape, model_filepath, previous_model_filepath)

    def _train_model(self, model, X, Y, w, callbacks=[], val_data=None, filepath='', plot_performance=True):
        fitargs = {'batch_size': int(0.1*len(X)), 'epochs': 100, 'verbose': 1}
        if callbacks:
            fitargs.setdefault('callbacks', []).extend(callbacks)

        val_dict = {'validation_data': val_data} if val_data is not None else {}

        model.fit(X, Y, sample_weight=w, **fitargs, **val_dict)

        if filepath:
            model.save_weights(filepath)

            if plot_performance and val_data is not None:
                preds_train = model.predict(X, batch_size=int(0.1*len(X)))[:,1]
                X_val, Y_val, w_val = val_data
                preds_val = model.predict(X_val, batch_size=int(0.1*len(X_val)))[:,1]
                figname_preds = filepath+'_preds'
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
