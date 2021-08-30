import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import plotting
from datahandler import DataHandler
from model import get_model, get_callbacks
import util
import metrics
from histogramming import set_hist_contents, set_hist_errors, get_values_and_errors, get_mean_from_hists, get_sigma_from_hists, get_bin_correlations_from_hists
import logging
logger = logging.getLogger('OmniFoldwBkg')
logger.setLevel(logging.DEBUG)

class OmniFoldwBkg(object):
    def __init__(
        self,
        variables_det,
        variables_truth,
        iterations=4,
        outdir=".",
        truth_known=True,
    ):
        # list of detector and truth level variable names used in training
        self.vars_reco = variables_det
        self.vars_truth = variables_truth
        self.truth_known = truth_known
        # number of iterations
        self.iterations = iterations
        # category labels
        self.label_obs = 1
        self.label_sig = 0
        self.label_bkg = 0
        # data handlers
        self.datahandle_obs = None
        self.datahandle_sig = None
        self.datahandle_bkg = None
        self.datahandle_obsbkg = None
        # unfoled weights
        self.unfolded_weights = None
        self.unfolded_weights_resample = None
        # output directory
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        self.outdir = outdir.rstrip('/')+'/'

    def prepare_inputs(
        self,
        obsHandle,
        simHandle,
        bkgHandle=None,
        obsBkgHandle=None,
        plot_corr=False,
        reweighter=None,
    ):
        # observed data
        self.datahandle_obs = obsHandle
        logger.info(
            "Total number of observed events: {}".format(len(self.datahandle_obs))
        )
        if plot_corr:
            logger.info("Plot input variable correlations")
            corr_obs_reco = obsHandle.get_correlations(self.vars_reco)
            plotting.plot_correlations(corr_obs_reco, os.path.join(self.outdir, 'correlations_det_obs'))

        # simulation
        self.datahandle_sig = simHandle
        logger.info(
            "Total number of simulated events: {}".format(len(self.datahandle_sig))
        )
        if plot_corr:
            logger.info("Plot input variable correlations")
            corr_sim_reco = simHandle.get_correlations(self.vars_reco)
            plotting.plot_correlations(corr_sim_reco, os.path.join(self.outdir, 'correlations_det_sig'))
            corr_sim_gen = simHandle.get_correlations(self.vars_truth)
            plotting.plot_correlations(corr_sim_gen, os.path.join(self.outdir, 'correlations_gen_sig'))

        # background simulation if needed
        self.datahandle_bkg = bkgHandle
        if self.datahandle_bkg is not None:
            logger.info(
                "Total number of simulated background events: {}".format(
                    len(self.datahandle_bkg)
                )
            )

        # background simulation to be mixed with pseudo data
        if self.datahandle_bkg is not None:
            self.datahandle_obsbkg = obsBkgHandle
            if self.datahandle_obsbkg is not None:
                logger.info(
                    "Total number of background events mixed with data: {}".format(
                        len(self.datahandle_obsbkg)
                    )
                )

        # prepare event weights
        logger.info("Prepare event weights")
        self._set_event_weights(reweighter)

    def _set_event_weights(self, reweighter=None, rescale=True, plot=True):
        """
        preprocess event weights.
        Reweight the pseudo data sample if reweighter is provided.
        Rescale simulation weights to be consistent with data weights

        Parameters
        ----------
        reweighter : reweight.Reweighter, optional
            A function that takes events and returns event weights, and the
            variables it expects.
        rescale : bool, default: False
            if True, rescale simulation sample weights to be compatible with
            data
        plot : bool, default: True
            if True, plot distributions of event weights
        """
        logger.info("Preprocess event weights")

        # (pseudo) data weights
        self.datahandle_obs.rescale_weights(reweighter=reweighter)

        # total weights of data
        sumw_obs = self.datahandle_obs.sum_weights()
        if self.datahandle_obsbkg is not None:
            sumw_obs += self.datahandle_obsbkg.sum_weights()

        logger.debug("Total weights of data events: {}".format(sumw_obs))

        if rescale:
            logger.info("Rescale simulation prior weights to data")

            # total weights of simulated events
            sumw_sig = self.datahandle_sig.sum_weights()

            sumw_bkg = 0.
            if self.datahandle_bkg is not None:
                sumw_bkg = self.datahandle_bkg.sum_weights()

            sumw_sim = sumw_sig + sumw_bkg

            self.datahandle_sig.rescale_weights( sumw_obs/sumw_sim )

            if self.datahandle_bkg is not None:
                self.datahandle_bkg.rescale_weights( sumw_obs/sumw_sim )

        logger.debug(
            "Total weights of simulated signal events: {}".format(
                self.datahandle_sig.sum_weights()
            )
        )

        if self.datahandle_bkg is not None:
            logger.debug(
                "Total weights of simulated background events: {}".format(
                    self.datahandle_bkg.sum_weights()
                )
            )

        if plot:
            logger.info("Plot event weights")
            plotting.plot_data_arrays(
                os.path.join(self.outdir, 'Event_weights'),
                [self.datahandle_sig.get_weights(), self.datahandle_obs.get_weights()],
                labels=['Sim.', 'Data'], title='Event weights', xlabel='w'
            )

    def _get_event_weights(self, resample=False, plot=True):
        """
        Get event weights for training
        """
        logger.info("Get event weights for training")

        # standardize data weights to mean of one
        wobs = self.datahandle_obs.get_weights(bootstrap=resample)
        if self.datahandle_obsbkg is not None:
            wobsbkg = self.datahandle_obsbkg.get_weights(bootstrap=resample)
            wobs = np.concatenate([wobs, wobsbkg])

        logger.debug("Standardize data weights to mean of one")
        wmean_obs = np.mean(wobs)
        wobs /= wmean_obs

        # simulation weights
        wsig = self.datahandle_sig.get_weights()
        wbkg = self.datahandle_bkg.get_weights() if self.datahandle_bkg is not None else None

        # rescale simulation weights by the same factor
        logger.debug("Scale simulation weights by the same factor as data")
        wsig /= wmean_obs
        if wbkg is not None:
            wbkg /= wmean_obs

        # TODO check the alternative: all weights are standardized
        #wsig /= np.mean(wsig)
        #if wbkg is not None:
        #    wbkg /= np.mean(wbkg)

        if plot and not resample:
            logger.info("Plot the distribution of event weights used in training")
            plotting.plot_data_arrays(
                os.path.join(self.outdir, 'Event_weights_train'),
                [wsig, wobs], labels=['Sim.', 'Data'],
                title='Prior event weights in training', xlabel='w (training)'
            )

        return wobs, wsig, wbkg

    def _get_feature_arrays_step1(self, preprocess=True, plot=False):
        """
        Get arrays for step 1 unfolding
        setp 1: observed data vs simulation at detector level
        """
        X_obs = self.datahandle_obs[self.vars_reco]
        if self.datahandle_obsbkg is not None:
            X_obsbkg = self.datahandle_obsbkg[self.vars_reco]
            X_obs = np.concatenate([X_obs, X_obsbkg])

        X_sig = self.datahandle_sig[self.vars_reco]

        if self.datahandle_bkg is not None:
            X_bkg = self.datahandle_bkg[self.vars_reco]
        else:
            X_bkg = None

        ####
        # preprocess feature arrays
        if preprocess:
            logger.info("Preprocess feature arrays for step 1")

            X_all = np.concatenate([X_obs, X_sig]) if X_bkg is None else np.concatenate([X_obs, X_sig, X_bkg])

            # divide by their orders of magnitude
            Xmean = np.mean(np.abs(X_all), axis=0)
            Xoom = 10**(np.log10(Xmean).astype(int))
            X_obs /= Xoom
            X_sig /= Xoom
            if X_bkg is not None:
                X_bkg /= Xoom

            # TODO: check alternatives
            # e.g. standardize features to mean of zero and variance of one
            #Xmean = np.mean(X_all, axis=0)
            #Xstd = np.std(X_all, axis=0)
            #
            #X_obs -= Xmean
            #X_obs /= Xstd
            #X_sig -= Xmean
            #X_sig /= Xstd
            #if X_bkg is not None:
            #    X_bkg -= Xmean
            #    X_bkg /= Xstd

        if plot:
            logger.info("Plot the distribution of variables for step 1 training")
            # weights
            wobs, wsig = self._get_event_weights(plot=False)[:2]
            for vname, vobs, vsig in zip(self.vars_reco, X_obs.T, X_sig.T):
                logger.debug("  Plot variable {}".format(vname))
                plotting.plot_data_arrays(
                    os.path.join(self.outdir, "Train_step1_"+vname),
                    [vsig, vobs], weight_arrs=[wsig, wobs],
                    labels=['Sim.', 'Data'],
                    title="Step-1 training inputs", xlabel=vname
                )

        return X_obs, X_sig, X_bkg

    def _get_feature_arrays_step2(self, preprocess=True, plot=False):
        """
        Get arrays for step 2 unfolding
        """
        X_gen = self.datahandle_sig[self.vars_truth]

        if preprocess:
            logger.info("Preprocess feature arrays for step 2")
            # # divide by their orders of magnitude
            Xmean = np.mean(np.abs(X_gen), axis=0)
            Xoom = 10**(np.log10(Xmean).astype(int))
            X_gen /= Xoom

            # TODO: check alternatives
            #Xmean = np.mean(X_gen, axis=0)
            #Xstd = np.std(X_gen, axis=0)
            #X_gen -= Xmean
            #X_gen /= Xstd

        if plot:
            logger.info("Plot the distribution of variables for step 2 training")
            wsig = self._get_event_weights(plot=False)[1]
            for vname, vgen in zip(self.vars_truth, X_gen.T):
                logger.debug("  Plot variable {}".format(vname))
                plotting.plot_data_arrays(
                    os.path.join(self.outdir, "Train_step2_"+vname),
                    [vgen], [wsig], labels=['Gen.'],
                    title="Step-2 training inputs", xlabel=vname
                )

        return X_gen

    def run(
        self,
        error_type='sumw2',
        nresamples=0,
        load_previous_iteration=True,
        load_models_from=None,
        batch_size=256,
        epochs=100
    ):
        """
        Run unfolding
        """
        fitargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}

        # Input arrays for step 1
        logger.info("Get input arrays for step 1")
        # features
        X_obs, X_sim, X_bkg = self._get_feature_arrays_step1(
            preprocess=True, plot=True)

        # labels
        Y_obs = np.full(len(X_obs), self.label_obs)
        Y_sim = np.full(len(X_sim), self.label_sig)
        Y_bkg = None if X_bkg is None else np.full(len(X_bkg), self.label_bkg)

        if X_bkg is None:
            X_step1 = np.concatenate([X_obs, X_sim])
            Y_step1 = np.concatenate([Y_obs, Y_sim])
        else:
            X_step1 = np.concatenate([X_obs, X_sim, X_bkg])
            Y_step1 = np.concatenate([Y_obs, Y_sim, Y_bkg])

        # make Y categorical
        Y_step1 = tf.keras.utils.to_categorical(Y_step1)

        logger.info("Size of the feature array for step 1: {:.3f} MB".format(
            X_step1.nbytes*2**-20)
        )
        logger.info("Size of the label array for step 1: {:.3f} MB".format(
            Y_step1.nbytes*2**-20)
        )

        # Input arrays for step 2
        logger.info("Get input arrays for step 2")
        # features
        X_gen = self._get_feature_arrays_step2(preprocess=True, plot=True)
        X_step2 = np.concatenate([X_gen, X_gen])

        # labels
        Y_step2 = np.concatenate([np.ones(len(X_gen)), np.zeros(len(X_gen))])
        Y_step2 = tf.keras.utils.to_categorical(Y_step2)

        logger.info("Size of the feature array for step 2: {:.3f} MB".format(
            X_step2.nbytes*2**-20)
        )
        logger.info("Size of the label array for step 2: {:.3f} MB".format(
            Y_step2.nbytes*2**-20)
        )

        # unfold
        self.unfolded_weights = self._unfold(
            X_step1, Y_step1, X_step2, Y_step2, X_sim, X_gen,
            resample_data=False,
            model_name="Models",
            load_models_dir=load_models_from,
            load_previous_iter=load_previous_iteration,
            **fitargs
        )

        # save weights
        wfile = os.path.join(self.outdir, 'weights.npz')
        np.savez(wfile, weights = self.unfolded_weights)

        # resamples
        if error_type in ['bootstrap_full', 'bootstrap_model']:
            self.unfolded_weights_resample = np.empty(
                shape=( nresamples, self.iterations, len(X_sim) )
            )

            for ir in range(nresamples):
                logger.info("Resample #{}".format(ir))
                # unfold
                self.unfolded_weights_resample[ir,:,:] = self._unfold(
                    X_step1, Y_step1, X_step2, Y_step2, X_sim, X_gen,
                    resample_data=(error_type=='bootstrap_full'),
                    model_name="Models_rs{}".format(ir),
                    load_models_dir=load_models_from,
                    load_previous_iter=load_previous_iteration,
                    **fitargs
                )

            # save weights
            wfile = os.path.join(self.outdir, 'weights_resample{}.npz'.format(nresamples))
            np.savez(wfile, weights_resample = self.unfolded_weights_resample)

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
        rw = self.unfolded_weights if all_iterations else self.unfolded_weights[-1]
        wsim = self.datahandle_sig.get_weights()
        h_uf = self.datahandle_sig.get_histogram(variable, bins, wsim*rw)
        # h_uf is a hist object or a list of hist objects

        bin_corr = None # bin correlations
        if bootstrap_uncertainty:
            if self.unfolded_weights_resample is not None:
                hists_uf_rs = self.get_unfolded_hists_resample(variable, bins, all_iterations)

                # combine the "nominal" hist with the resampled ones
                hists_uf_rs.append(h_uf)

                # take the mean of each bin as the new nominal
                hmean = get_mean_from_hists(hists_uf_rs)

                # take the standardard deviation of each bin as bin errors
                hsigma = get_sigma_from_hists(hists_uf_rs)

                # compute bin correlations
                bin_corr = get_bin_correlations_from_hists(hists_uf_rs)

                # update the unfolded histogram
                set_hist_contents(h_uf, hmean)
                set_hist_errors(h_uf, hsigma)
            else:
                logger.warn("  Unable to compute bootstrap uncertainty. Use sum of weights squared in each bin instead.")

        if normalize:
            # renormalize the unfolded histograms and its error to the nominal signal simulation weights
            if all_iterations:
                # rescale all iterations to simulation weights
                rs = wsim.sum() / (wsim*rw).sum(axis=1)
                for h, r in zip(h_uf, rs):
                    h *= r
            else:
                h_uf *= (wsim.sum() / (wsim*rw).sum())

        return h_uf, bin_corr

    def plot_distributions_reco(self, varname, varConfig, bins):
        # observed
        nobs = len(self.datahandle_obs)
        h_obs = self.datahandle_obs.get_histogram(varConfig['branch_det'], bins)

        if self.datahandle_obsbkg is not None:
            # add background
            h_obsbkg = self.datahandle_obsbkg.get_histogram(varConfig['branch_det'], bins)
            h_obs = h_obs + h_obsbkg

        # signal simulation
        h_sim = self.datahandle_sig.get_histogram(varConfig['branch_det'], bins)

        # background simulation
        if self.datahandle_bkg is None:
            h_simbkg = None
        else:
            h_simbkg = self.datahandle_bkg.get_histogram(varConfig['branch_det'], bins)

        # plot
        figname = os.path.join(self.outdir, 'Reco_{}'.format(varname))
        logger.info("  Plot detector-level distribution: {}".format(figname))
        plotting.plot_reco_variable(h_obs, h_sim, h_simbkg,
                                    figname=figname, log_scale=False,
                                    **varConfig)

    def plot_distributions_unfold(self, varname, varConfig, bins, ibu=None, iteration_history=False, plot_resamples=True, plot_bins=False):
        # unfolded distribution
        h_uf, h_uf_corr = self.get_unfolded_distribution(varConfig['branch_mc'], bins, all_iterations=False)

        if ibu:
            h_ibu, h_ibu_corr = ibu.get_unfolded_distribution()
        else:
            h_ibu, h_ibu_corr = None, None

        # signal prior distribution
        h_gen = self.datahandle_sig.get_histogram(varConfig['branch_mc'], bins)

        # MC truth if known
        if self.truth_known:
            h_truth = self.datahandle_obs.get_histogram(varConfig['branch_mc'], bins)
        else:
            h_truth = None

        # compute chi2s
        text_chi2 = []
        if self.truth_known:
            text_chi2 = metrics.write_texts_Chi2(h_truth, [h_uf, h_ibu, h_gen], labels=['OmniFold', 'IBU', 'Prior'])
            logger.info("  "+"    ".join(text_chi2))

        # Compute KS test statistic
        text_ks = []
        if self.truth_known:
            arr_truth = self.datahandle_obs[varConfig['branch_mc']]
            arr_sim = self.datahandle_sig[varConfig['branch_mc']]
            text_ks = metrics.write_texts_KS(
                arr_truth, self.datahandle_obs.get_weights(),
                [arr_sim, arr_sim],
                [self.datahandle_sig.get_weights()*self.unfolded_weights[-1],
                 self.datahandle_sig.get_weights()],
                labels=['OmniFold', 'Prior']
            )
            logger.info("  "+"    ".join(text_ks))

        # plot
        figname = os.path.join(self.outdir, 'Unfold_{}'.format(varname))
        logger.info("  Plot unfolded distribution: {}".format(figname))
        plotting.plot_results(h_gen, h_uf, h_ibu, h_truth,
                              figname=figname, texts=text_chi2, **varConfig)

        # bin correlations
        if h_uf_corr is not None:
            figname_of_corr = os.path.join(self.outdir, 'BinCorrelations_{}_OmniFold'.format(varname))
            logger.info("  Plot bin correlations: {}".format(figname_of_corr))
            plotting.plot_correlations(h_uf_corr, figname_of_corr)
        if h_ibu_corr is not None:
            figname_ibu_corr = os.path.join(self.outdir, 'BinCorrelations_{}_IBU'.format(varname))
            logger.info("  Plot bin correlations: {}".format(figname_ibu_corr))
            plotting.plot_correlations(h_ibu_corr, figname_ibu_corr)

        # plot all resampled unfolded distributions
        if plot_resamples and self.unfolded_weights_resample is not None:
            hists_resample = self.get_unfolded_hists_resample(varConfig['branch_mc'], bins, all_iterations=False)
            figname_resamples = os.path.join(self.outdir, 'Unfold_AllResamples_{}'.format(varname))
            logger.info("  Plot unfolded distributions for all trials: {}".format(figname_resamples))
            plotting.plot_hists_resamples(figname_resamples, hists_resample, h_gen, hist_truth=h_truth, **varConfig)

        # plot distributions of bin entries
        if plot_bins and self.unfolded_weights_resample is not None:
            histo_uf_all = self.get_unfolded_hists_resample(varConfig['branch_mc'], bins, all_iterations=iteration_history)
            # histo_uf_all shape:
            # if not iteration_history: (nresamples, nbins)
            # if iteration_history: (nresamples, niterations, nbins)

            # plot pulls of each bin entries
            figname_bindistr = os.path.join(self.outdir, 'Unfold_BinDistr_{}'.format(varname))
            logger.info("  Plot distributions of bin entries from all trials: {}".format(figname_bindistr))
            plotting.plot_hists_bin_distr(figname_bindistr, histo_uf_all, h_truth)

        # plot iteration history
        if iteration_history:
            iteration_dir = os.path.join(self.outdir, 'Iterations')
            if not os.path.isdir(iteration_dir):
                logger.info("Create directory {}".format(iteration_dir))
                os.makedirs(iteration_dir)

            figname_prefix = os.path.join(iteration_dir, varname)

            hists_uf = self.get_unfolded_distribution(varConfig['branch_mc'], bins, all_iterations=True)[0]
            # Add prior to the head of the list
            hists_uf = [h_gen] + list(hists_uf)
            # plot
            plotting.plot_iteration_distributions(figname_prefix+"_OmniFold_iterations", hists_uf, h_truth, **varConfig)

            if ibu:
                hists_ibu = ibu.get_unfolded_distribution(all_iterations=True)[0]
                # Add prior to the head of the list
                hists_ibu = [h_gen] + list(hists_ibu)
                # plot
                plotting.plot_iteration_distributions(figname_prefix+"_IBU_iterations", hists_ibu, h_truth, **varConfig)
            else:
                hists_ibu = []

            plotting.plot_iteration_diffChi2s(figname_prefix+"_diffChi2s", [hists_ibu, hists_uf], labels=["IBU", "OmniFold"])
            if self.truth_known:
                plotting.plot_iteration_chi2s(figname_prefix+"_chi2s_wrt_Truth", h_truth, [hists_ibu, hists_uf], labels=["IBU", "OmniFold"])

                if self.unfolded_weights_resample is not None:
                    hists_uf_all = self.get_unfolded_hists_resample(varConfig['branch_mc'], bins, all_iterations=True, normalize=True)
                    # add prior
                    hists_uf_all = [[h_gen]+list(hists_uf_rs) for hists_uf_rs in hists_uf_all]

                    # TODO:
                    # Should we use the same bin errors from boostrap for all trails?
                    #uf_err_all = [get_values_and_errors(hists_uf)[1]] * len(hists_uf_all)
                    #set_hist_errors(hists_uf_all, uf_err_all)

                    plotting.plot_iteration_chi2s(figname_prefix+"_AllResamples_chi2s_wrt_Truth", h_truth, hists_uf_all, lw=0.7, ms=0.7)

    def _unfold(
        self,
        X_step1, Y_step1,
        X_step2, Y_step2,
        X_sim,
        X_gen,
        resample_data=False,
        model_name='Models',
        load_models_dir=None,
        load_previous_iter=True,
        val_size=0.2,
        #fname_event_weights='weights.npz',
        **fitargs
    ):
        ################
        # model directory
        if load_models_dir is None:
            # train models
            reweight_only=False

            # directory to store trained models
            model_dir = os.path.join(self.outdir, model_name) if model_name else None
            if model_dir and not os.path.isdir(model_dir):
                logger.info("Create directory {}".format(model_dir))
                os.makedirs(model_dir)
        else:
            # used the trained models for reweighting
            reweight_only=True

            model_dir = os.path.join(load_models_dir, model_name)

            if not os.path.isdir(model_dir):
                raise RuntimeError("Cannot load models fromn {}: directory does not exist!".format(model_dir))

            logger.info("Reweight using trained models from {}".format(model_dir))

        ################
        # event weights for training
        logger.info("Get prior event weights")
        wobs, wsim, wbkg = self._get_event_weights(resample=resample_data)
        logger.debug("wobs.sum() = {}".format(wobs.sum()))
        logger.debug("wsim.sum() = {}".format(wsim.sum()))
        if wbkg is not None:
            logger.debug("wbkg.sum() = {}".format(wbkg.sum()))

        ################
        # start iterations
        wm_push = np.ones_like(wsim)
        wt_pull = np.ones_like(wsim)
        weights_unfold = np.empty(shape=(self.iterations, len(wsim)))
        ## shape: (n_iterations, n_events)

        for i in range(self.iterations):
            logger.info("Iteration {}".format(i))
            ####
            # step 1: reweight sim to look like data
            logger.info("Step 1")
            # set up the model for iteration i
            model_step1, cb_step1 = self._set_up_model_step1(X_step1.shape[1:], i, model_dir, load_previous_iter, reweight_only)

            if not reweight_only:
                # prepare weight array for training
                if wbkg is None:
                    w_step1 = np.concatenate([wobs, wm_push*wsim])
                else:
                    w_step1 = np.concatenate([wobs, wm_push*wsim, wbkg])
                assert(len(w_step1)==len(X_step1))

                # split data into training and test sets
                X_step1_train, X_step1_test, Y_step1_train, Y_step1_test, w_step1_train, w_step1_test = train_test_split(X_step1, Y_step1, w_step1, test_size=val_size)

                logger.info("Start training")
                fname_preds1 = model_dir+'/preds_step1_{}'.format(i) if model_dir else None
                self._train_model(model_step1, X_step1_train, Y_step1_train, w_step1_train, callbacks=cb_step1, val_data=(X_step1_test, Y_step1_test, w_step1_test), figname_preds=fname_preds1, **fitargs)
                logger.info("Model training done")

            # reweight
            logger.info("Reweighting")
            fname_rhist1 = model_dir+'/rhist_step1_{}'.format(i) if model_dir else None
            wm_i = wm_push * self._reweight_step1(model_step1, X_sim, fname_rhist1)
            logger.debug("Iteration {} step 1: wm.sum() = {}".format(i, wm_i.sum()))

            # TODO: check the performace
            #wm_i /= np.mean(wm_i)

            # pull the learned weights from detector level to the truth level
            wt_pull = wm_i

            ####
            # step 2: reweight the simulation prior to the learned weights
            logger.info("Step 2")
            # set up the model for iteration i
            model_step2, cb_step2 = self._set_up_model_step2(X_step2.shape[1:], i, model_dir, load_previous_iter, reweight_only)

            if not reweight_only:
                # prepare weight array for training
                w_step2 = np.concatenate([wt_pull*wsim, wsim])

                # split data into training and test sets
                X_step2_train, X_step2_test, Y_step2_train, Y_step2_test, w_step2_train, w_step2_test = train_test_split(X_step2, Y_step2, w_step2, test_size=val_size)

                # train model
                logger.info("Start training")
                fname_preds2 = model_dir+'/preds_step2_{}'.format(i) if model_dir else None
                self._train_model(model_step2, X_step2_train, Y_step2_train, w_step2_train, callbacks=cb_step2, val_data=(X_step2_test, Y_step2_test, w_step2_test), figname_preds=fname_preds2, **fitargs)

            # reweight
            logger.info("Reweighting")
            fname_rhist2 = model_dir+'/rhist_step2_{}'.format(i) if model_dir else None
            wt_i = self._reweight_step2(model_step2, X_gen, fname_rhist2)
            logger.debug("Iteration {} step 2: wt.sum() = {}".format(i, wt_i.sum()))

            # TODO: check the performace
            #wt_i /= np.mean(wt_i)

            # push the updated truth level weights to the detector level
            wm_push = wt_i
            # save truth level weights of this iteration
            weights_unfold[i,:] = wt_i
        # end of iterations
        #assert(not np.isnan(weights_unfold).any())

        logger.debug("Sum of unfolded weights = {}".format(weights_unfold[-1].sum()))

        # Plot training log
        if model_dir:
            logger.info("Plot model training history")
            for csvfile in glob.glob(os.path.join(model_dir, '*.csv')):
                logger.info("  Plot training log {}".format(csvfile))
                plotting.plot_train_log(csvfile)

        return weights_unfold

    def get_unfolded_hists_resample(self, variable, bins, all_iterations=False, normalize=False):
        hists_resample = []
        for iresample in range(len(self.unfolded_weights_resample)):
            if all_iterations:
                rw = self.unfolded_weights_resample[iresample]
            else:
                rw = self.unfolded_weights_resample[iresample][-1]

            wsim = self.datahandle_sig.get_weights()
            h = self.datahandle_sig.get_histogram(variable, bins, wsim*rw)

            if normalize:
                # rescale each iteration to the nominal signal simulation weights
                if all_iterations:
                    rs = wsim.sum() / (wsim*rw).sum(axis=1)
                    for hh, r in zip(h, rs):
                        hh *= r
                else:
                    h *= (wsim.sum() / (wsim*rw).sum())

            hists_resample.append(h)

        return hists_resample

    def _read_weights_from_file(self, weights_file, array_name='weights'):
        # load unfolded weights from saved file
        wfile = np.load(weights_file)
        weights = wfile[array_name]
        wfile.close()
        return weights

    def _set_up_model(self, input_shape, filepath_save=None, filepath_load=None,
                      nclass=2):
        # get model
        model = get_model(input_shape, nclass=nclass)

        # callbacks
        callbacks = get_callbacks(filepath_save)

        # load weights from the previous model if available
        if filepath_load:
            logger.info("Load model weights from {}".format(filepath_load))
            if filepath_save is None:
                # reweight only without training
                model.load_weights(filepath_load).expect_partial()
            else:
                model.load_weights(filepath_load)

        return model, callbacks

    def _set_up_model_step1(self, input_shape, iteration, model_dir,
                            load_previous_iter=True, reweight_only=False):
        # model filepath
        model_fp = os.path.join(model_dir, 'model_step1_{}') if model_dir else None

        if reweight_only:
            # load trained models for reweighting
            return self._set_up_model(input_shape, filepath_save=None, filepath_load=model_fp.format(iteration))
        else:
            # set up model for training
            if load_previous_iter and iteration > 0:
                # initialize model based on the previous iteration
                assert(model_fp)
                return self._set_up_model(input_shape, filepath_save=model_fp.format(iteration), filepath_load=model_fp.format(iteration-1))
            else:
                return self._set_up_model(input_shape, filepath_save=model_fp.format(iteration), filepath_load=None)

    def _set_up_model_step2(self, input_shape, iteration, model_dir,
                            load_previous_iter=True, reweight_only=False):
        # model filepath
        model_fp = os.path.join(model_dir, 'model_step2_{}') if model_dir else None

        # set up model for training
        if reweight_only:
            # load trained models for reweighting
            return self._set_up_model(input_shape, filepath_save=None, filepath_load=model_fp.format(iteration))
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

        # zip event weights with labels
        Yw = np.column_stack((Y, w))
        if val_data is not None:
            X_val, Y_val, w_val = val_data
            Yw_val = np.column_stack((Y_val, w_val))
            val_dict = {'validation_data': (X_val, Yw_val)}
        else:
            val_dict = {}

        model.fit(X, Yw, **fitargs, **val_dict)

        if figname_preds and val_data is not None:
            preds_train = model.predict(X, batch_size=int(0.1*len(X)))[:,1]
            preds_val = model.predict(X_val, batch_size=int(0.1*len(X_val)))[:,1]
            logger.info("Plot model output distribution: {}".format(figname_preds))
            plotting.plot_training_vs_validation(figname_preds, preds_train, Y, w, preds_val, Y_val, w_val)

    def _reweight(self, model, events, plotname=None):

        preds = model.predict(events, batch_size=int(0.1*len(events)))[:,1]
        r = np.nan_to_num( preds / (1. - preds) )

        if plotname: # plot the ratio distribution
            logger.info("Plot likelihood ratio distribution "+plotname)
            plotting.plot_LR_distr(plotname, [r])

        return r

    #def _reweight_binned(self):

    def _reweight_step1(self, model, events, plotname=None):
        return self._reweight(model, events, plotname)

    def _reweight_step2(self, model, events, plotname=None):
        return self._reweight(model, events, plotname)

###########
# Approaches to deal with backgrounds

###
# Default as implemented in the base class:
# Reweight data vs signal+background simulation at detector level
# Compare unfolded signal vs (data - background) at truth level

###
# Add background as negatively weighted data
class OmniFoldwBkg_negW(OmniFoldwBkg):
    def __init__(
        self,
        variables_det,
        variables_truth,
        iterations=4,
        outdir=".",
        truth_known=True
    ):
        super().__init__(variables_det, variables_truth, iterations, outdir, truth_known)

        # set background simulation label the same as data
        self.label_bkg = self.label_obs

    def _get_event_weights(self, normalize=True, resample=False):
        wobs, wsim, wbkg = super()._get_event_weights(normalize, resample)

        # flip the sign of thebackground weights
        wbkg *= -1

        return wobs, wsim, wbkg

###
# multi-class classification
class OmniFoldwBkg_multi(OmniFoldwBkg):
    def __init__(
        self,
        variables_det,
        variables_truth,
        iterations=4,
        outdir=".",
        truth_known=True,
    ):
        super().__init__(variables_det, variables_truth, iterations, outdir, truth_known)

        # new class label for background
        self.label_bkg = 2

    def _set_up_model_step1(self, input_shape, iteration, model_dir,
                            load_previous_iter=True):
        # model filepath
        model_fp = os.path.join(model_dir, 'model_step1_{}') if model_dir else None

        # set up model for training
        if load_previous_iter and iteration > 0:
            # initialize model based on the previous iteration
            assert(model_fp)
            return self._set_up_model(input_shape, filepath_save=model_fp.format(iteration), filepath_load=model_fp.format(iteration-1), nclass=3)
        else:
            return self._set_up_model(input_shape, filepath_save=model_fp.format(iteration), filepath_load=None, nclass=3)

    def _reweight_step1(self, model, events, plotname=None):

        preds_obs = model.predict(events, batch_size=int(0.1*len(events)))[:,self.label_obs]
        preds_sig = model.predict(events, batch_size=int(0.1*len(events)))[:,self.label_sig]
        preds_bkg = model.predict(events, batch_size=int(0.1*len(events)))[:,self.label_bkg]

        r = np.nan_to_num( preds_obs / preds_sig - preds_bkg / preds_sig)

        if plotname: # plot the ratio distribution
            logger.info("Plot likelihood ratio distribution "+plotname)
            plotting.plot_LR_distr(plotname, [r])

        return r
