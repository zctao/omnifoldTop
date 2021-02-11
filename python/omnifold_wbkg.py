import os

import numpy as np
import tensorflow as tf

# The original omnifold
import external.OmniFold.omnifold as omnifold
import external.OmniFold.modplot as modplot

from util import read_dataset, get_variable_arr, reweight_sample
from util import DataShufflerDet, DataShufflerGen
from util import write_triangular_discriminators, write_chi2
from util import normalize_histogram, add_histograms, divide_histograms
from ibu import ibu
from model import get_callbacks, get_model

from plotting import plot_results, plot_reco_variable, plot_correlations, plot_response, plot_graphs, plot_LR_distr, plot_LR_func, plot_training_vs_validation, plot_iteration_distributions, plot_iteration_chi2s, plot_iteration_diffChi2s

#from binning import get_bins
from util import get_bins

from util import getLogger
logger = getLogger('OmniFoldwBkg')

# Base class of OmniFold for non-negligible background
# Adapted from the original omnifold:
# https://github.dcom/ericmetodiev/OmniFold/blob/master/omnifold.py
class OmniFoldwBkg(object):
    def __init__(self, variables_det, variables_gen, wname, it, outdir='./', binned_rw=False):
        assert(len(variables_det)==len(variables_gen))
        self.vars_det = variables_det # list of detector-level variables
        self.vars_gen = variables_gen # list of truth-level variables
        self.weight_name = wname # name of sample weight
        self.label_obs = 1
        self.label_sig = 0
        self.label_bkg = 0
        # number of iterationns
        self.iterations = it
        # feature, label, and weight arrays
        self.X_det = None # detector-level feature array
        self.Y_det = None # detector-level label array
        self.X_gen = None # simulation-level feature array
        self.Y_gen = None # simulation-level label array
        self.wdata = None # ndarray for data sample weights
        self.wsig = None # ndarray for signal simulation weights
        self.wbkg = None  # ndarray for background simulation sample weights
        # unfolded event weights
        self.ws_unfolded = []
        # output directory
        self.outdir = outdir.rstrip('/')+'/'
        # reweight type
        self.binned_rw = binned_rw

    def _set_up_model_det_i(self, i, model_filepath=None):
        """ Set up the model for the i-th iteration of detector-level reweighting
        """
        # input dimension
        input_shape = self.X_det.shape[1:]

        # get model
        model = get_model(input_shape)

        # callbacks
        callbacks = get_callbacks(model_filepath.format(i))

        # load weights from the previous iteration if not the first one
        if i > 0 and model_filepath is not None:
            model.load_weights(model_filepath.format(i-1))

        return model, callbacks

    def _set_up_model_sim_i(self, i, model_filepath=None):
        """ Set up the model for the i-th iteration of simulation reweighting
        """
        # input dimension
        input_shape = self.X_gen.shape[1:]

        # get model
        model = get_model(input_shape)

        # callbacks
        callbacks = get_callbacks(model_filepath.format(i))

        # load weights from the previous iteration if not the first one
        if i > 0 and model_filepath is not None:
            model.load_weights(model_filepath.format(i-1))

        return model, callbacks

    def _compute_likelihood_ratio(self, preds, Y, weights, figname=None, nbins=50):
        logger.info("Compute likelihood ratio from model output distributions")

        # choose bin range based on preds max and min
        preds_max = preds.max()
        preds_min = preds.min()
        wbin = (preds_max - preds_min)/nbins
        bins_preds = np.linspace(preds_min-wbin/2, preds_max+wbin/2, nbins+1)
        # In case the bin width is too small, probably make more sense to just put all events in one bin
        if wbin < 0.001:
            logger.warn("The classifier has little discrimination power")
            wbin = 0.001
            bins_preds = np.asarray([preds_min - wbin/2, preds_max + wbin/2])

        if Y.ndim == 1: # Y is simply a 1D array of labels
            preds_cat1 = preds[Y==1]
            preds_cat0 = preds[Y==0]
            w_cat1 = weights[Y==1]
            w_cat0 = weights[Y==0]
        else: # Y is categorical
            preds_cat1 = preds[Y.argmax(axis=1)==1]
            preds_cat0 = preds[Y.argmax(axis=1)==0]
            w_cat1 = weights[Y.argmax(axis=1)==1]
            w_cat0 = weights[Y.argmax(axis=1)==0]

        hist_preds1, hist_preds1_unc = modplot.calc_hist(preds_cat1, bins=bins_preds, weights=w_cat1, density=True)[:2]
        hist_preds0, hist_preds0_unc = modplot.calc_hist(preds_cat0, bins=bins_preds, weights=w_cat0, density=True)[:2]
        # Estimated likelihood ratio based on the classifier
        f_r, f_r_unc = divide_histograms(hist_preds1, hist_preds0, hist_preds1_unc, hist_preds0_unc)

        # get reweighting factor for each event by looking up f_r using preds
        r = f_r[np.digitize(preds, bins_preds)-1]

        if figname is not None:
            logger.info("Plot likelihood ratio as a function the model output: {}".format(figname))
            plot_LR_func(figname, bins_preds, f_r, f_r_unc)

        return r

    def _reweight(self, X, Y, w, model, filepath, fitargs, val_data=None):

        val_dict = {'validation_data': val_data} if val_data is not None else {}
        model.fit(X, Y, sample_weight=w, **fitargs, **val_dict)
        model.save_weights(filepath)
        preds = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,1]

        # concatenate validation predictions into training predictions
        if val_data is not None:
            preds_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,1]
            Y_val = val_data[1]
            w_val = val_data[2]

            figname_preds = filepath+'_preds'
            logger.info("Plot model output distribution: {}".format(figname_preds))
            plot_training_vs_validation(figname_preds, preds, Y, w, preds_val, Y_val, w_val)

            preds = np.concatenate((preds, preds_val))
            Y = np.concatenate((Y, Y_val))
            w = np.concatenate((w, w_val))

        r = preds/(1 - preds + 10**-50)
        # The above is copied from the reweight function from orginal OmniFold package with minor modification: https://github.com/ericmetodiev/OmniFold/blob/master/omnifold.py#L15

        # alternatively
        r_alt = self._compute_likelihood_ratio(preds, Y, w, figname=filepath+'_LR')

        # plot the likelihood ratio function
        #plot_graphs(filepath+'_LR', [(preds, r), (preds, r_alt)], labels=['Direct', 'Binned'], xlabel='Prediction (y=1)', ylabel='r')

        # plot likelihood ratio distribution
        figname_rhist = filepath+'_rhist'
        logger.info("Plot likelihood ratio distribution: {}".format(figname_rhist))
        plot_LR_distr(figname_rhist, [r, r_alt], labels=['Direct', 'Binned'])

        if self.binned_rw:
            w *= np.clip(r_alt, fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
        else:
            w *= np.clip(r, fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))

        return w

    def _reweight_step1(self, X, Y, w, model, filepath, fitargs, callbacks=[],
                        val_data=None):
        # add callbacks to fit arguments if provided
        fitargs_step1 = dict(fitargs)
        if callbacks:
            fitargs_step1.setdefault('callbacks',[]).extend(callbacks)

        # the original one
        #return omnifold.reweight(X, Y, w, model, filepath, fitargs_step1, val_data=val_data)
        return self._reweight(X, Y, w, model, filepath, fitargs_step1, val_data=val_data)

    def _reweight_step2(self, X, Y, w, model, filepath, fitargs, callbacks=[],
                        val_data=None):
        # add callbacks to fit arguments if provided
        fitargs_step2 = dict(fitargs)
        if callbacks:
            fitargs_step2.setdefault('callbacks',[]).extend(callbacks)

        # the original one
        #return omnifold.reweight(X, Y, w, model, filepath, fitargs_step2, val_data=val_data)
        return self._reweight(X, Y, w, model, filepath, fitargs_step2, val_data=val_data)

    def _preprocess_det(self, dataset_obs, dataset_sig, dataset_bkg=None, standardize=True):
        """ Set self.X_det, self.Y_det, self.wdata, self.wsig, self.wbkg
        Args:
            dataset_obs, dataset_sig, dataset_bkg: structured numpy array whose field names are variables 
            standardize: bool, if true standardize feature array X
        """
        X_obs, Y_obs, self.wdata = read_dataset(dataset_obs, self.vars_det, label=self.label_obs, weight_name=self.weight_name)
        X_sig, Y_sig, self.wsig = read_dataset(dataset_sig, self.vars_det, label=self.label_sig, weight_name=self.weight_name)

        self.X_det = np.concatenate((X_obs, X_sig))
        self.Y_det = np.concatenate((Y_obs, Y_sig))

        if dataset_bkg is not None:
            X_bkg, Y_bkg, self.wbkg = read_dataset(dataset_bkg, self.vars_det, label=self.label_bkg, weight_name=self.weight_name)
            self.X_det = np.concatenate((self.X_det, X_bkg))
            self.Y_det = np.concatenate((self.Y_det, Y_bkg))

        if standardize: # standardize X
            self.X_det = (self.X_det - np.mean(self.X_det, axis=0)) / np.std(self.X_det, axis=0)

        # make Y categorical
        self.Y_det = tf.keras.utils.to_categorical(self.Y_det)

    def _preprocess_gen(self, dataset_sig, standardize=True):
        """ Set self.X_gen, self.Y_gen
        Args:
            dataset_sig: structured numpy array whose field names are variables 
            standardize: bool, if true standardize feature array X
        """
        X_gen_sig, _, _ = read_dataset(dataset_sig, self.vars_gen, 1)

        self.X_gen = np.concatenate((X_gen_sig, X_gen_sig))

        if standardize: # standardize X
            self.X_gen = (self.X_gen - np.mean(self.X_gen, axis=0)) / np.std(self.X_gen, axis=0)

        # make Y categorical
        nsim = len(X_gen_sig)
        self.Y_gen = tf.keras.utils.to_categorical(np.concatenate([np.ones(nsim), np.zeros(nsim)]))

    def _rescale_event_weights(self):
        # standardize data sample weights
        self.wdata = self.wdata / np.mean(self.wdata)
        ndata = len(self.wdata)

        # rescale simulation sample weights
        sumweights_sim = self.wsig.sum()
        if self.wbkg is not None:
            sumweights_sim += self.wbkg.sum()

        self.wsig = self.wsig / sumweights_sim * ndata
        if self.wbkg is not None:
            self.wbkg = self.wbkg / sumweights_sim * ndata

        logger.debug("wdata.sum() = {}".format(self.wdata.sum()))
        logger.debug("wsig.sum() = {}".format(self.wsig.sum()))
        if self.wbkg is not None:
            logger.debug("wbkg.sum() = {}".format(self.wbkg.sum()))

    def _get_reco_distributions(self, bins, arr_obs, arr_sig, arr_bkg=None):
        # observed distributions
        hist_obs, hist_obs_unc = modplot.calc_hist(arr_obs, weights=self.wdata, bins=bins, density=False)[:2]

        # signal simulation
        hist_sig, hist_sig_unc = modplot.calc_hist(arr_sig, weights=self.wsig, bins=bins, density=False)[:2]

        # background simulation
        hist_bkg, hist_bkg_unc = None, None
        if arr_bkg is not None:
            #wbkg = self.wbkg if self.wbkg.sum() > 0 else -self.wbkg
            hist_bkg, hist_bkg_unc = modplot.calc_hist(arr_bkg, weights=self.wbkg, bins=bins, density=False)[:2]

        return (hist_obs, hist_obs_unc), (hist_sig, hist_sig_unc), (hist_bkg, hist_bkg_unc)

    def _get_truth_distributions(self, bins, arr_truth, arr_bkg=None):
        hist_truth, hist_truth_unc = modplot.calc_hist(arr_truth, weights=self.wdata, bins=bins, density=False)[:2]

        # subtract background contribution if there is any
        if arr_bkg is not None:
            hist_genbkg, hist_genbkg_unc = modplot.calc_hist(arr_bkg, weights=self.wbkg, bins=bins, density=False)[:2]

            hist_truth, hist_truth_unc = add_histograms(hist_truth, hist_genbkg, hist_truth_unc, hist_genbkg_unc, c1=1., c2=-1.)

        return hist_truth, hist_truth_unc

    def _get_ibu_distributions(self, bins_det, bins_mc, arr_sim, arr_gen, hist_obs, hist_obs_unc, hist_simbkg=None, hist_simbkg_unc=None):
        if hist_simbkg is None:
            return ibu(hist_obs, hist_obs_unc, arr_sim, arr_gen, bins_det, bins_mc, self.wsig, it=self.iterations)
        else:
            # subtract background
            hist_obs_cor, hist_obs_cor_unc = add_histograms(hist_obs, hist_simbkg, hist_obs_unc, hist_simbkg_unc, c1=1., c2=-1.)
            return ibu(hist_obs_cor, hist_obs_cor_unc, arr_sim, arr_gen, bins_det, bins_mc, self.wsig, it=self.iterations)

    def _get_omnifold_distributions(self, bins, arr_gen, arr_genbkg=None):
        hists_of, hists_of_unc = [], []

        for w in self.ws_unfolded:
            h_of, herr_of = modplot.calc_hist(arr_gen, weights=w, bins=bins, density=False)[:2]
            hists_of.append(h_of)
            hists_of_unc.append(herr_of)

        return hists_of, hists_of_unc

    def prepare_inputs(self, dataset_obs, dataset_sig, dataset_bkg=None, standardize=True, plot_corr=True, truth_known=False, reweight_type=None, obs_config={}):
        """ Prepare input arrays for training
        Args:
            dataset_obs, dataset_sig, dataset_bkg: structured numpy array whose field names are variables
            standardize: bool, if true standardize feature array X
        """
        # detector-level variables (for step 1 reweighting)
        self._preprocess_det(dataset_obs, dataset_sig, dataset_bkg, standardize)
        logger.info("Total number of observed events: {}".format(len(self.wdata)))
        logger.info("Total number of simulated signal events: {}".format(len(self.wsig)))
        if self.wbkg is not None:
            logger.info("Total number of simulated background events: {}".format(len(self.wbkg)))
        logger.info("Feature array X_det size: {:.3f} MB".format(self.X_det.nbytes*2**-20))
        logger.info("Label array Y_det size: {:.3f} MB".format(self.Y_det.nbytes*2**-20))

        # generator-level variables (for step 2 reweighting)
        self._preprocess_gen(dataset_sig, standardize)
        logger.info("Feature array X_gen size: {:.3f} MB".format(self.X_gen.nbytes*10**-6))
        logger.info("Label array Y_gen size: {:.3f} MB".format(self.Y_gen.nbytes*10**-6))

        # event weights
        # reweight data if needed e.g. for stress tests
        if reweight_type is not None:
            self.wdata = reweight_sample(self.wdata, dataset_obs, obs_config, reweight_type)

        # normalize the total weights
        self._rescale_event_weights()

        ######################
        # plot input variables
        # correlations
        if plot_corr:
            logger.info("Plot input variable correlations")
            plot_correlations(dataset_obs, self.vars_det, os.path.join(self.outdir, 'correlations_det_obs'))
            plot_correlations(dataset_sig, self.vars_det, os.path.join(self.outdir, 'correlations_det_sig'))
            plot_correlations(dataset_sig, self.vars_gen, os.path.join(self.outdir, 'correlations_gen_sig'))

    def unfold(self, fitargs, val=0.2):
        # initialize the truth weights to the prior
        ws_t = [self.wsig]
        ws_m = [self.wsig]

        # split dataset for training and validation
        # detector level
        splitter_det = DataShufflerDet(len(self.X_det), val)
        X_det_train, X_det_val = splitter_det.shuffle_and_split(self.X_det)
        Y_det_train, Y_det_val = splitter_det.shuffle_and_split(self.Y_det)

        # simulation
        splitter_gen = DataShufflerGen(len(self.X_gen), val)
        X_gen_train, X_gen_val = splitter_gen.shuffle_and_split(self.X_gen)
        Y_gen_train, Y_gen_val = splitter_gen.shuffle_and_split(self.Y_gen)

        # model filepath
        model_dir = os.path.join(self.outdir, 'Models')
        if not os.path.isdir(model_dir):
            logger.info("Create directory {}".format(model_dir))
            os.makedirs(model_dir)

        model_det_fp = os.path.join(model_dir, 'model_step1_{}')
        model_sim_fp = os.path.join(model_dir, 'model_step2_{}')

        # start iterations
        for i in range(self.iterations):

            # set up models for this iteration
            model_det, cb_det = self._set_up_model_det_i(i, model_det_fp)
            model_sim, cb_sim = self._set_up_model_sim_i(i, model_sim_fp)

            # step 1: reweight sim to look like data
            # push the latest truth-level weights to the detector level
            wm_push_i = ws_t[-1] # for i=0, this is self.wsig
            w = np.concatenate((self.wdata, wm_push_i))
            if self.wbkg is not None:
                w = np.concatenate((w, self.wbkg))
            assert(len(w)==len(self.X_det))

            w_train, w_val = splitter_det.shuffle_and_split(w)

            rw = self._reweight_step1(X_det_train, Y_det_train, w_train, model_det, model_det_fp.format(i), fitargs, cb_det, val_data=(X_det_val, Y_det_val, w_val))

            wnew = splitter_det.unshuffle(rw)
            if self.wbkg is not None:
                wnew = wnew[len(self.wdata):-len(self.wbkg)]
            else:
                wnew = wnew[len(self.wdata):]

            # rescale the new weights to the original one
            wnew *= (self.wdata.sum()/wnew.sum())
            logger.debug("ws_m.sum() = {}".format(wnew.sum()))

            ws_m.append(wnew)

            # step 2: reweight the simulation prior to the learned weights
            # pull the updated detector-level weights back to the truth level
            wt_pull_i = ws_m[-1]
            w = np.concatenate((wt_pull_i, ws_t[-1]))
            w_train, w_val = splitter_gen.shuffle_and_split(w)

            rw = self._reweight_step2(X_gen_train, Y_gen_train, w_train, model_sim, model_sim_fp.format(i), fitargs, cb_sim, val_data=(X_gen_val, Y_gen_val, w_val))

            wnew = splitter_gen.unshuffle(rw)[len(ws_t[-1]):]

            # rescale the new weights to the original one
            wnew *= (self.wsig.sum()/wnew.sum())
            logger.debug("ws_t.sum() = {}".format(wnew.sum()))

            ws_t.append(wnew)

        self.ws_unfolded = ws_t
        #self.ws_unfolded = [w * self.wsig.sum() / w.sum() for w in ws_t]
        logger.debug("unfolded_weights.sum() = {}".format(self.ws_unfolded[-1].sum()))

        # save the weights
        weights_file = self.outdir.rstrip('/')+'/weights.npz'
        np.savez(weights_file, weights = self.ws_unfolded)

    def set_weights_from_file(self, weights_file, array_name='weights'):
        wfile = np.load(weights_file)
        ws_t = wfile[array_name]
        wfile.close()
        self.ws_unfolded = ws_t

    def results(self, vars_dict, dataset_obs, dataset_sig, dataset_bkg=None, binning_config='', truth_known=False, normalize=False, plot_iterations=True):
        """
        Args:
            vars_dict:
            dataset_obs/sig/bkg: structured numpy array labeled by variable names
        Return:
            
        """
        # directory to save the iteration history if needed
        iteration_dir = os.path.join(self.outdir, 'Iterations')
        if plot_iterations:
            if not os.path.isdir(iteration_dir):
                logger.info("Create directory {}".format(iteration_dir))
                os.makedirs(iteration_dir)

        for varname, config in vars_dict.items():
            logger.info("Unfold variable: {}".format(varname))
            dataobs = np.hstack(get_variable_arr(dataset_obs,config['branch_det']))
            truth = np.hstack(get_variable_arr(dataset_obs,config['branch_mc'])) if truth_known else None

            sim_sig = np.hstack(get_variable_arr(dataset_sig,config['branch_det']))
            gen_sig = np.hstack(get_variable_arr(dataset_sig,config['branch_mc']))
            sim_bkg = np.hstack(get_variable_arr(dataset_bkg,config['branch_det'])) if dataset_bkg is not None else None
            gen_bkg = np.hstack(get_variable_arr(dataset_bkg,config['branch_mc'])) if dataset_bkg is not None else None

            # histograms
            # set up bins
            bins_det = get_bins(varname, binning_config)
            if bins_det is None:
                bins_det = np.linspace(config['xlim'][0], config['xlim'][1], config['nbins_det']+1)

            bins_mc = get_bins(varname, binning_config)
            if bins_mc is None:
                bins_mc = np.linspace(config['xlim'][0], config['xlim'][1], config['nbins_mc']+1)

            ###########################
            # detector-level distributions
            histograms_reco = self._get_reco_distributions(bins_det, dataobs, sim_sig, sim_bkg)
            # observed distributions
            hist_obs, hist_obs_unc = histograms_reco[0]
            # signal simulation
            hist_sim, hist_sim_unc = histograms_reco[1]
            # background simulation
            hist_simbkg, hist_simbkg_unc = histograms_reco[2]

            # plot detector-level variable distributions
            figname_vardet = os.path.join(self.outdir, 'Reco_{}'.format(varname))
            logger.info("  Plot detector-level variable distribution: {}".format(figname_vardet))
            plot_reco_variable(bins_det,
                               (hist_obs,hist_obs_unc), (hist_sim,hist_sim_unc),
                               (hist_simbkg, hist_simbkg_unc),
                               figname=figname_vardet, log_scale = False,
                               **config)

            ###########################
            # generated distribution (prior)
            hist_gen, hist_gen_unc = modplot.calc_hist(gen_sig, weights=self.wsig, bins=bins_mc, density=False)[:2]
            # normalization if needed
            if normalize:
                normalize_histogram(bins_mc, hist_gen, hist_gen_unc)

            # truth distribution if known
            if truth is not None:
                hist_truth, hist_truth_unc = self._get_truth_distributions(bins_mc, truth, gen_bkg)
            else:
                hist_truth, hist_truth_unc = None, None
            # normalization if needed
            if normalize:
                normalize_histogram(bins_mc, hist_truth, hist_truth_unc)

            # unfolded distributions
            # iterative Bayesian unfolding
            hists_ibu, hists_ibu_unc, response = self._get_ibu_distributions(
                bins_det, bins_mc, sim_sig, gen_sig, hist_obs, hist_obs_unc,
                hist_simbkg, hist_simbkg_unc)
            # normalization if needed
            if normalize:
                for h_ibu, h_ibu_err in zip(hists_ibu, hists_ibu_unc):
                    normalize_histogram(bins_mc, h_ibu, h_ibu_err)

            # plot response matrix
            rname = os.path.join(self.outdir, 'Response_{}'.format(varname))
            logger.info("  Plot detector response: {}".format(rname))
            plot_response(rname, response, bins_det, bins_mc, varname)

            # omnifold
            hists_of, hists_of_unc = self._get_omnifold_distributions(bins_mc, gen_sig, gen_bkg)
            # normalization if needed
            if normalize:
                for h_of, h_of_err in zip(hists_of, hists_of_unc):
                    normalize_histogram(bins_mc, h_of, h_of_err)

            # compute the differences
            text_td = []
            if truth is not None:
                #text_td = write_triangular_discriminators(hist_truth, [hists_of[-1], hists_ibu[-1], hist_gen], labels=['OmniFold', 'IBU', 'Prior'])
                text_td = write_chi2(hist_truth, hist_truth_unc, [hists_of[-1], hists_ibu[-1], hist_gen], [hists_of_unc[-1], hists_ibu_unc[-1], hist_gen_unc], labels=['OmniFold', 'IBU', 'Prior'])
                logger.info("  "+"    ".join(text_td))

            # plot results
            figname = os.path.join(self.outdir, 'Unfold_{}'.format(varname))
            logger.info("  Plot unfolded distribution: {}".format(figname))
            plot_results(bins_mc, (hist_gen, hist_gen_unc),
                         (hists_of[-1], hists_of_unc[-1]),
                         (hists_ibu[-1], hists_ibu_unc[-1]),
                         (hist_truth, hist_truth_unc),
                         figname=figname, texts=text_td, **config)

            if plot_iterations:
                figname_prefix = os.path.join(iteration_dir, varname)
                plot_iteration_distributions(figname_prefix+"_IBU_iterations", bins_mc, hists_ibu, hists_ibu_unc, **config)
                plot_iteration_distributions(figname_prefix+"_OmniFold_iterations", bins_mc, hists_of, hists_of_unc, **config)
                plot_iteration_diffChi2s(figname_prefix+"_diffChi2s", [hists_ibu, hists_of], [hists_ibu_unc, hists_of_unc], labels=["IBU", "OmniFold"])
                if truth_known:
                    plot_iteration_chi2s(figname_prefix+"_chi2s_wrt_Truth", hist_truth, hist_truth_unc, [hists_ibu, hists_of], [hists_ibu_unc, hists_of_unc], labels=["IBU", "OmniFold"])

##############################################################################
#############
# Approach 1
#############
# unfold as is first, then subtract the background histogram from the unfolded distribution for any observable of interest.

# preprocess_data: data vs mc signal only w/o background
# classifier: data vs signal mc
# reweight: standard
# show result: subtract background histograms

class OmniFold_subHist(OmniFoldwBkg):
    def __init__(self, variables_det, variables_gen, wname, it, outdir='./'):
        super().__init__(variables_det, variables_gen, wname, it, outdir)

    def _preprocess_det(self, dataset_obs, dataset_sig, dataset_bkg=None, standardize=True):
        # exclude background events at this step
        super()._preprocess_det(dataset_obs, dataset_sig, None, standardize)
        # self.wbkg is None

    def _get_omnifold_distributions(self, bins, arr_gen, arr_genbkg=None):
        hists_of, hists_of_unc = [], []

        for w in self.ws_unfolded:
            h_of, herr_of = modplot.calc_hist(arr_gen, weights=w, bins=bins, density=False)[:2]

            # in case of background
            if arr_genbkg is not None:
                hist_genbkg, hist_genbkg_unc = modplot.calc_hist(arr_genbkg, weights=self.wbkg_gen, bins=bins, density=False)[:2]

                # subtract background
                h_of, herr_of = add_histograms(h_of, hist_genbkg, herr_of, hist_genbkg_unc, c1=1., c2=-1.)

            hists_of.append(h_of)
            hists_of_unc.append(herr_of)

        return hists_of, hists_of_unc

    def results(self, vars_dict, dataset_obs, dataset_sig, dataset_bkg, truth_known=False, normalize=False):
        # set self.wbkg properly and rescale self.wsig accordingly
        # This is needed because background was ignored during data preparation steps
        self.wbkg = np.hstack(dataset_bkg[self.weight_name])
        self._rescale_event_weights()

        super().results(vars_dict, dataset_obs, dataset_sig, dataset_bkg, truth_known, normalize)
    
#############
# Approach 2
#############
# unfold as is, but set the background event weights to be negative

# preprocess_data: set background weights to be negative
# background label is the same as data
# detector level (step 1 reweighting)
# show result: standard

class OmniFold_negW(OmniFoldwBkg):
    def __init__(self, variables_det, variables_gen, wname, it, outdir='./'):
        super().__init__(variables_det, variables_gen, wname, it, outdir)
        # make background label same as data
        self.label_bkg = self.label_obs

    def _rescale_event_weights(self):
        super()._rescale_event_weights()

        # negate background weights
        self.wbkg = -self.wbkg

    def results(self, vars_dict, dataset_obs, dataset_sig, dataset_bkg, truth_known=False, normalize=False):
        # flip the sign of background weights back first
        self.wbkg = -self.wbkg

        # proceed as usual
        super().results(vars_dict, dataset_obs, dataset_sig, dataset_bkg, truth_known=truth_known, normalize=normalize)
    
#############
# Approach 3
#############
# add a correction term to the ratio for updating weights in step 1

# preprocess_data: standard w/ signal + background mc
# classifier: data vs mc
# reweight: correct the original weight from classifier
# show result: standard

class OmniFold_corR(OmniFoldwBkg):
    def __init__(self, variables_det, variables_gen, wname, outdir='./'):
        OmniFoldwBkg.__init__(self, variables_det, variables_gen, wname, outdir)

    # redefine step 1 reweighting
    def _reweight_step1(self, X, Y, w, model, filepath, fitargs, val_data=None):
        # validation data
        val_dict = {'validation_data': val_data} if val_data is not None else {}
        model.fit(X, Y, sample_weight=w, **fitargs, **val_dict)
        model.save_weights(filepath)
        preds = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,1]

        # concatenate validation predictions into training predictions
        if val_data is not None:
            preds_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,1]
            preds = np.concatenate((preds, preds_val))
            w = np.concatenate((w, val_data[2]))

        r = preds/(1 - preds + 10**-50)
        # correction term
        # FIXME
        nbkg = self.wbkg.sum() 
        nsig = self.wdata.sum() - nbkg
        cor = (r-1)*nbkg/nsim
        r += cor
        w *= np.clip(r, fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
        return w
    
#############
# Approach 4
#############
# approximate likelihood ratio of data vs signal and background vs signal separately with separate classifiers

# preprocess_data: (data, signal_mc) and (background_mc, signal_mc) separately
# classifier: data vs signal, background vs signal
# reweight: weight from data_vs_signal minus weight from background_vs_signal
# show result: standard
    
#############
# Approach 5
#############
# use a multi-class classifier to approximate the likelihood ratio of signal events in data and signal mc

# preprocess_data: standard w/ background mc
# classifier: multi-class
# reweight: new_weight = old_weight * (y_data - y_bkg) / y_sig
# show result: standard

class OmniFold_multi(OmniFoldwBkg):
    def __init__(self, variables_det, variables_gen, wname, it, outdir='./'):
        super().__init__(variables_det, variables_gen, wname, it, outdir)

        # new label for background
        self.label_bkg = 2

    # multi-class classifier for step 1 reweighting
    def _set_up_model_det_i(self, i, model_filepath=None):
        # input dimension
        input_shape = self.X_det.shape[1:]

        # get model
        model = get_model(input_shape, nclass=3)

        # callbacks
        callbacks = get_callbacks(model_filepath.format(i))

        # load weights from the previous iteration if not the first one
        if i > 0 and model_filepath is not None:
            model.load_weights(model_filepath.format(i-1))

        return model, callbacks

    # reweighting with multi-class classifer
    def _reweight_step1(self, X, Y, w, model, filepath, fitargs, callbacks=[],
                        val_data=None):
        # add callbacks to fit arguments
        fitargs_step1 = dict(fitargs)
        if callbacks:
            fitargs_step1.setdefault('callbacks',[]).extend(callbacks)

        val_dict = {'validation_data': val_data} if val_data is not None else {}

        model.fit(X, Y, sample_weight=w, **fitargs_step1, **val_dict)
        model.save_weights(filepath)

        preds_obs = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,self.label_obs]
        preds_sig = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,self.label_sig]

        # concatenate validation predictions
        if val_data is not None:
            preds_obs_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,self.label_obs]
            preds_sig_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,self.label_sig]
            preds_obs = np.concatenate((preds_obs, preds_obs_val))
            preds_sig = np.concatenate((preds_sig, preds_sig_val))
            w = np.concatenate((w, val_data[2]))

        r = preds_obs / preds_sig

        w *= np.clip(r, fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
        return w
