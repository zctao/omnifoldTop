import os
import numpy as np
import pandas as pd

from histogramming import calc_hist, calc_hist2d, get_hist, set_hist_errors

import plotting
import logging
logger = logging.getLogger('IBU')
logger.setLevel(logging.DEBUG)

class IBU(object):
    def __init__(self, varname, bins_det, bins_mc, obs, sim, gen, simbkg=None, wobs=1., wsig=1., wbkg=1., iterations=4, nresample=25, outdir='.'):
        # variable name
        self.varname = varname
        # bin edges
        self.bins_det = bins_det # detector level
        self.bins_mc = bins_mc # truth level
        # ndarray of variable in observed data
        self.array_obs = obs
        # ndarray of variable in simulation at the detector level
        self.array_sim = sim
        # ndarray of variable in simulation at the truth level
        self.array_gen = gen
        # ndarray of variable in background simulation at the detector level
        self.array_bkg = simbkg
        # event weights
        self.weights_obs = wobs
        self.weights_sig = wsig
        self.weights_bkg = wbkg
        # number of iterations
        self.iterations = iterations
        # number of resamples for uncertainty calculation
        self.nresamples = nresample
        # output directory
        self.outdir = outdir
        # unfolded distributions
        self.hists_unfolded = None
        self.hists_unfolded_corr = None

    def run(self):
        # response matrix
        r = self._response_matrix(self.weights_sig)
        figname = os.path.join(self.outdir, 'Response_{}'.format(self.varname))
        logger.info("  Plot detector response: {}".format(figname))
        plotting.plot_response(figname, r, self.varname)

        # unfold
        self.hists_unfolded = self._unfold(r, self.weights_obs, self.weights_sig, self.weights_bkg)

        # bin uncertainty and correlation
        unfolded_err, self.hists_unfolded_corr = self._uncertainty(
            self.nresamples, response=r, resample_obs=True, resample_sig=False)

        set_hist_errors(self.hists_unfolded, unfolded_err)

    def get_unfolded_distribution(self, all_iterations=False):
        if all_iterations:
            return self.hists_unfolded, self.hists_unfolded_corr
        else:
            return self.hists_unfolded[-1], self.hists_unfolded_corr[-1]

    def _response_matrix(self, weights_sim):
        r = calc_hist2d(self.array_sim, self.array_gen, bins=(self.bins_det, self.bins_mc), weights=weights_sim)
        # normalize per truth bin
        #r.view()['value'] = r.values() / r.project(1).values()
        r.view()['value'] = r.values() / r.values().sum(axis=0)

        return r

    def _unfold(self, response, weights_obs, weights_sig, weights_bkg=None):
        ######
        # detector level
        # observed distribution
        h_obs = calc_hist(self.array_obs, self.bins_det, weights=weights_obs)

        # if background is not none, subtract background
        if self.array_bkg is not None:
            h_bkg = calc_hist(self.array_bkg, self.bins_det, weights=weights_bkg)
            h_obs  = h_obs + (-1*h_bkg)

        ######
        # truth level
        # prior distribution
        h_prior = calc_hist(self.array_gen, self.bins_mc, weights=weights_sig)

        # bin widths
        wbins_det = self.bins_det[1:] - self.bins_det[:-1]
        wbins_mc = self.bins_mc[1:] - self.bins_mc[:-1]

        # start iterations
        hists_ibu = [h_prior]

        for i in range(self.iterations):
            # update the estimate given the response matrix and the latest unfolded distribution
            m = response.values() * hists_ibu[-1].values()
            m /= (m.sum(axis=1)[:,np.newaxis] + 10**-50)

            # update the unfolded given m and the observed distribution
            entry_ibu = np.dot(m.T, h_obs.values())*wbins_det/wbins_mc
            h_ibu = get_hist(self.bins_mc, entry_ibu)
            hists_ibu.append(h_ibu)

        return hists_ibu[1:] # shape: (n_iteration, nbins_hist)

    def _uncertainty(self, nresamples, response=None, resample_obs=True, resample_sig=True):
        hists_resample = []

        for iresample in range(nresamples):
            # resample the weights
            reweights_obs = self.weights_obs * np.random.poisson(1, size=len(self.array_obs)) if resample_obs else self.weights_obs
            reweights_sig = self.weights_sig * np.random.poisson(1, size=len(self.array_gen)) if resample_sig else self.weights_sig

            # recompute response with resampled simulation weights if needed
            if response is None:
                response = self._response_matrix(reweights_sig)

            hists_resample.append(self._unfold(response, reweights_obs, reweights_sig, self.weights_bkg))

        # standard deviation of each bin
        hists_resample = np.asarray(hists_resample)['value']
        errors = np.std(hists_resample, axis=0, ddof=1) # shape: (n_iteration, nbins_hist)

        # bin correlations
        corrs = []
        # np.asarray(hists_resample) shape: (n_resamples, n_iterations, n_bins)
        # for each iteration
        for i in range(self.iterations):
            df_ihist = pd.DataFrame(hists_resample[:,i,:])
            corrs.append(df_ihist.corr())
            #assert( np.allclose(np.sqrt(np.asarray([df_ihist.cov()[i][i] for i in range(len(df_ihist.columns))])), errors[i]) ) # sanity check

        return errors, corrs
