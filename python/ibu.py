import os
import numpy as np

import external.OmniFold.modplot as modplot
import plotting
from util import getLogger, add_histograms

logger = getLogger('IBU')

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
        self.hists_unfolded_err = None

    def run(self):
        # response matrix
        r = self._response_matrix(self.weights_sig, plot=True)

        # unfold
        self.hists_unfolded = self._unfold(r, self.weights_obs, self.weights_sig, self.weights_bkg)

        # uncertainty
        self.hists_unfolded_err = self._uncertainty(self.nresamples, response=r,
                                                    resample_obs=True,
                                                    resample_sig=False)

    def get_unfolded_distribution(self, all_iterations=False):
        if all_iterations:
            return self.hists_unfolded, self.hists_unfolded_err
        else:
            return self.hists_unfolded[-1], self.hists_unfolded_err[-1]

    def _response_matrix(self, weights_sim, plot=True):
        r = np.histogram2d(self.array_sim, self.array_gen, bins=(self.bins_det, self.bins_mc), weights=weights_sim)[0]
        r /= (r.sum(axis=0) + 10**-50)

        if plot:
            figname = os.path.join(self.outdir, 'Response_{}'.format(self.varname))
            logger.info("  Plot detector response: {}".format(figname))
            plotting.plot_response(figname, r, self.bins_det, self.bins_mc, self.varname)

        return r

    def _unfold(self, response, weights_obs, weights_sig, weights_bkg=None):
        ######
        # detector level
        # observed distribution
        hist_obs, hist_obs_err = modplot.calc_hist(self.array_obs, weights=weights_obs, bins=self.bins_det, density=False)[:2]

        # if background is not none, subtract background
        if self.array_bkg is not None:
            hist_bkg, hist_bkg_err = modplot.calc_hist(self.array_bkg, weights=weights_bkg, bins=self.bins_det)[:2]
            hist_obs, hist_obs_err = add_histograms(hist_obs, hist_bkg, hist_obs_err, hist_bkg_err, c1=1., c2=-1.)

        ######
        # truth level
        # prior distribution
        hist_prior, hist_prior_err = modplot.calc_hist(self.array_gen, weights=weights_sig, bins=self.bins_mc)[:2]

        # bin widths
        wbins_det = self.bins_det[1:] - self.bins_det[:-1]
        wbins_mc = self.bins_mc[1:] - self.bins_mc[:-1]

        # start iterations
        hists_ibu = [hist_prior]

        for i in range(self.iterations):
            # update the estimate given the response matrix and the latest unfolded distribution
            m = response * hists_ibu[-1]
            m /= (m.sum(axis=1)[:,np.newaxis] + 10**-50)

            # update the unfolded given m and the observed distribution
            hists_ibu.append(np.dot(m.T, hist_obs)*wbins_det/wbins_mc)

        return hists_ibu # shape: (n_iteration, nbins_hist)

    def _uncertainty(self, nresamples, response=None, resample_obs=True, resample_sig=True):
        hists_resample = []

        for iresample in range(nresamples):
            # resample the weights
            reweights_obs = self.weights_obs * np.random.poisson(1, size=len(self.array_obs)) if resample_obs else self.weights_obs
            reweights_sig = self.weights_sig * np.random.poisson(1, size=len(self.array_gen)) if resampple_sig else self.weights_sig

            # recompute response with resampled simulation weights if needed
            if response is None:
                response = self._response_matrix(reweights_sig, plot=False)

            hists_resample.append(self._unfold(response, reweights_obs, reweights_sig, self.weights_bkg))

        # for now: return the standard deviation, bin-by-bin, as the uncertainty
        # TODO: bin correlations
        return np.std(np.asarray(hists_resample), axis=0) # shape: (n_iteration, nbins_hist)
