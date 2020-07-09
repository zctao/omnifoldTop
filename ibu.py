#########################################################
# iterative Bayesian unfolding
# based on the implementation in https://github.com/ericmetodiev/OmniFold/blob/master/ibu.py
# with some modifications
import numpy as np

import sys
sys.path.append('external')
import OmniFold.modplot as modplot

def response_matrix(datasim, datagen, bins_det, bins_gen):
    r = np.histogram2d(datasim, datagen, bins=(bins_det, bins_gen))[0]
    r /= (r.sum(axis=0) + 10**-50)
    return r

def ibu_core(hist_obs, hist_prior, response, wbin_det, wbin_gen, iteration):
    # initialize the truth distribution to the prior
    phis = [hist_prior]

    # iterate
    for i in range(iteration):
        # update the estimate given the repsonse matrix and the latest truth
        m = response * phis[-1]
        m /= (m.sum(axis=1)[:,np.newaxis] + 10**-50)

        # update the truth distribution given m and the observed histogram
        phis.append(np.dot(m.T, hist_obs)*wbin_det/wbin_gen)

    return phis[-1]

def ibu_unc(hist_obs, hist_prior, hist_prior_unc, response, wbin_det, wbin_gen, iteration, nresamples=50):
    # statistical uncertainty on the IBU distribution
    # (only from the prior for now)
    rephis = []
    for resample in range(nresamples):
        # vary the prior distribution based on the bin uncertainties
        hist_prior_rw = np.random.normal(hist_prior, hist_prior_unc)

        # redo IBU with the new prior
        phi = ibu_core(hist_obs, hist_prior_rw, response, wbin_det, wbin_gen, iteration)

        # record the result
        rephis.append(phi)

    # return the per-bin standard deviation as the uncertainty
    return np.std(np.asarray(rephis), axis=0)

def ibu(hist_obs, datasim, datagen, bins_det, bins_gen, winit, it, density=False, nresample=25):
    binwidth_det = bins_det[1]-bins_det[0]
    binwidth_gen = bins_gen[1]-bins_gen[0]

    # response matrix
    r = response_matrix(datasim, datagen, bins_det, bins_gen)

    # prior distribution
    hist_prior, hist_prior_unc = modplot.calc_hist(datagen, weights=winit, bins=bins_gen, density=density)[:2]

    # do ibu
    hist_ibu = ibu_core(hist_obs, hist_prior, r, binwidth_det, binwidth_gen, it)

    # uncertainty
    hist_ibu_unc = ibu_unc(hist_obs, hist_prior, hist_prior_unc, r, binwidth_det, binwidth_gen, it, nresample)

    return hist_ibu, hist_ibu_unc
