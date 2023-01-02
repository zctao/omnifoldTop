import os
import numpy as np
import pandas as pd
import hist

import histogramming as myhu

def _response_matrix(array_sim, array_gen, bins_reco, bins_truth, weights):

    r = myhu.calc_hist2d(
        array_sim, array_gen, bins=(bins_reco, bins_truth), weights=weights,
        density=False
    )

    # normalize per truth bin
    #r.view()['value'] = r.values() / r.project(1).values()
    #r.view()['value'] = r.values() / r.values().sum(axis=0)
    r_normed = np.zeros_like(r.values())
    np.divide(r.values(), r.values().sum(axis=0), out=r_normed, where=r.values().sum(axis=0)!=0)

    r.view()['value'] = r_normed

    return r

def _get_obs_distribution(
    bins, array_obs, array_bkg = None,
    weights_obs=1., weights_bkg=1.,
    bootstrap=False
    ):
    if bootstrap:
        weights_obs = weights_obs * np.random.poisson(1, size=len(weights_obs))

    h_obs = myhu.calc_hist(array_obs, bins, weights=weights_obs, density=False)

    # if there is background, subtract it
    if array_bkg is not None:
        h_bkg = myhu.calc_hist(array_bkg, bins, weights=weights_bkg, density=False)
        h_obs = h_obs + (-1 * h_bkg)

    return h_obs

def _unfold(
    response,
    h_obs,
    h_prior,
    niterations,
    acceptance_correction=None,
    efficiency_correction=None
    ):

    # apply acceptance correction if available
    if acceptance_correction is not None:
        # in case the acceptance correction histogram has different binning
        # get the correction factors using h_obs bin center
        f_acc = np.array([ acceptance_correction[hist.loc(c)].value for c in h_obs.axes[0].centers ])
        h_obs = h_obs * f_acc

    # bin widths
    wbins_reco = h_obs.axes[0].widths # reco level
    wbins_truth = h_prior.axes[0].widths # truth level
    bins_truth = h_prior.axes[0].edges

    hists_unfold = [h_prior]

    for i in range(niterations):
        m = response.values() * hists_unfold[-1].values()
        m /= (m.sum(axis=1)[:,np.newaxis] + 10**-50)

        i_unfold = np.dot(m.T, h_obs.values())
        h_ibu = myhu.get_hist(bins_truth, i_unfold)
        hists_unfold.append(h_ibu)

    # exclude h_prior
    hists_unfold = hists_unfold[1:]

    # FIXME
    if efficiency_correction is not None:
        # apply efficiency correction
        f_eff = np.array([ 1. / efficiency_correction[hist.loc(c)].value for c in h_obs.axes[0].centers ])

        for huf in hists_unfold:
            huf *= f_eff

    return hists_unfold # shape: (niterations, )

def run_ibu(
    bins_reco, bins_truth, # binning for reco and truth level variable
    array_data, # array of observed data
    array_sim, # array of signal simulation reco level
    array_gen, # array of signal truth level MC
    array_bkg = None, # array of background simulation at reco level
    weights_data = 1., # event weights of observed data
    weights_sim = 1., # event weights of signal MC at reco level
    weights_gen = 1., # event weights of signal MC at truth level
    weights_bkg = 1., # event weights of background MC at reco level
    niterations = 4, # number of iterations
    nresamples = 25, # number of resamples for uncertainty estimation
    acceptance_correction = None, # histogram for acceptance correction
    efficiency_correction = None, # histogram for efficiency correction
    flow = True, # If False, exclude underflow/overflow bins
    all_iterations = False, # if True, return results at every iteration
    density = False, # if True, normalize the final histograms by its bin widths
    norm = None
    ):

    # Response matrix
    r = _response_matrix(
        array_sim, array_gen, bins_reco, bins_truth, weights = weights_sim
    )

    # Histograms
    # observed reco level
    h_obs = _get_obs_distribution(
        bins_reco, array_data, array_bkg, weights_data, weights_bkg
    )

    # signal MC truth level prior
    r2d = myhu.calc_hist2d(
        array_sim, array_gen, bins=(bins_reco, bins_truth), weights=weights_sim,
        density=False
    )
    h_prior = myhu.projectToYaxis(r2d, flow=flow)

    # In case flow=True, the following (what was done previously) is equivalent
    #h_prior = myhu.calc_hist(array_gen, bins_truth, weights=weights_sim, density=False)

    #[h_prior = myhu.calc_hist(array_gen, bins_truth, weights=weights_gen, density=False)]

    # unfolded distribution
    hists_ibu = _unfold(r, h_obs, h_prior, niterations,
                        acceptance_correction, efficiency_correction)

    # compute bin errors and correlation
    hists_ibu_resample = []

    for iresample in range(nresamples):

        h_obs_rs = _get_obs_distribution(
            bins_reco, array_data, array_bkg, weights_data, weights_bkg,
            bootstrap = True)

        hists_ibu_resample.append(
            _unfold(r, h_obs_rs, h_prior, niterations,
                    acceptance_correction, efficiency_correction)
        )

    # standard deviation of each bin
    bin_errors = myhu.get_sigma_from_hists(hists_ibu_resample) # shape: (niterations, nbins_hist)

    # set error
    myhu.set_hist_errors(hists_ibu, bin_errors)

    # normalization
    if norm is not None:
        for h in hists_ibu:
            h = myhu.renormalize_hist(h, norm, density=False)

    if density:
        # divide the histogram by its bin widths
        for h in hists_ibu:
            h /= myhu.get_hist_widths(h)

    # bin correlations
    bin_corr = myhu.get_bin_correlations_from_hists(hists_ibu_resample)

    # Return results
    if all_iterations:
        return hists_ibu, bin_corr, r
    else:
        return hists_ibu[-1], bin_corr[-1], r
