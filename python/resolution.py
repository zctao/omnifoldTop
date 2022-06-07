import numpy as np
from scipy.optimize import curve_fit

from plotter import plot_graphs

from util import getLogger
logger = getLogger('Resolution', level=20)

def resolution(reco_arr, truth_arr, weights, bins_reco, bins_truth, figname=None, xlabel=''):
    """
    compute resolutions given reco and truth variables
    return a function that outputs resolution for a given truth value
    """
    # migration matrix
    hist2d, xbins, ybins = np.histogram2d(reco_arr, truth_arr, bins=(bins_reco, bins_truth), weights=weights)
    # bin errors?

    # look at the reco variable (x) distribution in each slice of truth value (y)
    midbins_y = (ybins[:-1] + ybins[1:]) / 2.
    midbins_x = (xbins[:-1] + xbins[1:]) / 2.

    resol_arr = []
    resol_unc_arr = [] # uncertainties

    for ybin in range(len(bins_truth)-1):
        value_truth = midbins_y[ybin]
        hist_reco = hist2d[:,ybin]
        #hist_reco_err = hist2d_err[:,ybin]

        # in case there is no entries in the histogram
        if hist_reco.sum() == 0:
            # unable to get resolution
            resol_arr.append(0)
            resol_unc_arr.append(0)
            continue

        # estimate resolution from the reco histogram for this truth bin
        #reco_res, reco_res_unc = get_hist_sigma(midbins_x, hist_reco)
        # or fit hist_reco with a gaussian to estimate sigma
        reco_res, reco_res_unc = get_hist_sigma_fit(midbins_x, hist_reco)

        resol_arr.append(reco_res)
        resol_unc_arr.append(reco_res_unc)

    assert(len(resol_arr)==len(midbins_y))
    assert(len(resol_unc_arr)==len(midbins_y))

    if figname:
        # plot the resolution vs truth value
        plot_graphs(figname, [(midbins_y, resol_arr)],
                    error_arrays=[resol_unc_arr],
                    xlabel=xlabel, ylabel='Resolution')

    def f_resol(truth_value):
        # first determine which bin the value is in
        ibin = np.digitize(truth_value, ybins) - 1
        # read resolution from the array
        resol = resol_arr[ibin]
        #resol_unc = resol_unc_arr[ibin]
        return resol#, resol_unc

    # alternatively, fit the resolution vs truth value with a polynomial
    # and return the fitted function

    return f_resol

def get_hist_sigma(midbins, hist, hist_err=None):
    # estimated sample mean
    sample_mean = np.average(midbins, weights=hist)
    #sample_mean = sum(midbins*hist)/sum(hist)

    # estimated sample variance (assume large sample number)
    sample_var = np.average((midbins - sample_mean)**2, weights=hist)
    #sample_var = sum(hist * (midbins - sample_mean)**2)/sum(hist)
    # standard deviation
    sample_std = np.sqrt(sample_var)

    # variance of sample variance TODO
    #sample_std_unc = (sum(hist*(midbins - sample_mean)**4)/sum(hist) - sample_var) / sum(hist)
    sample_std_unc = 0.

    return sample_std, sample_std_unc

def get_hist_sigma_fit(midbins, hist, hist_err=None):
    # initial value
    mean0 = sum(midbins*hist)/sum(hist)
    sigma0 = np.sqrt(sum(hist * (midbins - mean0)**2)/sum(hist))

    # fit
    try:
        #popt, pcov = curve_fit(gauss, midbins, hist, p0=[max(hist), mean0, sigma0], bounds=([0.,-np.inf,0.], np.inf), sigma=hist_err)
        popt, pcov = curve_fit(gauss, midbins, hist, p0=[max(hist), mean0, sigma0], bounds=([-np.inf,-np.inf,0.], np.inf), sigma=hist_err)

        A, mu, sigma = popt
        A_err, mu_err, sigma_err = np.sqrt(np.diag(pcov))
    except RuntimeError as e:
        logger.warn("Fit failed with message: {}".format(e))
        logger.warn("Use initial value estiamted from sample: sigma = {}".format(sigma0))
        logger.debug("mean0 = {}".format(mean0))
        logger.debug("histogram = {}".format(hist))
        sigma = sigma0
        sigma_err = sigma0

    return sigma, sigma_err

def gauss(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))
