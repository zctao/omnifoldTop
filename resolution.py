import numpy as np

from plotting import plot_graphs

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
            #resol_unc_arr.append(0)
            continue

        # estimate resolution for this truth bin
        # estimated sample mean
        reco_mean = np.average(midbins_x, weights=hist_reco)
        # estiamted sample variance
        # (assume total number of events is much larger than 1)
        reco_var = np.average((midbins_x - reco_mean)**2, weights=hist_reco)
        # standard deviation
        reco_res = np.sqrt(reco_var)
        resol_arr.append(reco_res)
        # variance of the sample variance TODO
        #resol_unc_arr.append()

        # alternatively, fit hist_reco with a gaussian to estimate sigma

    assert(len(resol_arr)==len(midbins_y))

    if figname:
        # plot the resolution vs truth value
        plot_graphs(figname, [(midbins_y, resol_arr)],
                    #error_arrays=[resol_unc_arr],
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
